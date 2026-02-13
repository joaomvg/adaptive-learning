#!/usr/bin/env python3
# simulate_recommend_loop.py
#
# Loop:
#   new student -> recommend next question -> get score (manual or simulated) ->
#   write ATTEMPTED -> update MASTERY -> repeat
#
# Requires:
# - Neo4j running with Questions/Terms/Categories loaded
# - model checkpoint from training (model.pt)
#
# Example (manual score input):
#   python simulate_recommend_loop.py \
#     --ckpt outputs/model.pt \
#     --neo4j-uri bolt://localhost:7687 \
#     --neo4j-user neo4j \
#     --neo4j-pass password \
#     --student-id S_NEW_001 \
#     --steps 10 \
#     --exclude-seen
#
# Example (auto simulate scores):
#   python simulate_recommend_loop.py ... --auto-score
#
import argparse
import math
import random
from typing import Dict, List

import numpy as np
import torch
from neo4j import GraphDatabase
from adaptive_engine.config import Config
from adaptive_engine.kg import (
    neo4j_fetch_candidates as fetch_candidates,
    neo4j_fetch_history as fetch_history,
    neo4j_fetch_term_mastery as fetch_mastery,
)
from adaptive_engine.inference import (build_history_tensors,
                                       pack_candidate_batch,
                                       load_checkpoint)


def bucketize_correctness(c: np.ndarray, n_bins: int) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.minimum((c * (n_bins - 1e-6)).astype(np.int64), n_bins - 1)


def recency_feature(delta_days: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 0.0
    return float(math.exp(-math.log(2.0) * max(delta_days, 0.0) / half_life_days))

# -----------------------------
# Neo4j operations
# -----------------------------


def ensure_student(driver, student_id: str):
    with driver.session() as sess:
        sess.run("MERGE (:Student {id:$sid})", sid=student_id)


def pick_target_terms(driver, student_id: str, k: int = 5) -> List[str]:
    # prioritize weak / low-evidence terms
    cypher = """
    MATCH (s:Student {id:$sid})-[m:MASTERY]->(t:Term)
    RETURN t.name AS term, m.mu AS mu, m.n AS n
    """
    with driver.session() as sess:
        rows = sess.run(cypher, sid=student_id).data()

    if not rows:
        cypher2 = "MATCH (t:Term) RETURN t.name AS term LIMIT $k"
        with driver.session() as sess:
            rows2 = sess.run(cypher2, k=k).data()
        return [str(r["term"]) for r in rows2]

    scored = []
    for r in rows:
        mu = float(r["mu"])
        n = int(r["n"])
        priority = (1.0 - mu) + 0.4 / math.sqrt(n + 1.0)
        scored.append((priority, str(r["term"])))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [t for _, t in scored[:k]]


def write_attempt(driver, student_id: str, qid: str, ts: int, correctness: float):
    # Use CREATE: attempts are events.
    cypher = """
    MATCH (s:Student {id:$sid})
    MATCH (q:Question {id:$qid})
    CREATE (s)-[:ATTEMPTED {ts:$ts, correctness:$corr}]->(q)
    """
    with driver.session() as sess:
        sess.run(cypher, sid=student_id, qid=qid, ts=int(ts), corr=float(correctness))


def upsert_mastery(driver, student_id: str, term: str, ts: int, y: float, alpha: float):
    cypher = """
    MERGE (s:Student {id:$sid})
    MERGE (t:Term {name:$term})
    MERGE (s)-[m:MASTERY]->(t)
    ON CREATE SET m.mu = $y, m.n = 1, m.last_ts = $ts
    ON MATCH SET
      m.mu = (1.0 - $alpha) * m.mu + $alpha * $y,
      m.n = m.n + 1,
      m.last_ts = $ts
    """
    with driver.session() as sess:
        sess.run(cypher, sid=student_id, term=term, ts=int(ts), y=float(y), alpha=float(alpha))


# -----------------------------
# Featurization + scoring
# -----------------------------


@torch.no_grad()
def score_and_rank(
    model,
    cfg: Config,
    vocabs,
    history,
    mastery,
    candidates,
    device,
    target_p: float,
    beta_term_weakness: float,
    recency_half_life_days: float,
):
    q_hist, c_hist, dt_hist, attn, last_ts = build_history_tensors(
        history=history,
        q_vocab=vocabs["q_vocab"],
        correctness_bins=cfg.correctness_bins,
        time_bins=cfg.time_bins,
        max_hist=cfg.max_hist,
    )
    q_hist, c_hist, dt_hist, attn = (
        q_hist.to(device),
        c_hist.to(device),
        dt_hist.to(device),
        attn.to(device),
    )

    ranked = []
    for cand in candidates:
        batch = pack_candidate_batch(
            q_hist,
            c_hist,
            dt_hist,
            attn,
            cand,
            vocabs,
            mastery,
            last_ts,
            recency_half_life_days,
            device=device,
        )
        p = float(torch.sigmoid(model(batch)).item())
        mu = mastery.get(cand["term"], (0.5, 0, 0))[0]
        util = -abs(p - target_p) + beta_term_weakness * (1.0 - float(mu))
        ranked.append((cand, p, util, float(mu)))

    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


# -----------------------------
# Score input (manual or auto)
# -----------------------------


def ask_score_manual() -> float:
    while True:
        s = input("Enter observed correctness score in [0,1] (e.g. 0.0, 0.5, 1.0): ").strip()
        try:
            y = float(s)
            if 0.0 <= y <= 1.0:
                return y
        except Exception:
            pass
        print("Invalid. Please enter a float between 0 and 1.")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def diff_to_num(diff: str) -> float:
    return {"easy": -0.8, "medium": 0.0, "hard": 0.8}.get(diff, 0.0)


def simulate_score_auto(
    rng: random.Random, term: str, diff: str, true_ability: Dict[str, float], global_a: float
) -> float:
    # simple latent ability per term
    if term not in true_ability:
        true_ability[term] = global_a + rng.gauss(0.0, 0.6)
    a = true_ability[term]
    p = sigmoid(a - diff_to_num(diff))
    y = max(0.0, min(1.0, p + rng.gauss(0.0, 0.10)))
    # small learning
    true_ability[term] = a + 0.03 * (y - 0.3)
    return float(y)


# -----------------------------
# Main
# -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-pass", required=True)
    ap.add_argument("--student-id", required=True)

    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--history-limit", type=int, default=200)
    ap.add_argument("--target-terms", type=int, default=5)
    ap.add_argument("--candidate-limit", type=int, default=800)
    ap.add_argument("--exclude-seen", action="store_true")
    ap.add_argument(
        "--allowed-difficulties", type=str, default="", help="comma-separated, e.g. easy,medium"
    )
    ap.add_argument("--target-p", type=float, default=0.7)
    ap.add_argument("--beta-term-weakness", type=float, default=0.4)
    ap.add_argument("--recency-half-life-days", type=float, default=14.0)
    ap.add_argument("--alpha-mastery", type=float, default=0.2)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--start-ts", type=int, default=1704067200)  # ~ 2024-01-01
    ap.add_argument("--gap-sec", type=int, default=600)

    ap.add_argument("--auto-score", action="store_true", help="simulate score instead of prompting")
    ap.add_argument("--sim-seed", type=int, default=123)
    ap.add_argument("--sim-global-ability", type=float, default=0.0)
    args = ap.parse_args()

    device = args.device
    model, cfg, vocabs = load_checkpoint(args.ckpt, device=device)

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))
    ensure_student(driver, args.student_id)

    ts = int(args.start_ts)
    rng = random.Random(args.sim_seed)
    true_ability: Dict[str, float] = {}

    print(
        f"\nSimulating for student={args.student_id} steps={args.steps} auto_score={args.auto_score}\n"
    )

    for step in range(1, args.steps + 1):
        history = fetch_history(driver, args.student_id, limit=args.history_limit)
        mastery = fetch_mastery(driver, args.student_id)

        terms = pick_target_terms(driver, args.student_id, k=args.target_terms)
        candidates = fetch_candidates(
            driver,
            args.student_id,
            terms=terms,
            limit=args.candidate_limit,
            exclude_seen=args.exclude_seen,
        )
        if not candidates:
            print("No candidates found. Try removing --exclude-seen or increase --candidate-limit.")
            break

        ranked = score_and_rank(
            model=model,
            cfg=cfg,
            vocabs=vocabs,
            history=history,
            mastery=mastery,
            candidates=candidates,
            device=device,
            target_p=args.target_p,
            beta_term_weakness=args.beta_term_weakness,
            recency_half_life_days=args.recency_half_life_days,
        )

        top = ranked[0]
        cand, p_pred, util, mu = top

        print(
            f"[{step:02d}] RECOMMEND q={cand['qid']} | term={cand['term']} | diff={cand['difficulty']} | "
            f"p_pred={p_pred:.3f} | util={util:.3f} | term_mu={mu:.3f}"
        )

        if args.auto_score:
            y = simulate_score_auto(
                rng, cand["term"], cand["difficulty"], true_ability, args.sim_global_ability
            )
            print(f"      observed (simulated) correctness={y:.3f}")
        else:
            y = ask_score_manual()

        # write to graph + update mastery
        write_attempt(driver, args.student_id, cand["qid"], ts=ts, correctness=y)
        upsert_mastery(driver, args.student_id, cand["term"], ts=ts, y=y, alpha=args.alpha_mastery)

        ts += int(args.gap_sec)

    driver.close()
    print("\nDone.\n")


if __name__ == "__main__":
    main()
