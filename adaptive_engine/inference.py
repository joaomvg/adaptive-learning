# recommend.py
# Inference script for the Adaptive Assessment Engine:
# - Pull student history + mastery state from Neo4j (KG)
# - Generate candidate questions by graph traversal + constraints
# - Score candidates with trained model: P(correctness | candidate, history, mastery_features)
# - Select questions that match a target challenge level (e.g., p ~= 0.7) and prioritize weak/uncertain terms
#
# Assumptions:
# - You trained and saved the checkpoint produced by train_adaptive_engine.py (model.pt)
# - Neo4j contains:
#   (:Student {id})
#   (:Question {id, difficulty})
#   (:Term {name})
#   (:Category {name})
#   (q)-[:IN_TERM]->(t)
#   (t)-[:IN_CATEGORY]->(c)
#   (s)-[:ATTEMPTED {ts, correctness}]->(q)
#   (s)-[:MASTERY {mu, n, last_ts}]->(t)
#
# Usage:
#   python recommend.py \
#     --ckpt model.pt \
#     --neo4j-uri bolt://localhost:7687 \
#     --neo4j-user neo4j \
#     --neo4j-pass password \
#     --student-id "S123" \
#     --topn 10 \
#     --target-p 0.7
#
# Note:
# - If you do not have MASTERY edges in Neo4j yet, you can still run this script:
#   it will fall back to priors (mu=0.5, n=0, recency=0).
# - Candidate generation is KG-based, so this scales without full softmax.

import argparse
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from neo4j import GraphDatabase

from adaptive_engine.config import Config
from adaptive_engine.model import AdaptiveScorer
from adaptive_engine.kg import (
    neo4j_fetch_candidates,
    neo4j_fetch_term_mastery,
    neo4j_fetch_history,
    neo4j_pick_target_terms,
)


def bucketize_correctness(c: np.ndarray, n_bins: int) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.minimum((c * (n_bins - 1e-6)).astype(np.int64), n_bins - 1)


def recency_feature(delta_days: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 0.0
    return float(math.exp(-math.log(2.0) * max(delta_days, 0.0) / half_life_days))


# ---------------------------
# Featurization for model
# ---------------------------


def build_history_tensors(
    history: List[Tuple[str, float, int]],
    q_vocab: Dict[str, int],
    correctness_bins: int,
    time_bins: int,
    max_hist: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Build history tensors (1, T) for q_hist, c_hist_bin, dt_hist_bin, attn.
    dt_bin here is a simple log1p bucket with fixed edges; for inference we approximate with coarse bins.
    """
    if len(history) == 0:
        q_hist = torch.zeros((1, 0), dtype=torch.long)
        c_hist = torch.zeros((1, 0), dtype=torch.long)
        dt_hist = torch.zeros((1, 0), dtype=torch.long)
        attn = torch.zeros((1, 0), dtype=torch.bool)
        last_ts = 0
        return q_hist, c_hist, dt_hist, attn, last_ts

    # keep only last max_hist items
    hist = history[-max_hist:]
    q_ids = [q_vocab.get(qid, 0) for (qid, _, _) in hist]
    corrs = np.array([float(c) for (_, c, _) in hist], dtype=np.float32)
    ts = np.array([int(t) for (_, _, t) in hist], dtype=np.int64)
    last_ts = int(ts[-1])

    # correctness bins for history tokens
    c_bins = bucketize_correctness(corrs, correctness_bins).tolist()

    # time delta bins: simple heuristic buckets on log1p(delta)
    dsec = np.diff(ts, prepend=ts[0]).astype(np.float32)
    x = np.log1p(np.clip(dsec, 0, None))
    # fixed bin edges (coarse). You can tune.
    edges = np.linspace(0.0, max(1.0, float(x.max())), time_bins + 1)
    dt_bins = (
        np.clip(np.digitize(x, edges[1:-1], right=False), 0, time_bins - 1)
        .astype(np.int64)
        .tolist()
    )

    q_hist = torch.tensor([q_ids], dtype=torch.long)
    c_hist = torch.tensor([c_bins], dtype=torch.long)
    dt_hist = torch.tensor([dt_bins], dtype=torch.long)
    attn = torch.ones((1, len(q_ids)), dtype=torch.bool)
    return q_hist, c_hist, dt_hist, attn, last_ts


def pack_candidate_batch(
    q_hist: torch.Tensor,
    c_hist: torch.Tensor,
    dt_hist: torch.Tensor,
    attn: torch.Tensor,
    candidate: Dict[str, str],
    vocabs: Dict[str, Dict[str, int]],
    mastery_by_term: Dict[str, Tuple[float, int, int]],
    last_history_ts: int,
    recency_half_life_days: float,
    device: str,
) -> Dict[str, torch.Tensor]:
    q_vocab = vocabs["q_vocab"]
    term_vocab = vocabs["term_vocab"]
    cat_vocab = vocabs["cat_vocab"]
    diff_vocab = vocabs["diff_vocab"]

    qid = candidate["qid"]
    term = candidate["term"]
    cat = candidate["category"]
    diff = candidate["difficulty"]

    q_id = q_vocab.get(qid, 0)
    term_id = term_vocab.get(term, 0)
    cat_id = cat_vocab.get(cat, 0)
    diff_id = diff_vocab.get(diff, 0)

    # mastery features for the candidate's term
    if term in mastery_by_term:
        mu, n, last_ts_term = mastery_by_term[term]
        # recency from term last seen to last history ts (or now if you have now)
        delta_days = (last_history_ts - last_ts_term) / (3600 * 24) if last_ts_term > 0 else 0.0
        rec = recency_feature(delta_days, recency_half_life_days)
    else:
        mu, n, rec = 0.5, 0, 0.0

    mu_t = torch.tensor([[float(mu)]], dtype=torch.float32, device=device)
    n_feat = torch.tensor([[math.log1p(float(n))]], dtype=torch.float32, device=device)
    rec_t = torch.tensor([[float(rec)]], dtype=torch.float32, device=device)

    batch = {
        "q_hist": q_hist.to(device),
        "c_hist": c_hist.to(device),
        "dt_hist": dt_hist.to(device),
        "attn": attn.to(device),
        "q_id": torch.tensor([q_id], dtype=torch.long, device=device),
        "term_id": torch.tensor([term_id], dtype=torch.long, device=device),
        "cat_id": torch.tensor([cat_id], dtype=torch.long, device=device),
        "diff_id": torch.tensor([diff_id], dtype=torch.long, device=device),
        "mu_term": mu_t,
        "n_feat": n_feat,
        "recency": rec_t,
    }
    return batch


@torch.no_grad()
def score_candidates(
    model: nn.Module,
    history_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int],
    candidates: List[Dict[str, str]],
    vocabs: Dict[str, Dict[str, int]],
    mastery_by_term: Dict[str, Tuple[float, int, int]],
    device: str,
    recency_half_life_days: float,
) -> List[Tuple[Dict[str, str], float]]:
    q_hist, c_hist, dt_hist, attn, last_ts = history_tensors

    results = []
    for cand in candidates:
        batch = pack_candidate_batch(
            q_hist,
            c_hist,
            dt_hist,
            attn,
            cand,
            vocabs,
            mastery_by_term,
            last_ts,
            recency_half_life_days,
            device=device,
        )
        logit = model(batch)
        p = float(torch.sigmoid(logit).item())
        results.append((cand, p))
    return results


def select_topn(
    scored: List[Tuple[Dict[str, str], float]],
    mastery_by_term: Dict[str, Tuple[float, int, int]],
    target_p: float,
    topn: int,
    beta_term_weakness: float = 0.4,
) -> List[Tuple[Dict[str, str], float, float]]:
    """
    Utility that avoids recommending only easy questions.
    We target predicted success probability near target_p, and optionally prioritize weak terms.

    Utility:
      U = -|p - target_p| + beta * (1 - mu_term)
    """
    out = []
    for cand, p in scored:
        term = cand["term"]
        mu = mastery_by_term.get(term, (0.5, 0, 0))[0]
        util = -abs(p - target_p) + beta_term_weakness * (1.0 - float(mu))
        out.append((cand, p, util))
    out.sort(key=lambda x: x[2], reverse=True)
    return out[:topn]


# ---------------------------
# Main CLI
# ---------------------------


def load_checkpoint(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    cfgd = ckpt["cfg"]
    cfg = Config(
        max_hist=int(cfgd["max_hist"]),
        d_model=int(cfgd["d_model"]),
        n_heads=int(cfgd["n_heads"]),
        n_layers=int(cfgd["n_layers"]),
        dropout=float(cfgd["dropout"]),
        correctness_bins=int(cfgd["correctness_bins"]),
        time_bins=int(cfgd["time_bins"]),
    )

    q_vocab = ckpt["q_vocab"]
    term_vocab = ckpt["term_vocab"]
    cat_vocab = ckpt["cat_vocab"]
    diff_vocab = ckpt["diff_vocab"]

    model = AdaptiveScorer(
        n_questions=len(q_vocab),
        n_terms=len(term_vocab),
        n_cats=len(cat_vocab),
        n_diffs=len(diff_vocab),
        cfg=cfg,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    vocabs = {
        "q_vocab": q_vocab,
        "term_vocab": term_vocab,
        "cat_vocab": cat_vocab,
        "diff_vocab": diff_vocab,
    }
    return model, cfg, vocabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--neo4j-uri", type=str, required=True)
    ap.add_argument("--neo4j-user", type=str, required=True)
    ap.add_argument("--neo4j-pass", type=str, required=True)
    ap.add_argument("--student-id", type=str, required=True)

    ap.add_argument("--history-limit", type=int, default=200)
    ap.add_argument("--max-hist", type=int, default=None)  # override
    ap.add_argument("--target-terms", type=int, default=5)
    ap.add_argument("--candidate-limit", type=int, default=500)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--target-p", type=float, default=0.7)
    ap.add_argument("--exclude-seen", action="store_true")
    ap.add_argument(
        "--allowed-difficulties", type=str, default="", help="comma-separated, e.g. easy,medium"
    )

    ap.add_argument("--recency-half-life-days", type=float, default=14.0)
    ap.add_argument("--beta-term-weakness", type=float, default=0.4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    model, cfg, vocabs = load_checkpoint(args.ckpt, device=device)
    if args.max_hist is not None:
        cfg.max_hist = int(args.max_hist)

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_pass))

    # 1) Fetch student history
    history = neo4j_fetch_history(driver, args.student_id, limit=args.history_limit)

    # 2) Fetch mastery edges (optional)
    mastery_by_term = neo4j_fetch_term_mastery(driver, args.student_id)

    # 3) Pick target terms
    target_terms_info = neo4j_pick_target_terms(driver, args.student_id, k=args.target_terms)
    target_terms = [t for (t, _, _, _) in target_terms_info]

    # 4) Candidate generation
    allowed = [x.strip() for x in args.allowed_difficulties.split(",") if x.strip()] or None
    candidates = neo4j_fetch_candidates(
        driver,
        args.student_id,
        terms=target_terms,
        allowed_difficulties=allowed,
        limit=args.candidate_limit,
        exclude_seen=args.exclude_seen,
    )

    if not candidates:
        print(
            "No candidates found. Try: --exclude-seen off, increase --candidate-limit, or relax difficulty/term filters."
        )
        return

    # 5) Build history tensors
    hist_tensors = build_history_tensors(
        history=history,
        q_vocab=vocabs["q_vocab"],
        correctness_bins=cfg.correctness_bins,
        time_bins=cfg.time_bins,
        max_hist=cfg.max_hist,
    )

    # 6) Score candidates
    scored = score_candidates(
        model=model,
        history_tensors=hist_tensors,
        candidates=candidates,
        vocabs=vocabs,
        mastery_by_term=mastery_by_term,
        device=device,
        recency_half_life_days=args.recency_half_life_days,
    )

    # 7) Select top-N with a challenge-targeting utility (avoids always choosing easy)
    chosen = select_topn(
        scored=scored,
        mastery_by_term=mastery_by_term,
        target_p=args.target_p,
        topn=args.topn,
        beta_term_weakness=args.beta_term_weakness,
    )

    # Print results
    print(f"\nStudent: {args.student_id}")
    print(f"History events: {len(history)} | Candidates scored: {len(scored)}")
    print(f"Target terms: {target_terms}\n")

    print("Top recommendations (ordered by utility):")
    for rank, (cand, p, util) in enumerate(chosen, start=1):
        mu = mastery_by_term.get(cand["term"], (0.5, 0, 0))[0]
        print(
            f"{rank:02d}. qid={cand['qid']} | term={cand['term']} | cat={cand['category']} | diff={cand['difficulty']}"
            f" | p_correct={p:.3f} | term_mu={mu:.3f} | utility={util:.3f}"
        )

    driver.close()


if __name__ == "__main__":
    main()
