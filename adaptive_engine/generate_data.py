# generate_synth_data.py
# Generate synthetic questions.csv and interactions.csv for the adaptive assessment engine.

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DIFF_LEVELS = ["easy", "medium", "hard"]
DIFF_TO_NUM = {"easy": -0.8, "medium": 0.0, "hard": 0.8}


@dataclass
class GenCfg:
    n_categories: int = 10
    terms_per_category: int = 30
    questions_per_term: int = 25  # total questions = n_categories*terms_per_category*questions_per_term

    n_students: int = 2000
    interactions_per_student: int = 80  # total interactions ~ n_students * interactions_per_student

    seed: int = 42
    start_date: str = "2024-01-01T00:00:00Z"

    # Behavior controls
    p_repeat_term: float = 0.35          # chance to stay on same term next interaction
    p_jump_category: float = 0.12        # chance to jump to another category
    p_new_term_bias: float = 0.55        # bias toward unseen/less-practiced terms

    # Learning dynamics
    learn_rate: float = 0.03             # per attempt improvement in latent skill for that term (small)
    forget_half_life_days: float = 30.0  # slow forgetting in latent skill (optional-ish)

    # Noise / partial credit
    slip: float = 0.08                   # chance of lower score even if capable
    guess: float = 0.06                  # chance of higher score even if not capable
    score_noise: float = 0.10            # Gaussian noise added to score

    # Timing
    mean_gap_minutes: float = 45.0
    gap_jitter_minutes: float = 20.0


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def parse_start_date(s: str) -> datetime:
    # Accept "2024-01-01T00:00:00Z"
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def make_questions(cfg: GenCfg) -> pd.DataFrame:
    categories = [f"Category_{i:02d}" for i in range(cfg.n_categories)]
    rows = []
    qid = 1

    for ci, cat in enumerate(categories):
        for ti in range(cfg.terms_per_category):
            term = f"{cat}::Term_{ti:03d}"
            # distribute difficulty roughly evenly within term
            for qi in range(cfg.questions_per_term):
                diff = DIFF_LEVELS[qi % len(DIFF_LEVELS)]
                rows.append(
                    {
                        "question_id": f"Q{qid}",
                        "category": cat,
                        "difficulty": diff,
                        "term": term,
                    }
                )
                qid += 1

    return pd.DataFrame(rows)


def sample_next_term(
    rng: random.Random,
    cat_terms: Dict[str, List[str]],
    current_cat: str,
    current_term: str,
    term_counts: Dict[str, int],
    cfg: GenCfg,
) -> Tuple[str, str]:
    # Decide whether to stick to same term, move within category, or jump category
    if current_term and rng.random() < cfg.p_repeat_term:
        return current_cat, current_term

    # Jump category?
    if rng.random() < cfg.p_jump_category:
        current_cat = rng.choice(list(cat_terms.keys()))
        current_term = rng.choice(cat_terms[current_cat])
        return current_cat, current_term

    # Otherwise stay in category but pick a term, biased toward less-seen terms
    terms = cat_terms[current_cat]
    counts = np.array([term_counts.get(t, 0) for t in terms], dtype=np.float32)

    # weight: prefer lower counts (new terms) with controllable bias
    # w ~ exp(-bias * count)
    w = np.exp(-cfg.p_new_term_bias * counts)
    w = w / w.sum()

    current_term = rng.choices(terms, weights=w.tolist(), k=1)[0]
    return current_cat, current_term


def pick_question_for_term(
    rng: random.Random,
    term_to_questions: Dict[str, List[Tuple[str, str]]],  # term -> [(qid, diff), ...]
    term: str,
    target_p: float,
    ability_term: float,
) -> Tuple[str, str]:
    """
    Pick a question in the term with a difficulty roughly matching a target success probability.
    This simulates a platform that already does some adaptivity.
    """
    # Choose desired difficulty numeric threshold based on ability and target
    # We'll pick diff whose implied success probability is closest to target.
    # implied p = sigmoid(ability_term - diff_num)
    best = None
    best_gap = 1e9
    candidates = term_to_questions[term]

    # sample a subset for speed if huge
    subset = candidates if len(candidates) <= 50 else rng.sample(candidates, 50)

    for qid, diff in subset:
        p = sigmoid(ability_term - DIFF_TO_NUM[diff])
        gap = abs(p - target_p)
        if gap < best_gap:
            best_gap = gap
            best = (qid, diff)

    # fallback: random
    if best is None:
        best = rng.choice(candidates)
    return best


def simulate_score(
    rng: random.Random,
    ability_term: float,
    diff: str,
    cfg: GenCfg,
) -> float:
    # base probability-like latent success
    p = sigmoid(ability_term - DIFF_TO_NUM[diff])

    # slip/guess mixture: nudge p
    if rng.random() < cfg.slip:
        p *= 0.6
    if rng.random() < cfg.guess:
        p = 1.0 - (1.0 - p) * 0.6

    # turn into partial credit score:
    # sample around p with noise and clamp
    score = p + rng.gauss(0.0, cfg.score_noise)
    return clamp01(score)


def apply_forgetting(ability: float, delta_days: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return ability
    # exponential decay toward 0 (neutral)
    decay = math.exp(-math.log(2.0) * max(delta_days, 0.0) / half_life_days)
    return ability * decay


def generate_interactions(questions_df: pd.DataFrame, cfg: GenCfg) -> pd.DataFrame:
    rng = random.Random(cfg.seed)

    # Mappings
    term_to_questions: Dict[str, List[Tuple[str, str]]] = {}
    term_to_cat: Dict[str, str] = {}
    cat_terms: Dict[str, List[str]] = {}

    for row in questions_df.itertuples(index=False):
        qid = row.question_id
        cat = row.category
        diff = row.difficulty
        term = row.term
        term_to_questions.setdefault(term, []).append((qid, diff))
        term_to_cat[term] = cat
        cat_terms.setdefault(cat, []).append(term)

    # Unique terms list
    all_terms = list(term_to_questions.keys())

    # Student latent abilities
    # Global ability per student + per-term offsets (sparse-ish)
    n_students = cfg.n_students
    student_ids = [f"S{i:05d}" for i in range(1, n_students + 1)]

    # per student: global ability ~ N(0,1)
    global_ability = {sid: rng.gauss(0.0, 1.0) for sid in student_ids}

    start_dt = parse_start_date(cfg.start_date)

    rows = []

    for sid in student_ids:
        # per-student term ability state: dict term -> (ability, last_ts)
        ability_state: Dict[str, Tuple[float, datetime]] = {}

        # start in a random category/term
        current_term = rng.choice(all_terms)
        current_cat = term_to_cat[current_term]

        # counts to bias term selection
        term_counts: Dict[str, int] = {}

        # time
        t = start_dt + timedelta(minutes=rng.randint(0, 60 * 24))  # spread start within first day

        for _ in range(cfg.interactions_per_student):
            # choose next term
            current_cat, current_term = sample_next_term(
                rng, cat_terms, current_cat, current_term, term_counts, cfg
            )

            # ability for this term (with forgetting)
            if current_term in ability_state:
                a_term, last_t = ability_state[current_term]
                delta_days = (t - last_t).total_seconds() / (3600 * 24)
                a_term = apply_forgetting(a_term, delta_days, cfg.forget_half_life_days)
            else:
                # initialize term ability around global ability with some term noise
                a_term = global_ability[sid] + rng.gauss(0.0, 0.6)

            # pick a question (simulate platform targeting ~70% success)
            qid, diff = pick_question_for_term(
                rng=rng,
                term_to_questions=term_to_questions,
                term=current_term,
                target_p=0.7,
                ability_term=a_term,
            )

            # outcome
            score = simulate_score(rng, a_term, diff, cfg)

            rows.append(
                {
                    "student_id": sid,
                    "question_id": qid,
                    "correctness": round(float(score), 3),
                    "timestamp": t.isoformat().replace("+00:00", "Z"),
                }
            )

            # update term ability slightly upward as function of score (learning)
            # if score high, learn a bit; if low, smaller improvement
            a_term = a_term + cfg.learn_rate * (score - 0.3)
            ability_state[current_term] = (a_term, t)

            # update counts
            term_counts[current_term] = term_counts.get(current_term, 0) + 1

            # advance time
            gap = max(
                1.0,
                rng.gauss(cfg.mean_gap_minutes, cfg.gap_jitter_minutes),
            )
            t = t + timedelta(minutes=float(gap))

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="data", help="Output directory")
    ap.add_argument("--n-categories", type=int, default=10)
    ap.add_argument("--terms-per-category", type=int, default=30)
    ap.add_argument("--questions-per-term", type=int, default=25)
    ap.add_argument("--n-students", type=int, default=2000)
    ap.add_argument("--interactions-per-student", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = GenCfg(
        n_categories=args.n_categories,
        terms_per_category=args.terms_per_category,
        questions_per_term=args.questions_per_term,
        n_students=args.n_students,
        interactions_per_student=args.interactions_per_student,
        seed=args.seed,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    questions_df = make_questions(cfg)
    inter_df = generate_interactions(questions_df, cfg)

    out_dir = args.out_dir
    import os
    os.makedirs(out_dir, exist_ok=True)

    q_path = os.path.join(out_dir, "questions.csv")
    i_path = os.path.join(out_dir, "interactions.csv")

    questions_df.to_csv(q_path, index=False)
    inter_df.to_csv(i_path, index=False)

    print("Wrote:")
    print(f"  {q_path}  (questions={len(questions_df):,})")
    print(f"  {i_path}  (interactions={len(inter_df):,})")


if __name__ == "__main__":
    main()