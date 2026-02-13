import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from adaptive_engine.config import Config
from adaptive_engine.logging_config import get_logger

logger = get_logger(__name__)


def parse_timestamp(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, errors="coerce", utc=True, format='ISO8601')
    # fallback if already numeric
    if ts.isna().all():
        ts = pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="s", utc=True)
    return ts


def bucketize_correctness(c: np.ndarray, n_bins: int) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    # bins 0..n_bins-1
    return np.minimum((c * (n_bins - 1e-6)).astype(np.int64), n_bins - 1)


def bucketize_time_deltas(delta_seconds: np.ndarray, n_bins: int) -> np.ndarray:
    # log1p binning
    d = np.maximum(delta_seconds, 0.0)
    x = np.log1p(d)
    # compute bin edges by percentiles for robustness
    # If you want deterministic edges across runs, you can precompute from train only.
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    # ensure strictly increasing edges
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros_like(d, dtype=np.int64)
    # digitize into 0..(n_bins-1) but adapt if edges collapsed
    b = np.digitize(x, edges[1:-1], right=False)
    b = np.clip(b, 0, n_bins - 1).astype(np.int64)
    return b


def recency_feature(delta_days: float, half_life_days: float) -> float:
    # exp decay: 1.0 when recent, -> 0 as old
    if half_life_days <= 0:
        return 0.0
    return float(math.exp(-math.log(2.0) * max(delta_days, 0.0) / half_life_days))


# -----------------------
# Vocab building
# -----------------------

def build_vocab(values: pd.Series) -> Dict[str, int]:
    # 0 reserved for PAD/UNK
    uniq = values.astype(str).fillna("").unique().tolist()
    vocab = {v: i + 1 for i, v in enumerate(uniq)}
    return vocab


# -----------------------
# Sample construction with leakage-free mastery
# -----------------------

@dataclass
class Sample:
    q_hist: List[int]
    c_hist_bin: List[int]
    dt_hist_bin: List[int]
    q_id: int
    term_id: int
    cat_id: int
    diff_id: int
    mu_term: float
    n_term: float
    recency_term: float
    y: float


def build_samples_time_split(
    questions_df: pd.DataFrame,
    inter_df: pd.DataFrame,
    cfg: Config,
) -> Tuple[List[Sample], List[Sample], List[Sample], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Build leakage-free samples and split time-based per student:
    - train: first 80% interactions per student
    - val: next 10%
    - test: last 10%

    Each sample uses mastery and history computed strictly from earlier events.
    """
    logger.info('Prepare questionds df')
    
    qmeta = questions_df.copy()
    qmeta["question_id"] = qmeta["question_id"].astype(str)
    qmeta["term"] = qmeta["term"].astype(str)
    qmeta["category"] = qmeta["category"].astype(str)
    qmeta["difficulty"] = qmeta["difficulty"].astype(str)

    # vocabularies
    logger.info('Create vocabularies')
    q_vocab = build_vocab(qmeta["question_id"])
    term_vocab = build_vocab(qmeta["term"])
    cat_vocab = build_vocab(qmeta["category"])
    diff_vocab = build_vocab(qmeta["difficulty"])

    # map question -> (term, cat, diff)
    qmeta["q_id"] = qmeta["question_id"].map(q_vocab).astype(int)
    qmeta["term_id"] = qmeta["term"].map(term_vocab).astype(int)
    qmeta["cat_id"] = qmeta["category"].map(cat_vocab).astype(int)
    qmeta["diff_id"] = qmeta["difficulty"].map(diff_vocab).astype(int)

    q_lookup = qmeta.set_index("question_id")[["q_id", "term_id", "cat_id", "diff_id"]].to_dict("index")

    # interactions prep
    df = inter_df.copy()
    df["student_id"] = df["student_id"].astype(str)
    df["question_id"] = df["question_id"].astype(str)
    df["correctness"] = pd.to_numeric(df["correctness"], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(np.float32)
    df["timestamp"] = parse_timestamp(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["student_id", "timestamp"])

    # filter interactions to questions present in questions_df
    df = df[df["question_id"].isin(q_lookup.keys())].copy()

    # compute time delta seconds per student (for history token feature)
    df["ts_sec"] = (df["timestamp"].astype("int64") // 10**9).astype(np.int64)
    df["delta_sec"] = df.groupby("student_id")["ts_sec"].diff().fillna(0).clip(lower=0).astype(np.float32)

    # pre-bucketize correctness + time deltas (used in history tokens)
    df["c_bin"] = bucketize_correctness(df["correctness"].to_numpy(), cfg.correctness_bins)
    df["dt_bin"] = bucketize_time_deltas(df["delta_sec"].to_numpy(), cfg.time_bins)

    # build samples per student sequentially, computing mastery on the fly (no leakage)
    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    test_samples: List[Sample] = []

    # per student split points
    for sid, g in df.groupby("student_id", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 5:
            continue
        n_train = max(1, int(round(n * 0.8)))
        n_val = max(1, int(round(n * 0.1)))
        train_end = n_train
        val_end = min(n, n_train + n_val)

        # student-term mastery state (stored as dict)
        # mu in [0,1], n count, last_ts_sec
        mastery: Dict[int, Tuple[float, int, int]] = {}

        # history buffers (of question ids + bins)
        hist_q: List[int] = []
        hist_cbin: List[int] = []
        hist_dtbin: List[int] = []

        for i in range(n):
            q_str = g.loc[i, "question_id"]
            meta = q_lookup[q_str]
            q_id = meta["q_id"]
            term_id = meta["term_id"]
            cat_id = meta["cat_id"]
            diff_id = meta["diff_id"]

            y = float(g.loc[i, "correctness"])
            ts = int(g.loc[i, "ts_sec"])
            c_bin = int(g.loc[i, "c_bin"])
            dt_bin = int(g.loc[i, "dt_bin"])

            # mastery BEFORE seeing this interaction (no leakage)
            if term_id in mastery:
                mu_old, n_old, last_ts = mastery[term_id]
                mu_term = float(mu_old)
                n_term = float(n_old)
                delta_days = (ts - last_ts) / (3600 * 24)
                rec = recency_feature(delta_days, cfg.recency_half_life_days)
            else:
                mu_term = 0.5  # prior
                n_term = 0.0
                rec = 0.0

            # build sample only if there is some history (optional; you can allow empty history too)
            # We'll allow empty history by using length 0.
            q_hist = hist_q[-cfg.max_hist:]
            c_hist = hist_cbin[-cfg.max_hist:]
            dt_hist = hist_dtbin[-cfg.max_hist:]

            sample = Sample(
                q_hist=q_hist,
                c_hist_bin=c_hist,
                dt_hist_bin=dt_hist,
                q_id=q_id,
                term_id=term_id,
                cat_id=cat_id,
                diff_id=diff_id,
                mu_term=mu_term,
                n_term=n_term,
                recency_term=rec,
                y=y,
            )

            if i < train_end:
                train_samples.append(sample)
            elif i < val_end:
                val_samples.append(sample)
            else:
                test_samples.append(sample)

            # update mastery AFTER observing y
            if term_id in mastery:
                mu_old, n_old, _last_ts = mastery[term_id]
                n_new = n_old + 1
                mu_new = (1.0 - cfg.alpha) * mu_old + cfg.alpha * y
                mastery[term_id] = (float(mu_new), int(n_new), ts)
            else:
                mastery[term_id] = (float(y), 1, ts)

            # append to history buffers
            hist_q.append(q_id)
            hist_cbin.append(c_bin)
            hist_dtbin.append(dt_bin)

    return train_samples, val_samples, test_samples, q_vocab, term_vocab, cat_vocab, diff_vocab


# -----------------------
# Dataset / collate
# -----------------------

class SamplesDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def collate(samples: List[Sample], pad_id: int = 0):
    B = len(samples)
    max_len = max((len(s.q_hist) for s in samples), default=0)

    q_hist = torch.full((B, max_len), pad_id, dtype=torch.long)
    c_hist = torch.full((B, max_len), 0, dtype=torch.long)
    dt_hist = torch.full((B, max_len), 0, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.bool)

    for i, s in enumerate(samples):
        L = len(s.q_hist)
        if L == 0:
            continue
        q_hist[i, :L] = torch.tensor(s.q_hist, dtype=torch.long)
        c_hist[i, :L] = torch.tensor(s.c_hist_bin, dtype=torch.long)
        dt_hist[i, :L] = torch.tensor(s.dt_hist_bin, dtype=torch.long)
        attn[i, :L] = True

    q_id = torch.tensor([s.q_id for s in samples], dtype=torch.long)
    term_id = torch.tensor([s.term_id for s in samples], dtype=torch.long)
    cat_id = torch.tensor([s.cat_id for s in samples], dtype=torch.long)
    diff_id = torch.tensor([s.diff_id for s in samples], dtype=torch.long)

    mu_term = torch.tensor([s.mu_term for s in samples], dtype=torch.float32).unsqueeze(1)
    n_term = torch.tensor([s.n_term for s in samples], dtype=torch.float32).unsqueeze(1)
    recency = torch.tensor([s.recency_term for s in samples], dtype=torch.float32).unsqueeze(1)

    y = torch.tensor([s.y for s in samples], dtype=torch.float32).clamp(0.0, 1.0)

    # log1p(n) as feature
    n_feat = torch.log1p(n_term)

    return {
        "q_hist": q_hist,
        "c_hist": c_hist,
        "dt_hist": dt_hist,
        "attn": attn,
        "q_id": q_id,
        "term_id": term_id,
        "cat_id": cat_id,
        "diff_id": diff_id,
        "mu_term": mu_term,
        "n_feat": n_feat,
        "recency": recency,
        "y": y,
    }