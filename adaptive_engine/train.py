# train_adaptive_engine.py
# Baseline training script for an adaptive assessment engine:
# Train a model for  P(correctness | candidate_question, history, mastery_features)
# - correctness is a float in [0,1]
# - history is a sequence of past (question_id, correctness_bin, time_delta_bin)
# - mastery_features are derived from the KG structure: term/category/difficulty + student-term running mastery (mu, n, recency)
#
# Usage:
#   python train_adaptive_engine.py --questions questions.csv --interactions interactions.csv
#
# Expected columns:
# Questions CSV:
#   question_id (str), category (str), difficulty (str), term (str)
# Interactions CSV:
#   student_id (str), question_id (str), correctness (float 0..1), timestamp (datetime or parseable)
#
# Notes:
# - This script computes mastery sequentially from interactions (no leakage).
# - It does NOT require a full softmax over questions: it's a regression-style scorer for a served question.
# - For "recommendation", you will later generate candidates from KG (term/category/difficulty traversal) and score them.

import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptive_engine.config import Config
from torch.utils.data import DataLoader

from adaptive_engine.data import build_samples_time_split, SamplesDataset, collate
from adaptive_engine.model import AdaptiveScorer
from adaptive_engine.logging_config import get_logger

logger = get_logger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# Metrics
# -----------------------


@torch.no_grad()
def eval_rmse_brier(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        logit = model(batch)
        p = torch.sigmoid(logit).clamp(0.0, 1.0)
        y = batch["y"].clamp(0.0, 1.0)
        ys.append(y.detach().cpu())
        ps.append(p.detach().cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()

    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    brier = float(np.mean((p - y) ** 2))
    return rmse, brier


def train_one_epoch(
    model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: str, cfg: Config
):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        y = batch["y"].clamp(0.0, 1.0)
        logit = model(batch)
        # BCE with soft labels in [0,1] (works well as probability regression)
        loss = F.binary_cross_entropy_with_logits(logit, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        bs = y.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(1, n)


# -----------------------
# Main
# -----------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=str, required=True)
    ap.add_argument("--interactions", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", type=str, default="model.pt")
    args = ap.parse_args()

    logger.info("Start training")

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs

    set_seed(cfg.seed)

    questions_df = pd.read_csv(args.questions)
    inter_df = pd.read_csv(args.interactions)

    train_s, val_s, test_s, q_vocab, term_vocab, cat_vocab, diff_vocab = build_samples_time_split(
        questions_df, inter_df, cfg
    )

    print(f"Samples: train={len(train_s):,} val={len(val_s):,} test={len(test_s):,}")
    print(
        f"Vocab: questions={len(q_vocab):,} terms={len(term_vocab):,} cats={len(cat_vocab):,} diffs={len(diff_vocab):,}"
    )

    train_ds = SamplesDataset(train_s)
    val_ds = SamplesDataset(val_s)
    test_ds = SamplesDataset(test_s)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate(b, pad_id=0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate(b, pad_id=0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate(b, pad_id=0),
    )

    model = AdaptiveScorer(
        n_questions=len(q_vocab),
        n_terms=len(term_vocab),
        n_cats=len(cat_vocab),
        n_diffs=len(diff_vocab),
        cfg=cfg,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, cfg.device, cfg)
        val_rmse, val_brier = eval_rmse_brier(model, val_loader, cfg.device)
        print(
            f"Epoch {epoch:02d} | train_bce={tr_loss:.4f} | val_rmse={val_rmse:.4f} | val_brier={val_brier:.4f}"
        )

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "q_vocab": q_vocab,
                    "term_vocab": term_vocab,
                    "cat_vocab": cat_vocab,
                    "diff_vocab": diff_vocab,
                },
                args.out,
            )
            print(f"  saved -> {args.out}")

    # final test
    ckpt = torch.load(args.out, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])
    test_rmse, test_brier = eval_rmse_brier(model, test_loader, cfg.device)
    print(f"Test | rmse={test_rmse:.4f} | brier={test_brier:.4f}")


if __name__ == "__main__":
    main()
