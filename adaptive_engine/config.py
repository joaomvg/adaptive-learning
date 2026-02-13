from dataclasses import dataclass
import torch

# -----------------------
# Config / utilities
# -----------------------


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # sequence encoding
    max_hist: int = 80
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1

    # discretizations
    correctness_bins: int = 6       # for history tokens only
    time_bins: int = 10             # log-binned delta seconds

    # optimization
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    grad_clip: float = 1.0

    # mastery update
    alpha: float = 0.2              # EWMA for mu
    # feature transforms
    recency_half_life_days: float = 14.0  # for recency feature