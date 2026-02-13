import math
import torch
import torch.nn as nn
from adaptive_engine.config import Config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(pos * div)
        self.pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", self.pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


class AdaptiveScorer(nn.Module):
    """
    Predicts expected correctness y in [0,1] for the *served* question q_id,
    given history and mastery features.
    """

    def __init__(self, n_questions: int, n_terms: int, n_cats: int, n_diffs: int, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # token embeddings for history
        self.q_emb = nn.Embedding(n_questions + 1, cfg.d_model, padding_idx=0)
        self.c_emb = nn.Embedding(cfg.correctness_bins, cfg.d_model)
        self.dt_emb = nn.Embedding(cfg.time_bins, cfg.d_model)

        self.pos = PositionalEncoding(cfg.d_model, max_len=2048)
        self.drop = nn.Dropout(cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        # candidate (served) question + KG attributes
        self.q_out = nn.Embedding(n_questions + 1, cfg.d_model, padding_idx=0)
        self.term_emb = nn.Embedding(n_terms + 1, cfg.d_model, padding_idx=0)
        self.cat_emb = nn.Embedding(n_cats + 1, cfg.d_model, padding_idx=0)
        self.diff_emb = nn.Embedding(n_diffs + 1, cfg.d_model, padding_idx=0)

        # mastery numeric features -> projection
        self.mastery_proj = nn.Sequential(
            nn.Linear(3, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # final head
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model * 3, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def encode_history(self, q_hist, c_hist, dt_hist, attn):
        x = self.q_emb(q_hist) + self.c_emb(c_hist) + self.dt_emb(dt_hist)
        x = self.pos(self.drop(x))
        key_padding_mask = ~attn  # True where PAD
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B,T,D)

        # pool: last valid position
        lengths = attn.long().sum(dim=1)
        last_idx = (lengths - 1).clamp(min=0)
        h_last = h[torch.arange(h.size(0), device=h.device), last_idx]  # (B,D)

        # if empty history (length 0), last_idx becomes 0 but token is PAD; mitigate by zeroing:
        empty = (lengths == 0).unsqueeze(1)
        h_last = torch.where(empty, torch.zeros_like(h_last), h_last)
        return h_last

    def forward(self, batch):
        q_hist = batch["q_hist"]
        c_hist = batch["c_hist"]
        dt_hist = batch["dt_hist"]
        attn = batch["attn"]

        q_id = batch["q_id"]
        term_id = batch["term_id"]
        cat_id = batch["cat_id"]
        diff_id = batch["diff_id"]

        mu = batch["mu_term"]  # (B,1)
        n_feat = batch["n_feat"]  # (B,1)
        rec = batch["recency"]  # (B,1)

        h = self.encode_history(q_hist, c_hist, dt_hist, attn)  # (B,D)

        qv = (
            self.q_out(q_id)
            + self.term_emb(term_id)
            + self.cat_emb(cat_id)
            + self.diff_emb(diff_id)
        )  # (B,D)
        m = self.mastery_proj(torch.cat([mu, n_feat, rec], dim=1))  # (B,D)

        z = torch.cat([h, qv, m], dim=1)  # (B, 3D)
        logit = self.head(z).squeeze(1)  # (B,)
        return logit
