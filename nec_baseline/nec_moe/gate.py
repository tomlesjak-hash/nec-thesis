"""Gate head: ``LayerNorm(n) -> Linear(n, K)`` emitting **logits**.

Fixes ledger defects 3 and 4:

- **Logits out, no in-model softmax** — the fused log-domain loss
  (:mod:`nec_moe.losses`) consumes logits; there is no ``nn.Softmax`` anywhere
  in the model tree.
- **LayerNorm, never BatchNorm** — per-sample normalization; no cross-sample
  coupling of dates inside a batch (the leakage channel the prototype's
  ``BatchNorm1d`` opened directly upstream of the regime decision).

Zero-initialized by default so the gate starts *exactly* uniform: entropy
``log K`` at step 0, no expert starts dead (Module 8 capstone — a confident
random gate at t=0 is a collapsed mixture before training starts).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .config import GateConfig
from .utils import assert_shape

__all__ = ["GateHead"]


class GateHead(nn.Module):
    """``h_T (B, hidden_dim) -> gate logits z (B, n_experts)``."""

    def __init__(self, hidden_dim: int, n_experts: int, cfg: GateConfig) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, n_experts)
        if cfg.zero_init:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, h_t: Tensor) -> Tensor:
        assert_shape(h_t, (None, self.hidden_dim), "h_T (gate input)")
        z = self.linear(self.norm(h_t))
        assert_shape(z, (h_t.shape[0], self.n_experts), "gate logits")
        return z
