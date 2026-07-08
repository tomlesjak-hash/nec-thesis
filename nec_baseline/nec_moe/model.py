"""NECModel: assembly of encoder → gate → prior → experts (design doc §2).

Forward = the *predict* side only: it forms the prior ``pi`` (memoryless from
the current window; HMM from the previous filtered posterior via the
transition matrix) and the expert means, and the point prediction
``y_hat = sum_k pi_k mu_k``. It never sees the target — the Bayes *update*
(responsibilities / filtered posterior) lives in :func:`nec_moe.losses.mixture_nll`,
which does. This split is what keeps the HMM path strictly causal: the
prediction for date t uses only ``r_{t-1}`` (data through t-1).

The model never slices feature tensors (ledger defect 7): ``x_snap`` arrives as
its own tensor; concatenating ``h_T`` onto it (Decision A's
``snapshot_plus_hidden`` mode) is construction, not slicing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .config import NECConfig
from .encoder import GRUEncoder
from .experts import build_emission
from .gate import GateHead
from .priors import PriorContext, PriorOutput, build_prior
from .utils import assert_shape

__all__ = ["NECOutput", "NECModel"]


@dataclass
class NECOutput:
    """Structured forward result — objectives and diagnostics pick what they need."""

    gate_logits: Tensor  # (B, K)
    prior: PriorOutput  # log prior (B, K) + aux/info
    mu: Tensor  # (B, K) expert means
    log_sigma: Tensor  # (K,) expert noise scales
    y_hat: Tensor  # (B,) mixture-mean point prediction
    h_t: Tensor  # (B, n) encoder state


class NECModel(nn.Module):
    """Corrected NEC: sequence encoder → regime gate → mixture of experts."""

    def __init__(self, cfg: NECConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.encoder = GRUEncoder(cfg.data, cfg.encoder)
        self.gate = GateHead(cfg.encoder.hidden_dim, cfg.experts.n_experts, cfg.gate)
        self.prior = build_prior(cfg)
        self.experts = build_emission(cfg)

    def expert_input(self, x_snap: Tensor, h_t: Tensor) -> Tensor:
        """Build the expert input per Decision A's ``input_mode``.

        Construction, never slicing (ledger defect 7): ``x_snap`` is used as-is
        or concatenated with the encoder state.
        """
        if self.cfg.experts.input_mode == "snapshot":
            return x_snap
        return torch.cat([x_snap, h_t], dim=-1)

    def forward(
        self,
        x_seq: Tensor,
        x_snap: Tensor,
        ctx: PriorContext | None = None,
    ) -> NECOutput:
        d = self.cfg.data
        assert_shape(x_seq, (None, d.seq_len, d.d_seq), "x_seq")
        assert_shape(x_snap, (x_seq.shape[0], d.d_snap), "x_snap")

        h_t = self.encoder(x_seq)
        gate_logits = self.gate(h_t)
        prior = self.prior(gate_logits, ctx)

        x_exp = self.expert_input(x_snap, h_t)
        mu, log_sigma = self.experts(x_exp)

        pi = prior.log_prior.exp()  # -inf -> exactly 0 for sparse priors
        y_hat = (pi * mu).sum(dim=-1)
        return NECOutput(
            gate_logits=gate_logits,
            prior=prior,
            mu=mu,
            log_sigma=log_sigma,
            y_hat=y_hat,
            h_t=h_t,
        )
