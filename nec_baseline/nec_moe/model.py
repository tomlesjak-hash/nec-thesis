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

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseModel
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
    mu: Tensor  # (B, K) expert means (base + correction in correction mode)
    log_sigma: Tensor  # (K,) expert noise scales
    y_hat: Tensor  # (B,) mixture-mean point prediction
    h_t: Tensor  # (B, n) encoder state
    # Residual design (brief 02); None outside correction mode.
    base_pred: Tensor | None = None  # (B,) f0(x), frozen
    corrections: Tensor | None = None  # (B, K) per-expert corrections r_k(x)

    @property
    def correction(self) -> Tensor | None:
        """``sum_k pi_k r_k`` ``(B,)`` — the correction actually applied.

        The reported quantity of brief 02 §5: a model whose corrections are
        numerically negligible has answered the thesis question in the
        negative whatever its R-squared does.
        """
        if self.corrections is None:
            return None
        return (self.prior.log_prior.exp() * self.corrections).sum(dim=-1)

    @property
    def expert_signal(self) -> Tensor:
        """What the **experts themselves** contribute: ``r_k``, else ``mu_k``.

        Every diagnostic and auxiliary loss that asks "how different are the
        experts from each other?" must read this rather than ``mu``. In
        correction mode ``mu_k = f0 + r_k`` shares one base across all K
        columns, so any between-expert statistic computed on ``mu`` is
        measuring the base: correlations are pulled toward 1 however the
        experts behave, and a decorrelation penalty applied to ``mu`` can only
        be satisfied by inflating ``r`` until it overwhelms ``f0`` — a
        pressure directly opposed to the shrinkage the design wants.

        ``corrections`` is materialised by the forward pass (the experts'
        raw output, captured *before* the base is added), never recovered as
        ``mu - f0``: a subtraction would be a lossy float round-trip and
        would silently produce garbage if the base were ever changed between
        the two reads.
        """
        return self.mu if self.corrections is None else self.corrections


def _assert_normalized(log_prior: Tensor, atol: float = 1e-5) -> None:
    """Raise unless every row of ``log_prior`` sums to 1 in probability domain.

    Checked in the log domain (``logsumexp`` == 0), which is where sparse
    priors live: ``hard`` and ``topk`` carry ``-inf`` entries that are exactly
    0 after exponentiation, so a probability-domain sum would work but would
    round-trip through ``exp`` for no reason.
    """
    err = torch.logsumexp(log_prior.detach(), dim=-1).abs().max()
    if float(err) > atol:
        raise ValueError(
            f"prior rows must sum to 1 (max |logsumexp| = {float(err):.3e} > "
            f"{atol:g}). The residual prediction f0 + sum_k pi_k r_k relies on "
            "sum_k pi_k = 1; an unnormalized prior rescales the frozen base "
            "instead, which no loss curve would show"
        )


class NECModel(nn.Module):
    """Corrected NEC: sequence encoder → regime gate → mixture of experts."""

    #: registered buffer: has a fitted base been attached? (declared for the
    #: type checker, the way Emission declares log_sigma)
    base_fitted: Tensor

    def __init__(self, cfg: NECConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.encoder = GRUEncoder(cfg.data, cfg.encoder)
        self.gate = GateHead(cfg.encoder.hidden_dim, cfg.experts.n_experts, cfg.gate)
        self.prior = build_prior(cfg)
        self.experts = build_emission(cfg)
        # Frozen base f0 of the residual design. Its *architecture* is fully
        # determined by the config, so the module is built here and its
        # fitted *weights* arrive later via attach_base — which is what lets
        # state_dict round-trip (a checkpoint restores the exact floor a
        # result was measured against) instead of failing on unexpected keys
        # when Trainer.load rebuilds the model. The flag is a buffer so it
        # round-trips too: a restored model knows whether it was fitted.
        self.base: BaseModel | None = None
        if cfg.base.enabled:
            self.base = BaseModel(cfg.data.d_snap, cfg.base).freeze()
        self.register_buffer("base_fitted", torch.zeros((), dtype=torch.bool))
        # Evaluation switch, not state: set only inside corrections_disabled()
        # and never saved (audit finding E-1).
        self._corrections_off = False

    # ------------------------------------------------------------ the base
    def attach_base(self, base: BaseModel) -> NECModel:
        """Load an already-fitted, already-frozen base's weights.

        Frozen means ``requires_grad=False``, which is also what keeps the
        base out of the optimizer: parameter groups are built by filtering on
        that flag, so a frozen base is structurally unable to train even if
        someone rebuilds the optimizer.
        """
        if self.base is None:
            raise ValueError(
                "attach_base requires base.enabled=True — attaching a base to "
                "a config that does not declare one would make the fitted "
                "model differ from its own recorded config"
            )
        if any(p.requires_grad for p in base.parameters()):
            raise ValueError(
                "the base must be frozen before it is attached (call "
                "BaseModel.freeze(); fit_base returns a frozen model): a live "
                "base would co-adapt with the experts and destroy the "
                "attribution the frozen design exists to provide"
            )
        self.base.load_state_dict(base.state_dict())
        self.base.freeze()
        self.base_fitted.fill_(True)
        return self

    @contextlib.contextmanager
    def corrections_disabled(self) -> Iterator[NECModel]:
        """Evaluate the SAME model with every correction forced to exactly zero.

        This is how the base is scored (audit finding E-1). With ``r_k = 0``
        every expert's mean is ``f0``, so the mixture density becomes
        ``sum_k pi_k N(y; f0, s_k^2)``: the base's predictions under the
        model's own noise model, prior and (for stateful priors) recursion.
        The NLL improvement ``base - mixture`` then measures what the
        corrections add and nothing else. It is exactly zero when every
        correction is zero, whatever the experts' noise scales. Scoring the
        base as one Gaussian at an averaged sigma instead would credit the
        experts' variance structure to the corrections.
        """
        if not self.cfg.experts.correction_mode:
            raise ValueError(
                "corrections_disabled() needs experts.correction_mode=True: "
                "outside correction mode there is no base to fall back to"
            )
        previous = self._corrections_off
        self._corrections_off = True
        try:
            yield self
        finally:
            self._corrections_off = previous

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

        # Residual design: expert k's predictive mean for y is f0(x) + r_k(x),
        # so adding the base here — rather than anywhere downstream — leaves
        # the likelihood, the fused mixture NLL and y_hat untouched. Since
        # sum_k pi_k = 1 for every prior, y_hat = sum_k pi_k (f0 + r_k)
        # collapses to exactly f0 + sum_k pi_k r_k, the design equation.
        base_pred, corrections = None, None
        if self.cfg.experts.correction_mode:
            if self.base is None or not bool(self.base_fitted):
                raise RuntimeError(
                    "correction_mode is on but no fitted base is attached: "
                    "the experts' corrections would be corrections to an "
                    "untrained random network. The walk-forward harness fits "
                    "and attaches one per fold; a manual caller must call "
                    "attach_base() with the result of fit_base()"
                )
            corrections = torch.zeros_like(mu) if self._corrections_off else mu
            base_pred = self.base(x_snap)
            mu = base_pred.unsqueeze(-1) + corrections
            # The residual identity is y_hat = sum_k pi_k (f0 + r_k)
            #                                = f0 * (sum_k pi_k) + sum_k pi_k r_k,
            # which collapses to the design equation ONLY because sum_k pi_k = 1.
            # Every registered prior normalizes, but that is a property of each
            # prior rather than of this call site, so it is checked here, where
            # the base's coefficient depends on it. A prior that silently failed
            # to normalize would scale f0 by sum_k pi_k and shift every
            # prediction — visible in no loss curve and in no gradient.
            _assert_normalized(prior.log_prior)

        pi = prior.log_prior.exp()  # -inf -> exactly 0 for sparse priors
        y_hat = (pi * mu).sum(dim=-1)
        return NECOutput(
            gate_logits=gate_logits,
            prior=prior,
            mu=mu,
            log_sigma=log_sigma,
            y_hat=y_hat,
            h_t=h_t,
            base_pred=base_pred,
            corrections=corrections,
        )
