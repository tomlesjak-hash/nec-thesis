"""Gate calibration: reliability diagrams + temperature scaling (M4 §4.9).

Why this exists
---------------
A gate can *rank* regimes correctly and still lie about probabilities — and the
gate's probabilities are mixture weights, so systematic overconfidence
misallocates the experts on every forward pass (M4's capstone: "a 0.9 that
means 0.6 is a 50% overallocation delivered with confidence").

What "calibrated" means for a latent-regime gate
------------------------------------------------
Regimes are latent, so there is no observed label. The training semantics
supply the target: the gate's whole job is to predict the **responsibilities**
(its gradient is ``pi − r``), so on *held-out* data the gate is calibrated iff

    E[ r_k | pi_k = q ]  =  q        for all q,

where ``r`` is the Bayes posterior over experts computed from the held-out
``y`` (the mixture's update step). :func:`gate_reliability` bins the pooled
``(sample, expert)`` claims ``pi_nk`` and compares each bin's mean claim to its
mean realized responsibility; the gap, count-weighted, is the expected
calibration error (ECE).

The repair
----------
Temperature scaling — the gauge-respecting multiclass Platt scaling: replace
``pi = softmax(z)`` with ``softmax(z / T)`` for a single scalar ``T`` fitted on
a **validation tail of the training window** (same discipline as tuning; never
the test block). :func:`fit_temperature` picks ``T`` by minimizing the held-out
**mixture NLL** — the principled objective for a mixture, since ``T`` is then
just one extra likelihood parameter — and reports NLL and ECE before/after.
``T > 1`` softens an overconfident gate; ``T < 1`` sharpens an underconfident
one; a well-calibrated gate fits ``T ≈ 1``.

Scope: priors whose *predictive* weights are ``softmax(gate_logits)`` — soft,
and Gumbel at eval. Hard/top-k routing has no smooth confidence to rescale,
``uniform`` ignores the logits entirely, and the HMM's regime probabilities
come from the forward filter, not the gate — all rejected with guidance.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from .data import Panel
from .likelihood import expert_log_likelihood
from .losses import mixture_nll
from .priors import GumbelSoftmaxRegimePrior, PriorContext, SoftRegimePrior
from .train import Trainer

__all__ = [
    "ReliabilityReport",
    "gate_reliability",
    "TemperatureFit",
    "fit_temperature",
]


def _check_softmax_family(trainer: Trainer) -> None:
    prior = trainer.model.prior
    if not isinstance(prior, (SoftRegimePrior, GumbelSoftmaxRegimePrior)):
        raise ValueError(
            f"gate calibration applies to softmax-predictive gates (soft, "
            f"gumbel-at-eval); got {type(prior).__name__}. Hard/top-k have no "
            "smooth confidence to rescale, uniform ignores the logits, and "
            "the HMM's regime probabilities come from the forward filter — "
            "calibrating those is a different exercise."
        )


@torch.no_grad()
def _gate_logits_and_loglik(
    trainer: Trainer, panel: Panel, batch_size: int = 65536
) -> tuple[Tensor, Tensor]:
    """Chunked eval-mode forward: gate logits (N, K) + expert log-liks (N, K)."""
    model = trainer.model
    model.eval()
    zs, lls = [], []
    for i in range(0, len(panel), batch_size):
        sl = slice(i, i + batch_size)
        out = model(
            panel.x_seq[sl], panel.x_snap[sl],
            PriorContext(date=panel.date[sl]),
        )
        zs.append(out.gate_logits)
        lls.append(expert_log_likelihood(out.mu, out.log_sigma, panel.y[sl]))
    return torch.cat(zs), torch.cat(lls)


@dataclass(frozen=True)
class ReliabilityReport:
    """A reliability diagram in numbers (pooled ``(sample, expert)`` claims)."""

    bin_edges: tuple[float, ...]  # n_bins + 1 edges on [0, 1]
    bin_confidence: tuple[float, ...]  # mean claimed pi per bin (nan if empty)
    bin_outcome: tuple[float, ...]  # mean realized responsibility per bin
    bin_weight: tuple[float, ...]  # share of claims per bin (sums to 1)
    ece: float  # sum_b weight_b * |confidence_b − outcome_b|
    nll: float  # held-out mixture NLL at this temperature
    temperature: float  # the T these numbers were computed under
    n_claims: int  # N * K pooled claims

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "bin_lo": self.bin_edges[:-1],
                "bin_hi": self.bin_edges[1:],
                "confidence": self.bin_confidence,
                "outcome": self.bin_outcome,
                "weight": self.bin_weight,
            }
        )


def _reliability_from(
    z: Tensor, log_lik: Tensor, temperature: float, n_bins: int
) -> ReliabilityReport:
    log_prior = F.log_softmax(z / temperature, dim=-1)
    nll_out = mixture_nll(log_prior, log_lik)
    pi = log_prior.exp().flatten()  # pooled claims
    r = nll_out.responsibilities.flatten()  # pooled realized posteriors

    edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    idx = torch.clamp(
        torch.bucketize(pi.double(), edges[1:-1], right=False), 0, n_bins - 1
    )
    conf, out, weight = [], [], []
    n = pi.numel()
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        weight.append(c / n)
        conf.append(float(pi[m].mean()) if c else float("nan"))
        out.append(float(r[m].mean()) if c else float("nan"))
    ece = sum(
        w * abs(c - o) for w, c, o in zip(weight, conf, out, strict=True) if w > 0
    )
    return ReliabilityReport(
        bin_edges=tuple(float(e) for e in edges),
        bin_confidence=tuple(conf),
        bin_outcome=tuple(out),
        bin_weight=tuple(weight),
        ece=float(ece),
        nll=float(nll_out.nll),
        temperature=temperature,
        n_claims=n,
    )


def gate_reliability(
    trainer: Trainer,
    panel: Panel,
    *,
    temperature: float = 1.0,
    n_bins: int = 10,
) -> ReliabilityReport:
    """Held-out reliability of the gate's mixture weights (see module docstring).

    Run it on data the model did not train on — a validation tail or a test
    block; in-sample reliability flatters the gate. ``temperature`` lets you
    inspect the diagram before (1.0) and after (a fitted T) scaling.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    _check_softmax_family(trainer)
    z, log_lik = _gate_logits_and_loglik(trainer, panel)
    return _reliability_from(z, log_lik, temperature, n_bins)


@dataclass(frozen=True)
class TemperatureFit:
    """The fitted correction plus its before/after evidence."""

    temperature: float
    nll_before: float  # held-out mixture NLL at T = 1
    nll_after: float  # at the fitted T
    ece_before: float
    ece_after: float
    n_claims: int

    @property
    def verdict(self) -> str:
        if abs(self.temperature - 1.0) < 0.1:
            return "gate approximately calibrated (T ≈ 1): leave it alone"
        kind = "overconfident (T > 1 softens)" if self.temperature > 1 else \
            "underconfident (T < 1 sharpens)"
        return f"gate {kind}: T = {self.temperature:.2f}"


def fit_temperature(
    trainer: Trainer,
    val_panel: Panel,
    *,
    n_bins: int = 10,
    grid_points: int = 61,
) -> TemperatureFit:
    """Fit the temperature on a validation panel by held-out mixture NLL.

    Protocol: ``val_panel`` must be a validation tail of the *training* window
    (:func:`nec_moe.tuning.validation_tail`) — fitting on the test block would
    launder test information into the model. Deterministic two-stage grid over
    ``log T`` (coarse on [0.1, 10], refined around the optimum); one scalar
    fitted on hundreds+ of dates cannot overfit meaningfully, which is the
    charm of temperature scaling.
    """
    _check_softmax_family(trainer)
    z, log_lik = _gate_logits_and_loglik(trainer, val_panel)

    def nll_at(t: float) -> float:
        log_prior = F.log_softmax(z / t, dim=-1)
        return float(mixture_nll(log_prior, log_lik).nll)

    coarse = torch.logspace(-1.0, 1.0, grid_points)
    best = min(coarse, key=lambda t: nll_at(float(t)))
    fine = torch.logspace(
        float(torch.log10(best)) - 0.15, float(torch.log10(best)) + 0.15, 41
    )
    t_star = float(min(fine, key=lambda t: nll_at(float(t))))

    before = _reliability_from(z, log_lik, 1.0, n_bins)
    after = _reliability_from(z, log_lik, t_star, n_bins)
    return TemperatureFit(
        temperature=t_star,
        nll_before=before.nll,
        nll_after=after.nll,
        ece_before=before.ece,
        ece_after=after.ece,
        n_claims=before.n_claims,
    )
