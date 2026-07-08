"""Failure modes as dashboards (design doc §4.4) and regime-recovery scoring.

All functions consume detached tensors and are cheap enough to log every eval
interval:

- gate entropy: uniform forever = dead gate; instant sharpness = collapse;
- utilization at two scopes (per-batch may legitimately be extreme on a single
  date; the *running* scope, from the load-balance buffer, is what the
  near-zero alarm watches);
- responsibility sharpness: healthy specialization sharpens gradually;
- expert output correlation ``R_jk``: the homogenization watch, logged even
  when the decorrelation aux is off.

Regime identity across fits (Decision D): :func:`canonical_expert_order` is the
declared scalar sort (ascending ``sigma_k``) applied after each fit before any
cross-fit aggregation. :func:`wasserstein_template_tracking` is the documented
escalation (arXiv:2603.04441) — a stub this pass.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "gate_entropy",
    "utilization",
    "sharpness",
    "expert_output_correlation",
    "binary_auc",
    "regime_recovery_auc",
    "canonical_expert_order",
    "wasserstein_template_tracking",
]


def gate_entropy(probs: Tensor, eps: float = 1e-12) -> Tensor:
    """Mean per-sample entropy of ``probs (B, K)`` in nats (max: log K)."""
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1).mean()


def utilization(responsibilities: Tensor) -> Tensor:
    """Per-batch effective share of each expert: ``mean_b r_bk`` -> (K,)."""
    return responsibilities.mean(dim=0)


def sharpness(responsibilities: Tensor) -> Tensor:
    """``mean_b max_k r_bk`` — 1/K when uniform, 1.0 when one-hot."""
    return responsibilities.max(dim=-1).values.mean()


def expert_output_correlation(mu: Tensor, eps: float = 1e-8) -> Tensor:
    """Batch correlation matrix ``R (K, K)`` of expert outputs (homogenization watch)."""
    centered = mu - mu.mean(dim=0, keepdim=True)
    norms = centered.norm(dim=0).clamp_min(eps)
    unit = centered / norms
    return unit.T @ unit


def binary_auc(scores: Tensor, labels: Tensor) -> float:
    """AUC of ``scores (N,)`` against boolean ``labels (N,)`` (Mann–Whitney).

    No sklearn dependency; average ranks handle ties.
    """
    scores = scores.detach().double().flatten()
    labels = labels.detach().bool().flatten()
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("binary_auc needs both classes present")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    # average ranks over exact ties
    uniq, inv, counts = torch.unique(scores, return_inverse=True, return_counts=True)
    if len(uniq) != len(scores):
        sums = torch.zeros(len(uniq), dtype=torch.float64).scatter_add_(0, inv, ranks)
        ranks = (sums / counts.double())[inv]
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def regime_recovery_auc(prob_expert: Tensor, true_regime: Tensor) -> float:
    """Label-switching-proof AUC of one expert's probability vs a binary regime.

    Reports ``max(auc, 1 - auc)``: which learned expert corresponds to which
    true regime is a permutation the likelihood cannot pin down (Module 8
    failure mode 2), so orientation is not part of the claim.
    """
    auc = binary_auc(prob_expert, true_regime == 1)
    return max(auc, 1.0 - auc)


def canonical_expert_order(log_sigma: Tensor) -> Tensor:
    """Decision D's declared statistic: expert indices sorted by ascending sigma.

    Apply after each fit, before any cross-fit/per-regime aggregation.
    """
    return torch.argsort(log_sigma.detach())


def wasserstein_template_tracking(*_args, **_kwargs):  # pragma: no cover - stub
    """Escalation path for cross-refit regime identity (arXiv:2603.04441).

    Maps each refit's regime Gaussians to persistent templates via the
    closed-form 2-Wasserstein distance between Gaussians, with exponential
    template smoothing. Deliberately unimplemented this pass — see design doc
    §9bis (Decision D): the baseline uses :func:`canonical_expert_order`.
    """
    raise NotImplementedError(
        "Wasserstein template tracking is the Decision D escalation path; "
        "the baseline uses canonical_expert_order (scalar sigma sort). "
        "See NEC_sources/NEC_baseline_design.md §9bis."
    )
