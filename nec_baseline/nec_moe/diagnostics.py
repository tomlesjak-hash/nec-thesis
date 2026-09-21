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

Chain persistence (for any prior exposing a transition matrix):
:func:`expected_durations` and :func:`stationary_distribution` report how sticky
a *fitted* chain actually is. Persistence is the property that distinguishes the
gate families, so a chain whose expected duration is near one period is a
degenerate regime process and must be visible immediately rather than inferred
from returns. :func:`persistence_metrics` packs both into flat, registry-safe
keys so the trainer's history and the trial registry use one spelling.

Regime identity across fits (Decision D): :func:`canonical_expert_order` is the
declared scalar sort (ascending ``sigma_k``) applied after each fit before any
cross-fit aggregation. :func:`wasserstein_template_tracking` is the documented
escalation (arXiv:2603.04441); it is **not implemented** and raises — see its
docstring.
"""

from __future__ import annotations

import math

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
    "expected_durations",
    "stationary_distribution",
    "persistence_metrics",
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


# --------------------------------------------------------------------------- #
# Chain persistence: how sticky is the fitted regime process?
# --------------------------------------------------------------------------- #


def _as_transition(transition: Tensor, *, atol: float = 1e-4) -> Tensor:
    """Validate a row-stochastic ``(K, K)`` matrix; return it detached, float64.

    Row-stochasticity is checked rather than assumed: these functions are the
    reporting path for *fitted* chains, and silently summarizing a matrix that
    is not a transition matrix would defeat the point.
    """
    a = transition.detach().to(torch.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 1:
        raise ValueError(
            f"transition must be a square (K, K) matrix, got {tuple(transition.shape)}"
        )
    if bool((a < -atol).any()):
        raise ValueError("transition has negative entries — not a stochastic matrix")
    rows = a.sum(dim=1)
    if not torch.allclose(rows, torch.ones_like(rows), atol=atol):
        raise ValueError(
            f"transition rows must sum to 1 (atol={atol}), got row sums {rows.tolist()}"
        )
    return a


def expected_durations(transition: Tensor) -> Tensor:
    """``1 / (1 - A_kk)`` ``(K,)`` — expected consecutive periods spent in regime k.

    For a first-order chain, leaving state ``k`` is a Bernoulli trial each
    period with success probability ``1 - A_kk``, so the sojourn time is
    geometric with mean ``1 / (1 - A_kk)`` (Hamilton 1989, §3; Kim & Nelson
    1999, §4). A fitted value near 1 means the chain re-draws its state every
    period — a degenerate regime process wearing a transition matrix.

    An absorbing state (``A_kk == 1``) yields ``inf``, which is the honest
    answer; :func:`persistence_metrics` is the registry-safe wrapper that
    handles it.
    """
    a = _as_transition(transition)
    return 1.0 / (1.0 - a.diagonal())


def stationary_distribution(transition: Tensor) -> Tensor:
    """``pi (K,)`` solving ``pi A = pi`` with ``sum_k pi_k = 1``.

    The normalized left eigenvector of ``A`` for eigenvalue 1 — the long-run
    share of time the chain spends in each regime, which is the marginal a
    memoryless prior would have to match to be comparable. Computed as the
    least-squares solution of the stacked system ``[A^T - I; 1^T] pi = [0; 1]``
    (SVD driver: exact for a well-posed chain, minimum-norm otherwise) rather
    than by power iteration, which converges arbitrarily slowly for precisely
    the sticky chains this project fits.

    The solution is unique iff the chain has a single recurrent class; for a
    reducible chain any convex combination of the per-class stationary vectors
    is stationary and the minimum-norm representative is returned.
    """
    a = _as_transition(transition)
    k = a.shape[0]
    lhs = torch.cat(
        [a.T - torch.eye(k, dtype=a.dtype), torch.ones(1, k, dtype=a.dtype)], dim=0
    )
    rhs = torch.zeros(k + 1, 1, dtype=a.dtype)
    rhs[k, 0] = 1.0
    return torch.linalg.lstsq(lhs, rhs, driver="gelsd").solution.squeeze(1)


def persistence_metrics(
    transition: Tensor, order: Tensor | None = None
) -> dict[str, float]:
    """Flat ``{name: float}`` persistence summary, safe for the trial registry.

    Keys per regime ``k``: ``stay_prob_k`` (the diagonal), ``stationary_k``,
    and ``expected_duration_k``. The duration key is **omitted** for an
    absorbing state, where the true value is ``inf`` and the registry
    (deliberately) rejects non-finite metrics; ``stay_prob_k`` is always
    present and bounded, so persistence is never silently missing from a
    logged trial.

    ``order`` is Decision D's canonical permutation
    (:func:`canonical_expert_order`): states are relabelled by it before
    reporting, so ``expected_duration_0`` means the same regime across refits
    and cross-fold aggregation is not averaging apples with oranges.
    """
    a = _as_transition(transition)
    if order is not None:
        idx = order.detach().to(torch.long)
        if idx.shape != (a.shape[0],):
            raise ValueError(
                f"order must have shape ({a.shape[0]},), got {tuple(idx.shape)}"
            )
        a = a[idx][:, idx]
    stay = a.diagonal()
    durations = 1.0 / (1.0 - stay)
    pi = stationary_distribution(a)
    out: dict[str, float] = {}
    for k in range(a.shape[0]):
        out[f"stay_prob_{k}"] = float(stay[k])
        out[f"stationary_{k}"] = float(pi[k])
        d = float(durations[k])
        if math.isfinite(d):
            out[f"expected_duration_{k}"] = d
    return out


def wasserstein_template_tracking(*_args: object, **_kwargs: object) -> tuple[int, ...]:
    """Escalation path for cross-refit regime identity (arXiv:2603.04441).

    **Not implemented** — calling this raises :class:`NotImplementedError`.
    The intended mechanism maps each refit's regime Gaussians to persistent
    templates via the closed-form 2-Wasserstein distance between Gaussians,
    with exponential template smoothing; the baseline instead uses the
    declared scalar sort :func:`canonical_expert_order` (design doc §9bis,
    Decision D).

    Distinct from the Wasserstein *gate* (a regime prior fitted by clustering
    segment-level empirical distributions): that one assigns regimes, this one
    tracks regime identity across refits. Keep the names apart.
    """
    raise NotImplementedError(
        "wasserstein_template_tracking is the Decision D escalation path and is "
        "not implemented; the baseline uses canonical_expert_order (scalar "
        "sigma sort). See NEC_sources/NEC_baseline_design.md §9bis. This is "
        "not the Wasserstein gate — that is a RegimePrior, a different object."
    )
