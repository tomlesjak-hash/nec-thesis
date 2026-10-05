"""Decay weights for the base and the experts (brief 09 C; Q16 (d)(e)).

The specification is ``Master Thesis/Training_Window_and_Weighting_Theory.md``
sections 4-6. Let ``T`` be the last training date of the fold.

**Base: calendar exponential decay** (section 4). A training row on date
``s`` gets ``w_base(s) = 2 ** (-age(s) / H_base)``, with ``age(s)`` the number
of trading days from ``s`` to ``T``, counted on the panel's own trading
calendar (its date codes). ``H_base = None`` or infinity means equal weights.

**Experts: regime-clock decay, mixture form** (section 5). With ``xi_u(k)``
the frozen gate's **filtered** probabilities on the training block (never the
window-average gate weights, never the smoothed probabilities),

    n_k(s, T) = sum over u = s+1 .. T of xi_u(k)      # later regime-k experience
    w_exp(s)  = sum over k of xi_s(k) * rho ** n_k(s, T),   rho = 2 ** (-1 / H)

with ``H`` in regime-days. A row ages only by later experience of its own
regime, so calm data turns over quickly and a crisis keeps its weight until
the next crisis replaces it. ``n_k`` is a reverse cumulative sum, ``O(T K)``.
The per-expert form (expert ``k``'s gradient weighted by ``rho ** n_k``
alone) is an open minor point (Q16) and is **not** implemented.

**Using the weights** (C.3). Both losses become weighted means,
``sum(w * per_row_loss) / sum(w)``, normalised by the sum so the step size
does not depend on the half-life (Pesaran, Pick & Pranovich 2013 normalise
their forecast weights to sum to one in the same way). The weights multiply
each row's term of the per-row composite likelihood (Q24; Varin, Reid &
Firth 2011): each term keeps a zero-mean score, so the weights change
efficiency, not consistency. They are fixed per fold, computed once before
training from data up to ``T`` only, and reported only as summaries
(:func:`weight_summary`): the Kish effective sample size overall and per
regime, and the weight mass by calendar year.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "EXPERT_DECAYS",
    "no_decay",
    "decay_factor",
    "calendar_decay_weights",
    "regime_clock_exponents",
    "regime_clock_components",
    "regime_clock_weights",
    "kish_ess",
    "weight_summary",
    "weighted_mean",
]

#: How the experts' training rows are weighted (``TrainConfig.expert_decay``).
EXPERT_DECAYS: tuple[str, ...] = ("none", "regime_clock")


def no_decay(half_life: float | None) -> bool:
    """``None`` or infinity: equal weights, and the weighting is skipped
    entirely, so such a run is bit-identical to an unweighted one."""
    return half_life is None or math.isinf(half_life)


def decay_factor(half_life: float) -> float:
    """``rho = 2 ** (-1 / H)``: the factor per day (or regime-day) of age."""
    if not half_life > 0:
        raise ValueError(f"a half-life must be > 0, got {half_life}")
    return 2.0 ** (-1.0 / half_life)


def calendar_decay_weights(date: Tensor, half_life: float | None) -> Tensor | None:
    """``w(s) = 2 ** (-age(s) / H)`` per row (float64), ``age(s) = T - s`` in
    date codes (the panel's trading days), ``T`` the latest date in ``date``.
    ``None`` when ``half_life`` means equal weights (:func:`no_decay`)."""
    if no_decay(half_life):
        return None
    assert half_life is not None
    age = (date.max() - date).to(torch.float64)
    return torch.pow(torch.tensor(2.0, dtype=torch.float64), -age / half_life)


def regime_clock_exponents(filtered: np.ndarray) -> np.ndarray:
    """``n_k(s, T) = sum_{u=s+1..T} xi_u(k)``, ``(T, K)``: a reverse
    cumulative sum minus the row itself. Row ``s`` uses only rows after it
    and up to the last row, so nothing past ``T`` can enter."""
    xi = np.asarray(filtered, dtype=float)
    if xi.ndim != 2:
        raise ValueError(f"filtered probabilities must be (T, K), got {xi.shape}")
    inclusive = np.cumsum(xi[::-1], axis=0)[::-1]
    return inclusive - xi


def regime_clock_components(filtered: np.ndarray, half_life: float) -> np.ndarray:
    """``(T, K)``: ``xi_s(k) * rho ** n_k(s, T)``, each regime's share of row
    ``s``'s weight (they sum over ``k`` to :func:`regime_clock_weights`)."""
    xi = np.asarray(filtered, dtype=float)
    if math.isinf(half_life):
        return xi.copy()
    return xi * decay_factor(half_life) ** regime_clock_exponents(xi)


def regime_clock_weights(filtered: np.ndarray, half_life: float) -> np.ndarray:
    """``w_exp(s) = sum_k xi_s(k) rho ** n_k(s, T)`` per date, ``(T,)``."""
    return regime_clock_components(filtered, half_life).sum(axis=1)


def kish_ess(weights: np.ndarray | Tensor, counts: np.ndarray | None = None) -> float:
    """Kish's effective sample size ``(sum w)^2 / sum w^2``. With ``counts``,
    weight ``w_i`` stands for ``counts_i`` identical rows (a date's weight
    on each of its stocks)."""
    w = np.asarray(weights, dtype=float)
    c = np.ones_like(w) if counts is None else np.asarray(counts, dtype=float)
    denom = float(np.sum(c * w * w))
    return float(np.sum(c * w)) ** 2 / denom if denom > 0 else 0.0


def weighted_mean(per_row: Tensor, weight: Tensor | None) -> Tensor:
    """``sum(w * l) / sum(w)``, or the plain mean when ``weight`` is None
    (the exact unweighted path, so equal weights cost nothing and change
    nothing)."""
    if weight is None:
        return per_row.mean()
    w = weight.to(per_row.dtype)
    return (w * per_row).sum() / w.sum()


def weight_summary(
    prefix: str,
    date_weights: np.ndarray,
    dates: Tensor,
    rows_per_date: np.ndarray,
    date_labels: Sequence[str] | None,
    components: np.ndarray | None = None,
) -> dict[str, float]:
    """Aggregate diagnostics of one fold's weights (brief 09 C.3), keyed by
    ``prefix``:

    - ``<prefix>_ess_dates``: Kish ESS over dates (each date counted once);
    - ``<prefix>_ess_rows``: Kish ESS over rows (a date's weight on each of
      its stocks);
    - ``<prefix>_ess_regime_<k>``: Kish ESS over dates of regime ``k``'s
      part of the weight (``components[:, k]``), when given;
    - ``<prefix>_mass_<year>``: the share of the total row weight on dates of
      that calendar year (needs ``date_labels``).

    ``date_weights``, ``dates`` and ``rows_per_date`` are aligned per date.
    No per-row or per-security value is returned.
    """
    w = np.asarray(date_weights, dtype=float)
    n = np.asarray(rows_per_date, dtype=float)
    out = {
        f"{prefix}_ess_dates": kish_ess(w),
        f"{prefix}_ess_rows": kish_ess(w, n),
    }
    if components is not None:
        for k in range(components.shape[1]):
            out[f"{prefix}_ess_regime_{k}"] = kish_ess(components[:, k])
    if date_labels is not None:
        total = float(np.sum(w * n))
        years = np.array([int(str(date_labels[int(d)])[:4]) for d in dates])
        for year in np.unique(years):
            mass = float(np.sum((w * n)[years == year]))
            out[f"{prefix}_mass_{int(year)}"] = mass / total if total > 0 else 0.0
    return out
