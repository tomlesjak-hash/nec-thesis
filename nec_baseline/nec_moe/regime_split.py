"""Regime-split diagnostics on 2000-2006 (brief 11): descriptive, never a forecast.

Pure functions of arrays and frames; the scripts (``scripts/ic_regime_split.py``,
``scripts/sector_regimes.py``) load the data under the hard date guard
(2000-01-01 to 2006-12-31, no override) and write the aggregate outputs.

**Part A, the market's regime probabilities.** The Hamilton gate
(:class:`~nec_moe.markov_gate.MarkovSwitchingRegimePrior`: switching mean and
variance, the existing multi-start, canonical order by ascending variance, so
the last regime is the stress one) is fitted on the whole span with full
memory (equal weights: the pilot has not chosen the gate's memory, and equal
weights are the reference arm). Only **filtered** probabilities are used,
never smoothed ones: the gate reads ``filtered_marginal_probabilities``
(statsmodels) or the native forward pass. A horizon's regime weight is Q27's
window average, :func:`~nec_moe.markov_gate.gate_weight_probs` with
``"window"``; the filtered probability itself is the robustness variant.

The parameters are estimated on all of 2000-2006 (:data:`IN_SAMPLE_CAVEAT`):
the probabilities are causal in the data they filter, not in the parameters.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import MarkovGateConfig
from .data import FeatureSchema, Panel
from .markov_gate import MarkovSwitchingRegimePrior, gate_weight_probs

__all__ = [
    "IN_SAMPLE_CAVEAT",
    "WEIGHT_KINDS",
    "RegimeFit",
    "series_panel",
    "fit_regimes",
    "regime_weights",
    "regime_names",
    "regime_summary",
    "trailing_vol_state",
]

#: Written into every output of brief 11 (A.4), and into the figure captions.
IN_SAMPLE_CAVEAT = (
    "The gate's parameters are estimated on all of 2000-2006, so the regime "
    "probabilities are causal in the data they filter (each date's probability "
    "uses market returns up to that date only; filtered, never smoothed) but not "
    "in the parameters. A descriptive in-sample diagnostic, not a forecast."
)

#: The regime weights a date can carry (A.3): Q27's window average over the
#: target window, or the filtered probability itself (robustness).
WEIGHT_KINDS: tuple[str, ...] = ("window", "filtered")


@dataclass(frozen=True)
class RegimeFit:
    """A fitted K-state gate on one series: filtered probabilities and the
    parameters, all in canonical order (ascending variance; the last regime
    is the stress one)."""

    k: int
    labels: tuple[str, ...]  # ISO dates, ascending, one per filtered row
    filtered: np.ndarray  # (T, K) filtered probabilities, never smoothed
    transition: np.ndarray  # (K, K) row-stochastic
    means: np.ndarray  # (K,)
    variances: np.ndarray  # (K,)
    expected_durations: np.ndarray  # (K,) days
    llf: float
    backend: str
    n_starts: int
    n_converged: int
    chosen_start: int
    start_llfs: tuple[float, ...]
    start_status: tuple[str, ...]
    n_distinct_optima: int

    @property
    def stress(self) -> np.ndarray:
        """``(T,)`` filtered probability of the highest-variance regime."""
        return self.filtered[:, self.k - 1]


def series_panel(
    labels: Sequence[str],
    values: np.ndarray,
    *,
    horizon: int = 5,
    metadata: dict[str, Any] | None = None,
) -> Panel:
    """A one-row-per-date panel carrying a date-level series.

    The series sits in sequence channel 0 (the gate's ``"sequence_channel"``
    key reads it there); ``metadata`` can carry the CRSP market series under
    its own key (``crsp_market_log_return``). Nothing else is in the panel:
    the gate reads only the dates and the series."""
    n = len(labels)
    x = np.asarray(values, dtype=float)
    if x.shape != (n,):
        raise ValueError(f"{n} labels but values of shape {x.shape}")
    x_seq = torch.zeros(n, 1, 1, dtype=torch.float32)
    x_seq[:, -1, 0] = torch.from_numpy(x).to(torch.float32)
    return Panel(
        x_seq=x_seq, x_snap=torch.zeros(n, 1), y=torch.zeros(n),
        date=torch.arange(n), entity=torch.zeros(n, dtype=torch.long),
        schema=FeatureSchema(sequence_features=("series",), snapshot_features=("none",),
                             target=f"fwd_ret_{horizon}d"),
        date_labels=tuple(str(pd.Timestamp(d).date()) for d in labels),
        metadata=dict(metadata or {}),
    )


def fit_regimes(panel: Panel, k: int, cfg: MarkovGateConfig) -> RegimeFit:
    """Fit the Hamilton gate with ``k`` states on ``panel``'s series, full
    memory, and return its filtered probabilities and parameters.

    ``cfg`` sets the series key, the estimator and the multi-start; its
    ``k_regimes`` and ``memory`` are overridden (``k``, ``"full"``)."""
    cfg = dataclasses.replace(cfg, k_regimes=k, memory="full")
    gate = MarkovSwitchingRegimePrior(k, cfg, horizon=panel.horizon)
    gate.fit(panel)
    fit = gate.fit_result
    assert fit is not None  # fit() either sets it or raises
    codes = torch.unique(panel.date, sorted=True)[cfg.order:]
    assert panel.date_labels is not None
    return RegimeFit(
        k=k,
        labels=tuple(panel.date_labels[int(c)] for c in codes),
        filtered=gate.filtered_probabilities(codes),
        transition=np.asarray(fit.transition, dtype=float),
        means=np.asarray(fit.means, dtype=float),
        variances=np.asarray(fit.variances, dtype=float),
        expected_durations=np.asarray(fit.expected_durations, dtype=float),
        llf=float(fit.llf),
        backend=fit.backend,
        n_starts=int(fit.n_starts),
        n_converged=int(fit.n_converged),
        chosen_start=int(fit.chosen_start),
        start_llfs=tuple(float(x) for x in fit.start_llfs),
        start_status=tuple(fit.start_status),
        n_distinct_optima=len(fit.distinct_optima),
    )


def regime_weights(fit: RegimeFit, kind: str, h: int) -> pd.DataFrame:
    """Dates x regimes weights for horizon ``h`` (A.3), rows summing to 1:
    Q27's window average ``(1/h) sum_j xi_t A^j`` (``"window"``) or the
    filtered probability ``xi_t`` (``"filtered"``)."""
    if kind not in WEIGHT_KINDS:
        raise ValueError(f"weight kind {kind!r}: use one of {WEIGHT_KINDS}")
    w = gate_weight_probs(fit.filtered, fit.transition, kind, h)
    return pd.DataFrame(w, index=pd.DatetimeIndex(pd.to_datetime(list(fit.labels))),
                        columns=regime_names(fit.k))


def regime_names(k: int) -> list[str]:
    """Canonical regime names: ascending variance, so calm first, stress last."""
    if k == 2:
        return ["calm", "stress"]
    if k == 3:
        return ["calm", "middle", "stress"]
    return [f"regime_{j}" for j in range(k)]


def regime_summary(fit: RegimeFit, stress_threshold: float = 0.5) -> dict[str, Any]:
    """Aggregate description of a fit (A.6): parameters, durations, the share
    of days each regime is the most likely one, the multi-start trace."""
    names = regime_names(fit.k)
    most_likely = fit.filtered.argmax(axis=1)
    return {
        "k": fit.k,
        "regimes": names,
        "dates": len(fit.labels),
        "first_date": fit.labels[0],
        "last_date": fit.labels[-1],
        "backend": fit.backend,
        "llf": fit.llf,
        "means": fit.means.tolist(),
        "variances": fit.variances.tolist(),
        "sd_daily": np.sqrt(fit.variances).tolist(),
        "transition": fit.transition.tolist(),
        "expected_durations_days": fit.expected_durations.tolist(),
        "share_of_days_most_likely": [float((most_likely == j).mean()) for j in range(fit.k)],
        "mean_filtered_probability": fit.filtered.mean(axis=0).tolist(),
        "share_of_days_stress_above_threshold": float((fit.stress > stress_threshold).mean()),
        "stress_threshold": stress_threshold,
        "multi_start": {
            "n_starts": fit.n_starts,
            "n_converged": fit.n_converged,
            "chosen_start": fit.chosen_start,
            "n_distinct_optima": fit.n_distinct_optima,
            "start_llfs": list(fit.start_llfs),
            "start_status": list(fit.start_status),
        },
    }


def trailing_vol_state(log_ret: pd.Series, window: int) -> tuple[pd.Series, float]:
    """The parameter-free robustness state (A.5): 1 where the trailing
    ``window``-day standard deviation of ``log_ret`` (data up to and
    including t) is above its median over the series, else 0; missing for the
    first ``window - 1`` dates. Returns the state and the median."""
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    vol = log_ret.rolling(window, min_periods=window).std()
    threshold = float(vol.median())
    state = (vol > threshold).astype(float).where(vol.notna())
    return state, threshold
