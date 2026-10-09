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

**Part B, the IC split by regime.** For a daily IC series and regime weights
``w_k(t)`` (summing to 1 on each date):

- the regime-weighted mean IC, ``sum_t w_k IC_t / sum_t w_k``, with Kish's
  effective number of dates (:func:`weighted_regime_ic`);
- the pure-regime IC with inference, the regression without intercept
  ``IC_t = sum_k c_k w_k(t) + e_t`` (:func:`regime_regression`). Its
  sandwich standard errors use :func:`nec_moe.evaluation.hac_variance` on the
  scalar series ``z_t = a' (X'X)^-1 x_t e_t`` of each linear combination
  ``a'c``: ``Var(a'c) = T * LRV(z)``, which is exactly the quadratic form
  ``a' (X'X)^-1 S (X'X)^-1 a`` with ``S`` the same kernel's long-run
  variance of ``x_t e_t`` (the long-run variance is linear in the
  autocovariances). ``z`` has mean zero by the normal equations, so the
  estimator's demeaning changes nothing; its ``T - 1`` divisor inflates the
  variance by ``T / (T - 1)``. One HAC implementation, as brief 11 asks;
- the hypothesis families (:data:`PRIMARY_HYPOTHESES`), fixed before any run.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import baum_welch as bw
from .config import MarkovGateConfig
from .data import FeatureSchema, Panel
from .decay import kish_ess
from .evaluation import hac_variance
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
    "HAC_CHOICES",
    "hac_choice",
    "RegimeRegression",
    "regime_regression",
    "weighted_regime_ic",
    "hard_split",
    "PRIMARY_HYPOTHESES",
    "PRIMARY_SPEC",
    "p_value",
    "contrast_size_simulation",
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


# --------------------------------------------------------------------------- #
# Part B: the IC split by regime
# --------------------------------------------------------------------------- #

#: The two HAC choices of B.2: Hansen-Hodrick (uniform kernel, lag h - 1,
#: the target's overlap) and Newey-West (Bartlett, lag max(h - 1, nw_min_lag),
#: because a persistent regressor makes the residuals' autocorrelation outlast
#: the overlap).
HAC_CHOICES: tuple[str, ...] = ("hansen_hodrick", "newey_west")


def hac_choice(choice: str, h: int, nw_min_lag: int) -> tuple[int, str]:
    """``(lags, kernel)`` of a HAC choice at horizon ``h``."""
    if choice == "hansen_hodrick":
        return h - 1, "uniform"
    if choice == "newey_west":
        return max(h - 1, nw_min_lag), "bartlett"
    raise ValueError(f"unknown HAC choice {choice!r}; use one of {HAC_CHOICES}")


@dataclass(frozen=True)
class RegimeRegression:
    """``IC_t = sum_k c_k w_k(t) + e_t`` without intercept, one HAC choice."""

    coef: np.ndarray  # (K,) c_k: the IC on a date fully in regime k
    se: np.ndarray  # (K,)
    t: np.ndarray  # (K,)
    contrast: float  # c_last - c_first: stress minus calm
    contrast_se: float
    contrast_t: float
    kernels: tuple[str, ...]  # kernel actually used per c_k, then the contrast
    lags: int
    n: int

    @staticmethod
    def undefined(k: int, lags: int, n: int) -> RegimeRegression:
        nan = np.full(k, np.nan)
        return RegimeRegression(nan, nan, nan, float("nan"), float("nan"), float("nan"),
                                ("none",) * (k + 1), lags, n)


def _combination_se(x: np.ndarray, resid: np.ndarray, xtx_inv: np.ndarray, a: np.ndarray,
                    lags: int, kernel: str) -> tuple[float, str]:
    """SE of ``a'c`` from the scalar series ``z_t = a'(X'X)^-1 x_t e_t``
    (module docstring): ``sqrt(T * hac_variance(z))``."""
    z = (x @ (xtx_inv @ a)) * resid
    lrv, used = hac_variance(torch.from_numpy(z), lags, kernel)
    return float(np.sqrt(max(len(z) * lrv, 0.0))), used


def regime_regression(ic: np.ndarray, w: np.ndarray, lags: int, kernel: str) -> RegimeRegression:
    """The pure-regime IC regression (B.2) with sandwich standard errors.

    ``ic`` is ``(T,)``, ``w`` the ``(T, K)`` regime weights in canonical order
    (calm first, stress last). Undefined (NaN) when ``X'X`` is singular, e.g.
    a hard split without a single stress date."""
    y = np.asarray(ic, dtype=float)
    x = np.asarray(w, dtype=float)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"ic {y.shape} and weights {x.shape} do not align")
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y, x = y[keep], x[keep]
    t_len, k = x.shape
    xtx = x.T @ x
    if t_len <= k or np.linalg.matrix_rank(xtx) < k:
        return RegimeRegression.undefined(k, lags, t_len)
    xtx_inv = np.linalg.inv(xtx)
    coef = xtx_inv @ (x.T @ y)
    resid = y - x @ coef
    se, kernels = np.empty(k), []
    for j in range(k):
        se[j], used = _combination_se(x, resid, xtx_inv, np.eye(k)[j], lags, kernel)
        kernels.append(used)
    a = np.zeros(k)
    a[-1], a[0] = 1.0, -1.0
    c_se, used = _combination_se(x, resid, xtx_inv, a, lags, kernel)
    kernels.append(used)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = coef / se
        c_t = float((coef[-1] - coef[0]) / c_se) if c_se > 0 else float("nan")
    return RegimeRegression(coef, se, t_stats, float(coef[-1] - coef[0]), c_se, c_t,
                            tuple(kernels), lags, t_len)


def weighted_regime_ic(ic: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(IC_k, effective dates_k)``: the regime-weighted mean IC
    ``sum_t w_k IC_t / sum_t w_k`` and Kish's ``(sum w)^2 / sum w^2`` (B.1)."""
    y = np.asarray(ic, dtype=float)
    x = np.asarray(w, dtype=float)
    keep = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y, x = y[keep], x[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        means = (x * y[:, None]).sum(axis=0) / x.sum(axis=0)
    eff = np.array([kish_ess(x[:, j]) for j in range(x.shape[1])])
    return means, eff


def hard_split(stress: np.ndarray, threshold: float) -> np.ndarray:
    """``(T, 2)`` hard weights (B.3): ``[1 - D, D]`` with
    ``D = 1[stress > threshold]``; stress against the rest (for K = 3 the calm
    and middle regimes are pooled). Missing stays missing."""
    p = np.asarray(stress, dtype=float)
    d = (p > threshold).astype(float)
    d[~np.isfinite(p)] = np.nan
    return np.column_stack([1.0 - d, d])


#: B.4, declared before the run. Every primary test is one-sided: the
#: contrast c_stress - c_calm is negative (reversal's negative IC more
#: negative in stress; momentum's positive IC smaller in stress).
PRIMARY_HYPOTHESES: tuple[dict[str, Any], ...] = (
    {"id": "H1", "claim": "short-term reversal is stronger (more negative IC) in stress",
     "signals": ("ret_5d", "ret_20d", "ret_20d_ind_rel"), "alternative": "contrast < 0",
     "sources": "Nagel (2012); Hameed & Mian (2015)"},
    {"id": "H2", "claim": "momentum is weaker in stress",
     "signals": ("mom_12_1", "ind_mom_12_1"), "alternative": "contrast < 0",
     "sources": "Cooper, Gutierrez & Hameed (2004); Wang & Xu (2015); Daniel & Moskowitz (2016)"},
)

#: The specification every primary test uses (B.4): K = 2, h = 5,
#: Hansen-Hodrick, the soft (pure-regime) regression on the window weights.
PRIMARY_SPEC: dict[str, Any] = {
    "k": 2, "h": 5, "hac": "hansen_hodrick", "split": "soft", "weight_kind": "window",
    "correction": "Holm over the five one-sided tests",
}


def p_value(t: float, alternative: str) -> float:
    """Normal p-value of a t-statistic: ``"less"`` (one-sided, H_a: below 0)
    or ``"two-sided"``."""
    if not np.isfinite(t):
        return float("nan")
    cdf = NormalDist().cdf(t)
    if alternative == "less":
        return cdf
    if alternative == "two-sided":
        return 2.0 * min(cdf, 1.0 - cdf)
    raise ValueError(f"unknown alternative {alternative!r}")


def contrast_size_simulation(
    n_sims: int = 500,
    t_len: int = 1750,
    h: int = 5,
    alpha: float = 0.05,
    nw_min_lag: int = 20,
    stay: tuple[float, float] = (0.99, 0.97),
    sd: tuple[float, float] = (0.006, 0.018),
    ic_sd: float = 0.15,
    drift_sd: float = 0.0,
    drift_rho: float = 0.98,
    seed: int = 0,
) -> dict[str, float]:
    """Size of the two-sided contrast test when the IC does not depend on the
    regime (B, tests): the rejection rate at ``alpha`` under both HAC choices.

    Each simulation draws a two-state market series (``stay``, ``sd``), takes
    its filtered stress probability with the true parameters and Q27's window
    weight at ``h`` as the regressor (persistent, like the real one), and a
    daily IC with no regime effect: a constant plus an ``h - 1``-order moving
    average, the overlap of an ``h``-day target (sd ``ic_sd``). With
    ``drift_sd > 0`` the IC also carries a slow AR(1) drift (stationary sd
    ``drift_sd``, autocorrelation ``drift_rho``), still independent of the
    regime: the case Newey-West's long lag is there for. Returns the rejection
    rate per HAC choice and the mean and sd of the contrast."""
    rng = np.random.default_rng(seed)
    a = np.array([[stay[0], 1 - stay[0]], [1 - stay[1], stay[1]]])
    pi0 = np.array([1 - stay[1], 1 - stay[0]]) / (2 - stay[0] - stay[1])
    params = bw.HMMParams(np.zeros(2), np.asarray(sd) ** 2, a, pi0)
    rejects = {c: 0 for c in HAC_CHOICES}
    contrasts: list[float] = []
    for _ in range(n_sims):
        s = np.zeros(t_len, dtype=int)
        s[0] = rng.uniform() < pi0[1]
        for t in range(1, t_len):
            s[t] = s[t - 1] if rng.uniform() < a[s[t - 1], s[t - 1]] else 1 - s[t - 1]
        market = rng.normal(0.0, np.asarray(sd)[s])
        filtered, _ = bw.filter_series(market, params)
        w = gate_weight_probs(filtered, a, "window", h)
        u = rng.normal(0.0, ic_sd, t_len + h - 1)
        ic = 0.02 + np.convolve(u, np.ones(h) / np.sqrt(h), mode="valid")
        if drift_sd > 0:
            drift = np.empty(t_len)
            drift[0] = rng.normal(0.0, drift_sd)
            shocks = rng.normal(0.0, drift_sd * np.sqrt(1 - drift_rho**2), t_len)
            for t in range(1, t_len):
                drift[t] = drift_rho * drift[t - 1] + shocks[t]
            ic = ic + drift
        for choice in HAC_CHOICES:
            lags, kernel = hac_choice(choice, h, nw_min_lag)
            res = regime_regression(ic, w, lags, kernel)
            if p_value(res.contrast_t, "two-sided") < alpha:
                rejects[choice] += 1
        contrasts.append(res.contrast)
    return {c: rejects[c] / n_sims for c in HAC_CHOICES} | {
        "mean_contrast": float(np.mean(contrasts)), "sd_contrast": float(np.std(contrasts)),
        "n_sims": float(n_sims), "t_len": float(t_len), "h": float(h), "alpha": alpha}
