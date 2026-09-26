"""Selection-aware inference (Module 5): deflated Sharpe, Bonferroni, BH-FDR.

The moment a result is selected from N candidates, its naive significance is
inflated; this module holds the three corrections the syllabus assigns:

- **Deflated Sharpe ratio** (Bailey & López de Prado 2014) — corrects a
  *max-of-N* selection on Sharpe: the probabilistic Sharpe ratio evaluated
  against the expected maximum Sharpe of N unskilled trials (which grows like
  ``sqrt(V[SR]) * Phi^-1(1 - 1/N)``), with the PSR's denominator additionally
  correcting for the **non-normality** of returns (skew, kurtosis). The
  ``n_trials`` input is exactly :meth:`nec_moe.registry.TrialRegistry.n_trials`.
- **Bonferroni** — family-wise error control for a *tested family*: reject at
  ``alpha / N``; conservative under correlated tests.
- **Benjamini–Hochberg** — false-discovery-rate control, the more appropriate
  criterion when screening many signals/configs (syllabus: the factor-zoo
  discipline). Step-up procedure with monotone adjusted p-values.

All per-period, no annualization (synthetic dates carry no calendar); the
normal quantile comes from torch (no scipy dependency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .evaluation import IcSummary, hac_variance, long_run_variance

__all__ = [
    "sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "DeflatedSharpe",
    "deflated_sharpe_ratio",
    "effective_sample_size",
    "ic_pvalue",
    "bonferroni",
    "benjamini_hochberg",
]

_EULER_GAMMA = 0.577215664901532860606512090082

_STD_NORMAL = torch.distributions.Normal(0.0, 1.0)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    return float(_STD_NORMAL.icdf(torch.tensor(p, dtype=torch.float64)))


def _moments(returns: Tensor) -> tuple[float, float, float, float, int]:
    r = returns.double().flatten()
    t = int(r.numel())
    if t < 4:
        raise ValueError(f"need >= 4 return observations, got {t}")
    mean = r.mean()
    centered = r - mean
    std = centered.pow(2).mean().sqrt()  # population std (LdP convention)
    if float(std) == 0.0:
        raise ValueError("returns have zero variance")
    skew = float(centered.pow(3).mean() / std**3)
    kurt = float(centered.pow(4).mean() / std**4)  # Pearson: normal -> 3
    return float(mean), float(std), skew, kurt, t


def effective_sample_size(
    returns: Tensor, hac_lags: int, hac_kernel: str = "uniform"
) -> float:
    """``T * gamma_0 / LRV``: the i.i.d.-equivalent number of periods, at most ``T``.

    ``LRV`` is the long-run variance over ``hac_lags`` autocovariances
    (:func:`nec_moe.evaluation.long_run_variance`, Hansen-Hodrick by
    default). For a daily series of overlapping ``h``-period returns,
    ``hac_lags = h - 1`` gives roughly ``T / h`` (audit finding E-2): the
    series carries about one independent observation per ``h`` days. Capped
    at ``T``, so negative autocorrelation never makes a track record look
    longer than it is. ``hac_lags = 0`` returns ``T``.
    """
    r = returns.detach().double().flatten()
    t = int(r.numel())
    if hac_lags == 0:
        return float(t)
    lrv, _ = hac_variance(r, hac_lags, hac_kernel)
    gamma0 = long_run_variance(r, 0)
    if lrv <= 0:
        return float(t)
    return min(float(t), t * gamma0 / lrv)


def sharpe_ratio(returns: Tensor) -> float:
    """Per-period Sharpe ``mean / std`` (population std, no annualization)."""
    mean, std, _, _, _ = _moments(returns)
    return mean / std


def probabilistic_sharpe_ratio(
    returns: Tensor,
    sr_benchmark: float = 0.0,
    *,
    hac_lags: int = 0,
    hac_kernel: str = "uniform",
) -> float:
    """PSR: P(true SR > ``sr_benchmark``) given the observed track record.

    Bailey & LdP: ``Phi( (SR - SR*) * sqrt(T - 1) /
    sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2) )`` — fat tails and negative skew
    widen the denominator and deflate the confidence (the non-normality half
    of the correction).

    ``hac_lags > 0`` replaces ``T`` with :func:`effective_sample_size`, for a
    serially correlated series such as a daily-formed book of ``h``-day
    returns (pass ``h - 1``; audit E-2). Bailey and LdP's formula assumes
    i.i.d. periods, so at ``T`` it overstates the confidence for such a series.
    """
    mean, std, skew, kurt, t_obs = _moments(returns)
    t = effective_sample_size(returns, hac_lags, hac_kernel)
    sr = mean / std
    denom_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom_sq <= 0:
        raise ValueError(
            f"PSR denominator non-positive (skew={skew:.2f}, kurt={kurt:.2f}, "
            f"SR={sr:.2f}) — track record too pathological for the approximation"
        )
    if t <= 1.0:
        raise ValueError(
            f"effective sample size {t:.2f} of {t_obs} periods is too small for a PSR"
        )
    return _phi((sr - sr_benchmark) * math.sqrt(t - 1.0) / math.sqrt(denom_sq))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] of ``n_trials`` unskilled (true SR = 0) trials.

    Bailey & LdP's extreme-value approximation:
    ``sqrt(V[SR]) * ((1 - g) * z(1 - 1/N) + g * z(1 - 1/(N e)))`` with ``g``
    the Euler–Mascheroni constant. ``sr_variance`` is the cross-trial variance
    of the *estimated* Sharpe ratios (from the trial registry). ``N = 1``
    returns 0: a single un-selected trial has no selection effect.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if sr_variance < 0:
        raise ValueError(f"sr_variance must be >= 0, got {sr_variance}")
    if n_trials == 1 or sr_variance == 0.0:
        return 0.0
    z1 = _phi_inv(1.0 - 1.0 / n_trials)
    z2 = _phi_inv(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


@dataclass(frozen=True)
class DeflatedSharpe:
    """The full deflation story, for reporting."""

    sharpe: float  # observed per-period SR
    psr_zero: float  # naive PSR against SR* = 0 (no selection correction)
    expected_max_sr: float  # the selection hurdle implied by n_trials
    dsr: float  # PSR against that hurdle — the honest number
    n_trials: int
    skewness: float
    kurtosis: float
    n_periods: int
    effective_periods: float = float("nan")  # T after the serial-correlation correction
    hac_lags: int = 0


def deflated_sharpe_ratio(
    returns: Tensor,
    n_trials: int,
    sr_variance: float,
    *,
    hac_lags: int = 0,
    hac_kernel: str = "uniform",
) -> DeflatedSharpe:
    """DSR: PSR of the *selected* track record against E[max SR] of the trials.

    ``n_trials`` and ``sr_variance`` (variance of estimated SRs across the
    family) come from the trial registry:
    ``n = reg.n_trials(tag)``; ``var = statistics.pvariance(reg.metric_values(tag, "sharpe"))``.
    A DSR near 1 says the record survives its own selection multiplicity; a
    DSR near 0.5 or below says the "discovery" looks like the lucky best of N.

    ``hac_lags``: pass ``h - 1`` for a daily series of ``h``-period returns
    (see :func:`probabilistic_sharpe_ratio`).
    """
    mean, std, skew, kurt, t = _moments(returns)
    sr = mean / std
    hurdle = expected_max_sharpe(n_trials, sr_variance)
    return DeflatedSharpe(
        sharpe=sr,
        psr_zero=probabilistic_sharpe_ratio(
            returns, 0.0, hac_lags=hac_lags, hac_kernel=hac_kernel
        ),
        expected_max_sr=hurdle,
        dsr=probabilistic_sharpe_ratio(
            returns, hurdle, hac_lags=hac_lags, hac_kernel=hac_kernel
        ),
        n_trials=n_trials,
        skewness=skew,
        kurtosis=kurt,
        n_periods=t,
        effective_periods=effective_sample_size(returns, hac_lags, hac_kernel),
        hac_lags=hac_lags,
    )


# --------------------------------------------------------------------------- #
# p-value families
# --------------------------------------------------------------------------- #


def ic_pvalue(summary: IcSummary) -> float:
    """Two-sided p-value of ``mean IC = 0`` from the IC t-stat (normal approx).

    The t-stat carries the summary's Newey-West correction
    (``IcSummary.hac_lags``), so an overlapping-horizon IC series gets an
    honest p-value when the summary was built with ``hac_lags = h - 1``.
    """
    return 2.0 * (1.0 - _phi(abs(summary.t_stat)))


def bonferroni(pvals: list[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """FWER control: ``(reject_mask, adjusted_pvals = min(1, p * N))``."""
    _check_pvals(pvals, alpha)
    n = len(pvals)
    adjusted = [min(1.0, p * n) for p in pvals]
    return [a <= alpha for a in adjusted], adjusted


def benjamini_hochberg(
    pvals: list[float], alpha: float = 0.10
) -> tuple[list[bool], list[float]]:
    """FDR control (BH step-up): ``(reject_mask, monotone q-values)``.

    Rejects all hypotheses with rank <= the largest k such that
    ``p_(k) <= k/N * alpha``. FDR is the right criterion when screening many
    candidate signals: it controls the expected *fraction* of false
    discoveries, where FWER controls the chance of even one — too strict for
    a factor-zoo screen (syllabus, Module 5).
    """
    _check_pvals(pvals, alpha)
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    # q-values: p_(i) * N / rank, made monotone from the largest rank down
    q_sorted = [pvals[order[i]] * n / (i + 1) for i in range(n)]
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q_sorted = [min(1.0, q) for q in q_sorted]
    qvals = [0.0] * n
    for rank, idx in enumerate(order):
        qvals[idx] = q_sorted[rank]
    # step-up rejection
    k_star = 0
    for i in range(n):
        if pvals[order[i]] <= (i + 1) / n * alpha:
            k_star = i + 1
    reject = [False] * n
    for i in range(k_star):
        reject[order[i]] = True
    return reject, qvals


def _check_pvals(pvals: list[float], alpha: float) -> None:
    if not pvals:
        raise ValueError("empty p-value family")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    bad = [p for p in pvals if not 0.0 <= p <= 1.0]
    if bad:
        raise ValueError(f"p-values outside [0, 1]: {bad[:3]}")
