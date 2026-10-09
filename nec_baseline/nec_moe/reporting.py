"""Reporting additions (brief 12; Q9 update 2026-10-09): **reporting only**.

Computed after a run, from what it saved (``scripts/report_run.py``): each
fold's per-security predictions and labels, the panel's daily forward
returns, the fitted gates. Nothing here is called by training or by the
walk-forward harness, so no fitted object, no existing metric, no settings
hash and no resume state can change. The existing metrics are recomputed by
the same functions (:func:`~nec_moe.evaluation.rank_ic_by_date`,
:func:`~nec_moe.evaluation.long_short_book`) and come out identical.

**Part A, long leg against short leg.** On the existing book's own legs
(:func:`~nec_moe.evaluation._quantile_legs`, the same ``n_quantiles``): the
long leg is the mean market-neutral target of the top quantile, the short
leg minus the mean of the bottom one, so long + short is the book's gross
spread. The target is demeaned with equal weights over the universe, so each
leg is already measured against the universe average. The rank IC is also
computed within the upper and within the lower half of each date's forecast
distribution (ranks recomputed inside each half).

**Part B, decile monotonicity.** Each date's names are sorted by forecast
into ``n_groups`` groups of (nearly) equal size (:func:`sort_group_returns`;
the sizes differ by at most one, the extra names in the lowest groups), and
each group's mean target is recorded. :func:`monotonic_relation_test` is
Patton & Timmermann's (2010) test: the null that mean returns do not rise,
against the alternative that every adjacent difference is positive; the
statistic is the smallest adjacent difference of the means, and its p-value
comes from Politis & Romano's (1994) stationary bootstrap of the dates (whole
cross-sections, so the groups' correlation is kept), recentred at the
sample differences (the least favourable null). Blocks average
``mean_block`` dates, ``max(10, 2h)`` by default, so they span the target's
overlap.

**Part C, by year, named episode and regime.** The same statistics by
calendar year and by the eight episodes fixed in Q9 (:data:`EPISODES`; a
date belongs to an episode when its **formation date** lies in the window),
and, for gated arms, the regime-weighted means of brief 11
(:func:`nec_moe.regime_split.weighted_regime_ic`) with the arm's own Q27
window-average weights, rebuilt from each fold's saved gate
(:func:`window_weights_from_gate_state`).

**Standard errors.** Hansen-Hodrick, through
:func:`~nec_moe.evaluation.hac_variance`: lag ``h - 1`` on every per-date
series of ``h``-day returns (ICs, half-sample ICs, sort-group returns), lag 0
on the book's own periods (non-overlapping ``h``-day periods, or the
staggered book's daily returns), which do not overlap (Tom's decision
2026-10-09). Every summary reports the lag it used.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .evaluation import _quantile_legs, hac_variance, long_short_book, rank_ic_by_date
from .markov_gate import gate_weight_probs

__all__ = [
    "SeriesSummary",
    "series_summary",
    "leg_returns",
    "half_sample_ics",
    "sort_group_returns",
    "stationary_bootstrap_indices",
    "monotonic_relation_test",
    "shape_spearman",
    "mr_simulation",
    "EPISODES",
    "in_window",
    "window_weights_from_gate_state",
]


@dataclass(frozen=True)
class SeriesSummary:
    """Mean of a per-date (or per-period) series with its HAC standard error."""

    mean: float
    se: float
    t: float
    std: float  # the series' own standard deviation (for an IC: ICIR = mean / std)
    n: int
    lags: int
    kernel: str

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def series_summary(x: Tensor | np.ndarray, lags: int) -> SeriesSummary:
    """Mean, Hansen-Hodrick standard error (uniform kernel, ``lags``; the
    Bartlett fallback if the uniform variance is not positive) and count."""
    t = (x.detach().double().flatten().clone() if isinstance(x, Tensor)
         else torch.from_numpy(np.array(x, dtype=float)).flatten())  # copies
    t = t[torch.isfinite(t)]
    n = int(t.numel())
    if n < 2:
        nan = float("nan")
        return SeriesSummary(float(t.mean()) if n else nan, nan, nan, nan, n, lags, "none")
    mean, std = float(t.mean()), float(t.std(unbiased=True))
    lrv, used = hac_variance(t, min(lags, n - 1), "uniform")
    se = math.sqrt(lrv / n) if lrv > 0 else float("nan")
    return SeriesSummary(mean, se, mean / se if se > 0 else float("nan"), std, n, lags, used)


def leg_returns(
    pred: Tensor,
    y: Tensor,
    date: Tensor,
    entity: Tensor,
    *,
    n_quantiles: int,
    horizon: int,
    scheme: str = "nonoverlapping",
    y_daily: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``(formation dates, gross, turnover, long leg, short leg)`` per period.

    ``gross`` and ``turnover`` are :func:`~nec_moe.evaluation.long_short_book`'s
    own output (the existing book, unchanged); the legs come from the same
    :func:`~nec_moe.evaluation._quantile_legs`:

    - ``"nonoverlapping"``: per period, ``long = mean(y[top])`` and
      ``short = -mean(y[bottom])`` in the book's float32, so
      ``long + short == gross`` exactly;
    - ``"staggered"``: each date's long (short) leg is the average over the
      live cohorts of the top (minus the bottom) quantile's daily return, as
      the book averages their difference; the sum equals the gross to float32
      rounding (the book averages the cohorts' rounded differences).
    """
    dates, gross, tno = long_short_book(
        pred, y, date, entity, n_quantiles=n_quantiles, horizon=horizon, scheme=scheme,
        y_daily=y_daily,
    )
    legs = _quantile_legs(pred, date, n_quantiles)
    if scheme == "nonoverlapping":
        long = torch.stack([y[top].mean() for _, _, top in legs[::horizon]])
        short = torch.stack([-y[bottom].mean() for _, bottom, _ in legs[::horizon]])
        return dates, gross, tno, long, short
    daily = y.unsqueeze(1) if horizon == 1 else y_daily
    assert daily is not None  # long_short_book has refused a staggered book without it
    c_long = [daily[top].mean(dim=0) for _, _, top in legs]
    c_short = [-daily[bottom].mean(dim=0) for _, bottom, _ in legs]

    def book_average(cohorts: list[Tensor]) -> Tensor:
        return torch.tensor([
            sum(float(cohorts[j - m][m]) for m in range(min(j, horizon - 1) + 1)) / horizon
            for j in range(len(legs))
        ])

    return dates, gross, tno, book_average(c_long), book_average(c_short)


def half_sample_ics(
    pred: Tensor, y: Tensor, date: Tensor
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """``((dates, ICs) of the lower half, (dates, ICs) of the upper half)``:
    each date's names split by forecast into the bottom and the top
    ``n // 2`` (the median name dropped when ``n`` is odd, as
    :func:`~nec_moe.evaluation._quantile_legs` with two groups does), and the
    rank IC of each half by :func:`~nec_moe.evaluation.rank_ic_by_date`,
    ranks recomputed within the half."""
    halves = _quantile_legs(pred, date, 2)
    lower = torch.cat([bottom for _, bottom, _ in halves])
    upper = torch.cat([top for _, _, top in halves])
    return (
        rank_ic_by_date(pred[lower], y[lower], date[lower]),
        rank_ic_by_date(pred[upper], y[upper], date[upper]),
    )


# --------------------------------------------------------------------------- #
# Part B: decile monotonicity
# --------------------------------------------------------------------------- #


def sort_group_returns(pred: Tensor, y: Tensor, date: Tensor, n_groups: int
                       ) -> tuple[Tensor, np.ndarray]:
    """``(dates (T,), returns (T, n_groups))``: each date's names sorted by
    forecast (ascending, so group 1 holds the lowest forecasts) and split
    into ``n_groups`` contiguous groups of nearly equal size; each group's
    mean target."""
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2, got {n_groups}")
    dates, out = [], []
    for d in torch.unique(date, sorted=True):
        rows = torch.nonzero(date == d, as_tuple=True)[0]
        if len(rows) < n_groups:
            raise ValueError(f"date {int(d)} has {len(rows)} names, fewer than {n_groups} groups")
        order = rows[torch.argsort(pred[rows])].numpy()
        yd = y.double().numpy()
        out.append([float(yd[g].mean()) for g in np.array_split(order, n_groups)])
        dates.append(d)
    return torch.stack(dates), np.asarray(out, dtype=float)


def stationary_bootstrap_indices(t_len: int, mean_block: float, reps: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """``(reps, t_len)`` resampled date indices (Politis & Romano 1994): each
    position starts a new block at a uniform date with probability
    ``1 / mean_block``, else continues the previous one (circularly)."""
    if mean_block < 1:
        raise ValueError(f"mean_block must be >= 1, got {mean_block}")
    q = 1.0 / mean_block
    idx = np.empty((reps, t_len), dtype=np.int64)
    starts = rng.integers(0, t_len, size=(reps, t_len))
    new = rng.random((reps, t_len)) < q
    idx[:, 0] = starts[:, 0]
    for t in range(1, t_len):
        idx[:, t] = np.where(new[:, t], starts[:, t], (idx[:, t - 1] + 1) % t_len)
    return idx


def monotonic_relation_test(returns: np.ndarray, *, mean_block: float, reps: int,
                            seed: int) -> dict[str, float | list[float]]:
    """Patton & Timmermann's (2010) monotonic relation test on a ``(T, G)``
    date-by-group series: H0 mean returns not increasing, H1 every adjacent
    difference positive. ``J = min_j (mean_{j+1} - mean_j)``; the p-value is
    the share of stationary-bootstrap draws with
    ``min_j (diff*_j - diff_j) > J``. Reproducible under ``seed``."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r).all(axis=1)]
    t_len, g = r.shape
    if g < 2 or t_len < 2:
        raise ValueError(f"need at least 2 groups and 2 dates, got {r.shape}")
    diffs = np.diff(r.mean(axis=0))
    j_stat = float(diffs.min())
    idx = stationary_bootstrap_indices(t_len, mean_block, reps, np.random.default_rng(seed))
    boot = np.stack([np.diff(r[i].mean(axis=0)) for i in idx])  # (reps, g - 1)
    j_boot = (boot - diffs).min(axis=1)
    return {"statistic": j_stat, "p": float((j_boot > j_stat).mean()),
            "differences": diffs.tolist(), "reps": reps, "mean_block": mean_block,
            "dates": t_len}


def shape_spearman(group_means: np.ndarray) -> float:
    """Spearman correlation between group number and mean return (average
    ranks), a descriptive summary of the curve's shape."""
    m = np.asarray(group_means, dtype=float)
    ranks = _average_ranks(m)
    pos = np.arange(1, len(m) + 1, dtype=float)
    return float(np.corrcoef(pos, ranks)[0, 1])


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties shared."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        tie = x == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks


def mr_simulation(true_means: np.ndarray, *, n_sims: int = 300, t_len: int = 750,
                  h: int = 5, noise_sd: float = 0.02, common_sd: float = 0.01,
                  reps: int = 299, alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Rejection rate of :func:`monotonic_relation_test` at ``alpha`` for
    groups with ``true_means``: each date's group returns are the means plus
    an ``h - 1``-order moving average (an ``h``-day target's overlap) of a
    common shock (sd ``common_sd``) and group-specific shocks (sd
    ``noise_sd``). The bootstrap's mean block is ``max(10, 2h)``."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(true_means, dtype=float)
    g = len(mu)
    kernel = np.ones(h) / np.sqrt(h)
    rejects = 0
    for i in range(n_sims):
        common = rng.normal(0.0, common_sd, t_len + h - 1)
        own = rng.normal(0.0, noise_sd, (t_len + h - 1, g))
        noise = np.column_stack([np.convolve(own[:, j] + common, kernel, mode="valid")
                                 for j in range(g)])
        out = monotonic_relation_test(mu + noise, mean_block=max(10, 2 * h), reps=reps,
                                      seed=seed + 1 + i)
        rejects += int(out["p"] < alpha)  # type: ignore[operator]
    return {"rejection_rate": rejects / n_sims, "n_sims": float(n_sims), "t_len": float(t_len),
            "h": float(h), "reps": float(reps), "alpha": alpha}


# --------------------------------------------------------------------------- #
# Part C: by year, named episode and regime
# --------------------------------------------------------------------------- #

#: The named episodes, fixed in Q9 (update 2026-10-09) before any test-period
#: number was computed: (name, first date, last date), both inclusive.
EPISODES: tuple[tuple[str, str, str], ...] = (
    ("euro_debt_flash_crash_2010", "2010-04-26", "2010-07-02"),
    ("us_downgrade_2011", "2011-07-25", "2011-10-31"),
    ("china_oil_2015_16", "2015-08-17", "2016-02-29"),
    ("q4_selloff_2018", "2018-10-01", "2018-12-31"),
    ("covid_crash_2020", "2020-02-20", "2020-04-30"),
    ("covid_rebound_2020", "2020-05-01", "2020-12-31"),
    ("bear_market_2022", "2022-01-03", "2022-10-31"),
    ("regional_banks_2023", "2023-03-08", "2023-05-05"),
)


def in_window(formation: pd.Series | pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    """Which formation dates lie in ``[start, end]`` (both inclusive). A
    period belongs to an episode by its formation date, even when its return
    window reaches past the episode's end."""
    d = pd.DatetimeIndex(formation)
    return np.asarray((d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end)))


def window_weights_from_gate_state(state: dict, codes: list[int], horizon: int) -> np.ndarray:
    """``(len(codes), K)`` Q27 window-average weights of a saved Hamilton
    gate on these date codes: its filtered probabilities moved by its
    transition matrix and averaged over ``horizon`` steps
    (:func:`~nec_moe.markov_gate.gate_weight_probs`), canonical order (calm
    first). Raises for a date the gate never filtered."""
    extra = state.get("extra", {})
    fit, filtered = extra.get("fit_result"), extra.get("filtered", {})
    if fit is None:
        raise ValueError("the gate state carries no fitted transition matrix")
    missing = [c for c in codes if int(c) not in filtered]
    if missing:
        raise ValueError(f"{len(missing)} date(s) without a filtered probability, first "
                         f"{missing[:5]}")
    xi = np.stack([np.asarray(filtered[int(c)], dtype=float) for c in codes])
    return gate_weight_probs(xi, np.asarray(fit.transition, dtype=float), "window", horizon)
