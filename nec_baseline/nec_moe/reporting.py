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
import torch
from torch import Tensor

from .evaluation import _quantile_legs, hac_variance, long_short_book, rank_ic_by_date

__all__ = [
    "SeriesSummary",
    "series_summary",
    "leg_returns",
    "half_sample_ics",
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
    t = torch.from_numpy(np.array(x, dtype=float)).flatten()  # a copy: pandas may be read-only
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
