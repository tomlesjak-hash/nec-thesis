"""Walk-forward evaluation harness: purged splits, rank-IC, ICIR (Module 5 machinery).

The skeleton of the thesis evaluation protocol, synthetic-testable now and
reusable verbatim once the real data pipeline exists. What it enforces:

- **Chronological, expanding walk-forward** — train strictly precedes test,
  never shuffled across time.
- **Purging** — a training sample dated ``d`` carries a forward-return label
  spanning ``(d, d + horizon]``; if that window reaches into the test period,
  the label was computed from test-period prices and the fold leaks. The last
  ``purge_dates`` (= label horizon) training dates before each test block are
  therefore dropped. (Embargo — protecting *training* data that follows a test
  block — does not arise in pure expanding walk-forward, where no training date
  follows a test date; it returns if k-fold-style CV is ever used.)
- **Fit-once-per-window** (Decision D): each fold trains one fresh model and
  freezes it for that fold's out-of-sample dates, so regime-label permutation
  cannot occur *within* a fold; identity *across* folds is handled by the
  declared canonical order (ascending ``sigma_k``), recorded per fold.

Metrics follow the syllabus: per-date cross-sectional Spearman rank-IC, its
mean and dispersion, ICIR (= mean/std of the daily IC series), and the t-stat
``ICIR * sqrt(T)`` (the syllabus's ``IC-bar / SE(IC-bar)`` form).

The harness serves **both** thesis variations: memoryless priors train on
shuffled minibatches and predict per-batch; the stateful HMM prior trains on
the chronological sequence and is evaluated by a strictly-causal filtering
pass whose state is warmed through the (past) training window — legal, since
training data precedes the test block by construction.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from .base import BaseCache, BaseFit
from .data import Panel
from .registry import TrialRegistry, trial_provenance

if TYPE_CHECKING:  # pragma: no cover
    from .baselines import BaselineModel
from .diagnostics import (
    canonical_expert_order,
    pairwise_expert_distance,
    persistence_metrics,
)
from .priors import PriorContext
from .train import Trainer
from .utils import atomic_torch_save

__all__ = [
    "WalkForwardFold",
    "walk_forward_folds",
    "rank_ic_by_date",
    "IcSummary",
    "ic_summary",
    "HAC_KERNELS",
    "long_run_variance",
    "resolve_hac_lags",
    "long_short_by_date",
    "PortfolioSummary",
    "portfolio_summary",
    "FoldResult",
    "WalkForwardResult",
    "walk_forward_evaluate",
    "walk_forward_evaluate_baseline",
    "base_and_correction",
]


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-window fold: purged train dates + a contiguous test block."""

    fold: int
    train_dates: Tensor  # (n_train,) int64, strictly < min(test) - purge
    test_dates: Tensor  # (n_test,) int64, contiguous block
    purged_dates: Tensor  # (purge,) int64 dropped between train and test


def walk_forward_folds(
    date: Tensor,
    *,
    n_folds: int,
    test_dates_per_fold: int,
    purge_dates: int,
    min_train_dates: int = 1,
) -> list[WalkForwardFold]:
    """Expanding-window folds over the unique dates of a panel.

    The last ``n_folds * test_dates_per_fold`` dates become consecutive test
    blocks; each fold trains on **all** earlier dates minus the ``purge_dates``
    immediately preceding its test block (label-overlap purging, AFML §7.4).
    """
    if n_folds < 1 or test_dates_per_fold < 1 or purge_dates < 0:
        raise ValueError(
            f"invalid split params: n_folds={n_folds}, "
            f"test_dates_per_fold={test_dates_per_fold}, purge_dates={purge_dates}"
        )
    dates = torch.unique(date, sorted=True)
    needed = min_train_dates + purge_dates + n_folds * test_dates_per_fold
    if len(dates) < needed:
        raise ValueError(
            f"not enough dates for the requested splits: have {len(dates)}, "
            f"need >= {needed} (min_train + purge + n_folds * test_block)"
        )
    first_test_start = len(dates) - n_folds * test_dates_per_fold
    folds: list[WalkForwardFold] = []
    for i in range(n_folds):
        ts = first_test_start + i * test_dates_per_fold
        folds.append(
            WalkForwardFold(
                fold=i,
                train_dates=dates[: max(ts - purge_dates, 0)],
                test_dates=dates[ts : ts + test_dates_per_fold],
                purged_dates=dates[max(ts - purge_dates, 0) : ts],
            )
        )
    return folds


# --------------------------------------------------------------------------- #
# Rank-IC
# --------------------------------------------------------------------------- #


def _ranks(x: Tensor) -> Tensor:
    """Ranks 1..n (ties broken by order — fine for continuous predictions)."""
    order = torch.argsort(x)
    r = torch.empty_like(x)
    r[order] = torch.arange(1, len(x) + 1, dtype=x.dtype)
    return r


def rank_ic_by_date(
    pred: Tensor, y: Tensor, date: Tensor, min_names: int = 3
) -> tuple[Tensor, Tensor]:
    """Per-date cross-sectional Spearman rank correlation.

    Returns ``(dates (T,), ics (T,))``; dates with fewer than ``min_names``
    samples or a degenerate cross-section are skipped.
    """
    if not (pred.shape == y.shape == date.shape):
        raise ValueError(
            f"pred/y/date shapes must match, got {tuple(pred.shape)}, "
            f"{tuple(y.shape)}, {tuple(date.shape)}"
        )
    out_dates, out_ics = [], []
    for d in torch.unique(date, sorted=True):
        m = date == d
        if int(m.sum()) < min_names:
            continue
        rp = _ranks(pred[m].double())
        ry = _ranks(y[m].double())
        rp = rp - rp.mean()
        ry = ry - ry.mean()
        denom = rp.norm() * ry.norm()
        if float(denom) == 0.0:
            continue
        out_dates.append(d)
        out_ics.append((rp @ ry) / denom)
    if not out_ics:
        raise ValueError(
            "no date had a scoreable cross-section — constant per-date "
            "predictions (e.g. a classical emission, whose mu_k ignore "
            "features) have no cross-sectional ranking; compare such models "
            "on held-out NLL and regime recovery instead of rank-IC"
        )
    return torch.stack(out_dates), torch.stack(out_ics).float()


@dataclass(frozen=True)
class IcSummary:
    """Summary of a daily rank-IC series.

    ``icir`` is the per-period information ratio ``mean/std``, a descriptive
    statistic. ``t_stat`` is ``IC-bar / SE(IC-bar)`` with a heteroskedasticity-
    and autocorrelation-consistent standard error over ``hac_lags``
    autocovariances, weighted by ``hac_kernel`` (:func:`long_run_variance`).
    With ``hac_lags = 0`` it is the i.i.d. ``icir * sqrt(T)``.
    """

    mean_ic: float
    ic_std: float
    icir: float
    t_stat: float
    n_dates: int
    hac_lags: int = 0
    hac_kernel: str = "uniform"


#: Kernels for :func:`long_run_variance`. ``"uniform"`` is Hansen and Hodrick
#: (1980): every autocovariance up to the lag at full weight, which is exact
#: for the MA(h-1) structure that overlapping h-period returns create.
#: ``"bartlett"`` is Newey and West (1987): weights ``1 - l/(L+1)``, always
#: non-negative, but they shrink exactly the autocovariances the overlap
#: produces. Measured on 2,000 simulated overlapping 5-period series per
#: length (2026-09-26), rejection rate of a nominal 5% two-sided test:
#:
#: ======  =====  ==================  =========================  ================
#: T       iid    bartlett, lag h-1   bartlett, lag 2(h-1)       uniform, lag h-1
#: ======  =====  ==================  =========================  ================
#: 120     0.402  0.132               0.112                      0.073
#: 360     0.384  0.120               0.094                      0.060
#: 500     0.374  0.108               0.077                      0.054
#: ======  =====  ==================  =========================  ================
HAC_KERNELS: tuple[str, ...] = ("uniform", "bartlett")


def long_run_variance(x: Tensor, lags: int, kernel: str = "uniform") -> float:
    """Long-run variance of a series, for the standard error of its mean.

    ``gamma_0 + 2 * sum_{l=1..L} w_l * gamma_l``, every autocovariance divided
    by ``T - 1`` so that ``lags = 0`` returns exactly the unbiased sample
    variance. ``kernel`` sets ``w_l``: ``"uniform"`` (Hansen-Hodrick,
    ``w_l = 1``) or ``"bartlett"`` (Newey-West, ``w_l = 1 - l/(L+1)``); see
    :data:`HAC_KERNELS` for why the default is uniform.

    Why (audit finding E-2): a daily series built on ``h``-period forward
    returns overlaps itself by ``h - 1`` periods, so neighbouring values
    share most of their return days. On the real panel the daily rank IC of
    5-day targets has lag-1 autocorrelation of 0.6 to 0.8, and an i.i.d.
    standard error overstated t by 1.6 to 1.8 times. ``lags = h - 1`` spans
    exactly that overlap. The uniform estimate can come out non-positive in
    finite samples; callers then fall back to Bartlett (see :func:`ic_summary`).
    """
    if lags < 0:
        raise ValueError(f"lags must be >= 0, got {lags}")
    if kernel not in HAC_KERNELS:
        raise ValueError(f"unknown HAC kernel {kernel!r}; registered: {list(HAC_KERNELS)}")
    xd = x.detach().double().flatten()
    t = int(xd.numel())
    if t < 2:
        raise ValueError(f"need >= 2 observations, got {t}")
    xc = xd - xd.mean()
    denom = t - 1
    lrv = float((xc * xc).sum()) / denom
    for lag in range(1, min(lags, t - 1) + 1):
        weight = 1.0 if kernel == "uniform" else 1.0 - lag / (lags + 1.0)
        lrv += 2.0 * weight * float((xc[:-lag] * xc[lag:]).sum()) / denom
    return lrv


def hac_variance(x: Tensor, lags: int, kernel: str) -> tuple[float, str]:
    """:func:`long_run_variance`, falling back to Bartlett if uniform is <= 0.

    Returns the variance and the kernel actually used, so a summary can say
    which standard error its t-statistic rests on.
    """
    lrv = long_run_variance(x, lags, kernel)
    if lrv <= 0 and kernel == "uniform" and lags > 0:
        return long_run_variance(x, lags, "bartlett"), "bartlett"
    return lrv, kernel


def resolve_hac_lags(panel: Panel, hac_lags: int | None) -> int:
    """``hac_lags``, or ``horizon - 1`` from the panel's target when ``None``."""
    if hac_lags is None:
        return panel.horizon - 1
    if hac_lags < 0:
        raise ValueError(f"hac_lags must be >= 0, got {hac_lags}")
    return hac_lags


def ic_summary(ics: Tensor, hac_lags: int = 0, hac_kernel: str = "uniform") -> IcSummary:
    t = int(ics.numel())
    if t < 2:
        raise ValueError(f"need >= 2 daily ICs to summarize, got {t}")
    mean = float(ics.mean())
    std = float(ics.std(unbiased=True))
    icir = mean / std if std > 0 else float("inf") if mean != 0 else 0.0
    lrv, used = hac_variance(ics, hac_lags, hac_kernel)
    if lrv > 0:
        t_stat = mean / math.sqrt(lrv / t)
    else:
        t_stat = float("inf") if mean != 0 else 0.0
    return IcSummary(
        mean_ic=mean,
        ic_std=std,
        icir=icir,
        t_stat=t_stat,
        n_dates=t,
        hac_lags=hac_lags,
        hac_kernel=used,
    )


# --------------------------------------------------------------------------- #
# Portfolio metrics: quantile long-short, turnover, cost drag
# --------------------------------------------------------------------------- #


def long_short_by_date(
    pred: Tensor,
    y: Tensor,
    date: Tensor,
    entity: Tensor,
    *,
    n_quantiles: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-date quantile long-short returns and turnover.

    Each date: rank by ``pred``; go long the top ``n // n_quantiles`` names and
    short the bottom leg, equal-weighted (``+1/leg`` / ``-1/leg`` — dollar
    neutral, unit gross per side). Turnover is the standard one-sided
    ``0.5 * sum_e |w_t(e) - w_{t-1}(e)|`` over the union of names, weights
    aligned by ``entity``; the first date's position entry counts (turnover 1.0
    for a fresh two-sided book).

    Returns ``(dates (T,), gross (T,), turnover (T,))``.
    """
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
    if not (pred.shape == y.shape == date.shape == entity.shape):
        raise ValueError("pred/y/date/entity shapes must match")
    out_dates, gross_l, tno_l = [], [], []
    prev_w: dict[int, float] = {}
    for d in torch.unique(date, sorted=True):
        rows = torch.nonzero(date == d, as_tuple=True)[0]
        leg = len(rows) // n_quantiles
        if leg < 1:
            raise ValueError(
                f"date {int(d)} has {len(rows)} names — too few for "
                f"{n_quantiles} quantiles (need >= {n_quantiles})"
            )
        order = rows[torch.argsort(pred[rows])]  # ascending by prediction
        short_rows, long_rows = order[:leg], order[-leg:]
        gross_l.append(float(y[long_rows].mean() - y[short_rows].mean()))
        w = {int(entity[r]): 1.0 / leg for r in long_rows}
        w |= {int(entity[r]): -1.0 / leg for r in short_rows}
        names = set(prev_w) | set(w)
        traded = sum(abs(w.get(e, 0.0) - prev_w.get(e, 0.0)) for e in names)
        tno_l.append(0.5 * traded)
        prev_w = w
        out_dates.append(d)
    return (
        torch.stack(out_dates),
        torch.tensor(gross_l),
        torch.tensor(tno_l),
    )


@dataclass(frozen=True)
class PortfolioSummary:
    """Per-period (no annualization — synthetic dates carry no calendar).

    ``cost_rate`` is cost per unit of *traded notional*; traded notional per
    date is ``2 * turnover`` (turnover is one-sided), so
    ``net_t = gross_t - cost_rate * 2 * turnover_t``.
    """

    mean_gross: float
    mean_net: float
    gross_std: float
    ir_gross: float  # mean/std of the gross per-date series
    ir_net: float  # mean/std of the net per-date series
    mean_turnover: float
    cost_rate: float
    n_dates: int


def portfolio_summary(
    gross: Tensor, turnover: Tensor, cost_rate: float = 0.0
) -> PortfolioSummary:
    if gross.numel() < 2 or gross.shape != turnover.shape:
        raise ValueError("need >= 2 matching per-date observations")
    net = gross - cost_rate * 2.0 * turnover
    g_std = float(gross.std(unbiased=True))
    n_std = float(net.std(unbiased=True))
    return PortfolioSummary(
        mean_gross=float(gross.mean()),
        mean_net=float(net.mean()),
        gross_std=g_std,
        ir_gross=float(gross.mean()) / g_std if g_std > 0 else 0.0,
        ir_net=float(net.mean()) / n_std if n_std > 0 else 0.0,
        mean_turnover=float(turnover.mean()),
        cost_rate=cost_rate,
        n_dates=int(gross.numel()),
    )


# --------------------------------------------------------------------------- #
# The harness
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FoldResult:
    fold: int
    nll: float  # held-out predictive NLL on the test block
    ic: IcSummary  # per-date rank-IC summary on the test block
    n_train: int
    n_test: int
    portfolio: PortfolioSummary | None = None  # when backtest_quantiles is set
    # Decision D: canonical (sigma-sorted) order per fitted window; empty for
    # single-model baselines, which have no experts
    expert_order: tuple[int, ...] = ()
    # Trainable elements connected to the loss (nec_moe.train.GradientAudit):
    # the parameter-count confound across prior kinds, measured per window.
    # None for single-model baselines, which never run the audit.
    live_param_count: int | None = None
    # Residual design (brief 02 §5). The frozen base's OWN out-of-sample
    # score on this fold, on the same metrics by the same code — the mixture's
    # number is uninterpretable without the floor it is measured from — plus
    # the correction magnitude actually applied out of sample.
    base_ic: IcSummary | None = None
    base_nll: float | None = None
    base_portfolio: PortfolioSummary | None = None
    correction: tuple[tuple[str, float], ...] = ()
    # Brief 03 §4: the canonical relabelling applied to this fold's fitted
    # gate regimes. Recorded so cross-fold per-regime statistics are auditable
    # — the fold's "regime 0" is only comparable to another fold's if the
    # permutation that produced it is on the record.
    gate_permutation: tuple[int, ...] = ()
    # The fitted gate's own summary (durations, stay probabilities, llf,
    # multi-start counts), in canonical order; empty for non-fitted gates.
    gate_metrics: tuple[tuple[str, float], ...] = ()
    # Wall clock per fold, split into gate fit / base fit / expert training
    # (brief 04 C.4, for the compute budget in Q11). Metadata about the RUN,
    # not a result, so excluded from equality: a resumed fold must still
    # compare equal to the uninterrupted one although the clock differed.
    timing: tuple[tuple[str, float], ...] = field(default=(), compare=False)

    @property
    def ic_improvement(self) -> float | None:
        """Mean rank-IC minus the base's — the thesis's primary quantity."""
        if self.base_ic is None:
            return None
        return self.ic.mean_ic - self.base_ic.mean_ic

    @property
    def nll_improvement(self) -> float | None:
        """Base NLL minus the mixture's: positive = the correction helped."""
        if self.base_nll is None:
            return None
        return self.base_nll - self.nll
    # Chain persistence of this window's fitted prior, in canonical state
    # order; empty for priors with no transition matrix. Pairs, not a dict,
    # to match the file's frozen/hashable convention — read with ``dict(...)``.
    persistence: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class WalkForwardResult:
    folds: list[FoldResult]
    pooled_ic: IcSummary  # over all test dates of all folds
    pooled_portfolio: PortfolioSummary | None = None
    # the frozen base pooled over the same dates by the same code (brief 04
    # C.4: pooled base values beside every arm); None when the base is off
    pooled_base_ic: IcSummary | None = None

    @property
    def mean_fold_nll(self) -> float:
        return sum(f.nll for f in self.folds) / len(self.folds)


class _FoldAccumulator:
    """Shared per-fold scoring + pooling for the NEC and baseline harnesses —
    both model families are graded by exactly the same code path."""

    def __init__(
        self,
        backtest_quantiles: int | None,
        cost_rate: float,
        hac_lags: int = 0,
        hac_kernel: str = "uniform",
    ) -> None:
        self.backtest_quantiles = backtest_quantiles
        self.cost_rate = cost_rate
        self.hac_lags = hac_lags
        self.hac_kernel = hac_kernel
        self.folds: list[FoldResult] = []
        self._ics: list[Tensor] = []
        self._base_ics: list[Tensor] = []
        self._gross: list[Tensor] = []
        self._tno: list[Tensor] = []

    def add(
        self,
        fold: WalkForwardFold,
        pred: Tensor,
        train: Panel,
        test: Panel,
        nll: float,
        expert_order: tuple[int, ...] = (),
        live_param_count: int | None = None,
        persistence: tuple[tuple[str, float], ...] = (),
        base_pred: Tensor | None = None,
        base_nll: float | None = None,
        correction: tuple[tuple[str, float], ...] = (),
        gate_permutation: tuple[int, ...] = (),
        gate_metrics: tuple[tuple[str, float], ...] = (),
        timing: tuple[tuple[str, float], ...] = (),
    ) -> dict[str, Any]:
        """Score a freshly-evaluated fold. Returns the picklable payload that
        fold-level resume persists and :meth:`add_completed` re-ingests."""
        _, ics = rank_ic_by_date(pred, test.y, test.date)
        portfolio = None
        gross: Tensor | None = None
        tno: Tensor | None = None
        if self.backtest_quantiles is not None:
            _, gross, tno = long_short_by_date(
                pred, test.y, test.date, test.entity,
                n_quantiles=self.backtest_quantiles,
            )
            portfolio = portfolio_summary(gross, tno, self.cost_rate)
        # The base scored through the identical code path — same dates, same
        # ranking, same backtest — so "improvement over the base" is a
        # difference of like-for-like numbers rather than of two protocols.
        base_ic = base_portfolio = None
        base_ics: Tensor | None = None
        if base_pred is not None:
            _, base_ics = rank_ic_by_date(base_pred, test.y, test.date)
            base_ic = ic_summary(base_ics, self.hac_lags, self.hac_kernel)
            if self.backtest_quantiles is not None:
                _, b_gross, b_tno = long_short_by_date(
                    base_pred, test.y, test.date, test.entity,
                    n_quantiles=self.backtest_quantiles,
                )
                base_portfolio = portfolio_summary(b_gross, b_tno, self.cost_rate)
        payload = {
            "fold_result": FoldResult(
                fold=fold.fold,
                nll=nll,
                ic=ic_summary(ics, self.hac_lags, self.hac_kernel),
                n_train=len(train),
                n_test=len(test),
                portfolio=portfolio,
                expert_order=expert_order,
                live_param_count=live_param_count,
                persistence=persistence,
                base_ic=base_ic,
                base_nll=base_nll,
                base_portfolio=base_portfolio,
                correction=correction,
                gate_permutation=gate_permutation,
                gate_metrics=gate_metrics,
                timing=timing,
            ),
            "ics": ics,
            "base_ics": base_ics,
            "gross": gross,
            "turnover": tno,
        }
        self._ingest(payload)
        return payload

    def add_completed(self, payload: dict[str, Any]) -> None:
        """Re-ingest a persisted fold (resume path) — pooled results come out
        identical to the run that produced it, since the stored per-date
        tensors are the pooling inputs."""
        if (payload["gross"] is None) != (self.backtest_quantiles is None):
            raise ValueError(
                "resume mismatch: the persisted fold was scored with a "
                "different backtest_quantiles setting — rerun with the "
                "original settings or clear the resume directory"
            )
        self._ingest(payload)

    def _ingest(self, payload: dict[str, Any]) -> None:
        self.folds.append(payload["fold_result"])
        self._ics.append(payload["ics"])
        # .get: fold files persisted before brief 04 carry no base series
        if payload.get("base_ics") is not None:
            self._base_ics.append(payload["base_ics"])
        if payload["gross"] is not None:
            self._gross.append(payload["gross"])
            self._tno.append(payload["turnover"])

    def result(self) -> WalkForwardResult:
        pooled_portfolio = (
            portfolio_summary(
                torch.cat(self._gross), torch.cat(self._tno), self.cost_rate
            )
            if self.backtest_quantiles is not None
            else None
        )
        # pooled only when every fold has a base series: a partial pool
        # would silently compare the mixture's dates with a subset of them
        pooled_base_ic = (
            ic_summary(torch.cat(self._base_ics), self.hac_lags, self.hac_kernel)
            if self._base_ics and len(self._base_ics) == len(self._ics)
            else None
        )
        return WalkForwardResult(
            folds=self.folds,
            pooled_ic=ic_summary(torch.cat(self._ics), self.hac_lags, self.hac_kernel),
            pooled_portfolio=pooled_portfolio,
            pooled_base_ic=pooled_base_ic,
        )


@torch.no_grad()
def _predict_memoryless(trainer: Trainer, test: Panel) -> Tensor:
    trainer.model.eval()
    return trainer.model(
        test.x_seq, test.x_snap, PriorContext(date=test.date)
    ).y_hat


def _fold_predictions(
    trainer: Trainer, train: Panel | None, test: Panel
) -> tuple[Tensor, float]:
    """Per-sample causal predictions (in ``test``'s row order) + held-out NLL.

    Memoryless: one-shot. Stateful (HMM): a strictly-causal filtering pass,
    with the filter state warmed through the *training* sequence first — legal
    because every training date precedes the test block. With ``train=None``
    the stateful pass starts cold from the prior's initial distribution.
    """
    if trainer.model.prior.stateful:
        init = None
        if train is not None:
            init = trainer.evaluate_sequence(train.time_sequence()).log_filtered
        seq = test.time_sequence()
        ev = trainer.evaluate_sequence(seq, init_state=init)
        pred = torch.cat([ev.y_hat[t] for t in range(len(seq))])
        # lock the ordering contract: per-date batches concatenated in date
        # order must reproduce the panel's (date-major) row order exactly
        if not torch.equal(torch.cat([b.y for b in seq]), test.y):
            raise AssertionError(
                "stateful prediction order does not match panel row order"
            )
        return pred, ev.nll
    return _predict_memoryless(trainer, test), trainer.evaluate(test.full_batch())


def _fit_gate(
    trainer: Trainer, panel: Panel, train: Panel, fold: WalkForwardFold
) -> None:
    """Fit the prior on this fold's training block, then apply it causally.

    Two calls, deliberately separate (brief 03 §1, §3):

    1. ``fit(train)`` sees the training block and **only** the training block,
       which is what makes the harness's existing purge discipline cover the
       gate for free — the panel handed over has already had the label-overlap
       dates removed.
    2. ``apply_causal(panel)`` extends the gate to the test dates by running
       the *frozen* fitted model forward. That is legal because a filter at
       date ``t`` conditions only on the series through ``t``; re-fitting
       would not be, and the precomputed prior refuses to overwrite a fitted
       row so the two populations can never be confused.

    Non-precomputed priors inherit a no-op ``fit`` and are untouched.
    """
    if int(train.date.max()) >= int(fold.test_dates.min()):
        raise AssertionError(
            f"gate fit panel reaches date {int(train.date.max())} but the test "
            f"block starts at {int(fold.test_dates.min())}: the gate would be "
            "fitted on data it must not see"
        )
    trainer.model.prior.fit(train)
    prior = trainer.model.prior
    if getattr(prior, "precomputed", False):
        # the gate must cover the test dates too, and only this path may
        # produce them
        prior.apply_causal(  # type: ignore[operator]
            panel.subset_dates(torch.cat([fold.train_dates, fold.test_dates]))
        )


def _gate_report(
    trainer: Trainer,
    fold: WalkForwardFold,
    registry: TrialRegistry | None,
    tag: str | None,
    seed: int,
    provenance: dict[str, Any] | None = None,
) -> tuple[tuple[int, ...], tuple[tuple[str, float], ...]]:
    """The fitted gate's permutation + metrics, and its multi-start trials.

    Brief 03 §5: the Markov switching likelihood is multimodal, so keeping the
    best of N random starts is a **selection event** exactly like "best of 20
    initializations" elsewhere in this project. Every start's converged
    log-likelihood is logged as its own trial so the multiplicity reaches the
    deflated Sharpe and the multiple-testing corrections, and the chosen
    start's index and the convergence count travel with the fold.
    """
    fit = getattr(trainer.model.prior, "fit_result", None)
    if fit is None:
        return (), ()
    metrics = fit.metrics()
    if registry is not None:
        gate_tag = str(tag or trainer.cfg.markov_gate.registry_tag)
        for i, llf in enumerate(fit.start_llfs):
            if math.isfinite(llf):
                registry.log(
                    gate_tag,
                    {"llf": llf, "start": float(i), "fold": float(fold.fold)},
                    config={
                        "arm": gate_tag,
                        "chosen": i == fit.chosen_start,
                        "objective": trainer.cfg.train.objective,
                        "correction_penalty_weight": (
                            trainer.cfg.train.correction_penalty_weight
                        ),
                        **(provenance or {}),
                    },
                    seed=seed,
                    notes=f"markov gate start {i} of {fit.n_starts}, fold {fold.fold}",
                )
    return fit.permutation, tuple(metrics.items())


def _attach_base(
    trainer: Trainer,
    train: Panel,
    fold: WalkForwardFold,
    cache: BaseCache | None,
    seed: int,
) -> BaseFit | None:
    """Fit (or reuse) this fold's frozen base and attach it. No-op when off.

    The window key is the fold's *training date range*, not its index: two
    runs that slice the same panel differently must not share a base, and the
    date range is what actually determines what the base saw.
    """
    if not trainer.cfg.base.enabled:
        return None
    window = (int(fold.train_dates.min()), int(fold.train_dates.max()))
    cache = cache if cache is not None else BaseCache()
    fit = cache.get_or_fit(train, trainer.cfg.base, window=window, seed=seed)
    trainer.model.attach_base(fit.model)
    return fit


@torch.no_grad()
def base_and_correction(
    trainer: Trainer,
    test: Panel,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    *,
    train: Panel | None = None,
) -> tuple[Tensor | None, float | None, tuple[tuple[str, float], ...]]:
    """The frozen base's own test predictions/NLL, and the correction applied.

    Both are brief 02 §5 reporting requirements: the base's out-of-sample
    performance is the floor every mixture number is measured from, and the
    correction magnitude answers the thesis question directly — a model whose
    corrections are numerically negligible has answered it in the negative
    whatever the R-squared does.

    **The base's NLL is the same model's NLL with every correction forced to
    zero** (:meth:`NECModel.corrections_disabled`): the mixture density
    ``sum_k pi_k N(y; f0, s_k^2)`` under the model's own noise scales, prior
    and evaluation path. ``base_nll - nll`` therefore measures what the
    corrections add and nothing else, and it is exactly zero when every
    correction is zero. (Audit finding E-1: the earlier single Gaussian at a
    prior-averaged sigma credited the experts' variance structure to the
    corrections.) Pass ``train`` for a stateful prior so the base is scored
    through the same warmed-up recursion as the mixture.
    """
    model = trainer.model
    if model.base is None:
        return None, None, ()
    model.eval()
    base_pred = model.base(test.x_snap)
    with model.corrections_disabled():
        _, base_nll = _fold_predictions(trainer, train, test)
    out = model(test.x_seq, test.x_snap, PriorContext(date=test.date))
    stats: tuple[tuple[str, float], ...] = ()
    correction = out.correction
    if correction is not None:
        a = correction.abs()
        stats = (
            ("correction_mean_abs", float(a.mean())),
            ("correction_std", float(correction.std(unbiased=True))),
            ("correction_p95_abs", float(a.quantile(0.95))),
            ("correction_max_abs", float(a.max())),
            # scale-free: how big is the correction next to the base's own
            # dispersion? 0.01 means the mixture barely moved the base.
            (
                "correction_rel_base_std",
                float(a.mean() / base_pred.std(unbiased=True).clamp_min(1e-12)),
            ),
        )
        # the signed distribution, not only its magnitude (brief 04 C.4):
        # a correction centred on zero and one that is one-signed are
        # different findings even at the same mean |.|
        qs = torch.quantile(correction.double(), torch.tensor(quantiles, dtype=torch.float64))
        stats += tuple(
            (f"correction_q{round(100 * q):02d}", float(v))
            for q, v in zip(quantiles, qs, strict=True)
        )
        # out-of-sample distance between the fitted corrections themselves:
        # whether the experts actually learned different functions
        if out.corrections is not None and out.corrections.shape[1] > 1:
            d = pairwise_expert_distance(out.corrections)
            off = d[~torch.eye(d.shape[0], dtype=torch.bool)]
            stats += (
                ("correction_min_pairwise_distance", float(off.min())),
                ("correction_mean_pairwise_distance", float(off.mean())),
            )
    return base_pred, base_nll, stats


def walk_forward_evaluate(
    panel: Panel,
    make_trainer: Callable[[], Trainer],
    *,
    n_folds: int,
    test_dates_per_fold: int,
    purge_dates: int,
    steps: int,
    warmstart_key: Callable[[Panel], Tensor] | None = None,
    min_train_dates: int = 1,
    backtest_quantiles: int | None = None,
    cost_rate: float = 0.0,
    resume_dir: str | Path | None = None,
    base_cache: BaseCache | None = None,
    seed: int = 0,
    registry: TrialRegistry | None = None,
    registry_tag: str | None = None,
    hac_lags: int | None = None,
    hac_kernel: str = "uniform",
) -> WalkForwardResult:
    """Fit-once-per-window walk-forward evaluation (Decision D protocol).

    ``hac_lags`` / ``hac_kernel``: the autocorrelation-consistent standard
    error behind every IC t-statistic (fold, base and pooled). ``hac_lags =
    None``, the default, uses ``horizon - 1`` from the panel's target
    (``fwd_ret_5d`` gives 4; a target that declares no horizon gives 0), the
    overlap between consecutive forward returns; the default kernel is
    Hansen-Hodrick (see :data:`HAC_KERNELS`; audit E-2).

    For each fold: build a **fresh** model/trainer via ``make_trainer`` (a
    fresh optimization per window — refits are independent, which is exactly
    why the canonical expert order is recorded per fold), optionally run the
    expert warm-start with ``warmstart_key(train_panel)`` as the sort proxy,
    train ``steps`` steps on the purged training window, freeze, and score the
    test block causally.

    With ``backtest_quantiles`` set, each fold also gets a quantile long-short
    backtest (:func:`long_short_by_date`) with turnover and a cost drag of
    ``cost_rate`` per unit traded notional. Each fold's book starts fresh
    (full entry turnover on its first date) — a slight overstatement for
    adjacent folds, deterministic and conservative.

    **Resume** (``resume_dir``): each completed fold is persisted there
    (``fold_<i>.pt``: scored payload + the trained model's state dict and
    config) and skipped on restart; a fold interrupted *mid-fit* leaves a
    trainer checkpoint (``fold_<i>_trainer.pt``, written every
    ``TrainConfig.checkpoint_every`` steps) that the next run picks up with
    :meth:`Trainer.load` for the remaining steps. Resume the same settings
    (folds, steps, backtest) on the same panel — the directory stores results,
    not the data. Because ``make_trainer`` seeds deterministically (as the
    sweep arms do), a resumed run reproduces the uninterrupted one exactly.

    **Residual mode** (``base.enabled``, brief 02): before the experts train,
    each fold fits its frozen base ``f0`` on that fold's training block and
    attaches it — order per fold is gate → base → experts → score.
    ``base_cache`` amortizes that fit: the base never sees regime
    information, so within one window and seed it is *identical* across every
    gate arm, and reusing it both saves the compute budget and makes every arm
    measure against literally the same floor. Pass one
    :class:`~nec_moe.base.BaseCache` to every arm of a sweep (``run_sweep``
    does); ``seed`` identifies the run, so two seeds get two bases.
    """
    folds = walk_forward_folds(
        panel.date,
        n_folds=n_folds,
        test_dates_per_fold=test_dates_per_fold,
        purge_dates=purge_dates,
        min_train_dates=min_train_dates,
    )
    resume = Path(resume_dir) if resume_dir is not None else None
    if resume is not None:
        resume.mkdir(parents=True, exist_ok=True)
    acc = _FoldAccumulator(
        backtest_quantiles, cost_rate, resolve_hac_lags(panel, hac_lags), hac_kernel
    )
    for fold in folds:
        done_file = resume / f"fold_{fold.fold}.pt" if resume is not None else None
        if done_file is not None and done_file.exists():
            acc.add_completed(torch.load(done_file, weights_only=False))
            continue
        train = panel.subset_dates(fold.train_dates)
        test = panel.subset_dates(fold.test_dates)
        fit_ckpt = (
            resume / f"fold_{fold.fold}_trainer.pt" if resume is not None else None
        )
        if fit_ckpt is not None and fit_ckpt.exists():
            # interrupted mid-fit: warm-start already happened before step 0,
            # so it must NOT rerun — everything is in the checkpoint
            trainer = Trainer.load(fit_ckpt)
            remaining = max(steps - trainer.step_count, 0)
            t_gate = t_base = 0.0  # not refitted in this process
            _attach_base(trainer, train, fold, base_cache, seed)
        else:
            trainer = make_trainer()
            # Order matters (brief 02 §4, brief 03 §1): the gate is fitted on
            # this fold's training block and frozen, THEN the base is fitted on
            # the same block and frozen, and only then do the experts train
            # against what is left.
            t0 = time.perf_counter()
            _fit_gate(trainer, panel, train, fold)
            t_gate = time.perf_counter() - t0
            t0 = time.perf_counter()
            base_fit = _attach_base(trainer, train, fold, base_cache, seed)
            t_base = time.perf_counter() - t0
            x = trainer.cfg.experts
            if warmstart_key is not None and not (
                x.correction_mode and x.zero_init_head
            ):
                trainer.warmstart_experts(train.full_batch(), warmstart_key(train))
            del base_fit
            remaining = steps
        t0 = time.perf_counter()
        if trainer.model.prior.stateful:
            trainer.fit_sequence(
                train.time_sequence(), steps=remaining, checkpoint_path=fit_ckpt
            )
        else:
            trainer.fit(train, steps=remaining, checkpoint_path=fit_ckpt)
        t_train = time.perf_counter() - t0

        pred, nll = _fold_predictions(trainer, train, test)
        canonical = canonical_expert_order(trainer.model.experts.log_sigma)
        order = tuple(int(i) for i in canonical)
        # Persistence of this window's chain, relabelled into canonical order
        # so the same key means the same regime across independently refitted
        # folds (Decision D); empty for priors with no transition matrix.
        transition = getattr(trainer.model.prior, "transition_matrix", None)
        persistence = (
            tuple(persistence_metrics(transition.detach(), canonical).items())
            if transition is not None
            else ()
        )
        base_pred, base_nll, correction = base_and_correction(trainer, test, train=train)
        gate_perm, gate_metrics = _gate_report(
            trainer, fold, registry, registry_tag, seed, trial_provenance(panel)
        )
        payload = acc.add(
            fold, pred, train, test, nll,
            expert_order=order,
            live_param_count=trainer.live_param_count,
            persistence=persistence,
            base_pred=base_pred,
            base_nll=base_nll,
            correction=correction,
            gate_permutation=gate_perm,
            gate_metrics=gate_metrics,
            timing=(
                ("gate_fit_s", t_gate),
                ("base_fit_s", t_base),
                ("expert_train_s", t_train),
            ),
        )
        if done_file is not None:
            payload = dict(
                payload,
                model=trainer.model.state_dict(),
                config=trainer.cfg.to_dict(),
            )
            atomic_torch_save(payload, done_file)
            if fit_ckpt is not None:
                fit_ckpt.unlink(missing_ok=True)  # superseded by the fold file
    return acc.result()


def walk_forward_evaluate_baseline(
    panel: Panel,
    make_model: Callable[[], BaselineModel],
    *,
    n_folds: int,
    test_dates_per_fold: int,
    purge_dates: int,
    min_train_dates: int = 1,
    backtest_quantiles: int | None = None,
    cost_rate: float = 0.0,
    hac_lags: int | None = None,
    hac_kernel: str = "uniform",
) -> WalkForwardResult:
    """The single-model counterpart of :func:`walk_forward_evaluate`.

    ``hac_lags`` and ``hac_kernel`` as in :func:`walk_forward_evaluate`.

    Same folds, same purging, same fit-once-per-window discipline, and —
    via the shared accumulator — byte-identical scoring: a baseline row and an
    NEC row in the thesis comparison table are graded by the same code.
    ``make_model`` returns a fresh :class:`~nec_moe.baselines.BaselineModel`
    per window.
    """
    folds = walk_forward_folds(
        panel.date,
        n_folds=n_folds,
        test_dates_per_fold=test_dates_per_fold,
        purge_dates=purge_dates,
        min_train_dates=min_train_dates,
    )
    acc = _FoldAccumulator(
        backtest_quantiles, cost_rate, resolve_hac_lags(panel, hac_lags), hac_kernel
    )
    for fold in folds:
        train = panel.subset_dates(fold.train_dates)
        test = panel.subset_dates(fold.test_dates)
        model = make_model()
        model.fit(train)
        acc.add(fold, model.predict(test), train, test, model.nll(test))
    return acc.result()
