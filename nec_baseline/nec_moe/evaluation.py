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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import torch
from torch import Tensor

from .data import Panel

if TYPE_CHECKING:  # pragma: no cover
    from .baselines import BaselineModel
from .diagnostics import canonical_expert_order
from .train import Trainer

__all__ = [
    "WalkForwardFold",
    "walk_forward_folds",
    "rank_ic_by_date",
    "IcSummary",
    "ic_summary",
    "long_short_by_date",
    "PortfolioSummary",
    "portfolio_summary",
    "FoldResult",
    "WalkForwardResult",
    "walk_forward_evaluate",
    "walk_forward_evaluate_baseline",
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

    ``icir`` is the per-period information ratio ``mean/std``; ``t_stat`` is
    the syllabus's ``IC-bar / SE(IC-bar) = icir * sqrt(T)``.
    """

    mean_ic: float
    ic_std: float
    icir: float
    t_stat: float
    n_dates: int


def ic_summary(ics: Tensor) -> IcSummary:
    t = int(ics.numel())
    if t < 2:
        raise ValueError(f"need >= 2 daily ICs to summarize, got {t}")
    mean = float(ics.mean())
    std = float(ics.std(unbiased=True))
    icir = mean / std if std > 0 else float("inf") if mean != 0 else 0.0
    return IcSummary(
        mean_ic=mean,
        ic_std=std,
        icir=icir,
        t_stat=icir * t**0.5,
        n_dates=t,
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


@dataclass(frozen=True)
class WalkForwardResult:
    folds: list[FoldResult]
    pooled_ic: IcSummary  # over all test dates of all folds
    pooled_portfolio: PortfolioSummary | None = None

    @property
    def mean_fold_nll(self) -> float:
        return sum(f.nll for f in self.folds) / len(self.folds)


class _FoldAccumulator:
    """Shared per-fold scoring + pooling for the NEC and baseline harnesses —
    both model families are graded by exactly the same code path."""

    def __init__(self, backtest_quantiles: int | None, cost_rate: float) -> None:
        self.backtest_quantiles = backtest_quantiles
        self.cost_rate = cost_rate
        self.folds: list[FoldResult] = []
        self._ics: list[Tensor] = []
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
    ) -> None:
        _, ics = rank_ic_by_date(pred, test.y, test.date)
        portfolio = None
        if self.backtest_quantiles is not None:
            _, gross, tno = long_short_by_date(
                pred, test.y, test.date, test.entity,
                n_quantiles=self.backtest_quantiles,
            )
            portfolio = portfolio_summary(gross, tno, self.cost_rate)
            self._gross.append(gross)
            self._tno.append(tno)
        self.folds.append(
            FoldResult(
                fold=fold.fold,
                nll=nll,
                ic=ic_summary(ics),
                n_train=len(train),
                n_test=len(test),
                portfolio=portfolio,
                expert_order=expert_order,
            )
        )
        self._ics.append(ics)

    def result(self) -> WalkForwardResult:
        pooled_portfolio = (
            portfolio_summary(
                torch.cat(self._gross), torch.cat(self._tno), self.cost_rate
            )
            if self.backtest_quantiles is not None
            else None
        )
        return WalkForwardResult(
            folds=self.folds,
            pooled_ic=ic_summary(torch.cat(self._ics)),
            pooled_portfolio=pooled_portfolio,
        )


@torch.no_grad()
def _predict_memoryless(trainer: Trainer, test: Panel) -> Tensor:
    trainer.model.eval()
    return trainer.model(test.x_seq, test.x_snap).y_hat


def _fold_predictions(
    trainer: Trainer, train: Panel, test: Panel
) -> tuple[Tensor, float]:
    """Per-sample causal predictions (in ``test``'s row order) + held-out NLL.

    Memoryless: one-shot. Stateful (HMM): a strictly-causal filtering pass,
    with the filter state warmed through the *training* sequence first — legal
    because every training date precedes the test block.
    """
    if trainer.model.prior.stateful:
        warm = trainer.evaluate_sequence(train.time_sequence())
        seq = test.time_sequence()
        ev = trainer.evaluate_sequence(seq, init_state=warm.log_filtered[-1])
        pred = torch.cat([ev.y_hat[t] for t in range(len(seq))])
        # lock the ordering contract: per-date batches concatenated in date
        # order must reproduce the panel's (date-major) row order exactly
        if not torch.equal(torch.cat([b.y for b in seq]), test.y):
            raise AssertionError(
                "stateful prediction order does not match panel row order"
            )
        return pred, ev.nll
    return _predict_memoryless(trainer, test), trainer.evaluate(test.full_batch())


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
) -> WalkForwardResult:
    """Fit-once-per-window walk-forward evaluation (Decision D protocol).

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
    """
    folds = walk_forward_folds(
        panel.date,
        n_folds=n_folds,
        test_dates_per_fold=test_dates_per_fold,
        purge_dates=purge_dates,
        min_train_dates=min_train_dates,
    )
    acc = _FoldAccumulator(backtest_quantiles, cost_rate)
    for fold in folds:
        train = panel.subset_dates(fold.train_dates)
        test = panel.subset_dates(fold.test_dates)
        trainer = make_trainer()
        if warmstart_key is not None:
            trainer.warmstart_experts(train.full_batch(), warmstart_key(train))
        if trainer.model.prior.stateful:
            trainer.fit_sequence(train.time_sequence(), steps=steps)
        else:
            trainer.fit(train, steps=steps)

        pred, nll = _fold_predictions(trainer, train, test)
        order = tuple(
            int(i) for i in canonical_expert_order(trainer.model.experts.log_sigma)
        )
        acc.add(fold, pred, train, test, nll, expert_order=order)
    return acc.result()


def walk_forward_evaluate_baseline(
    panel: Panel,
    make_model: Callable[[], "BaselineModel"],
    *,
    n_folds: int,
    test_dates_per_fold: int,
    purge_dates: int,
    min_train_dates: int = 1,
    backtest_quantiles: int | None = None,
    cost_rate: float = 0.0,
) -> WalkForwardResult:
    """The single-model counterpart of :func:`walk_forward_evaluate`.

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
    acc = _FoldAccumulator(backtest_quantiles, cost_rate)
    for fold in folds:
        train = panel.subset_dates(fold.train_dates)
        test = panel.subset_dates(fold.test_dates)
        model = make_model()
        model.fit(train)
        acc.add(fold, model.predict(test), train, test, model.nll(test))
    return acc.result()
