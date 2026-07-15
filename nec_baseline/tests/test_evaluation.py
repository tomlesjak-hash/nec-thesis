"""Walk-forward harness: purged splits, rank-IC math, end-to-end folds.

The split tests encode the leakage rules (train strictly before test, label-
overlap purging); the harness tests run real (tiny) models through the full
fit-once-per-window protocol on synthetic panels — both the memoryless and the
HMM paths.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import small_config

from nec_moe import (
    NECModel,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    ic_summary,
    long_short_by_date,
    portfolio_summary,
    rank_ic_by_date,
    walk_forward_evaluate,
    walk_forward_folds,
)

# --------------------------------------------------------------------------- #
# Splits and purging
# --------------------------------------------------------------------------- #


def test_walk_forward_folds_chronology_and_purge():
    date = torch.arange(100).repeat_interleave(4)  # 100 dates, 4 names each
    folds = walk_forward_folds(
        date, n_folds=3, test_dates_per_fold=10, purge_dates=5
    )
    assert len(folds) == 3
    prev_test_end = -1
    for f in folds:
        # expanding train, contiguous later test, strictly ordered
        assert int(f.train_dates.max()) < int(f.test_dates.min())
        assert int(f.test_dates.min()) > prev_test_end
        prev_test_end = int(f.test_dates.max())
        # the purge: exactly the `purge` dates before the test block are
        # neither trained on nor tested on
        gap = int(f.test_dates.min()) - int(f.train_dates.max())
        assert gap == 5 + 1
        assert len(f.purged_dates) == 5
        assert int(f.purged_dates.max()) == int(f.test_dates.min()) - 1
    # later folds train on strictly more dates (expanding window)
    assert len(folds[2].train_dates) > len(folds[0].train_dates)


def test_purge_removes_label_overlap():
    """The leakage rule made concrete: with horizon h, a train sample dated d
    has a label spanning (d, d+h]; purging by h guarantees d + h < test_start
    for every training date — no train label touches test-period prices."""
    date = torch.arange(60)
    h = 4
    folds = walk_forward_folds(date, n_folds=2, test_dates_per_fold=8, purge_dates=h)
    for f in folds:
        test_start = int(f.test_dates.min())
        assert int(f.train_dates.max()) + h < test_start
    # and purge=0 would violate exactly that
    unpurged = walk_forward_folds(date, n_folds=2, test_dates_per_fold=8, purge_dates=0)
    assert int(unpurged[0].train_dates.max()) + h >= int(unpurged[0].test_dates.min())


def test_walk_forward_folds_rejects_impossible_requests():
    date = torch.arange(20)
    with pytest.raises(ValueError, match="not enough dates"):
        walk_forward_folds(date, n_folds=3, test_dates_per_fold=10, purge_dates=2)
    with pytest.raises(ValueError, match="invalid split"):
        walk_forward_folds(date, n_folds=0, test_dates_per_fold=5, purge_dates=0)


# --------------------------------------------------------------------------- #
# Rank-IC
# --------------------------------------------------------------------------- #


def test_rank_ic_perfect_anti_and_monotone_invariance():
    g = torch.Generator().manual_seed(0)
    y = torch.randn(200, generator=g)
    date = torch.arange(10).repeat_interleave(20)
    _, ic_perfect = rank_ic_by_date(y, y, date)
    assert torch.allclose(ic_perfect, torch.ones(10), atol=1e-6)
    _, ic_anti = rank_ic_by_date(-y, y, date)
    assert torch.allclose(ic_anti, -torch.ones(10), atol=1e-6)
    # Spearman is rank-based: any strictly monotone transform is IC-invariant
    _, ic_mono = rank_ic_by_date(torch.exp(y), y, date)
    assert torch.allclose(ic_mono, torch.ones(10), atol=1e-6)


def test_rank_ic_random_predictions_near_zero():
    g = torch.Generator().manual_seed(1)
    y = torch.randn(5000, generator=g)
    pred = torch.randn(5000, generator=g)
    date = torch.arange(50).repeat_interleave(100)
    _, ics = rank_ic_by_date(pred, y, date)
    assert abs(float(ics.mean())) < 0.05


def test_rank_ic_skips_degenerate_dates():
    y = torch.tensor([1.0, 2.0, 3.0, 1.0, 1.0])
    pred = torch.tensor([1.0, 2.0, 3.0, 5.0, 6.0])
    date = torch.tensor([0, 0, 0, 1, 1])  # date 1 has only 2 names
    dates, ics = rank_ic_by_date(pred, y, date, min_names=3)
    assert dates.tolist() == [0] and len(ics) == 1


def test_ic_summary_math():
    ics = torch.tensor([0.10, 0.20, 0.00, 0.10])
    s = ic_summary(ics)
    assert s.mean_ic == pytest.approx(0.10)
    assert s.ic_std == pytest.approx(float(ics.std(unbiased=True)))
    assert s.icir == pytest.approx(s.mean_ic / s.ic_std)
    assert s.t_stat == pytest.approx(s.icir * 2.0)  # sqrt(4)
    assert s.n_dates == 4


# --------------------------------------------------------------------------- #
# Portfolio metrics: quantile long-short, turnover, cost drag
# --------------------------------------------------------------------------- #


def test_long_short_math_single_date():
    y = torch.tensor([0.5, -1.0, 2.0, 0.0, 3.0, -2.0])
    pred = y.clone()  # perfect ranking
    date = torch.zeros(6, dtype=torch.long)
    entity = torch.arange(6)
    dates, gross, tno = long_short_by_date(pred, y, date, entity, n_quantiles=3)
    # terciles of 6 names: leg = 2. Long = top-2 by pred {3.0, 2.0},
    # short = bottom-2 {-2.0, -1.0}
    assert dates.tolist() == [0]
    assert float(gross[0]) == pytest.approx((3.0 + 2.0) / 2 - (-2.0 - 1.0) / 2)
    # first date = full entry: 0.5 * sum|w| = 0.5 * 4 * (1/2) = 1.0
    assert float(tno[0]) == pytest.approx(1.0)


def test_turnover_and_cost_drag():
    # 4 names, halves (leg=2, weights +-1/2), 3 dates:
    # date 0 -> entry; date 1 -> identical ranks (turnover 0);
    # date 2 -> fully reversed ranks (every weight flips sign: traded = 4)
    y = torch.tensor([1.0, 2.0, 3.0, 4.0] * 3)
    pred = torch.tensor(
        [3.0, 2.0, 1.0, 0.0, 3.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    )
    date = torch.arange(3).repeat_interleave(4)
    entity = torch.arange(4).repeat(3)
    _, gross, tno = long_short_by_date(pred, y, date, entity, n_quantiles=2)
    assert tno.tolist() == pytest.approx([1.0, 0.0, 2.0])
    # gross: dates 0-1 long entities {0,1} (y 1,2) short {2,3} (y 3,4) -> -2;
    # date 2 the reverse -> +2
    assert gross.tolist() == pytest.approx([-2.0, -2.0, 2.0])

    s = portfolio_summary(gross, tno, cost_rate=0.01)
    # net_t = gross_t - cost_rate * 2 * turnover_t
    net = torch.tensor([-2.0 - 0.02, -2.0, 2.0 - 0.04])
    assert s.mean_net == pytest.approx(float(net.mean()))
    assert s.mean_gross == pytest.approx(float(gross.mean()))
    assert s.mean_turnover == pytest.approx(1.0)
    assert s.mean_net < s.mean_gross  # costs always drag


def test_long_short_too_few_names_raises():
    y = torch.randn(3)
    date = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="too few"):
        long_short_by_date(y, y, date, torch.arange(3), n_quantiles=4)
    with pytest.raises(ValueError, match="n_quantiles"):
        long_short_by_date(y, y, date, torch.arange(3), n_quantiles=1)


# --------------------------------------------------------------------------- #
# End-to-end harness
# --------------------------------------------------------------------------- #


def _vol_proxy(panel):
    return panel.x_seq[:, :, 0].std(dim=1)


def test_walk_forward_evaluate_memoryless():
    torch.manual_seed(0)
    spec = SyntheticSpec(vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=3)
    panel = SyntheticRegimePanel(spec).generate(n_dates=160, n_entities=8)

    def make_trainer() -> Trainer:
        torch.manual_seed(0)  # fresh, identically-initialized model per window
        return Trainer(
            NECModel(small_config(sigma_init=1.5, lr=3e-3, batch_size=256))
        )

    result = walk_forward_evaluate(
        panel,
        make_trainer,
        n_folds=2,
        test_dates_per_fold=20,
        purge_dates=5,
        steps=200,
        warmstart_key=_vol_proxy,
        backtest_quantiles=4,
        cost_rate=0.001,
    )
    assert len(result.folds) == 2
    for f in result.folds:
        assert math.isfinite(f.nll)
        assert sorted(f.expert_order) == [0, 1]  # a valid permutation, per fold
        assert f.n_test == 20 * 8
        assert f.portfolio is not None and math.isfinite(f.portfolio.mean_net)
    # a well-specified synthetic model must carry real cross-sectional signal
    assert result.pooled_ic.mean_ic > 0.3, result.pooled_ic
    assert result.pooled_ic.n_dates == 40
    assert math.isfinite(result.mean_fold_nll)
    # ... and it must survive the quantile long-short backtest with cost drag
    p = result.pooled_portfolio
    assert p is not None and p.n_dates == 40
    assert p.mean_gross > 0, p
    assert p.mean_net < p.mean_gross  # nonzero turnover => costs bite
    assert 0.0 <= p.mean_turnover <= 2.0


def test_walk_forward_evaluate_stateful_hmm():
    """Plumbing test for the HMM path: chronological fit, causal filtered
    evaluation warmed through the (past) training window."""
    torch.manual_seed(0)
    spec = SyntheticSpec(
        regime_process="markov",
        transition_stay=0.97,
        vol_levels=(1.0, 1.3),
        beta_scale=1.2,
        noise_std=0.8,
        seed=11,
    )
    panel = SyntheticRegimePanel(spec).generate(n_dates=140, n_entities=6)

    def make_trainer() -> Trainer:
        torch.manual_seed(0)
        return Trainer(
            NECModel(
                small_config(
                    prior_kind="hmm", sigma_init=2.0, lr=3e-3, sigma_freeze_steps=10
                )
            )
        )

    result = walk_forward_evaluate(
        panel,
        make_trainer,
        n_folds=2,
        test_dates_per_fold=15,
        purge_dates=5,
        steps=60,
        warmstart_key=_vol_proxy,
    )
    assert len(result.folds) == 2
    assert all(math.isfinite(f.nll) for f in result.folds)
    assert result.pooled_ic.n_dates == 30
    assert math.isfinite(result.pooled_ic.mean_ic)
    # backtest off by default
    assert result.pooled_portfolio is None
    assert all(f.portfolio is None for f in result.folds)
