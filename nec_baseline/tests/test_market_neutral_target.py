"""The market-neutral target (brief 08 A, decision Q25).

``fwd_mn_ret_{h}d`` is the raw forward return minus the date's equal-weighted
mean over the universe rows with a valid target. Pinned here: it averages to
zero on every date, a return common to every stock cancels from it, its mean
counts rows the panel later drops for a missing characteristic, it follows the
point-in-time universe, its daily pieces still sum to it, the long-short book
is dollar neutral and blind to a common return, and every trial row records
the target kind. Invented numbers throughout.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import small_config

from nec_moe import (
    SpellUniverse,
    StageBSpec,
    TrialRegistry,
    assemble_panel,
    data_config_from_panel,
    filter_point_in_time,
    market_frame,
    nec_arm,
    panel_target_kind,
    run_sweep,
    trial_provenance,
)
from nec_moe.config import target_horizon
from nec_moe.evaluation import _leg_weights, _quantile_legs, long_short_book
from nec_moe.train import DeadParameterWarning

N_DAYS = 260
IDX = pd.bdate_range("2021-01-04", periods=N_DAYS)
SPEC = StageBSpec(seq_len=10, horizon=3)  # the default target kind


def _daily(seed: int) -> pd.DataFrame:
    g = np.random.default_rng(seed)
    ret = g.normal(0.0004, 0.02, N_DAYS)
    vol = 1e6 * np.exp(g.normal(0.0, 0.4, N_DAYS))
    return pd.DataFrame(
        {"ret": ret, "volume": vol, "dollar_volume": 50.0 * np.exp(np.cumsum(ret)) * vol},
        index=IDX,
    )


def _universe(n: int = 7) -> dict[str, pd.DataFrame]:
    return {str(10001 + i): _daily(20 + i) for i in range(n)}


def _mkt(spec: StageBSpec = SPEC) -> pd.DataFrame:
    g = np.random.default_rng(99)
    return market_frame(pd.Series(g.normal(0.0003, 0.01, N_DAYS), index=IDX), spec)


def _raw_by_date(daily: dict[str, pd.DataFrame], spec: StageBSpec) -> pd.DataFrame:
    """Every entity's raw forward return on the calendar, one column each."""
    raw = dataclasses.replace(spec, target_kind="raw")
    return pd.DataFrame(
        {t: d["ret"].rolling(raw.horizon).sum().shift(-raw.horizon) for t, d in daily.items()}
    )


def test_target_name_and_default():
    assert StageBSpec().target_kind == "market_neutral"
    assert StageBSpec().target == "fwd_mn_ret_5d"
    assert target_horizon("fwd_mn_ret_5d") == 5
    assert SPEC.target == "fwd_mn_ret_3d" and SPEC.raw_target == "fwd_ret_3d"
    with pytest.raises(ValueError, match="target_kind"):
        StageBSpec(target_kind="excess").validate()  # type: ignore[arg-type]


def test_target_averages_to_zero_on_every_date():
    panel = assemble_panel(_universe(), _mkt(), SPEC)
    assert panel.schema.target == "fwd_mn_ret_3d"
    for d in torch.unique(panel.date):
        y = panel.y[panel.date == d].double()
        assert len(y) == 7  # every entity is a row here, so the mean is over the rows
        assert abs(float(y.mean())) < 1e-7


def test_target_is_raw_minus_the_dates_mean():
    daily = _universe()
    panel = assemble_panel(daily, _mkt(), SPEC)
    raw = _raw_by_date(daily, SPEC)
    assert panel.date_labels is not None and panel.entity_labels is not None
    for row in range(0, len(panel), 37):
        d = pd.Timestamp(panel.date_labels[int(panel.date[row])])
        e = panel.entity_labels[int(panel.entity[row])]
        expected = raw.loc[d, e] - raw.loc[d].mean()
        assert float(panel.y[row]) == pytest.approx(expected, abs=1e-6)


def test_a_return_common_to_every_stock_leaves_the_target_unchanged():
    daily = _universe()
    base = assemble_panel(daily, _mkt(), SPEC)
    shocked = {t: d.copy() for t, d in daily.items()}
    for day in (IDX[120], IDX[121], IDX[200]):  # a different constant per day
        c = 0.03 if day != IDX[200] else -0.05
        for d in shocked.values():
            d.loc[day, "ret"] += c
    moved = assemble_panel(shocked, _mkt(), SPEC)
    assert torch.equal(base.date, moved.date) and torch.equal(base.entity, moved.entity)
    assert torch.allclose(base.y, moved.y, atol=1e-6)
    assert base.y_daily is not None and moved.y_daily is not None
    assert torch.allclose(base.y_daily, moved.y_daily, atol=1e-6)
    # the raw target does move, so the test has teeth
    raw_spec = dataclasses.replace(SPEC, target_kind="raw")
    assert not torch.allclose(
        assemble_panel(daily, _mkt(raw_spec), raw_spec).y,
        assemble_panel(shocked, _mkt(raw_spec), raw_spec).y,
        atol=1e-3,
    )


def test_the_mean_counts_rows_dropped_for_a_missing_characteristic():
    """Brief 08 A.2: the mean runs over the date's rows with a valid target
    before any row is dropped for a missing feature, so a stock's target
    never depends on whether another stock's features are complete."""
    daily = _universe()
    gap_day = IDX[150]
    daily["10003"].loc[gap_day, "volume"] = np.nan  # its volume z-score goes missing
    panel = assemble_panel(daily, _mkt(), SPEC)
    raw = _raw_by_date(daily, SPEC)
    assert panel.date_labels is not None and panel.entity_labels is not None
    code = panel.date_labels.index(str(gap_day.date()))
    rows = panel.date == code
    assert int(rows.sum()) == 6  # 10003 dropped on that date ...
    present = [panel.entity_labels[int(e)] for e in panel.entity[rows]]
    assert "10003" not in present
    for e, y in zip(present, panel.y[rows], strict=True):
        # ... but its raw target is in the mean the others are demeaned by
        assert float(y) == pytest.approx(raw.loc[gap_day, e] - raw.loc[gap_day].mean(), abs=1e-6)


def test_the_mean_runs_over_each_dates_members_only():
    daily = _universe()
    late = IDX[180]  # 10007 joins the index late
    universe = SpellUniverse(pd.DataFrame({
        "entity": sorted(daily),
        "start": [pd.Timestamp("2010-01-04")] * 6 + [late],
        "end": [pd.Timestamp("2030-12-31")] * 7,
    }))
    panel = filter_point_in_time(assemble_panel(daily, _mkt(), SPEC, universe=universe), universe)
    assert panel.metadata["target_demeaned_over"] == "universe"
    raw = _raw_by_date(daily, SPEC)
    assert panel.date_labels is not None and panel.entity_labels is not None
    for d in torch.unique(panel.date):
        rows = panel.date == d
        day = pd.Timestamp(panel.date_labels[int(d)])
        members = sorted(universe.members_asof(day))
        assert abs(float(panel.y[rows].double().mean())) < 1e-7
        e = panel.entity_labels[int(panel.entity[rows][0])]
        assert float(panel.y[rows][0]) == pytest.approx(
            raw.loc[day, e] - raw.loc[day, members].mean(), abs=1e-6
        )


def test_filter_refuses_a_target_demeaned_over_the_candidates():
    daily = _universe()
    universe = SpellUniverse(pd.DataFrame({
        "entity": sorted(daily), "start": [pd.Timestamp("2010-01-04")] * 7,
        "end": [pd.Timestamp("2030-12-31")] * 7,
    }))
    panel = assemble_panel(daily, _mkt(), SPEC)
    assert panel.metadata["target_demeaned_over"] == "entities"
    with pytest.raises(ValueError, match="universe=universe"):
        filter_point_in_time(panel, universe)


def test_daily_pieces_sum_to_the_market_neutral_target():
    panel = assemble_panel(_universe(), _mkt(), SPEC)
    assert panel.y_daily is not None and panel.y_daily.shape == (len(panel), SPEC.horizon)
    assert torch.allclose(panel.y_daily.sum(dim=1), panel.y, atol=1e-6)
    # and each day's pieces are demeaned across the date's rows too
    for d in torch.unique(panel.date)[:20]:
        assert float(panel.y_daily[panel.date == d].double().mean(0).abs().max()) < 1e-7


def test_long_short_book_is_dollar_neutral_and_blind_to_a_common_return():
    """Brief 08 A.4: equal-weighted legs of equal size, so the weights sum to
    zero on every date and adding the same per-day return to every stock
    changes neither book (it is invariant, so it needs no restatement on the
    market-neutral basis)."""
    panel = assemble_panel(_universe(), _mkt(), dataclasses.replace(SPEC, target_kind="raw"))
    assert panel.y_daily is not None
    pred = torch.randn(len(panel), generator=torch.Generator().manual_seed(1))
    for _, short, long in _quantile_legs(pred, panel.date, 3):
        w = _leg_weights(panel.entity, short, long)
        assert sum(w.values()) == pytest.approx(0.0, abs=1e-12)

    h = SPEC.horizon
    g = torch.Generator().manual_seed(2)
    common = torch.randn(int(panel.date.max()) + h + 2, generator=g, dtype=torch.float64) * 0.02
    # row (t, k) holds the return on day t + k + 1: every stock gets that day's constant
    shift = torch.stack([common[panel.date + k + 1] for k in range(h)], dim=1).float()
    y_daily2 = panel.y_daily + shift
    y2 = panel.y + shift.sum(dim=1)
    for scheme in ("nonoverlapping", "staggered"):
        a = long_short_book(pred, panel.y, panel.date, panel.entity, n_quantiles=3,
                            horizon=h, scheme=scheme, y_daily=panel.y_daily)
        b = long_short_book(pred, y2, panel.date, panel.entity, n_quantiles=3,
                            horizon=h, scheme=scheme, y_daily=y_daily2)
        assert torch.allclose(a[1], b[1], atol=1e-5), scheme  # gross
        assert torch.equal(a[2], b[2]), scheme  # turnover


def test_registry_records_target_kind(tmp_path: Path):
    panel = assemble_panel(_universe(), _mkt(), SPEC)
    assert panel_target_kind(panel) == "market_neutral"
    assert trial_provenance(panel)["target_kind"] == "market_neutral"
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    with pytest.raises(ValueError, match="target_kind"):
        reg.log("t", {"x": 1.0}, config={
            "data_source": "s", "post_delisting_return": None, "hidden_init": None,
            "portfolio_scheme": None,
        })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        cfg = dataclasses.replace(small_config(), data=data_config_from_panel(panel))
        run_sweep(
            panel, [nec_arm("soft", cfg)], seeds=(0,), registry=reg, tag="mn", steps=3,
            n_folds=2, test_dates_per_fold=12, purge_dates=SPEC.horizon,
            backtest_quantiles=2, verbose=False,
        )
    rows = reg.trials("mn")
    assert rows and all(r.config["target_kind"] == "market_neutral" for r in rows)


def test_target_kind_falls_back_to_the_target_name():
    panel = assemble_panel(_universe(), _mkt(), SPEC)
    bare = dataclasses.replace(panel, metadata={})
    assert panel_target_kind(bare) == "market_neutral"
    for kind in ("raw", "residual"):
        spec = dataclasses.replace(SPEC, target_kind=kind, beta_window=60)
        p = dataclasses.replace(assemble_panel(_universe(), _mkt(spec), spec), metadata={})
        assert panel_target_kind(p) == kind
