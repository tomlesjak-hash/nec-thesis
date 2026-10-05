"""Stage B (real-data feature layer), tested offline on invented daily frames.

The feature layer takes one daily frame per entity (``DAILY_COLUMNS``: log
return, share volume, dollar volume, share factor, tradable, fill) and a
market log-return series; the CRSP layer builds those from ``DlyRet``,
``DlyPrc``, ``DlyVol`` and ``DlyCumFacShr``. These tests check the feature
values by hand, the panel contract and the harness chain on invented
numbers; the CRSP-specific rules and the no-lookahead test on a CRSP-format
fixture live in ``test_crsp.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from nec_moe import (
    NECModel,
    StageBSpec,
    Trainer,
    assemble_panel,
    data_config_from_panel,
    market_frame,
    stock_features,
    walk_forward_evaluate,
)
from nec_moe.config import EncoderConfig, ExpertConfig, NECConfig, TrainConfig
from nec_moe.features import SEQUENCE_FEATURES, SNAPSHOT_FEATURES, FeatureSpec

#: these tests pin the legacy 14-feature set (brief 08 D keeps it for that);
#: the Q26 set is tested in test_q26_features.py
LEGACY = FeatureSpec(feature_set="legacy14")

N_DAYS = 420
IDX = pd.bdate_range("2020-01-01", periods=N_DAYS)


def _daily(seed: int) -> pd.DataFrame:
    """An invented stock: log returns, raw volume, and raw price x volume."""
    g = np.random.default_rng(seed)
    ret = g.normal(0.0004, 0.02, size=N_DAYS)
    price = 100.0 * np.exp(np.cumsum(ret))
    volume = np.round(1e6 * np.exp(g.normal(0.0, 0.4, size=N_DAYS)))
    return pd.DataFrame(
        {"ret": ret, "volume": volume, "dollar_volume": price * volume}, index=IDX
    )


def _market(seed: int = 99) -> pd.Series:
    g = np.random.default_rng(seed)
    return pd.Series(g.normal(0.0003, 0.01, size=N_DAYS), index=IDX)


def _universe(n: int = 6) -> dict[str, pd.DataFrame]:
    return {f"{10001 + i}": _daily(10 + i) for i in range(n)}


def test_feature_values_hand_checked():
    spec = StageBSpec(features=LEGACY)
    daily = _daily(10)
    f = stock_features(daily, market_frame(_market(), spec), spec)
    r = daily["ret"]
    logc = r.cumsum()  # log of a total-return index, for the hand checks only
    t = 200  # deep enough past every warm-up window
    d = IDX[t]
    assert f.loc[d, "ret_1d"] == pytest.approx(r.iloc[t])
    assert f.loc[d, "ret_20d"] == pytest.approx(logc.iloc[t] - logc.iloc[t - 20])
    assert f.loc[d, "mom_120d"] == pytest.approx(logc.iloc[t] - logc.iloc[t - 120])
    level = np.exp(logc)
    assert f.loc[d, "drawdown_60d"] == pytest.approx(
        float(level.iloc[t] / level.iloc[t - 59 : t + 1].max() - 1.0)
    )
    assert f.loc[d, "vol_20d"] == pytest.approx(float(r.iloc[t - 19 : t + 1].std()))
    v = daily["volume"].iloc[t - 19 : t + 1]
    assert f.loc[d, "volume_z_20d"] == pytest.approx(float((v.iloc[-1] - v.mean()) / v.std()))
    # target: forward h-day log return, the only forward-looking column; one
    # stock's frame carries the raw return the market-neutral target is built
    # from (the cross-sectional mean is taken in assemble_panel)
    assert spec.stock_target == spec.raw_target == "fwd_ret_5d"
    assert f.loc[d, spec.stock_target] == pytest.approx(
        logc.iloc[t + spec.horizon] - logc.iloc[t]
    )
    assert f[spec.stock_target].iloc[-spec.horizon :].isna().all()


def test_dollar_volume_is_raw_price_times_raw_volume():
    """Brief 06 A.4.5 (and audit D-2): dollar volume at t is the dollars
    actually traded each day of the window, no adjustment applied."""
    spec = StageBSpec(features=LEGACY)
    daily = _daily(11)
    f = stock_features(daily, market_frame(_market(), spec), spec)
    d = IDX[200]
    assert f.loc[d, "dollar_vol_20d"] == pytest.approx(
        math.log(daily["dollar_volume"].iloc[181:201].mean())
    )


def test_drawdown_compounds_only_inside_its_window():
    """A missing return outside the 60-day window leaves the drawdown alone;
    one inside it makes the drawdown missing, never computed as if zero."""
    spec = StageBSpec(features=LEGACY)
    daily = _daily(12)
    mkt = market_frame(_market(), spec)
    base = stock_features(daily, mkt, spec)["drawdown_60d"]
    gap = daily.copy()
    gap.iloc[100, gap.columns.get_loc("ret")] = np.nan
    hit = stock_features(gap, mkt, spec)["drawdown_60d"]
    assert hit.iloc[100:159].isna().all()  # every window holding the gap
    assert np.allclose(hit.iloc[159:], base.iloc[159:])  # windows past it
    assert np.allclose(hit.iloc[59:100], base.iloc[59:100])  # windows before it


def test_no_lookahead():
    """Every non-target feature at t is unchanged when all data after t is
    deleted (the CRSP-fixture version is in test_crsp.py)."""
    spec = StageBSpec(features=LEGACY)
    daily, mkt_ret = _daily(13), _market()
    full = stock_features(daily, market_frame(mkt_ret), spec)
    cols = [c for c in full.columns if c != spec.stock_target]
    for t in (250, 340):
        cut = IDX[t]
        trunc = stock_features(daily.loc[:cut], market_frame(mkt_ret.loc[:cut]), spec)
        assert np.allclose(
            full.loc[cut, cols].to_numpy(dtype=float),
            trunc.loc[cut, cols].to_numpy(dtype=float),
            equal_nan=True,
        ), "future data changed a feature"


def test_panel_contract_ordering_and_window_alignment():
    spec = StageBSpec(cs_rank=False, features=LEGACY)  # raw units so cross-checks are exact
    panel = assemble_panel(_universe(), market_frame(_market(), spec), spec)

    panel.full_batch().validate_schema(panel.schema)
    assert panel.schema.snapshot_features == SNAPSHOT_FEATURES
    assert panel.x_seq.shape[1:] == (spec.seq_len, len(SEQUENCE_FEATURES))
    d = panel.date
    assert (d[1:] >= d[:-1]).all()  # date-major ordering
    # the window's last step IS the row's own date
    ret_col = SNAPSHOT_FEATURES.index("ret_1d")
    assert torch.allclose(panel.x_seq[:, -1, 0], panel.x_snap[:, ret_col], atol=1e-6)
    assert panel.entity_labels == tuple(f"{10001 + i}" for i in range(6))
    assert panel.date_labels is not None and int(panel.date.max()) < len(panel.date_labels)
    _, counts = torch.unique(panel.date, return_counts=True)
    assert int(counts.min()) >= spec.min_names_per_date


def test_tradable_false_days_are_never_rows_but_their_returns_count():
    """A non-tradable day (a delisting-return row, a fill day) is not a row,
    and its return still reaches the targets of the rows before it."""
    spec = StageBSpec(seq_len=10, cs_rank=False, features=LEGACY)
    daily = _universe()
    daily["10001"]["tradable"] = True
    daily["10001"].iloc[300, daily["10001"].columns.get_loc("tradable")] = False
    panel = assemble_panel(daily, market_frame(_market(), spec), spec)
    assert panel.date_labels is not None
    rows = panel.date[panel.entity == 0]
    days = {panel.date_labels[int(c)] for c in rows}
    assert str(IDX[300].date()) not in days
    assert str(IDX[299].date()) in days  # its target includes day 300's return


def test_window_keeps_only_rows_inside_it():
    spec = StageBSpec(seq_len=10, features=LEGACY)
    panel = assemble_panel(
        _universe(), market_frame(_market(), spec), spec,
        window=(str(IDX[200].date()), str(IDX[260].date())),
    )
    assert panel.date_labels is not None
    assert panel.date_labels[0] == str(IDX[200].date())
    assert panel.date_labels[-1] == str(IDX[260].date())


def test_cross_sectional_ranks():
    spec = StageBSpec(cs_rank=True, features=LEGACY)
    panel = assemble_panel(_universe(), market_frame(_market(), spec), spec)
    assert float(panel.x_snap.min()) >= -0.5 and float(panel.x_snap.max()) <= 0.5
    block = panel.x_snap[panel.date == panel.date[0]]
    assert block.shape[0] >= 5
    assert torch.allclose(block.min(dim=0).values, torch.full((block.shape[1],), -0.5))
    assert torch.allclose(block.max(dim=0).values, torch.full((block.shape[1],), 0.5))
    assert float(block.mean().abs()) < 1e-6


def test_assemble_panel_records_whether_it_rank_normalized():
    """The flag filter_point_in_time relies on to re-rank (audit D-1)."""
    for flag in (True, False):
        spec = StageBSpec(cs_rank=flag, features=LEGACY)
        panel = assemble_panel(_universe(), market_frame(_market(), spec), spec)
        assert panel.schema.rank_normalized is flag


def test_real_format_panel_runs_through_harness():
    """Plumbing: a real-format panel trains and evaluates end to end. The
    data is a random walk, so NO performance is asserted."""
    spec = StageBSpec(seq_len=10, features=LEGACY)
    panel = assemble_panel(_universe(), market_frame(_market(), spec), spec)
    cfg = NECConfig(
        data=data_config_from_panel(panel),
        encoder=EncoderConfig(hidden_dim=8),
        experts=ExpertConfig(hidden_dims=(8, 4), dropout=0.0),
        train=TrainConfig(sigma_init=0.1, sigma_freeze_steps=0, batch_size=256),
    )

    def make_trainer() -> Trainer:
        torch.manual_seed(0)
        return Trainer(NECModel(cfg))

    result = walk_forward_evaluate(
        panel, make_trainer, n_folds=2, test_dates_per_fold=15,
        purge_dates=spec.horizon,  # purge = the label horizon, by construction
        steps=30,
        warmstart_key=lambda p: p.x_seq[:, :, 2].std(dim=1),  # vol channel
        backtest_quantiles=3, cost_rate=0.001,
    )
    assert len(result.folds) == 2
    assert all(math.isfinite(f.nll) for f in result.folds)
    assert result.pooled_portfolio is not None
    assert math.isfinite(result.pooled_portfolio.mean_net)


def test_data_config_from_panel_builds_model():
    spec = StageBSpec(seq_len=10, features=LEGACY)
    panel = assemble_panel(_universe(), market_frame(_market(), spec), spec)
    model = NECModel(NECConfig(data=data_config_from_panel(panel)))
    out = model(panel.x_seq[:32], panel.x_snap[:32])
    assert out.y_hat.shape == (32,)
