"""Stage B (real data pipeline), tested entirely offline against fixture CSVs.

The download layer reads from a disk cache; these tests pre-seed the cache
with deterministic Stooq-format CSVs, so the feature layer, panel assembly,
and the no-lookahead guarantee are all verified without any network access.
The one genuinely-network test lives at the bottom and is skipped unless
``NEC_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nec_moe import (
    NECModel,
    StageBSpec,
    Trainer,
    build_panel,
    build_stage_b_panel,
    data_config_from_panel,
    load_ohlcv,
    load_universe,
    walk_forward_evaluate,
)
from nec_moe.config import EncoderConfig, ExpertConfig, NECConfig, TrainConfig
from nec_moe.features import (
    SEQUENCE_FEATURES,
    SNAPSHOT_FEATURES,
    market_features,
    ticker_features,
)

START, END = "2020-01-01", "2021-12-31"
N_DAYS = 420


def _fixture_prices(seed: int, n: int = N_DAYS) -> pd.DataFrame:
    g = np.random.default_rng(seed)
    dates = pd.bdate_range(START, periods=n)
    ret = g.normal(0.0004, 0.02, size=n)
    close = 100.0 * np.exp(np.cumsum(ret))
    volume = 1e6 * np.exp(g.normal(0.0, 0.4, size=n))
    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": close * (1 + g.normal(0, 0.002, n)),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": volume.round(),
        }
    )


def _seed_cache(cache: Path, tickers: list[str], *, seed0: int = 10) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(tickers):
        df = _fixture_prices(seed0 + i)
        (cache / f"{t}.us.{START}.{END}.csv").write_text(df.to_csv(index=False))


@pytest.fixture()
def cache_dir(tmp_path: Path) -> Path:
    _seed_cache(tmp_path, ["spy", "aaa", "bbb", "ccc", "ddd", "eee", "fff"])
    return tmp_path


# --------------------------------------------------------------------------- #
# Download / cache layer (offline: files pre-seeded)
# --------------------------------------------------------------------------- #


def test_load_ohlcv_parses_cached_csv(cache_dir: Path):
    px = load_ohlcv("aaa", START, END, cache_dir)
    assert list(px.columns) == ["open", "high", "low", "close", "volume"]
    assert px.index.is_monotonic_increasing and px.index.is_unique
    assert len(px) == N_DAYS and (px["close"] > 0).all()


def test_load_ohlcv_dedupes_and_sorts(cache_dir: Path):
    # append a duplicate of the first row: parser must dedupe (keep last)
    path = cache_dir / f"aaa.us.{START}.{END}.csv"
    lines = path.read_text().strip().splitlines()
    path.write_text("\n".join(lines + [lines[1]]) + "\n")
    px = load_ohlcv("aaa", START, END, cache_dir)
    assert px.index.is_unique and len(px) == N_DAYS


def test_load_universe_skips_bad_tickers(cache_dir: Path):
    # a cached-but-corrupt response (Stooq's 'No data') is a per-ticker failure
    (cache_dir / f"bad.us.{START}.{END}.csv").write_text("No data\n")
    prices, market = load_universe(
        ("aaa", "bbb", "ccc", "ddd", "eee", "bad"), START, END, cache_dir
    )
    assert set(prices) == {"aaa", "bbb", "ccc", "ddd", "eee"}
    assert len(market) == N_DAYS
    with pytest.raises(RuntimeError, match="only"):
        load_universe(("aaa", "bad"), START, END, cache_dir, min_tickers=2)


# --------------------------------------------------------------------------- #
# Feature layer
# --------------------------------------------------------------------------- #


def _px_and_mkt(cache_dir: Path):
    px = load_ohlcv("aaa", START, END, cache_dir)
    market = load_ohlcv("spy", START, END, cache_dir)
    return px, market_features(market)


def test_feature_values_hand_checked(cache_dir: Path):
    spec = StageBSpec()
    px, mkt = _px_and_mkt(cache_dir)
    f = ticker_features(px, mkt, spec)
    c = px["close"]
    t = 200  # deep enough past every warm-up window
    d = px.index[t]
    assert f.loc[d, "ret_1d"] == pytest.approx(math.log(c.iloc[t] / c.iloc[t - 1]))
    assert f.loc[d, "ret_20d"] == pytest.approx(math.log(c.iloc[t] / c.iloc[t - 20]))
    assert f.loc[d, "mom_120d"] == pytest.approx(math.log(c.iloc[t] / c.iloc[t - 120]))
    assert f.loc[d, "drawdown_60d"] == pytest.approx(
        float(c.iloc[t] / c.iloc[t - 59 : t + 1].max() - 1.0)
    )
    r1 = np.log(c).diff()
    assert f.loc[d, "vol_20d"] == pytest.approx(float(r1.iloc[t - 19 : t + 1].std()))
    # target: forward h-day log return, the only forward-looking column
    assert f.loc[d, spec.target] == pytest.approx(
        math.log(c.iloc[t + spec.horizon] / c.iloc[t])
    )
    assert f[spec.target].iloc[-spec.horizon :].isna().all()


def test_no_lookahead(cache_dir: Path):
    """THE Stage-B leakage test: every non-target feature at date t is
    unchanged when all data after t is deleted."""
    spec = StageBSpec()
    px = load_ohlcv("aaa", START, END, cache_dir)
    market = load_ohlcv("spy", START, END, cache_dir)
    full = ticker_features(px, market_features(market), spec)
    feature_cols = [c for c in full.columns if c != spec.target]
    for t in (250, 340):
        cutoff = px.index[t]
        trunc = ticker_features(
            px.loc[:cutoff], market_features(market.loc[:cutoff]), spec
        )
        a = full.loc[cutoff, feature_cols].to_numpy(dtype=float)
        b = trunc.loc[cutoff, feature_cols].to_numpy(dtype=float)
        assert np.allclose(a, b, equal_nan=True), "future data changed a feature"


# --------------------------------------------------------------------------- #
# Panel assembly
# --------------------------------------------------------------------------- #


def test_panel_contract_ordering_and_window_alignment(cache_dir: Path):
    prices, market = load_universe(
        ("aaa", "bbb", "ccc", "ddd", "eee", "fff"), START, END, cache_dir
    )
    spec = StageBSpec(cs_rank=False)  # raw units so cross-checks are exact
    panel = build_panel(prices, market, spec)

    # contract: schema widths match tensors; batch validates
    panel.full_batch().validate_schema(panel.schema)
    assert panel.schema.snapshot_features == SNAPSHOT_FEATURES
    assert panel.x_seq.shape[1:] == (spec.seq_len, len(SEQUENCE_FEATURES))

    # date-major ordering (the stateful trainer's contract)
    d = panel.date
    assert (d[1:] >= d[:-1]).all()

    # window alignment: the window's last step IS the row's own date —
    # seq channel 0 (ret_1d) at step -1 equals the raw snapshot ret_1d
    ret_col = SNAPSHOT_FEATURES.index("ret_1d")
    assert torch.allclose(panel.x_seq[:, -1, 0], panel.x_snap[:, ret_col], atol=1e-6)

    # labels map codes back
    assert panel.entity_labels is not None and len(panel.entity_labels) == 6
    assert panel.date_labels is not None
    assert int(panel.date.max()) < len(panel.date_labels)

    # every kept date has a thick cross-section
    _, counts = torch.unique(panel.date, return_counts=True)
    assert int(counts.min()) >= spec.min_names_per_date


def test_cross_sectional_ranks(cache_dir: Path):
    prices, market = load_universe(
        ("aaa", "bbb", "ccc", "ddd", "eee", "fff"), START, END, cache_dir
    )
    panel = build_panel(prices, market, StageBSpec(cs_rank=True))
    assert float(panel.x_snap.min()) >= -0.5 and float(panel.x_snap.max()) <= 0.5
    d0 = panel.date == panel.date[0]
    block = panel.x_snap[d0]
    assert block.shape[0] >= 5
    # full cross-section: each column spans exactly [-0.5, 0.5] and centers ~0
    assert torch.allclose(block.min(dim=0).values, torch.full((block.shape[1],), -0.5))
    assert torch.allclose(block.max(dim=0).values, torch.full((block.shape[1],), 0.5))
    assert float(block.mean().abs()) < 1e-6


def test_real_panel_runs_through_harness(cache_dir: Path):
    """Plumbing: a real-format panel trains and evaluates end to end. The
    fixture data is a random walk, so NO performance is asserted — only that
    the full Stage-B -> model -> walk-forward -> backtest chain holds."""
    prices, market = load_universe(
        ("aaa", "bbb", "ccc", "ddd", "eee", "fff"), START, END, cache_dir
    )
    spec = StageBSpec(seq_len=10)
    panel = build_panel(prices, market, spec)
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
        panel,
        make_trainer,
        n_folds=2,
        test_dates_per_fold=15,
        purge_dates=spec.horizon,  # purge = the label horizon, by construction
        steps=30,
        warmstart_key=lambda p: p.x_seq[:, :, 2].std(dim=1),  # vol channel
        backtest_quantiles=3,
        cost_rate=0.001,
    )
    assert len(result.folds) == 2
    assert all(math.isfinite(f.nll) for f in result.folds)
    assert result.pooled_portfolio is not None
    assert math.isfinite(result.pooled_portfolio.mean_net)


def test_data_config_from_panel_builds_model(cache_dir: Path):
    prices, market = load_universe(
        ("aaa", "bbb", "ccc", "ddd", "eee", "fff"), START, END, cache_dir
    )
    panel = build_panel(prices, market, StageBSpec(seq_len=10))
    cfg = NECConfig(data=data_config_from_panel(panel))
    model = NECModel(cfg)
    out = model(panel.x_seq[:32], panel.x_snap[:32])
    assert out.y_hat.shape == (32,)


# --------------------------------------------------------------------------- #
# Live network check — opt-in only (NEC_NETWORK_TESTS=1)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    os.environ.get("NEC_NETWORK_TESTS", "0") != "1",
    reason="network test; set NEC_NETWORK_TESTS=1 to run",
)
def test_live_small_panel(tmp_path: Path):
    """Live fetch via the yfinance fallback (Stooq's endpoint is behind a
    JS anti-bot wall as of mid-2026 — deliberately not circumvented; its
    manual browser-download-into-cache workflow is exercised by the offline
    tests above, since a cached file short-circuits the network either way)."""
    panel = build_stage_b_panel(
        "2022-01-01",
        "2023-12-31",
        str(tmp_path),
        tickers=("aapl", "msft", "jpm", "xom", "ko", "pg"),
        source="yfinance",
        spec=StageBSpec(seq_len=10),
    )
    assert len(panel) > 1000
    panel.full_batch().validate_schema(panel.schema)
