"""Residual-return target (`target_kind="residual"`): the market-neutral y.

The syllabus ablation `y = fwd − β_t · mkt_fwd`. Planted-truth validations:
a market clone (β=1) must have a ~zero residual target; a known-β ticker's
rolling beta must recover the truth; and the trailing beta must pass the same
no-lookahead probe every feature passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nec_moe import StageBSpec, build_panel, rolling_beta
from nec_moe.features import market_features, ticker_features


def _px_from_returns(r: np.ndarray, seed: int = 0) -> pd.DataFrame:
    close = 100.0 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2018-01-02", periods=len(r))
    g = np.random.default_rng(seed)
    vol = 1e6 * (1.0 + 0.1 * g.standard_normal(len(r))).clip(0.5)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": vol},
        index=idx,
    )


def _market(n: int = 420, seed: int = 7) -> tuple[np.ndarray, pd.DataFrame]:
    g = np.random.default_rng(seed)
    r = 0.0002 + 0.01 * g.standard_normal(n)
    return r, _px_from_returns(r, seed=seed)


def test_rolling_beta_recovers_true_beta_and_is_trailing():
    mkt_r, _ = _market()
    g = np.random.default_rng(1)
    r = pd.Series(2.0 * mkt_r + 0.002 * g.standard_normal(len(mkt_r)))
    beta = rolling_beta(r, pd.Series(mkt_r), window=120)
    assert beta[:119].isna().all()  # warm-up: no beta before a full window
    assert float((beta[150:] - 2.0).abs().max()) < 0.15  # truth recovered

    # no lookahead: beta at t from a series truncated at t equals the full one
    t = 300
    trunc = rolling_beta(r.iloc[: t + 1], pd.Series(mkt_r[: t + 1]), window=120)
    assert trunc.iloc[t] == pytest.approx(float(beta.iloc[t]), abs=1e-12)


def test_residual_target_is_zero_for_a_market_clone():
    """A ticker that IS the market has β=1 and zero residual forward return."""
    mkt_r, market = _market()
    px = _px_from_returns(mkt_r, seed=7)  # identical prices
    spec = StageBSpec(target_kind="residual", beta_window=60, horizon=5)
    f = ticker_features(px, market_features(market, spec), spec)
    resid = f[spec.target].dropna()
    assert len(resid) > 200
    assert float(resid.abs().max()) < 1e-10


def test_residual_target_removes_the_market_component():
    """β=2 ticker: raw fwd correlates with the market's fwd; residual doesn't."""
    mkt_r, market = _market()
    g = np.random.default_rng(3)
    r = 2.0 * mkt_r + 0.004 * g.standard_normal(len(mkt_r))
    px = _px_from_returns(r, seed=3)

    raw_spec = StageBSpec(target_kind="raw", horizon=5)
    res_spec = StageBSpec(target_kind="residual", beta_window=120, horizon=5)
    mkt = market_features(market, res_spec)
    f_raw = ticker_features(px, mkt, raw_spec)
    f_res = ticker_features(px, mkt, res_spec)

    joined = pd.DataFrame(
        {
            "raw": f_raw[raw_spec.target],
            "res": f_res[res_spec.target],
            "mkt_fwd": mkt["mkt_fwd_ret"].reindex(f_raw.index),
        }
    ).dropna()
    corr_raw = joined["raw"].corr(joined["mkt_fwd"])
    corr_res = joined["res"].corr(joined["mkt_fwd"])
    assert corr_raw > 0.9  # β=2 name: the raw target IS mostly market
    assert abs(corr_res) < 0.15  # the residual target is market-neutral


def test_residual_panel_builds_names_target_and_pays_the_warmup():
    mkt_r, market = _market()
    prices = {}
    for i in range(6):
        g = np.random.default_rng(10 + i)
        r = (0.5 + 0.3 * i) * mkt_r + 0.006 * g.standard_normal(len(mkt_r))
        prices[f"t{i}"] = _px_from_returns(r, seed=10 + i)

    raw = build_panel(prices, market, StageBSpec(seq_len=10, target_kind="raw"))
    res = build_panel(
        prices, market,
        StageBSpec(seq_len=10, target_kind="residual", beta_window=250),
    )
    assert res.schema.target == "fwd_resid_ret_5d"
    assert raw.schema.target == "fwd_ret_5d"
    # a beta window longer than the longest feature warm-up (mom_120d) costs
    # leading rows; a 120-day beta would coincide with mom_120d and cost none
    assert len(res) < len(raw)
    assert bool(res.y.isfinite().all())


def test_residual_guards():
    with pytest.raises(ValueError, match="beta_window"):
        StageBSpec(target_kind="residual", beta_window=5).validate()
    mkt_r, market = _market(n=200)
    px = _px_from_returns(mkt_r, seed=7)
    spec = StageBSpec(target_kind="residual", beta_window=60)
    with pytest.raises(ValueError, match="market_features"):
        # market frame built WITHOUT the spec lacks mkt_fwd_ret
        ticker_features(px, market_features(market), spec)
