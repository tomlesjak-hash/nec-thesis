"""
Synthetic multi-asset OHLCV data generator.

Why synthetic:
The OP pipeline assumed access to per-stock, per-day parquet files at
`/home/jupyterData/...` and a proprietary `dw_data.fastpai` library — neither
of which are available outside that environment. We replace that layer with
a generator that produces a realistic-looking long panel and, importantly,
a *planted predictive structure* so the pipeline downstream can be validated.

Design choices:
- Multi-asset cross section (panel data: date × asset), the same shape the
  OP framework expects.
- Each asset has:
  * an idiosyncratic random walk in log-price,
  * exposure to a few latent "factors" that drift slowly,
  * a small forward-looking signal mixed into a subset of features so the
    pipeline has *something* to learn (otherwise IC is structurally zero
    and you can't tell whether the model works or the data is hopeless).
- OHLCV is built by adding intraday noise around the close. Volume is
  log-normal scaled by realized vol.

The signal is deliberately moderate — strong enough that a reasonable model
gets IC > 0, not so strong that it makes the test trivial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .config import DataConfig


def _trading_days(n: int, start: str = "2022-01-03") -> pd.DatetimeIndex:
    """Generate `n` business days starting at `start`."""
    return pd.bdate_range(start=start, periods=n)


def generate_panel(cfg: DataConfig | None = None) -> pd.DataFrame:
    """
    Generate a long-format panel with columns:

        date, asset, open, high, low, close, volume,
        f_mom, f_value, f_quality, f_noise1, f_noise2

    The `f_*` columns are exogenous factor exposures; some of them carry a
    forward-looking signal so that downstream feature engineering + modeling
    has predictive structure to recover.

    Returns
    -------
    pd.DataFrame
        Long-format panel sorted by (date, asset). Shape ≈ n_days * n_assets.
    """
    cfg = cfg or DataConfig()
    rng = np.random.default_rng(cfg.seed)

    n_d, n_a = cfg.n_days, cfg.n_assets
    dates = _trading_days(n_d)
    assets = [f"A{ i:03d}" for i in range(n_a)]

    # 1. latent factor returns (3 factors, all assets see them with different loadings)
    factor_rets = rng.normal(0.0, 0.008, size=(n_d, 3))  # (T, K=3)
    loadings = rng.normal(0.0, 1.0, size=(n_a, 3))       # (N, K)

    # 2. idiosyncratic returns per asset
    idio = rng.normal(0.0, cfg.base_vol, size=(n_d, n_a))

    # 3. forward-looking signal — this is what we WANT the model to find.
    # We construct a "true" signal s_t at each date that mildly predicts
    # next-period idio return for a subset of assets.
    true_signal = rng.normal(0.0, 1.0, size=(n_d, n_a))  # (T, N)
    # mix planted signal into NEXT-day idio return
    next_idio_boost = np.zeros_like(idio)
    next_idio_boost[1:, :] = cfg.factor_signal_strength * cfg.base_vol * true_signal[:-1, :]
    idio = idio + next_idio_boost

    # 4. assemble per-asset log-returns
    factor_contrib = factor_rets @ loadings.T          # (T, N)
    log_rets = factor_contrib + idio                   # (T, N)

    # 5. price paths
    log_prices = log_rets.cumsum(axis=0) + rng.normal(2.0, 0.5, size=(1, n_a))
    close = np.exp(log_prices)

    # 6. fake OHLCV: open ≈ prev close * (1 + small noise), H/L bracket close
    noise_open = rng.normal(0.0, cfg.base_vol * 0.5, size=close.shape)
    open_ = np.empty_like(close)
    open_[0] = close[0] * (1.0 - rng.normal(0.0, cfg.base_vol * 0.3, size=n_a))
    open_[1:] = close[:-1] * (1.0 + noise_open[1:])

    spread = np.abs(rng.normal(0.0, cfg.base_vol * 1.5, size=close.shape))
    high = np.maximum(close, open_) * (1.0 + spread)
    low = np.minimum(close, open_) * (1.0 - spread)
    # log-normal volume scaled by abs(return) so big-move days have more volume
    vol = np.exp(rng.normal(13.0, 0.7, size=close.shape)) * (1.0 + 5.0 * np.abs(log_rets))

    # 7. factor "exposures" (engineered features): some carry signal, some don't
    f_mom = pd.DataFrame(log_rets, index=dates, columns=assets) \
        .rolling(20, min_periods=5).sum().to_numpy()
    f_value = -log_prices + log_prices.mean(axis=0, keepdims=True)  # mean-reverting to long-run avg
    f_quality = (loadings[:, 0] + 0.3 * rng.normal(0.0, 1.0, size=(n_d, n_a)))  # slowly drifting
    f_noise1 = rng.normal(0.0, 1.0, size=close.shape)
    f_noise2 = rng.normal(0.0, 1.0, size=close.shape)

    # The planted signal: bake the true_signal into a noisy observed feature
    # so that the model can in principle learn next-day return from features.
    f_planted = 0.6 * true_signal + 0.8 * rng.normal(0.0, 1.0, size=close.shape)

    # 8. stack into long format
    df = pd.DataFrame({
        "date": np.repeat(dates, n_a),
        "asset": np.tile(assets, n_d),
        "open": open_.flatten(),
        "high": high.flatten(),
        "low": low.flatten(),
        "close": close.flatten(),
        "volume": vol.flatten(),
        "f_mom": f_mom.flatten(),
        "f_value": f_value.flatten(),
        "f_quality": f_quality.flatten(),
        "f_planted": f_planted.flatten(),
        "f_noise1": f_noise1.flatten(),
        "f_noise2": f_noise2.flatten(),
    })

    df = df.sort_values(["date", "asset"]).reset_index(drop=True)
    return df


def split_by_date(
    df: pd.DataFrame,
    train_frac: float,
    valid_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered split. Critical for finance: never sample randomly across
    time, that leaks future into past.
    """
    dates = np.sort(df["date"].unique())
    n = len(dates)
    n_tr = int(n * train_frac)
    n_va = int(n * valid_frac)

    train_dates = dates[:n_tr]
    valid_dates = dates[n_tr:n_tr + n_va]
    test_dates = dates[n_tr + n_va:]

    return (
        df[df["date"].isin(train_dates)].copy(),
        df[df["date"].isin(valid_dates)].copy(),
        df[df["date"].isin(test_dates)].copy(),
    )
