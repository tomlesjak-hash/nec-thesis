"""
Feature engineering — rolling factor construction.

This is the tabular analogue of the MLP-OP rolling operators (Mean / Std /
Min / Max / Rate / Mean_res). The MLP-OP version computes those *inside* the
neural net as learnable layers; here we precompute them as classical features
so the tabular models (LGBM) can consume them too. The MLP-OP model later
uses its own internal versions on the raw sequence.

All rolling ops are per-asset (group by asset, roll over time). We use
`min_periods=` to allow shorter histories at the start rather than dropping
the first many rows.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .config import FeatureConfig


def _rolling_per_asset(
    df: pd.DataFrame,
    col: str,
    windows: List[int],
    op: str,
) -> pd.DataFrame:
    """
    Compute one rolling stat for one column across multiple windows.

    Returns a frame indexed identically to df, with new columns
    `{col}_{op}_{w}` for each window.
    """
    out_cols = {}
    grouped = df.groupby("asset", group_keys=False)[col]
    for w in windows:
        roll = grouped.rolling(window=w, min_periods=max(2, w // 2))
        if op == "mean":
            s = roll.mean()
        elif op == "std":
            s = roll.std()
        elif op == "min":
            s = roll.min()
        elif op == "max":
            s = roll.max()
        else:
            raise ValueError(f"Unknown op: {op}")
        # rolling on a groupby returns a multiindex (asset, original_index);
        # reset to original index so we can assign columns cleanly
        s = s.reset_index(level=0, drop=True)
        out_cols[f"{col}_{op}_{w}"] = s
    return pd.DataFrame(out_cols, index=df.index)


def build_features(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """
    Build the feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel from `data.generate_panel`.

    Returns
    -------
    pd.DataFrame
        Same panel with extra columns:
        - log returns
        - rolling mean/std/min/max of returns over each window
        - rolling mean/std of volume
        - momentum-style ratios
        - the exogenous f_* features are passed through
    """
    cfg = cfg or FeatureConfig()
    df = df.sort_values(["asset", "date"]).reset_index(drop=True).copy()

    # daily log return per asset (groupby.shift avoids version-fragile apply())
    log_close = np.log(df["close"])
    df["ret_1d"] = log_close - log_close.groupby(df["asset"]).shift(1)

    feats: List[pd.DataFrame] = []
    for op in ("mean", "std", "min", "max"):
        feats.append(_rolling_per_asset(df, "ret_1d", cfg.windows, op))

    # volume features
    df["log_vol"] = np.log1p(df["volume"])
    for op in ("mean", "std"):
        feats.append(_rolling_per_asset(df, "log_vol", cfg.windows, op))

    # mean residual ("Mean_res" analogue): current return minus rolling mean
    for w in cfg.windows:
        col = f"ret_1d_mean_{w}"
        # ensure the column already exists in the assembled feats — compute on the fly
        mean_w = (
            df.groupby("asset", group_keys=False)["ret_1d"]
            .rolling(window=w, min_periods=max(2, w // 2))
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"ret_resid_{w}"] = df["ret_1d"] - mean_w

    feature_df = pd.concat([df] + feats, axis=1)

    # Replace inf/-inf with NaN (rolling std on flat windows can blow up)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    return feature_df


def feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of model-input columns: everything that's a feature
    but not identifier / OHLCV / labels.
    """
    drop = {"date", "asset", "open", "high", "low", "close", "volume"}
    # labels columns are everything starting with 'y_' (added later)
    return [c for c in df.columns if c not in drop and not c.startswith("y_")]


def build_sequence_tensor(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (N, seq_len, F) tensor for the MLP-OP sequence model.

    For each (asset, date) row, we look back `seq_len` rows (per asset) and
    stack their features as a sequence. Rows without enough history are
    dropped.

    Returns
    -------
    X : np.ndarray of shape (N, seq_len, F)
    row_idx : np.ndarray of shape (N,)
        Original df indices corresponding to the last step of each sequence.
        Use these to align labels.
    asset_idx : np.ndarray of shape (N,)
        Asset id per row (string).
    """
    df = df.sort_values(["asset", "date"]).reset_index(drop=False)  # 'index' = orig idx
    F = len(feature_cols)
    X_list, row_idx_list, asset_idx_list = [], [], []

    for asset, g in df.groupby("asset", sort=False):
        arr = g[feature_cols].to_numpy(dtype=np.float32)  # (T_a, F)
        orig_idx = g["index"].to_numpy()
        T_a = arr.shape[0]
        if T_a < seq_len:
            continue
        # sliding windows
        for t in range(seq_len - 1, T_a):
            X_list.append(arr[t - seq_len + 1: t + 1, :])
            row_idx_list.append(orig_idx[t])
            asset_idx_list.append(asset)

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, seq_len, F), dtype=np.float32)
    return X, np.array(row_idx_list), np.array(asset_idx_list)
