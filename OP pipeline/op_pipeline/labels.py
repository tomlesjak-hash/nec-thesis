"""
Label construction — direct port of the OP LGBM label scheme.

Two outputs:

* y_ret : continuous forward return over `horizon` days.
* y_bucket : signed-magnitude bucket label (the OP `create_labels` scheme).
  Plus the OP balancing/downsampling helpers.

Why the OP file does bucketing:
Financial returns are heavy-tailed and dominated by tiny moves. Predicting a
continuous return exactly is brittle. Bucketing into signed magnitude levels
("how big and which direction") gives the model a more stable target and
lets the practitioner downweight the noisy near-zero bulk. This is a real
quant-shop pattern, not pedagogy.

The balancing helpers (`downsample_zero`, `balance_signed`) are reproduced
here in a cleaner form than the original (consistent variable names, no
'miltipile' typo, no global-state dependence).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from .config import LabelConfig


def add_forward_return(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Add `y_ret` = log forward return over `horizon` business days, per asset.
    """
    df = df.sort_values(["asset", "date"]).reset_index(drop=True).copy()
    log_close = np.log(df["close"])
    log_close_fwd = log_close.groupby(df["asset"]).shift(-horizon)
    df["y_ret"] = log_close_fwd - log_close
    return df


def signed_bucket_labels(series: pd.Series, thresholds: Sequence[float]) -> pd.Series:
    """
    Convert a continuous series to signed-magnitude bucket labels.

    With thresholds = [0.005, 0.01, 0.02]:
       |x| in [0, 0.005)         -> 0
       |x| in [0.005, 0.01)      -> 1
       |x| in [0.01,  0.02)      -> 2
       |x| in [0.02,  inf)       -> 3
    then multiplied by sign(x). Direct port of OP `create_labels`.
    """
    thresholds = sorted(thresholds)
    abs_s = series.abs().to_numpy()
    bounds = np.array([0.0] + list(thresholds) + [np.inf])
    indices = np.searchsorted(bounds, abs_s, side="right") - 1
    indices = np.clip(indices, 0, len(thresholds))
    signs = np.sign(series.to_numpy())
    # sign(0) == 0, which collapses the zero bucket regardless of indices — that's fine,
    # the zero bucket is class 0 anyway.
    labels = indices * signs
    return pd.Series(labels, index=series.index)


def downsample_zero(
    labels: pd.Series,
    multiple: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Cap the number of class-0 rows at `multiple * (count of non-zero rows)`.
    Returns the *kept* class-0 indices.

    OP analogue: `downsample_zero_label`. The original variable was misspelled
    ('miltipile') — fixed here.
    """
    rng = rng or np.random.default_rng(42)
    zero_idx = labels[labels == 0].index.to_numpy()
    nonzero_count = int((labels != 0).sum())
    target = int(nonzero_count * multiple)
    if len(zero_idx) <= target:
        return zero_idx
    return rng.choice(zero_idx, size=target, replace=False)


def balance_signed(
    labels: pd.Series,
    rng: np.random.Generator | None = None,
) -> List:
    """
    For each absolute bucket level |k| ≥ 1, take min(count(+k), count(-k))
    from each side. Returns the list of kept indices. Excludes class 0.

    OP analogue: `balance_labels`.
    """
    rng = rng or np.random.default_rng(42)
    kept: list = []

    abs_levels = sorted({abs(int(v)) for v in labels.unique() if v != 0})
    for k in abs_levels:
        pos = labels[labels == k].index.to_numpy()
        neg = labels[labels == -k].index.to_numpy()
        m = min(len(pos), len(neg))
        if len(pos) > m:
            pos = rng.choice(pos, size=m, replace=False)
        if len(neg) > m:
            neg = rng.choice(neg, size=m, replace=False)
        kept.extend(pos.tolist())
        kept.extend(neg.tolist())
    return kept


def build_balanced_sample(
    df: pd.DataFrame,
    cfg: LabelConfig | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    End-to-end OP label processing for a training fold:

    1. Bucket the continuous `y_ret` into signed buckets `y_bucket`.
    2. Downsample class 0 to `zero_multiple * nonzero_count`.
    3. Balance positive vs negative counts within each absolute level.
    4. Return only the surviving rows.

    Assumes `add_forward_return` has already been called.
    """
    cfg = cfg or LabelConfig()
    rng = np.random.default_rng(seed)

    work = df.dropna(subset=["y_ret"]).copy()
    # filter extreme labels — OP file used |y| < 0.2; we use the same.
    work = work[work["y_ret"].abs() < 0.2]

    work["y_bucket"] = signed_bucket_labels(work["y_ret"], cfg.thresholds)

    zero_keep = downsample_zero(work["y_bucket"], cfg.zero_multiple, rng=rng)
    nonzero_keep = balance_signed(work["y_bucket"], rng=rng)

    keep_idx = np.concatenate([zero_keep, np.array(nonzero_keep, dtype=zero_keep.dtype)])
    return work.loc[keep_idx].sort_index()
