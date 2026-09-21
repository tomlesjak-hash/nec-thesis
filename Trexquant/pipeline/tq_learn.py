#!/usr/bin/env python3
"""Fit a small model on the panel, emit paste-ready Trexsim weights.

    python3 tq_learn.py                 # ridge, the default
    python3 tq_learn.py --model mlp     # 14 -> 4 -> 1
    python3 tq_learn.py --model both

WHY SMALL, DELIBERATELY
-----------------------
The GP searched ~1.5M expressions here and not one transferred to Trexsim: the
best of 1.5M draws is overfit by construction. Ridge over 14 terminals has
**14 parameters**. Orders of magnitude less variance, so a far better chance of
surviving the local->platform gap — which is the entire argument for doing this.

It does NOT fix the panel's limitations: ~370 large caps against the platform's
top-1000, 212 delisted names missing, no intraday data. Fewer parameters helps
with overfitting, not with the universe being different.

WHAT IS FITTED
--------------
Target is the cross-sectionally rank-normalised, industry-demeaned NEXT-DAY
return — not the raw return. Trexsim neutralises by industry and scores a
dollar-neutral book, so fitting raw returns would fit market direction, which
the simulator removes before scoring. Selection is on rank IC, not R^2: R^2 is
dominated by the ~99% of return variance that is noise, while rank IC measures
whether the ordering is right, which is what a cross-sectional alpha needs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tq_gp import (TEST, TRAIN, VALID, _industry_neutral,  # noqa: E402
                   build_terminals, load_gp_data, sanitize)

OUT = Path(__file__).resolve().parents[1] / "gp_results"

#: Event-stress days get extra weight in the fit. The submitted alpha only
#: trades on those days, so the coefficients should be tuned to them.
EVENT_WEIGHT = 3.0


# ------------------------------------------------------------------ data --
def _stack(df: pd.DataFrame) -> pd.Series:
    """Version-safe wide -> long.

    pandas changed `.stack()` three times. In 2.0 the default dropped NA rows
    and `dropna=False` kept them; 2.1 added `future_stack=True` for the new
    behaviour; 3.0 made it the default and REMOVED `dropna`, raising
    ValueError if passed. The NAs must be kept here — dropping them per column
    would misalign features against each other before the joint dropna.
    """
    for kw in ({"future_stack": True}, {"dropna": False}, {}):
        try:
            return df.stack(**kw)
        except (TypeError, ValueError):
            continue
    return df.stack()


def cs_rank_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank, centred to [-0.5, 0.5]. Robust to outliers."""
    return df.rank(axis=1, pct=True) - 0.5


def build_xy(d: dict, terms: dict,
             names: "list[str] | None" = None
             ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """-> (X, y, sample_weight), stacked long, aligned, finite.

    `names` MUST be passed and identical across train/valid/test.
    `build_terminals` adds some columns conditionally (`beta60` needs
    ret1_spx, `idio` needs a usable industry map, `hi52` needs 120+ days of
    history), so a short window can silently yield a different feature set —
    which then blows up as a shape mismatch against the fitted coefficients.
    """
    if names is None:
        names = sorted(terms)
    missing = [n for n in names if n not in terms]
    if missing:
        raise KeyError(f"window is missing features {missing}; "
                       "cannot score with coefficients fitted elsewhere")
    uni = d["universe"].astype(bool)

    # target: next-day return, industry-demeaned, then rank-normalised
    fwd = d["ret1"].shift(-1)
    if "industry" in d:
        fwd = _industry_neutral(fwd.where(uni), d["industry"])
    y = cs_rank_norm(fwd.where(uni))

    # features: rank-normalised too, so coefficients are directly comparable
    # and no single terminal dominates through scale alone
    X = {n: cs_rank_norm(sanitize(terms[n]).where(uni)) for n in names}

    stack = pd.DataFrame({n: _stack(X[n]) for n in names})
    stack["__y"] = _stack(y)
    stack = stack.replace([np.inf, -np.inf], np.nan).dropna()

    w = pd.Series(1.0, index=stack.index)
    return stack[names], stack["__y"], w


def event_weights(index, d) -> pd.Series:
    """Up-weight event-stress days — the days the alpha actually trades."""
    dates = index.get_level_values(0)
    idx = d["close"].index
    flags = pd.Series(0.0, index=idx)
    for fld, thr in [("cal_days_until_next_fomc_meeting", 1),
                     ("cal_days_since_last_fomc_meeting", 0),
                     ("cal_days_until_next_monthly_options_expiration", 1),
                     ("cal_days_until_next_quarterly_options_expiration", 1),
                     ("cal_days_until_last_trading_day_of_quarter", 7),
                     ("cal_days_until_last_trading_day_of_month", 2),
                     ("cal_days_since_first_trading_day_of_month", 0),
                     ("cal_days_until_next_trading_holiday", 7)]:
        if fld in d:
            flags += (d[fld].iloc[:, 0] <= thr).astype(float)
    w = 1.0 + EVENT_WEIGHT * (flags > 0).astype(float)
    return pd.Series(w.reindex(dates).values, index=index).fillna(1.0)


# ---------------------------------------------------------------- models --
def fit_ridge(X, y, w, lam: float) -> np.ndarray:
    """Closed-form weighted ridge. No sklearn dependency."""
    Xv = X.values.astype(np.float64)
    Xv = np.hstack([Xv, np.ones((len(Xv), 1))])          # intercept
    yv = y.values.astype(np.float64)
    sw = np.sqrt(w.values.astype(np.float64))[:, None]
    Xw, yw = Xv * sw, yv * sw[:, 0]
    p = Xw.shape[1]
    R = lam * np.eye(p)
    R[-1, -1] = 0.0                                       # don't shrink bias
    return np.linalg.solve(Xw.T @ Xw + R, Xw.T @ yw)


def fit_mlp(X, y, w, n_hidden=4, epochs=150, lr=0.05, seed=0):
    """Tiny MLP by plain gradient descent — 14 -> n_hidden -> 1, ReLU."""
    rng = np.random.default_rng(seed)
    Xv = X.values.astype(np.float32)
    yv = y.values.astype(np.float32)
    sw = w.values.astype(np.float32)
    sw = sw / sw.mean()
    n, p = Xv.shape

    A = rng.normal(0, np.sqrt(2.0 / p), (p, n_hidden)).astype(np.float32)
    b = np.zeros(n_hidden, dtype=np.float32)
    v = rng.normal(0, np.sqrt(2.0 / n_hidden), n_hidden).astype(np.float32)
    c = np.float32(0.0)

    batch = min(200_000, n)
    for ep in range(epochs):
        sel = rng.choice(n, batch, replace=False) if n > batch else slice(None)
        xb, yb, wb = Xv[sel], yv[sel], sw[sel]
        z = xb @ A + b
        h = np.maximum(0.0, z)
        pred = h @ v + c
        err = (pred - yb) * wb / len(yb)
        gv = h.T @ err
        gc = err.sum()
        dh = np.outer(err, v) * (z > 0)
        gA = xb.T @ dh
        gb = dh.sum(axis=0)
        A -= lr * gA; b -= lr * gb; v -= lr * gv; c -= lr * gc
    return A, b, v, c


def rank_ic(pred: pd.Series, y: pd.Series) -> float:
    """Mean daily rank correlation — the metric that matters here."""
    df = pd.DataFrame({"p": pred, "y": y})
    daily = df.groupby(level=0).apply(
        lambda g: g["p"].rank().corr(g["y"].rank()) if len(g) > 20 else np.nan)
    return float(daily.mean())


# ------------------------------------------------------------------ emit --
TERM_TS = {
    "c_ma20": "(close / at_zero2nan(ts_mean(close, 20)))",
    "c_ma60": "(close / at_zero2nan(ts_mean(close, 60)))",
    "hl": "(high / at_zero2nan(low))",
    "co": "((close - open) / at_zero2nan(open))",
    "gap": "((open - ts_delay(close,1)) / at_zero2nan(ts_delay(close,1)))",
    "stoch": "((close - ts_min(low,20)) / at_zero2nan(ts_max(high,20) - ts_min(low,20)))",
    "ret1": "ret1", "ret5": "ret5", "ret20": "ret20",
    "v_ma20": "(volume / at_zero2nan(ts_mean(volume, 20)))",
    "dv_adv": "((close*volume) / at_zero2nan(ts_mean(close*volume, 60)))",
    "vol20": "ts_std(ret1, 20)",
    "size": "(cs_rank(ts_mean(close*volume, 60)) - 1)",
    "beta60": "ts_corr_binary(ret1, ret1_spx, 60)",
    "amihud": "((at_signsqrt(ret1) * 0 + abs(ret1)) / at_zero2nan(close*volume) * 1e9)",
    "hi52": "(close / at_zero2nan(ts_max(close, 252)))",
    "idio": "ret1",
}


def emit_ridge(names, coef) -> str:
    lines = ["# --- features (rank-normalised, matching training) ----------"]
    for n in names:
        lines.append(f"f_{n} = cs_rank({TERM_TS.get(n, n)}) - 1.5")
    lines.append("")
    lines.append("# --- learned ridge weights ----------------------------------")
    terms = " \\\n      + ".join(f"{coef[i]:+.6f} * f_{n}" for i, n in enumerate(names))
    lines.append(f"score = ( {terms} )")
    lines.append("")
    lines.append("alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))")
    return "\n".join(lines)


def emit_mlp(names, A, b, v, c) -> str:
    lines = ["# --- features (rank-normalised, matching training) ----------"]
    for n in names:
        lines.append(f"f_{n} = cs_rank({TERM_TS.get(n, n)}) - 1.5")
    lines.append("")
    lines.append("# --- hidden layer (ReLU) ------------------------------------")
    for j in range(A.shape[1]):
        t = " + ".join(f"{A[i, j]:+.5f}*f_{n}" for i, n in enumerate(names))
        lines.append(f"h{j} = np.maximum(0.0, {t} {b[j]:+.5f})")
    lines.append("")
    lines.append("# --- output layer -------------------------------------------")
    out = " + ".join(f"{v[j]:+.5f}*h{j}" for j in range(len(v)))
    lines.append(f"score = {out} {c:+.5f}")
    lines.append("")
    lines.append("alpha = at_zero2nan(cs_winsor(score, 0.01, remove_extreme=False))")
    return "\n".join(lines)


# ------------------------------------------------------------------ main --
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ridge", "mlp", "both"], default="ridge")
    ap.add_argument("--hidden", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"loading  train={TRAIN}  valid={VALID}  test={TEST}")
    d_tr, d_va, d_te = (load_gp_data(window=w) for w in (TRAIN, VALID, TEST))
    T_tr, T_va, T_te = (build_terminals(x) for x in (d_tr, d_va, d_te))
    # intersect across all three windows so the feature set is identical
    names = sorted(set(T_tr) & set(T_va) & set(T_te))
    dropped = (set(T_tr) | set(T_va) | set(T_te)) - set(names)
    print(f"  {len(names)} features: {', '.join(names)}")
    if dropped:
        print(f"  dropped (not present in every window): {sorted(dropped)}")

    Xtr, ytr, wtr = build_xy(d_tr, T_tr, names)
    Xva, yva, _ = build_xy(d_va, T_va, names)
    Xte, yte, _ = build_xy(d_te, T_te, names)
    assert Xtr.shape[1] == Xva.shape[1] == Xte.shape[1] == len(names)
    wtr = event_weights(Xtr.index, d_tr)
    print(f"  train {len(Xtr):,} rows | valid {len(Xva):,} | test {len(Xte):,}")
    print(f"  event-day weight {EVENT_WEIGHT}x on "
          f"{100*(wtr > 1).mean():.0f}% of rows\n")

    if a.model in ("ridge", "both"):
        print("RIDGE — selecting penalty on VALID by rank IC")
        best = (None, -9e9, None)
        for lam in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
            co = fit_ridge(Xtr, ytr, wtr, lam)
            pv = pd.Series(Xva.values @ co[:-1] + co[-1], index=Xva.index)
            ic = rank_ic(pv, yva)
            print(f"   lambda {lam:>8.0e}   valid rank IC {ic:+.4f}")
            if ic > best[1]:
                best = (lam, ic, co)
        lam, ic_va, co = best
        pt = pd.Series(Xte.values @ co[:-1] + co[-1], index=Xte.index)
        ptr = pd.Series(Xtr.values @ co[:-1] + co[-1], index=Xtr.index)
        print(f"\n   chosen lambda {lam:.0e}")
        print(f"   rank IC   train {rank_ic(ptr, ytr):+.4f} | "
              f"valid {ic_va:+.4f} | TEST {rank_ic(pt, yte):+.4f}")
        print("\n   coefficients (rank-normalised features):")
        for n, c_ in sorted(zip(names, co[:-1]), key=lambda t: -abs(t[1])):
            print(f"      {n:10s} {c_:+.6f}")
        txt = emit_ridge(names, co)
        (OUT / "learned_ridge.txt").write_text(txt)
        print("\n" + "=" * 68 + "\nPASTE INTO TREXSIM\n" + "=" * 68)
        print(txt)

    if a.model in ("mlp", "both"):
        print(f"\n\nMLP — {len(names)} -> {a.hidden} -> 1")
        A, b, v, c = fit_mlp(Xtr, ytr, wtr, n_hidden=a.hidden)
        def pred(X):
            return pd.Series(np.maximum(0, X.values @ A + b) @ v + c, index=X.index)
        print(f"   rank IC   train {rank_ic(pred(Xtr), ytr):+.4f} | "
              f"valid {rank_ic(pred(Xva), yva):+.4f} | "
              f"TEST {rank_ic(pred(Xte), yte):+.4f}")
        txt = emit_mlp(names, A, b, v, c)
        (OUT / "learned_mlp.txt").write_text(txt)
        print("\n" + "=" * 68 + "\nPASTE INTO TREXSIM\n" + "=" * 68)
        print(txt)

    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
