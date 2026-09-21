#!/usr/bin/env python3
"""Coordinate-descent tuner — automates the by-hand window search.

    python3 tq_tune.py --validate      # FIRST: check the tuner against alphas
                                       # you already tuned by hand on Trexsim
    python3 tq_tune.py                 # then tune rank 25

READ THIS BEFORE TRUSTING THE OUTPUT
------------------------------------
The local panel has NOT predicted Trexsim's absolute IR — 28 locally-qualifying
GP alphas all failed the platform's 0.07 gate. So this tuner is not used to
decide whether an alpha is good.

It is used for a strictly easier task: RANKING PARAMETER VARIANTS OF ONE FIXED
EXPRESSION. Comparing `ts_mean(x, 10)` against `ts_mean(x, 20)` inside the same
formula is a far weaker claim than comparing two different alphas, and much more
likely to transfer.

`--validate` tests exactly that claim. It reruns the tuning you did by hand on
the winning alpha and checks whether local search reaches the same conclusions
(tighter gate better, no accumulation better, skew window longer better). If it
agrees, use its output as a shortlist for the platform. If it disagrees, the
local panel cannot rank variants either and you should keep tuning by hand.

Either way the output is a paste-ready Trexsim expression, never a decision.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tq_gp_ops as O  # noqa: E402
from tq_gp import fitness, load_gp_data, report  # noqa: E402

TRAIN_FULL = ("2006-01-01", "2022-12-31")   # match the platform's window


# --------------------------------------------------------------- helpers --
def _z(x):
    return x.replace(0, np.nan)


def pieces(d, p):
    """Shared sub-expressions, parameterised."""
    c, o, h, l, v = d["close"], d["open"], d["high"], d["low"], d["volume"]
    ws = p["w_stoch"]
    stoch = (c - O.ts_min(l, ws)) / _z(O.ts_max(h, ws) - O.ts_min(l, ws))
    dv = c * v
    size = O.cs_rank(O.ts_mean(dv, p["w_size"]))          # already [0,1]
    volrank = O.cs_rank(O.ts_std(d["ret1"], p["w_volstd"]))
    co = (c - o) / _z(o)
    return stoch, size, volrank, co


def build_rank25(d, p):
    """Rank-25 expression with every window exposed as a parameter."""
    stoch, size, volrank, co = pieces(d, p)

    C = ((-stoch / _z(O.ts_mean(volrank, p["w_vrmean"])))
         / _z(O.ts_mean(O.ts_mean(stoch + size, p["w_in1"]), p["w_in2"])))
    A = O.ts_zscore(C, p["w_zs"]) - size

    E = O.ts_min(O.ts_mean(O.ts_mean(O.ts_mean(size, p["w_s1"]), p["w_s2"]),
                           p["w_s3"]), p["w_min"])
    D = stoch + ((O.ts_skew(O.ts_median(co, p["w_med"]), p["w_skew"])
                  + (size + d["ret5"])) + E)
    B = O.cs_rank(D)

    return O.ts_mean(A, p["w_out"]) - B


def gated(core, d, p):
    """Winsorise, gate, and mask to the universe — mirrors the platform code."""
    lo = core.quantile(p["winsor"], axis=1)
    hi = core.quantile(1.0 - p["winsor"], axis=1)
    a = core.clip(lo, hi, axis=0)
    g = d["gate_days"]
    mask = (g >= p["gate_lo"]) & (g <= p["gate_hi"])
    return a.where(mask)


#: knob -> candidate values. Order matters only for readability.
GRID_25 = {
    "gate_hi":  [6, 9, 12, 16, 20, 26],
    "gate_lo":  [0, 1, 2],
    "w_stoch":  [10, 20, 40],
    "w_size":   [30, 60, 120],
    "w_volstd": [10, 20, 40],
    "w_vrmean": [10, 20, 40],
    "w_in1":    [5, 10, 20],
    "w_in2":    [10, 20, 40],
    "w_zs":     [5, 10, 20],
    "w_out":    [5, 10, 20],
    "w_med":    [10, 20, 40],
    "w_skew":   [30, 60, 90],
    "w_min":    [10, 20, 40],
    "w_s1":     [5, 10, 20],
    "w_s2":     [10, 20],
    "w_s3":     [10, 20],
    "winsor":   [0.005, 0.01, 0.02],
}

BASE_25 = {"gate_hi": 12, "gate_lo": 0, "w_stoch": 20, "w_size": 60,
           "w_volstd": 20, "w_vrmean": 20, "w_in1": 10, "w_in2": 20,
           "w_zs": 10, "w_out": 10, "w_med": 20, "w_skew": 60, "w_min": 20,
           "w_s1": 10, "w_s2": 20, "w_s3": 20, "winsor": 0.01}


def score(d, p, builder):
    try:
        a = gated(builder(d, p), d, p)
        return fitness(a, d)
    except Exception:
        return -1.0


def coordinate_descent(d, base, grid, builder, passes=2, verbose=True):
    """Tune one knob at a time, repeat. Exactly the by-hand procedure."""
    cur = dict(base)
    best = score(d, cur, builder)
    if verbose:
        print(f"  start fitness {best:.4f}")
    n_eval = 1
    for it in range(passes):
        improved = False
        for k, vals in grid.items():
            k_best, k_val = best, cur[k]
            for val in vals:
                if val == cur[k]:
                    continue
                trial = dict(cur, **{k: val})
                s = score(d, trial, builder)
                n_eval += 1
                if s > k_best:
                    k_best, k_val = s, val
            if k_val != cur[k]:
                if verbose:
                    print(f"    pass {it+1}  {k:9s} {cur[k]:>6} -> {k_val:<6} "
                          f"fitness {best:.4f} -> {k_best:.4f}")
                cur[k], best, improved = k_val, k_best, True
        if not improved:
            if verbose:
                print(f"  converged after pass {it+1}")
            break
    return cur, best, n_eval


# -------------------------------------------------------------- validate --
def validate(d):
    """Does local search reproduce conclusions you verified on Trexsim?

    Your hand-tuning of the winning alpha established three facts. If the local
    panel can rank variants at all, it should agree with all three.
    """
    stoch = ((d["close"] - O.ts_min(d["low"], 20))
             / _z(O.ts_max(d["high"], 20) - O.ts_min(d["low"], 20)))
    relvol = d["volume"] / _z(O.ts_mean(d["volume"], 20))
    volret = d["ret1"].rolling(20).std() / _z(d["ret20"])
    lo, hi = volret.quantile(0.02, axis=1), volret.quantile(0.98, axis=1)
    volret = volret.clip(lo, hi, axis=0)

    def core_win(w_skew, w_mean):
        return (-(O.ts_zscore(stoch, 10))
                - (((O.ts_median(O.cs_zscore(relvol), 20)
                     - O.ts_mean(d["ret20"], 20))
                    + O.ts_skew(relvol - O.ts_delay(relvol, 3), w_skew))
                   - O.ts_mean(-stoch - O.ts_median(volret, 3), w_mean)))

    def run(core, glo, ghi, acc=None):
        x = O.ts_sum(core, acc) if acc else core
        p = {"winsor": 0.01, "gate_lo": glo, "gate_hi": ghi}
        return fitness(gated(x, d, p), d)

    base = core_win(33, 15)
    checks = [
        ("gate <=12 beats <=90",
         run(base, 0, 12) > run(base, 0, 90)),
        ("gate >=0 beats >=3",
         run(base, 0, 18) > run(base, 3, 18)),
        ("no accumulation beats ts_sum(5)",
         run(base, 0, 12) > run(base, 0, 12, acc=5)),
        ("ts_sum(5) beats ts_sum(20)",
         run(base, 0, 12, acc=5) > run(base, 0, 12, acc=20)),
        ("skew window 33 beats 20",
         run(core_win(33, 15), 0, 12) > run(core_win(20, 15), 0, 12)),
    ]
    print("\nVALIDATION — does local search agree with your Trexsim results?\n")
    ok = 0
    for name, passed in checks:
        print(f"   {'AGREES ' if passed else 'DISAGREES'}  {name}")
        ok += bool(passed)
    print(f"\n   {ok}/{len(checks)} agree.")
    if ok >= 4:
        print("   -> local ranking of variants looks trustworthy. Use the")
        print("      tuner's top configs as a platform shortlist.")
    else:
        print("   -> local panel cannot rank variants reliably either.")
        print("      Do NOT trust the tuner; keep tuning on Trexsim by hand.")
    return ok


# ------------------------------------------------------------------ main --
TREXSIM_25 = """gate = (trading_days_until_next_earnings_announcement >= {gate_lo}) * (trading_days_until_next_earnings_announcement <= {gate_hi})

stoch = (close - ts_min(low, {w_stoch})) / at_zero2nan(ts_max(high, {w_stoch}) - ts_min(low, {w_stoch}))
size  = cs_rank(ts_mean(close * volume, {w_size})) - 1
vrank = cs_rank(ts_std(ret1, {w_volstd})) - 1
co    = (close - open) / at_zero2nan(open)

C = (-(stoch) / at_zero2nan(ts_mean(vrank, {w_vrmean}))) / at_zero2nan(ts_mean(ts_mean(stoch + size, {w_in1}), {w_in2}))
A = ts_zscore(C, {w_zs}) - size
E = ts_min(ts_mean(ts_mean(ts_mean(size, {w_s1}), {w_s2}), {w_s3}), {w_min})
D = stoch + ((ts_skew(ts_median(co, {w_med}), {w_skew}) + (size + ret5)) + E)

core  = ts_mean(A, {w_out}) - (cs_rank(D) - 1)
alpha = at_zero2nan(cs_winsor(core, {winsor}, remove_extreme=False) * gate)"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()

    print(f"loading panel over {TRAIN_FULL} (matching the platform window)…")
    d = load_gp_data(window=TRAIN_FULL)
    if "trading_days_until_next_earnings" in d:
        d["gate_days"] = d["trading_days_until_next_earnings"]
    else:
        print("\n!! The local panel has NO earnings-date variable — it was")
        print("   never downloaded. The gate cannot be evaluated locally, so")
        print("   gate_lo / gate_hi will be ignored and only the internal")
        print("   windows can be tuned here. Tune the gate on Trexsim.\n")
        d["gate_days"] = pd.DataFrame(0.0, index=d["close"].index,
                                      columns=d["close"].columns)

    if a.validate:
        validate(d)
        return

    t0 = time.perf_counter()
    best_p, best_s, n = coordinate_descent(d, BASE_25, GRID_25, build_rank25,
                                           passes=a.passes)
    print(f"\n{n} evaluations in {time.perf_counter()-t0:.0f}s")
    print(f"best local fitness {best_s:.4f}\n")
    changed = {k: (BASE_25[k], v) for k, v in best_p.items() if v != BASE_25[k]}
    print("changed from base:")
    for k, (o_, n_) in changed.items():
        print(f"   {k:9s} {o_} -> {n_}")
    print("\n" + "=" * 70)
    print("PASTE INTO TREXSIM")
    print("=" * 70)
    print(TREXSIM_25.format(**best_p))


if __name__ == "__main__":
    main()
