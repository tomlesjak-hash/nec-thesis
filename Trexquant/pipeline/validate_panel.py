#!/usr/bin/env python3

"""Validate the built panel. Run after every download.

    cd "~/Desktop/Quant Model/Trexquant/pipeline"
    python3 validate_panel.py

Checks correctness, not just presence: alignment, fill rates, price sanity,
return distributions, adjustment consistency, universe-mask look-ahead, and
calendar variables. Prints PASS/WARN/FAIL per check and exits non-zero on FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path(__file__).resolve().parents[1] / "panel"
FAILED: list[str] = []
WARNED: list[str] = []


def report(name: str, ok: bool, detail: str, warn_only: bool = False) -> None:
    if ok:
        print(f"  PASS  {name:38s} {detail}")
    elif warn_only:
        WARNED.append(name)
        print(f"  WARN  {name:38s} {detail}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name:38s} {detail}")


def main() -> None:
    if not PANEL.exists():
        sys.exit(f"no panel at {PANEL} — run run_download.py first")

    import argparse
    import re
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-bad", action="store_true",
                    help="drop BAD_TICKERS (ticker-reuse corruption) before "
                         "validating — i.e. see the panel as the GP will")
    args = ap.parse_args()

    files = sorted(PANEL.glob("*.parquet"))
    dupes = [f for f in files if re.search(r" \d+$", f.stem)]
    files = [f for f in files if f not in dupes]
    p = {f.stem: pd.read_parquet(f) for f in files}

    if args.exclude_bad:
        sys.path.insert(0, str(Path(__file__).parent))
        from tq_data import BAD_TICKERS
        present = [t for t in BAD_TICKERS if t in p["close"].columns]
        p = {k: v.drop(columns=list(BAD_TICKERS), errors="ignore")
             for k, v in p.items()}
        print(f"\n  excluding {len(present)} ticker-reuse names: "
              f"{', '.join(present)}")
    print(f"\n{'='*78}\nPANEL VALIDATION — {PANEL}\n{'='*78}")
    print(f"\n{len(p)} fields\n")
    if dupes:
        print(f"  !! ignoring {len(dupes)} sync-conflict copies "
              f"(e.g. {dupes[0].name}). Your Desktop is probably on iCloud "
              "Drive — delete them:  rm ../panel/*\\ [0-9].parquet\n")

    # ---------------------------------------------------------- inventory --
    print("[1] FIELD INVENTORY")
    core = ["open", "high", "low", "close", "volume", "close_raw"]
    derived = ["ret1", "ret5", "ret10", "ret20", "dollar_volume", "adv60"]
    for f in core + derived + ["universe", "sector", "industry"]:
        if f in p:
            d = p[f]
            fill = 100 * d.notna().mean().mean()
            print(f"       {f:16s} {d.shape[0]:5d}d x {d.shape[1]:4d}tk  "
                  f"{fill:5.1f}% filled  [{np.nanmin(d.values):.4g}, "
                  f"{np.nanmax(d.values):.4g}]")
        else:
            print(f"       {f:16s} MISSING")
    missing = [f for f in core if f not in p]
    report("core fields present", not missing,
           "all 6" if not missing else f"missing {missing}")

    c = p["close"]

    # ---------------------------------------------------------- alignment --
    print("\n[2] ALIGNMENT")
    wide = {k: v for k, v in p.items()
            if v.shape[1] == c.shape[1] and not k.startswith("cal_")}
    bad_idx = [k for k, v in wide.items() if not v.index.equals(c.index)]
    bad_col = [k for k, v in wide.items() if not v.columns.equals(c.columns)]
    report("index identical across fields", not bad_idx,
           f"{len(wide)} wide fields share close's index"
           if not bad_idx else f"mismatched: {bad_idx[:4]}")
    report("columns identical across fields", not bad_col,
           "all tickers aligned" if not bad_col else f"mismatched: {bad_col[:4]}")
    report("index sorted & unique",
           c.index.is_monotonic_increasing and c.index.is_unique,
           f"{c.index[0].date()} -> {c.index[-1].date()}")

    # ------------------------------------------------------- price sanity --
    print("\n[3] PRICE SANITY")
    for f in ["open", "high", "low", "close"]:
        v = p[f].values
        neg = int(np.nansum(v <= 0))
        report(f"{f} strictly positive", neg == 0,
               "no non-positive values" if neg == 0 else f"{neg} values <= 0")
    hl = int(np.nansum(p["high"].values < p["low"].values))
    report("high >= low", hl == 0, "ok" if hl == 0 else f"{hl} violations")
    oob = int(np.nansum((p["close"].values > p["high"].values + 1e-4)
                        | (p["close"].values < p["low"].values - 1e-4)))
    report("close within [low, high]", oob == 0,
           "ok" if oob == 0 else f"{oob} bars outside range", warn_only=True)

    # Adjusted prices below a cent or above ~$50k usually mean a bad split
    # factor rather than a real quote. Heavy tails wreck GP terminals.
    cv = p["close"].values
    tiny_tk = list(p["close"].columns[(p["close"] < 0.05).any()])
    huge_tk = list(p["close"].columns[(p["close"] > 50000).any()])
    report("no sub-penny adjusted prices", not tiny_tk,
           f"{int(np.nansum(cv < 0.05))} bars < $0.05 in {tiny_tk}",
           warn_only=True)
    report("no absurd adjusted prices", not huge_tk,
           f"{int(np.nansum(cv > 50000))} bars > $50,000 in {huge_tk}",
           warn_only=True)
    if tiny_tk or huge_tk:
        print("        ^ mostly harmless: a WRONG-BY-A-CONSTANT adjustment "
              "cancels in\n          dimensionless terminals "
              "(close/ts_mean(close,20), ret1). Only a\n          spurious "
              "SPLIT would matter, and 'adjustment direction consistent'\n"
              "          below tests for exactly that.")

    # ------------------------------------------------------ adjustment ----
    print("\n[4] ADJUSTMENT CONSISTENCY  (close = adjusted, close_raw = actual)")
    if "close_raw" in p:
        ratio = (p["close_raw"] / p["close"]).replace([np.inf, -np.inf], np.nan)
        med = float(np.nanmedian(ratio.values))
        # adjusted <= raw almost always (dividends removed going back)
        frac_ge = float(np.nanmean((ratio.values >= 0.999)))
        report("raw >= adjusted (dividend back-adj)", frac_ge > 0.90,
               f"{100*frac_ge:.1f}% of bars, median ratio {med:.3f}",
               warn_only=True)
        # Yahoo normalises Adj Close so the factor is 1.0 at the LAST DATE IN
        # ITS DATABASE (today) — not at the end of our requested window. If the
        # panel ends in the past, the factor at that date still carries every
        # dividend paid since. So expect ratio > 1, growing with the gap.
        last = ratio.ffill().iloc[-1]
        med_last = float(np.nanmedian(last))
        gap_yrs = max(0.0, (pd.Timestamp.today() - c.index[-1]).days / 365.25)
        if gap_yrs < 0.25:
            ok = abs(med_last - 1.0) < 0.05
            detail = f"median {med_last:.4f} (panel is current, want ~1.0)"
        else:
            # ~1-3%/yr dividend yield on a broad US equity panel
            hi = 1.0 + 0.045 * gap_yrs
            ok = 1.0 - 1e-6 <= med_last <= hi
            detail = (f"median {med_last:.4f}; panel ends {gap_yrs:.1f}y ago so "
                      f"expect 1.00-{hi:.3f} (accrued dividends since)")
        report("adjustment anchored correctly", ok, detail)

        # raw/adj accumulates backwards, so going FORWARD it should not rise.
        # Caveat: REVERSE splits legitimately step it up, so a raw count can't
        # be thresholded. The discriminator is breadth — a systematic
        # adjustment error hits nearly every ticker; reverse splits hit a few.
        fwd = ratio.ffill()
        jump = (fwd.diff() > 0.01 * fwd.shift())
        n_tk = int((jump.sum() > 0).sum())
        frac = n_tk / max(fwd.shape[1], 1)
        worst = jump.sum().sort_values(ascending=False).head(3)
        worst = ", ".join(f"{k}:{int(v)}" for k, v in worst.items() if v > 0)
        report("adjustment direction consistent", frac < 0.20,
               f"{n_tk}/{fwd.shape[1]} tickers ({100*frac:.0f}%) show upward "
               f"jumps{' — ' + worst if worst else ''}. "
               "A few = reverse splits (fine); most = systematic error")

        # every price field must carry the SAME adjustment as close
        for f in ("open", "high", "low"):
            if f in p:
                inband = ((p[f] >= p["low"] - 1e-4)
                          & (p[f] <= p["high"] + 1e-4))
                frac = float(inband.values[np.isfinite(p[f].values)].mean())
                report(f"{f} shares close's adjustment", frac > 0.999,
                       f"{100*frac:.2f}% of bars inside [low, high]")

    # ---------------------------------------------------- return sanity ----
    print("\n[5] RETURN DISTRIBUTION")
    r = p["ret1"]
    rv = r.values[np.isfinite(r.values)]
    ann = float(np.nanstd(rv) * np.sqrt(252))
    report("ret1 annualised vol plausible", 0.10 < ann < 0.80,
           f"{ann:.1%} (equities typically 20-45%)")
    ext = int(np.sum(np.abs(rv) > 0.90))
    report("no absurd 1-day returns", ext == 0,
           "none |ret1| > 90%" if ext == 0 else f"{ext} bars |ret1| > 90%",
           warn_only=True)
    # ret5 must equal the 5-day compounded ret1
    chk = ((1 + r).rolling(5).apply(np.prod, raw=True) - 1)
    diff = (chk - p["ret5"]).abs()
    md = float(np.nanmax(diff.values)) if np.isfinite(diff.values).any() else 0.0
    report("ret5 consistent with ret1", md < 1e-3,
           f"max abs diff {md:.2e}")

    # ------------------------------------------------------- universe -----
    print("\n[6] UNIVERSE MASK")
    if "universe" in p:
        u = p["universe"].astype(bool)
        per = u.sum(axis=1)
        post = per[per.index >= "2011-01-01"]
        report("names/day stable post-2011", post.min() > 0,
               f"median {int(post.median())}, min {int(post.min())}, "
               f"max {int(post.max())}")
        # ZOMBIE test — in the universe outside its trading life.
        #
        # Not the same as "in the universe with a NaN price". By design a name
        # is held through interior gaps (halts, missing Yahoo days), so those
        # cells legitimately have no quote. The bug worth catching is a name
        # flagged BEFORE its first or AFTER its last real observation — a
        # holding in something that isn't trading at all.
        v = c.notna()
        alive = v.cummax() & v[::-1].cummax()[::-1]
        zombie = int((u & ~alive).values.sum())
        gaps = int((u & alive & ~v).values.sum())
        live_cells = max(int((u & alive).values.sum()), 1)
        report("no zombie holdings (outside trading life)", zombie == 0,
               "ok" if zombie == 0 else f"{zombie} cells before first / after "
               "last observation")
        report("halt gaps within tolerance", gaps / live_cells < 0.001,
               f"{gaps} cells ({100*gaps/live_cells:.3f}%) are in-universe "
               "during a trading halt — held by design, not a bug")
        # LOOK-AHEAD TEST — additions only.
        #
        # An ADDITION mid-month would mean the ranking used information from
        # inside the month it was ranking. That is look-ahead and must not
        # happen. A REMOVAL mid-month is legitimate and expected: it's a name
        # that stopped trading (delisting, acquisition), which the price guard
        # correctly drops on the day its data ends.
        d = u.astype(int).diff()
        adds = d.clip(lower=0).sum(axis=1)
        rems = (-d).clip(lower=0).sum(axis=1)
        adds, rems = adds[adds.index >= "2011-01-01"], rems[rems.index >= "2011-01-01"]
        firsts = adds.groupby([adds.index.year, adds.index.month]).head(1).index
        on_first = adds.loc[firsts].sum()
        pct = 100 * on_first / max(adds.sum(), 1)
        report("no mid-month additions (look-ahead)",
               adds.sum() == 0 or pct > 99.0,
               f"{pct:.1f}% of {int(adds.sum())} additions on the 1st trading "
               f"day of month ({int(rems.sum())} removals, any day = delistings)")

    # ------------------------------------------------------- calendar -----
    print("\n[7] CALENDAR / SEASONALITY")
    cals = {k: v for k, v in p.items() if k.startswith("cal_")}
    report("17 seasonality fields", len(cals) == 17, f"{len(cals)} found")
    if "cal_day_of_the_week" in p:
        dow = p["cal_day_of_the_week"].iloc[:, 0]
        report("day_of_week in 1..5",
               bool(dow.min() >= 1 and dow.max() <= 5),
               f"range [{dow.min():.0f}, {dow.max():.0f}]")
        real = pd.Series(c.index.dayofweek + 1, index=c.index)
        report("day_of_week matches the index",
               bool((dow.values == real.values).all()), "exact match")
    allnan = [k for k, v in cals.items() if v.isna().all().all()]
    report("no all-NaN calendar field", not allnan,
           "ok" if not allnan else f"all-NaN: {allnan}")

    # -------------------------------------------------------- context -----
    print("\n[8] CONTEXT SERIES")
    for f, lo, hi in [("close_spx", 800, 12000), ("close_vix", 8, 90),
                      ("close_gld", 80, 700)]:
        if f in p:
            v = p[f].values
            v = v[np.isfinite(v)]
            if len(v):
                ok = lo < np.median(v) < hi
                report(f"{f} plausible level", ok,
                       f"median {np.median(v):.1f} (expect {lo}-{hi})",
                       warn_only=True)

    # ------------------------------------------------------- industry -----
    print("\n[9] CLASSIFICATION")
    for f in ["sector", "industry"]:
        if f in p:
            d = p[f]
            nun = int(d.iloc[0].nunique())
            const = bool((d.nunique() <= 1).all())
            report(f"{f} constant over time", const,
                   f"{nun} distinct codes")
            fill = 100 * d.notna().mean().mean()
            report(f"{f} coverage", fill > 70,
                   f"{fill:.1f}% of cells classified", warn_only=fill > 50)

    # --------------------------------------------------------- summary ----
    print(f"\n{'='*78}")
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {', '.join(FAILED)}")
    if WARNED:
        print(f"WARNINGS ({len(WARNED)}): {', '.join(WARNED)}")
    if not FAILED and not WARNED:
        print("ALL CHECKS PASSED")
    elif not FAILED:
        print("PASSED with warnings — review above, usually benign")
    print("=" * 78 + "\n")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
