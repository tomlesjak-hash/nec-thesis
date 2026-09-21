#!/usr/bin/env python3
"""Build the Trexquant training panel. Run this on YOUR machine.

    cd "~/Desktop/Quant Model/Trexquant/pipeline"
    pip install yfinance pandas pyarrow lxml
    python run_download.py

Reuses nec_baseline's 592 cached tickers (migrated, not re-downloaded) and
does NOT modify anything under nec_baseline/.

Output: ../panel/*.parquet  — wide dates x tickers, float32.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tq_calendar import calendar_features  # noqa: E402
from tq_data import (  # noqa: E402
    CONTEXT_TICKERS, EARLIEST_RELIABLE, add_derived, add_marketcap, add_sectors,
    build_panel, download_batch, download_sectors, download_shares,
    load_sp500_universe, migrate_nec_cache, save_panel, top_n_mask,
)

ROOT = Path(__file__).resolve().parents[1]          # .../Trexquant
QUANT = ROOT.parent                                  # .../Quant Model
NEC_CACHE = QUANT / "nec_baseline" / "data_cache"
CACHE = ROOT / "cache"
PANEL = ROOT / "panel"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2006-01-01",
                    help="competition window start (no burn-in buffer: features "
                         "with a long lookback are NaN early in 2006)")
    ap.add_argument("--end", default="2022-12-31",
                    help="competition window end")
    ap.add_argument("--top-n", type=int, default=500,
                    help="universe size per date (see 05_UNIVERSE_DECISION.md)")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-shares", action="store_true",
                    help="skip the slow per-ticker shares download")
    ap.add_argument("--no-sectors", action="store_true",
                    help="skip the sector/industry download")
    ap.add_argument("--no-membership", action="store_true",
                    help="rank by ADV across all downloaded names instead of "
                         "intersecting with point-in-time index membership. "
                         "Hits the target count but ADDS survivorship bias.")
    ap.add_argument("--max-tickers", type=int, default=0,
                    help="SMOKE TEST: cap the ticker count (e.g. 20). "
                         "Run this first to verify the pipeline end-to-end "
                         "before committing to the full download.")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print(f"\nTrexquant panel build  {args.start} -> {args.end}\n" + "=" * 52)

    # 1. reuse what's already on disk -------------------------------------
    if NEC_CACHE.exists():
        n = migrate_nec_cache(NEC_CACHE, CACHE)
        print(f"[1] migrated {n} tickers from nec_baseline cache (not re-downloaded)")
    else:
        print(f"[1] no nec_baseline cache at {NEC_CACHE}")

    # 2. point-in-time universe -------------------------------------------
    print("[2] fetching S&P 500 point-in-time membership…")
    uni = load_sp500_universe(CACHE)
    tickers = sorted(uni.members_union(args.start, args.end))
    print(f"    {len(uni.current)} current members, {len(uni.changes)} changes, "
          f"{len(tickers)} unique tickers over the window")
    if pd.Timestamp(args.start) < EARLIEST_RELIABLE:
        yrs = (EARLIEST_RELIABLE - pd.Timestamp(args.start)).days / 365.25
        print(f"    ! {yrs:.0f} years of the window precede "
              f"{EARLIEST_RELIABLE.date()}. Wikipedia's change table is too "
              "sparse there (~40 events for the whole 2000s vs ~22/yr after),")
        print("      so pre-2011 membership degrades toward "
              "'today's index backfilled' = survivorship bias. Weight your "
              "training toward 2011+ and treat pre-2011 results with caution.")
    if args.max_tickers:
        tickers = tickers[: args.max_tickers]
        print(f"    SMOKE TEST: capped to {len(tickers)} tickers")

    # 3. download ----------------------------------------------------------
    print(f"[3] downloading {len(tickers)} equities + {len(CONTEXT_TICKERS)} context…")
    ok, fail = download_batch(list(tickers) + list(CONTEXT_TICKERS),
                              args.start, args.end, CACHE, refresh=args.refresh)
    print(f"    {len(ok)} ok, {len(fail)} failed")
    if fail:
        print("    failures (delisted names Yahoo no longer serves — the "
              "residual survivorship component):")
        for k, v in list(fail.items())[:10]:
            print(f"      {k}: {v[:60]}")
        if len(fail) > 10:
            print(f"      … and {len(fail) - 10} more")

    # 4. panel -------------------------------------------------------------
    print("[4] building wide panel…")
    eq = [t for t in ok if t not in CONTEXT_TICKERS]
    panel = build_panel(eq, CACHE, args.start, args.end)
    if not panel:
        sys.exit("no data — check that yfinance is installed and reachable")
    panel = add_derived(panel)
    dates = panel["close"].index
    print(f"    {panel['close'].shape[0]} dates x {panel['close'].shape[1]} tickers")

    # 4b. shares outstanding -> market cap (downloaded, not derived) -------
    if not args.no_shares:
        print("[4b] downloading shares outstanding history…")
        shares = download_shares(eq, args.start, args.end, CACHE,
                                 refresh=args.refresh)
        panel = add_marketcap(panel, shares)
    else:
        print("[4b] skipping shares download (--no-shares)")

    # 4c. sector / industry — needed for industry-neutral local fitness ----
    if not args.no_sectors:
        print("[4c] downloading sector/industry classification…")
        panel = add_sectors(panel, download_sectors(
            eq, CACHE, wiki=uni.sectors, refresh=args.refresh))
    else:
        print("[4c] skipping sector download (--no-sectors)")

    # 5. context series (1D) ----------------------------------------------
    ctx = build_panel(list(CONTEXT_TICKERS), CACHE, args.start, args.end)
    if "close" in ctx:
        cc = ctx["close"].reindex(dates)
        rename = {"^GSPC": "spx", "^VIX": "vix", "GLD": "gld"}
        for raw, short in rename.items():
            if raw in cc.columns:
                panel[f"close_{short}"] = cc[[raw]].rename(columns={raw: short}).astype("float32")
                panel[f"ret1_{short}"] = (
                    cc[[raw]].pct_change().rename(columns={raw: short}).astype("float32")
                )
        print(f"    context: {[c for c in cc.columns]}")

    # 6. calendar / FOMC (1D, gates only) ----------------------------------
    print("[5] computing calendar + FOMC features…")
    cal = calendar_features(dates)
    for c in cal.columns:
        panel[f"cal_{c}"] = cal[[c]].astype("float32")
    print(f"    {len(cal.columns)} seasonality variables")

    # 7. universe mask -----------------------------------------------------
    print(f"[6] building top-{args.top_n}-by-ADV mask (monthly rebalance)…")
    if args.no_membership:
        memb = None
        print("    --no-membership: ranking by ADV across ALL downloaded names,")
        print("    not just index members. Fills to the target count, but the")
        print("    substitutes are survivors -> MORE survivorship bias, not less.")
    else:
        memb = uni.mask(dates, list(panel["close"].columns))
    panel["universe"] = top_n_mask(
        panel["adv60"], args.top_n, memb, price=panel["close"]
    ).astype("float32")
    per_day = panel["universe"].sum(axis=1)
    post = per_day[per_day.index >= "2006-04-01"]      # skip adv60 warm-up
    print(f"    median names/day: {post.median():.0f}  "
          f"(min {post.min():.0f}, max {post.max():.0f})")
    if post.median() < 0.9 * args.top_n:
        short = 100 * (1 - post.median() / args.top_n)
        print(f"    ! {short:.0f}% short of {args.top_n}. Cause is Yahoo not "
              "serving delisted tickers, not a bug. See 09_UNIVERSE_COVERAGE.md")

    # 8. save --------------------------------------------------------------
    print("[7] saving…")
    save_panel(panel, PANEL)
    mb = sum(p.stat().st_size for p in PANEL.glob("*.parquet")) / 1e6
    print(f"\ndone in {time.perf_counter() - t0:.0f}s — {mb:.0f} MB in {PANEL}\n")


if __name__ == "__main__":
    main()
