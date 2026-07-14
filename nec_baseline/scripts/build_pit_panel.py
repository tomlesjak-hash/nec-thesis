"""Build the point-in-time Stage-B panel (handbook III.3, item 2).

The full pipeline of handbook II.4's point-in-time path, as a resumable script:

1. load the S&P 500 point-in-time universe (Wikipedia, cached);
2. take the union of members over the window — the candidate list to *attempt*;
3. download every candidate's daily bars (yfinance → the shared cache; a
   present cache file short-circuits the network, so **rerunning this script
   resumes** where it left off and only retries past failures);
4. build the feature panel, filter it point-in-time, and write:
   - ``results/pit_coverage_<window>.csv`` — the per-year coverage table to
     publish next to any result from this panel (the honest residual-bias
     number);
   - ``data_cache/pit_panel_<window>.pt`` — the built panel, for instant
     reloads (derived artifact; the cache CSVs remain the source of truth).

Failures are expected and *are the point being measured*: departed names often
have no data at a free source — that is survivorship component 2, quantified
by the coverage column instead of silently ignored.

Usage:  python3.14 scripts/build_pit_panel.py [start] [end]
        (defaults 2015-01-01 .. 2024-12-31; run from nec_baseline/)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nec_moe import (  # noqa: E402
    StageBSpec,
    build_panel,
    filter_point_in_time,
    load_ohlcv,
    load_sp500_universe,
    universe_coverage_report,
)

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
RESULTS = Path(__file__).resolve().parents[1] / "results"
MARKET = "spy"


def main(start: str = "2015-01-01", end: str = "2024-12-31") -> None:
    t0 = time.time()
    tag = f"{start[:4]}_{end[:4]}"

    u = load_sp500_universe(CACHE)
    candidates = sorted(u.members_union(start, end))
    print(f"[pit] {len(candidates)} candidate tickers were index members "
          f"at some point in {start}..{end}")

    market = load_ohlcv(MARKET, start, end, CACHE, source="yfinance")
    prices, failures = {}, {}
    for i, t in enumerate(candidates, 1):
        try:
            prices[t] = load_ohlcv(t, start, end, CACHE, source="yfinance")
        except Exception as exc:  # noqa: BLE001 — failure IS the measurement
            failures[t] = str(exc)[:80]
        if i % 25 == 0 or i == len(candidates):
            print(f"[pit] {i}/{len(candidates)} attempted "
                  f"({len(prices)} loaded, {len(failures)} failed, "
                  f"{time.time() - t0:.0f}s)")

    print(f"[pit] download done: {len(prices)} loaded, {len(failures)} failed")
    if failures:
        print("[pit] failed (survivorship component 2 — departed/renamed names):")
        for k in sorted(failures):
            print(f"        {k}: {failures[k]}")

    spec = StageBSpec(seq_len=20, horizon=5)
    panel = build_panel(prices, market, spec)
    print(f"[pit] raw panel: {len(panel)} rows, {len(panel.date_labels)} dates, "
          f"{len(panel.entity_labels)} tickers")
    panel = filter_point_in_time(panel, u)
    print(f"[pit] point-in-time panel: {len(panel)} rows")

    years = range(int(start[:4]) + 1, int(end[:4]) + 1)
    coverage = universe_coverage_report(
        u, [f"{y}-01-05" for y in years], available=set(prices)
    )
    RESULTS.mkdir(exist_ok=True)
    cov_path = RESULTS / f"pit_coverage_{tag}.csv"
    coverage.to_csv(cov_path)
    print(f"[pit] coverage (publish this next to any result):\n{coverage}")

    panel_path = CACHE / f"pit_panel_{tag}.pt"
    torch.save(panel, panel_path)
    print(f"[pit] saved {cov_path} and {panel_path} "
          f"({panel_path.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:3])
