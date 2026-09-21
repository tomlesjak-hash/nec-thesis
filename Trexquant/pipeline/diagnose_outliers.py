#!/usr/bin/env python3
"""Locate the extreme values flagged by validate_panel.py and decide what to drop.

    cd "~/Desktop/Quant Model/Trexquant/pipeline"
    python3 diagnose_outliers.py

Answers three questions:
  1. Which tickers produce the |ret1| > 90% bars, and are they real events or
     adjustment artifacts?
  2. Which tickers have implausible adjusted price levels?
  3. How often is adv60 zero (a divide-by-zero waiting to happen in the GP)?

Prints a ready-to-paste EXCLUDE list at the end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path(__file__).resolve().parents[1] / "panel"


def main() -> None:
    r = pd.read_parquet(PANEL / "ret1.parquet")
    c = pd.read_parquet(PANEL / "close.parquet")
    adv = pd.read_parquet(PANEL / "adv60.parquet")
    uni = pd.read_parquet(PANEL / "universe.parquet").astype(bool)

    print("=" * 74)
    print("1. EXTREME 1-DAY RETURNS  (|ret1| > 90%)")
    print("=" * 74)
    bad = (r.abs() > 0.9)
    per_tk = bad.sum()
    per_tk = per_tk[per_tk > 0].sort_values(ascending=False)
    print(f"{int(bad.values.sum())} bars across {len(per_tk)} tickers\n")
    print(per_tk.to_string())

    print("\n   worst individual bars — with price context:")
    rows = []
    for tk in per_tk.index:
        for dt in r.index[bad[tk].values]:
            i = r.index.get_loc(dt)
            prev = c[tk].iloc[i - 1] if i else np.nan
            rows.append({"ticker": tk, "date": dt.date(),
                         "ret1": r[tk].iloc[i],
                         "close_prev": prev, "close": c[tk].iloc[i],
                         "in_universe": bool(uni[tk].iloc[i])})
    ext = pd.DataFrame(rows).reindex(
        pd.DataFrame(rows)["ret1"].abs().sort_values(ascending=False).index
    )
    print(ext.head(15).to_string(index=False))

    print("\n" + "=" * 74)
    print("2. IMPLAUSIBLE ADJUSTED PRICE LEVELS")
    print("=" * 74)
    lo, hi = c.min(), c.max()
    span = (hi / lo.replace(0, np.nan))
    susp = pd.DataFrame({"min": lo, "max": hi, "span_x": span})
    susp = susp[(susp["min"] < 0.05) | (susp["max"] > 50000)
                | (susp["span_x"] > 1000)]
    if susp.empty:
        print("none")
    else:
        print(susp.sort_values("span_x", ascending=False).to_string())
        print("\n   A span above ~1000x over 17 years is almost always a bad")
        print("   split factor rather than genuine price action.")

    print("\n" + "=" * 74)
    print("3. ZERO / NaN adv60 INSIDE THE UNIVERSE  (GP divide-by-zero risk)")
    print("=" * 74)
    z = uni & ((adv == 0) | adv.isna())
    print(f"{int(z.values.sum())} cells "
          f"({100*z.values.sum()/max(uni.values.sum(),1):.4f}% of in-universe)")
    zt = z.sum()
    zt = zt[zt > 0].sort_values(ascending=False)
    if len(zt):
        print(zt.head(10).to_string())
    print("\n   Guard every denominator in the GP:")
    print("     rel = dv / adv60.replace(0, np.nan)")

    # ---------------------------------------------------------- exclusions --
    excl = set(susp.index)
    for tk, n in per_tk.items():
        if n >= 2 or tk in excl:      # repeat offenders are artifacts
            excl.add(tk)
    print("\n" + "=" * 74)
    print("SUGGESTED EXCLUSION LIST")
    print("=" * 74)
    if not excl:
        print("nothing to exclude")
    else:
        print(f"{len(excl)} tickers, "
              f"{100*len(excl)/c.shape[1]:.1f}% of the panel:\n")
        print("EXCLUDE = " + repr(sorted(excl)))
        still_in = {t for t in excl if uni[t].any()}
        print(f"\n{len(still_in)} of them appear in the universe at some point, "
              "so they would reach the GP:")
        print("  " + ", ".join(sorted(still_in)) if still_in else "  none")
        print("\nDrop them before training:")
        print("  panel = {k: v.drop(columns=EXCLUDE, errors='ignore') "
              "for k, v in panel.items()}")


if __name__ == "__main__":
    main()
