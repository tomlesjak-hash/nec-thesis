"""The brief 08 CRSP re-extract: ``StkDlySecurityData``, with prices and quotes.

Streams the full daily security file (about 42 GB uncompressed, from
``Data/crspdata/ciz202512_ascii/`` or ``Data/ciz202512_ascii.zip``, never
unzipped whole) and the cumulative adjustment factors, once each, resumably,
for the PERMNOs that were S&P 500 members in the window plus the other share
classes of their companies (for market equity). The lookback derives from the
longest feature window (about five years before ``start``); the training
window is unchanged. Writes a **new** folder,
``Data/derived/crsp_extract_ciz202512_<start>_<end>_StkDlySecurityData_lb<days>d/``:
the brief 06 extract is never overwritten. ``extract.json`` records the stock
file, the lookback and, for the unit check, the ``MetaItemInfo`` rows of
``DlyCap``, ``DlyShrOut`` and ``DlyVol``.

Interrupt it any time (Ctrl+C); rerun the same command to continue.

LICENCE: everything this writes is derived from CRSP and stays inside
``Data/``, which is gitignored. It prints counts only.

Usage (from nec_baseline/):  python3.14 scripts/extract_crsp_v2.py [start end]
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nec_moe import CRSPSpec, extract_crsp  # noqa: E402


def main(*window: str) -> None:
    spec = CRSPSpec()
    if window:
        spec = dataclasses.replace(spec, start=window[0], end=window[1])
    lo, hi = spec.extract_bounds
    print(f"[extract-v2] {spec.stock_file} of {spec.release}, window {spec.start}..{spec.end}, "
          f"rows {lo.date()}..{hi.date()} (lookback {spec.lookback_days} calendar days)")
    print(f"[extract-v2] writing {spec.extract_dir}")
    t0 = time.time()
    root = extract_crsp(spec)
    info = json.loads((root / "extract.json").read_text())
    print(f"[extract-v2] done in {time.time() - t0:.0f}s: {info['n_permnos']} member PERMNOs, "
          f"{info['n_company_permnos']} other share classes of their companies")
    for sub, stream in info["streams"].items():
        print(f"[extract-v2]   {sub}: {stream['rows_kept']:,} of {stream['rows_read']:,} rows kept")
    units = info.get("item_metadata")
    print("[extract-v2] MetaItemInfo rows for the unit check are in extract.json "
          f"(item_metadata): {'present' if units else 'MetaItemInfo.dat not found'}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
