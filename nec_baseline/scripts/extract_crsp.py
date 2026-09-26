"""Extract the CRSP rows the panel needs into ``Data/derived/`` (brief 06 A.5).

Streams the daily stock file and the cumulative adjustment-factor file once
each, from the extracted ``Data/crspdata/<release>_ascii/`` copy when it exists
and otherwise from ``Data/<release>_ascii.zip`` (never unzipped whole). Keeps
the PERMNOs that were S&P 500 members (``CRSPSpec.membership_indno``) at any
point in the window, from ``start - extract_lookback_days`` to ``end +
extract_lead_days``, and writes parquet parts plus the membership, market,
delisting and security-info tables to
``Data/derived/crsp_extract_<release>_<start>_<end>/``.

Resumable: every block's byte offset is recorded, and a rerun with the same
settings continues where the last one stopped.

LICENCE: everything this writes is derived from CRSP and stays inside
``Data/``, which is gitignored. Never copy it elsewhere.

Usage:  python3.14 scripts/extract_crsp.py [start] [end]
        (defaults 2015-01-01 .. 2024-12-31; run from nec_baseline/; needs the
        ``crsp`` extra, i.e. pyarrow)
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nec_moe import CRSPSpec, extract_crsp  # noqa: E402


def main(*window: str) -> None:
    spec = CRSPSpec()
    if window:
        spec = dataclasses.replace(spec, start=window[0], end=window[1])
    t0 = time.time()
    root = extract_crsp(spec)
    print(f"[extract] done in {time.time() - t0:.0f}s: {root}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
