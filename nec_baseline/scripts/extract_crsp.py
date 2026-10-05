"""Extract the CRSP rows the panel needs into ``Data/derived/`` (brief 06 A.5).

Since brief 08 the default ``CRSPSpec`` streams ``StkDlySecurityData`` with a
lookback derived from the feature windows, so this does what
``scripts/extract_crsp_v2.py`` does (that script is the documented command).
The brief 06 extract (primary file, 550 days) keeps its own folder.

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
        python3.14 scripts/extract_crsp.py --return-duration-flags
        (defaults: CRSPSpec's window, 2000-01-01 .. 2024-12-31 since brief 09;
        run from nec_baseline/; needs the
        ``crsp`` extra, i.e. pyarrow)

``--return-duration-flags`` adds ``DlyRetDurFlg`` to an existing extract, for
the missing-return diagnostic only. That flag exists only in the 42 GB
``StkDlySecurityData``, which is then streamed once (resumable) and filtered to
four columns of the extract's PERMNOs and dates.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nec_moe import CRSPSpec, extract_crsp, extract_return_duration_flags  # noqa: E402


def main(*args: str) -> None:
    durations = "--return-duration-flags" in args
    window = [a for a in args if not a.startswith("--")]
    spec = CRSPSpec()
    if window:
        spec = dataclasses.replace(spec, start=window[0], end=window[1])
    t0 = time.time()
    root = extract_return_duration_flags(spec) if durations else extract_crsp(spec)
    print(f"[extract] done in {time.time() - t0:.0f}s: {root}")


if __name__ == "__main__":
    main(*sys.argv[1:])
