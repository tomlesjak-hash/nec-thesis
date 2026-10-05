"""The brief 08 Compustat extract: CCM link, GICS, quarterly fundamentals.

Reads the CRSP/Compustat Merged release (``cfz202607``) from
``Data/crspdata/cfz202607_ascii/`` or ``Data/cfz202607_ascii.zip`` (the
standard locations; differently named copies such as ``Data/crspdata 3/`` are
not used). Keeps the GVKEYs linked (LINKTYPE LC/LU, LINKPRIM P/C) to the
PERMNOs of the CRSP v2 extract, so run ``scripts/extract_crsp_v2.py`` first.
Streams the quarterly files once, resumably, from the derived history start
(about mid-2011) and writes
``Data/derived/compustat_cfz202607_<start>_<end>/`` with an ``extract.json``
that carries the aggregate report: link ambiguities, the period-end mapping
check, the keyset-8 ("PRE") firm-quarter count, the filing types.

LICENCE: derived from Compustat and CRSP; stays inside ``Data/``. It prints
counts only.

Usage (from nec_baseline/):  python3.14 scripts/extract_compustat.py [start end]
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nec_moe import CompustatSpec, CRSPSpec, extract_compustat, load_extract  # noqa: E402


def main(*window: str) -> None:
    crsp = CRSPSpec()
    if window:
        crsp = dataclasses.replace(crsp, start=window[0], end=window[1])
    spec = CompustatSpec(start=crsp.start, end=crsp.end)
    extract = load_extract(crsp)  # refuses an extract built with other settings
    permnos = sorted(int(p) for p in extract.stock["PERMNO"].unique())
    print(f"[compustat] {len(permnos)} PERMNOs from {crsp.extract_dir.name}")
    print(f"[compustat] writing {spec.extract_dir}")
    t0 = time.time()
    root = extract_compustat(spec, permnos)
    rep = json.loads((root / "extract.json").read_text())["report"]
    print(f"[compustat] done in {time.time() - t0:.0f}s: {rep['gvkeys']} GVKEYs, history from "
          f"{rep['history_start']}")
    print(f"[compustat] link ambiguities (PERMNO-periods with two GVKEYs): "
          f"{rep['link_ambiguities']}; GICS rows whose lpermno disagrees with the link: "
          f"{rep['gics_lpermno_disagreements']}")
    mapping = dict(rep["period_end_mapping"])
    window = mapping.pop("since_history_start", None)
    print(f"[compustat] period ends, all history: {mapping}")
    print(f"[compustat] period ends, from {rep['history_start']} (what the features use): "
          f"{window}")
    print(f"[compustat] keyset-{rep['restated_keyset']} ('PRE') universe firm-quarters: "
          f"{rep['restated_keyset_firm_quarters']}")
    print(f"[compustat] filing types: {rep['filing_srctypes']}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
