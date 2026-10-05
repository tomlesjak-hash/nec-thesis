"""Build the point-in-time CRSP panel from the extracts (briefs 06 and 08).

Reads the CRSP v2 extract (``scripts/extract_crsp_v2.py``) and, for the Q26
features, the Compustat extract (``scripts/extract_compustat.py``); nothing
is downloaded. Builds the panel over every PERMNO that was an S&P 500 member
at some point in the window, with each row only on dates its PERMNO was a
member (ranks, sector aggregates and the market-neutral target taken over
that date's members), and writes, all inside ``Data/derived/``:

- ``pit_panel_crsp_<start>_<end>_<feature_set>_<target_kind>.pt``: the
  panel (the brief 06 file, ``pit_panel_crsp_<start>_<end>.pt``, is left
  alone);
- ``pit_coverage_crsp_<start>_<end>_<feature_set>_<target_kind>.csv``: per
  year, the members with and without usable rows;
- ``pit_panel_crsp_<start>_<end>_<feature_set>_<target_kind>.report.json``:
  the build's aggregate statistics (counts, ranges, rates; no per-security
  values).

LICENCE: all three are derived from CRSP and Compustat and stay inside
``Data/``, which is gitignored. Committed reports may quote their aggregate
numbers only.

Usage:  python3.14 scripts/build_pit_panel.py [start] [end]
        (defaults 2000-01-01 .. 2024-12-31, the sample of Q16 (a); run from
        nec_baseline/)
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nec_moe import (  # noqa: E402
    CompustatSpec,
    CRSPSpec,
    StageBSpec,
    build_crsp_panel,
    load_compustat_extract,
    load_extract,
)

#: the panel's feature/target settings: the 5-day horizon (Q21) and the
#: default market-neutral target (Q25, brief 08 A)
STAGE_B = StageBSpec(seq_len=20, horizon=5)


def main(*window: str) -> None:
    t0 = time.time()
    spec = CRSPSpec()
    extract = load_extract(spec)  # the default window's extract covers sub-windows
    if window:
        spec = dataclasses.replace(spec, start=window[0], end=window[1])
    cspec = CompustatSpec()  # like the CRSP one, the default window's extract
    compustat = load_compustat_extract(cspec) if STAGE_B.feature_set == "q26" else None
    build = build_crsp_panel(spec, STAGE_B, extract=extract, compustat=compustat,
                             compustat_spec=cspec)

    stem = f"crsp_{spec.start}_{spec.end}_{STAGE_B.feature_set}_{STAGE_B.target_kind}"
    out = spec.derived_dir
    out.mkdir(parents=True, exist_ok=True)
    coverage_path = out / f"pit_coverage_{stem}.csv"
    build.coverage.to_csv(coverage_path)
    report_path = out / f"pit_panel_{stem}.report.json"
    report_path.write_text(json.dumps(build.report, indent=2, default=str))
    panel_path = spec.panel_path_for(STAGE_B)
    torch.save(build.panel, panel_path)
    print(f"[pit] coverage by year:\n{build.coverage}")
    print(f"[pit] saved {panel_path.name} "
          f"({panel_path.stat().st_size / 1e6:.0f} MB), {coverage_path.name} and "
          f"{report_path.name} to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:3])
