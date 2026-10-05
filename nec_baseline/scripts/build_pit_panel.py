"""Build the point-in-time CRSP panel from the extract (brief 06 A.5).

Reads the extract written by ``scripts/extract_crsp.py`` (nothing is
downloaded), builds the feature panel over every PERMNO that was an S&P 500
member at some point in the window, keeps each row only on dates its PERMNO
was a member (re-ranking the snapshot features among the members, audit
finding D-1), and writes, all inside ``Data/derived/``:

- ``pit_panel_crsp_<start>_<end>.pt``: the built panel;
- ``pit_coverage_crsp_<start>_<end>.csv``: per year, the members with and
  without usable rows (the number to publish next to any result);
- ``pit_panel_crsp_<start>_<end>.report.json``: the build's aggregate
  statistics (counts, ranges, rates; no per-security values).

LICENCE: all three are derived from CRSP and stay inside ``Data/``, which is
gitignored. Committed reports may quote their aggregate numbers only.

Usage:  python3.14 scripts/build_pit_panel.py [start] [end]
        (defaults 2015-01-01 .. 2024-12-31; run from nec_baseline/)
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nec_moe import CRSPSpec, StageBSpec, build_crsp_panel, load_extract  # noqa: E402

#: the panel's feature/target settings: the 5-day horizon (Q21) and the
#: default market-neutral target (Q25, brief 08 A)
STAGE_B = StageBSpec(seq_len=20, horizon=5)


def main(*window: str) -> None:
    t0 = time.time()
    spec = CRSPSpec()
    extract = load_extract(spec)  # the default window's extract covers sub-windows
    if window:
        spec = dataclasses.replace(spec, start=window[0], end=window[1])
    build = build_crsp_panel(spec, STAGE_B, extract=extract)

    stem = f"crsp_{spec.start}_{spec.end}"
    out = spec.derived_dir
    out.mkdir(parents=True, exist_ok=True)
    coverage_path = out / f"pit_coverage_{stem}.csv"
    build.coverage.to_csv(coverage_path)
    report_path = out / f"pit_panel_{stem}.report.json"
    report_path.write_text(json.dumps(build.report, indent=2, default=str))
    torch.save(build.panel, spec.panel_path)
    print(f"[pit] coverage by year:\n{build.coverage}")
    print(f"[pit] saved {spec.panel_path.name} "
          f"({spec.panel_path.stat().st_size / 1e6:.0f} MB), {coverage_path.name} and "
          f"{report_path.name} to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:3])
