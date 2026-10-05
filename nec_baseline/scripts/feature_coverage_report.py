"""The Q26 feature coverage report (brief 08 E.3): aggregate only.

For the default window, prints and writes to
``results/feature_coverage/<date>/``:

- the share of universe rows per year with each characteristic present
  before the 0-fill (``present_share_by_year.csv``);
- the missing-value flag rates and the sector-missing share per year
  (``rates_by_year.csv``);
- the keyset-8 ("PRE") firm-quarter count and the data-layer checks
  (``coverage.json``).

It reads the Q26 panel built by ``scripts/build_pit_panel.py`` when that file
exists, else builds the panel in memory (the same build, not saved). It
prints no model result and no return statistic, and it refuses to write
anything that names a security (``nec_moe.coverage.assert_aggregate_only``):
the PERMNOs of the panel and the GVKEYs of the extract are checked against
every key and text value.

Needs the CRSP v2 extract and the Compustat extract
(``scripts/extract_crsp_v2.py``, ``scripts/extract_compustat.py``).

Usage (from nec_baseline/):  python3.14 scripts/feature_coverage_report.py
"""

from __future__ import annotations

import datetime as dt
import sys
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
from nec_moe.coverage import feature_coverage, write_feature_coverage  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "feature_coverage"


def main() -> None:
    crsp = CRSPSpec()
    stage_b = StageBSpec(seq_len=20, horizon=5)
    cspec = CompustatSpec(start=crsp.start, end=crsp.end)
    compustat = load_compustat_extract(cspec)
    panel_path = crsp.panel_path_for(stage_b)
    if panel_path.exists():
        print(f"[coverage] reading {panel_path.name}")
        panel = torch.load(panel_path, weights_only=False)
    else:
        print("[coverage] no saved Q26 panel: building it in memory (not saved)")
        panel = build_crsp_panel(crsp, stage_b, extract=load_extract(crsp),
                                 compustat=compustat, compustat_spec=cspec).panel
    report = feature_coverage(panel.metadata, compustat.info)
    forbidden = set(panel.entity_labels or ()) | set(compustat.links["KYGVKEY"].astype(str))
    out = OUT / dt.date.today().isoformat()
    paths = write_feature_coverage(report, out, forbidden)
    print(f"[coverage] {report['rows']:,} rows; flag_price_missing "
          f"{report['flag_price_rate']:.3f}, flag_fund_missing "
          f"{report['flag_fund_rate']:.3f}, no sector {report['sector_missing_share']:.3f}")
    print("[coverage] characteristic present before the fill (all years):")
    for name, share in report["present_share"].items():
        print(f"    {name:<20s} {share:6.3f}")
    c = report["compustat"]
    print(f"[coverage] keyset-{c['restated_keyset']} universe firm-quarters: "
          f"{c['restated_keyset_firm_quarters']}")
    print(f"[coverage] wrote {', '.join(p.name for p in paths)} to {out}")


if __name__ == "__main__":
    main()
