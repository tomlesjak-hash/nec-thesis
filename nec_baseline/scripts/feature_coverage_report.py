"""The Q26 feature coverage report (brief 08 E.3): aggregate only.

For the default window, prints and writes to
``results/feature_coverage/<date>/``:

- the share of universe rows per year with each characteristic present
  before the 0-fill (``present_share_by_year.csv``);
- the missing-value flag rates and the sector-missing share per year
  (``rates_by_year.csv``);
- the keyset-8 ("PRE") firm-quarter count and the data-layer checks
  (``coverage.json``);
- per fiscal year, the S&P 500 members' firm-quarters and the share with a
  report date (RDQ) and with a 10-Q/10-K filing date
  (``coverage.json``, ``rdq_coverage_by_fiscal_year``; brief 09 A.3). A
  member firm-quarter is one whose GVKEY is linked, on the period end, to a
  PERMNO that is a member that day;
- the early years in one table (``early_years.csv``, default 2000-2009):
  rows, flag rates and the sector-missing share by calendar year beside the
  members' RDQ and filing-date coverage by fiscal year.

It reads the Q26 panel built by ``scripts/build_pit_panel.py`` when that file
exists, else builds the panel in memory (the same build, not saved). It
prints no model result and no return statistic, and it refuses to write
anything that names a security (``nec_moe.coverage.assert_aggregate_only``):
the PERMNOs of the panel and the GVKEYs of the extract are checked against
every key and text value.

Needs the CRSP v2 extract and the Compustat extract
(``scripts/extract_crsp_v2.py``, ``scripts/extract_compustat.py``).

Usage (from nec_baseline/):
    python3.14 scripts/feature_coverage_report.py [--early FIRST LAST]
"""

from __future__ import annotations

import argparse
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
from nec_moe.compustat import quarterly_fundamentals  # noqa: E402
from nec_moe.coverage import (  # noqa: E402
    early_years_table,
    feature_coverage,
    rdq_coverage_by_fiscal_year,
    universe_firm_quarters,
    write_feature_coverage,
)
from nec_moe.crsp import read_membership  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "feature_coverage"
#: the early-years table's default range (brief 09 A.3: 2000-2009)
EARLY_YEARS = (2000, 2009)


def main(early: tuple[int, int] = EARLY_YEARS) -> None:
    crsp = CRSPSpec()
    stage_b = StageBSpec(seq_len=20, horizon=5)
    cspec = CompustatSpec(start=crsp.start, end=crsp.end)
    compustat = load_compustat_extract(cspec)
    extract = load_extract(crsp)
    panel_path = crsp.panel_path_for(stage_b)
    if panel_path.exists():
        print(f"[coverage] reading {panel_path.name}")
        panel = torch.load(panel_path, weights_only=False)
    else:
        print("[coverage] no saved Q26 panel: building it in memory (not saved)")
        panel = build_crsp_panel(crsp, stage_b, extract=extract,
                                 compustat=compustat, compustat_spec=cspec).panel
    report = feature_coverage(panel.metadata, compustat.info)
    # the members' report-date coverage by fiscal year (brief 09 A.3)
    calendar = extract.market.index
    quarters, _ = quarterly_fundamentals(compustat, cspec, calendar)
    members = universe_firm_quarters(quarters, compustat.links, read_membership(crsp), cspec)
    rdq = rdq_coverage_by_fiscal_year(members)
    report["rdq_coverage_by_fiscal_year"] = rdq
    early_frame = early_years_table(report, rdq, *early)
    report["early_years"] = {"first": early[0], "last": early[1]}
    forbidden = set(panel.entity_labels or ()) | set(compustat.links["KYGVKEY"].astype(str))
    out = OUT / dt.date.today().isoformat()
    paths = write_feature_coverage(report, out, forbidden, tables={"early_years": early_frame})
    print(f"[coverage] {report['rows']:,} rows; flag_price_missing "
          f"{report['flag_price_rate']:.3f}, flag_fund_missing "
          f"{report['flag_fund_rate']:.3f}, no sector {report['sector_missing_share']:.3f}")
    print("[coverage] characteristic present before the fill (all years):")
    for name, share in report["present_share"].items():
        print(f"    {name:<20s} {share:6.3f}")
    window = (report.get("quarters") or {}).get("since_history_start")
    if window:
        print(f"[coverage] firm-quarters from the history start: {window['firm_quarters']:,}; "
              f"with a 10Q/10K filing date {window['with_first_filing']:,}; "
              f"availability rules {window['anchor_rules']}")
    c = report["compustat"]
    print(f"[coverage] keyset-{c['restated_keyset']} universe firm-quarters: "
          f"{c['restated_keyset_firm_quarters']}")
    print(f"[coverage] early years {early[0]}-{early[1]} (rates by calendar year; "
          "RDQ and filing coverage of member firm-quarters by fiscal year):")
    print(early_frame.to_string())
    print(f"[coverage] wrote {', '.join(p.name for p in paths)} to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--early", nargs=2, type=int, default=list(EARLY_YEARS),
                    metavar=("FIRST", "LAST"), help="the early-years table's range")
    main(tuple(ap.parse_args().early))  # type: ignore[arg-type]
