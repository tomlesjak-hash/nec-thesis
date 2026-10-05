"""Brief 08 E: the trial registry's data-layer fields, the feature coverage
report (aggregate only, refusing per-security values) and the real-data
scripts. Invented fixtures; the real-data check skips without ``Data/``.
"""

from __future__ import annotations

import dataclasses
import json
import py_compile
import warnings
from pathlib import Path

import cfz_fixture as cfz
import crsp_fixture as fx
import pytest
from conftest import small_config

from nec_moe import (
    PROVENANCE_KEYS,
    StageBSpec,
    TrialRegistry,
    data_config_from_panel,
    nec_arm,
    run_sweep,
)
from nec_moe import compustat as cs
from nec_moe.characteristics import Q26_CHARACTERISTICS
from nec_moe.compustat import CompustatSpec
from nec_moe.coverage import (
    PerSecurityValues,
    assert_aggregate_only,
    feature_coverage,
    write_feature_coverage,
)
from nec_moe.crsp import CRSPSpec, build_crsp_panel, crsp_file_source, extract_crsp, load_extract
from nec_moe.features import FeatureSpec
from nec_moe.train import DeadParameterWarning

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SMALL = FeatureSpec(beta_corr_window=120, beta_corr_min_obs=80, seas_first_year=1,
                    seas_last_year=1)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("cov") / "Data"
    fx.write_fixture(root)
    cfz.write_fixture(root)
    crsp_spec = CRSPSpec(crsp_dir=str(root), release=fx.RELEASE, start=fx.START, end=fx.END,
                         chunk_bytes=20_000, member_count_min=5, member_count_max=8)
    extract_crsp(crsp_spec, verbose=False)
    cspec = CompustatSpec(crsp_dir=str(root), release=cfz.RELEASE, start=fx.START,
                          end=fx.END)
    cs.extract_compustat(cspec, [*range(10001, 10011)], verbose=False)
    cex = cs.load_compustat_extract(cspec)
    stage_b = StageBSpec(seq_len=10, horizon=5, min_names_per_date=4, features=SMALL)
    build = build_crsp_panel(crsp_spec, stage_b, extract=load_extract(crsp_spec),
                             compustat=cex, compustat_spec=cspec, verbose=False)
    return build.panel, cex


def test_every_trial_row_records_the_data_layer(built, tmp_path: Path):
    panel, _ = built
    for key in ("target_kind", "feature_set", "gate_weight", "crsp_stock_file",
                "compustat_release", "sector_source"):
        assert key in PROVENANCE_KEYS
    cfg = dataclasses.replace(small_config(), data=data_config_from_panel(panel))
    reg = TrialRegistry(tmp_path / "t.jsonl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        run_sweep(panel, [nec_arm("soft", cfg)], seeds=(0,), registry=reg, tag="e1",
                  steps=3, n_folds=2, test_dates_per_fold=10, purge_dates=5, verbose=False)
    (row,) = reg.trials("e1")
    c = row.config
    assert (c["target_kind"], c["feature_set"], c["crsp_stock_file"], c["compustat_release"],
            c["sector_source"]) == ("market_neutral", "q26", "StkDlySecurityData",
                                    cfz.RELEASE, "gics")
    assert c["gate_weight"] is None  # a soft gate has no Hamilton gate weight


def test_coverage_report_is_aggregate_and_complete(built, tmp_path: Path):
    panel, cex = built
    report = feature_coverage(panel.metadata, cex.info)
    assert set(report["present_share"]) == set(Q26_CHARACTERISTICS)
    assert all(0.0 <= v <= 1.0 for v in report["present_share"].values())
    assert set(report["by_year"]) == {"year_2020"}
    year = report["by_year"]["year_2020"]
    assert set(year) == {"rows", "present_share", "flag_price_rate", "flag_fund_rate",
                         "sector_missing_share"}
    assert year["sector_missing_share"] > 0  # 10008 has no GICS row
    # the keyset-8 quarters from the extract's history start on (2016-07-01
    # for this window): the fixture's 2017 Q4 only
    since = cs.fundamentals_start(CompustatSpec(start=fx.START, end=fx.END))
    expected = sum(cfz.period_end(fy, q, cfz.FYR[gv]) >= since for gv, fy, q in cfz.KEYSET8)
    assert expected == 1
    assert report["compustat"]["restated_keyset_firm_quarters"] == expected
    # nothing that looks like a return or a model result
    flat = json.dumps(report)
    for word in ("ret_mean", "ic", "sharpe", "nll", "y_hat"):
        assert f'"{word}"' not in flat
    forbidden = set(panel.entity_labels or ()) | set(cex.links["KYGVKEY"])
    paths = write_feature_coverage(report, tmp_path / "out", forbidden)
    assert [p.name for p in paths] == ["coverage.json", "present_share_by_year.csv",
                                       "rates_by_year.csv"]
    for p in paths:
        text = p.read_text()
        assert not any(f"{e}" in text.split('"') for e in forbidden)


def test_the_report_refuses_per_security_values(built, tmp_path: Path):
    panel, cex = built
    report = feature_coverage(panel.metadata, cex.info)
    forbidden = set(panel.entity_labels or ())
    out = tmp_path / "refused"
    leaky = dict(report, extra={"10003": 0.5})  # a PERMNO as a key
    with pytest.raises(PerSecurityValues, match="10003"):
        write_feature_coverage(leaky, out, forbidden)
    assert not out.exists()  # nothing written
    with pytest.raises(PerSecurityValues, match="permno"):
        assert_aggregate_only({"rows": [{"permno": 1, "x": 0.1}]})
    with pytest.raises(PerSecurityValues, match="not a summary"):
        assert_aggregate_only({"series": list(range(500))})
    with pytest.raises(PerSecurityValues, match="001003"):
        assert_aggregate_only({"note": "001003"}, {"001003"})
    assert_aggregate_only(report, forbidden)  # the real report passes
    # a GVKEY written without leading zeros can equal a year: the year keys
    # are spelled "year_2020", so the guard does not trip on them
    write_feature_coverage(report, tmp_path / "ok", forbidden | {"2020", "1", "8"})


def test_member_firm_quarters_and_their_report_date_coverage(tmp_path: Path):
    """Brief 09 A.3: a firm-quarter counts when its GVKEY is linked, on the
    period end, to a PERMNO that is an index member that day (both bounds
    inclusive); the RDQ and filing-date shares are taken by fiscal year, and
    the early-years table puts them beside the calendar-year flag rates.
    Invented identifiers and dates."""
    import pandas as pd

    from nec_moe.coverage import (
        early_years_table,
        rdq_coverage_by_fiscal_year,
        universe_firm_quarters,
    )

    t, nat = pd.Timestamp, pd.NaT
    quarters = pd.DataFrame({
        "KYGVKEY": ["900001", "900001", "900001", "900002", "900002", "900003"],
        "FYYYYQ": [20004, 20011, 20012, 20004, 20011, 20004],
        "FYEARQ": [2000, 2001, 2001, 2000, 2001, 2000],
        "datadate": [t("2000-12-31"), t("2001-03-31"), t("2001-06-30"), t("2000-12-31"),
                     t("2001-03-31"), nat],
        "rdq": [t("2001-01-25"), nat, t("2001-07-20"), t("2001-01-30"), t("2001-04-28"),
                t("2001-01-20")],
        "first_filing": [t("2001-02-10"), t("2001-05-01"), nat, t("2001-02-12"),
                         t("2001-05-10"), nat],
    })
    links = pd.DataFrame({
        "KYGVKEY": ["900001", "900002", "900002"],
        "LINKDT": [t("1990-01-01"), t("1990-01-01"), t("2001-01-01")],
        "LINKENDDT": [t("2200-12-31"), t("2000-12-31"), t("2200-12-31")],
        "LPERMNO": [90011, 90022, 90023],
        "LINKTYPE": ["LC", "LU", "NR"],  # 900002's later link is filtered out
        "LINKPRIM": ["P", "P", "P"],
    })
    spells = pd.DataFrame({
        "PERMNO": [90011, 90022],
        "MbrStartDt": [t("2000-06-01"), t("1995-01-01")],
        "MbrEndDt": [t("2001-04-30"), t("2030-01-01")],
    })
    members = universe_firm_quarters(quarters, links, spells, CompustatSpec())
    # 900001: Q4 2000 and Q1 2001 while a member, not Q2 2001 (spell ended);
    # 900002: Q4 2000 (link ends on the period end, inclusive), not Q1 2001
    # (only an NR link then); 900003 has no period end and no link
    assert sorted(zip(members["KYGVKEY"], members["FYYYYQ"], strict=True)) == [
        ("900001", 20004), ("900001", 20011), ("900002", 20004)]
    rdq = rdq_coverage_by_fiscal_year(members)
    assert rdq == {
        "fiscal_2000": {"firm_quarters": 2, "rdq_share": 1.0, "filing_share": 1.0},
        "fiscal_2001": {"firm_quarters": 1, "rdq_share": 0.0, "filing_share": 1.0},
    }
    report = {"by_year": {"year_2000": {"rows": 10, "flag_price_rate": 0.1,
                                        "flag_fund_rate": 0.2,
                                        "sector_missing_share": 0.3}}}
    table = early_years_table(report, rdq, 2000, 2001)
    assert list(table.index) == ["year_2000", "year_2001"]
    assert table.loc["year_2000", "rows"] == 10 and table.loc["year_2000", "rdq_share"] == 1.0
    assert table.loc["year_2001", "member_firm_quarters"] == 1
    assert pd.isna(table.loc["year_2001", "rows"])  # no panel rows that year
    with pytest.raises(ValueError, match="empty"):
        early_years_table(report, rdq, 2001, 2000)
    full = dict(report, rdq_coverage_by_fiscal_year=rdq, present_share={})
    full["by_year"] = {"year_2000": dict(report["by_year"]["year_2000"], present_share={})}
    paths = write_feature_coverage(full, tmp_path / "early", {"900001", "90011"},
                                   tables={"early_years": table})
    assert [p.name for p in paths][-1] == "early_years.csv"


def test_the_real_data_scripts_compile():
    for name in ("extract_crsp_v2.py", "extract_compustat.py", "feature_coverage_report.py",
                 "build_pit_panel.py"):
        py_compile.compile(str(SCRIPTS / name), doraise=True)


def _real() -> bool:
    try:
        crsp_file_source(CRSPSpec(), "StkIndMembership")
        crsp_file_source(CompustatSpec(), "linkhistory")
    except FileNotFoundError:
        return False
    return (CRSPSpec().extract_dir / "extract.json").exists() and (
        CompustatSpec().extract_dir / "extract.json").exists()


@pytest.mark.crsp_data
@pytest.mark.skipif(not _real(), reason="CRSP v2 and Compustat extracts not under Data/")
def test_real_extracts_load_and_report_aggregates_only():
    ex = load_extract(CRSPSpec())
    cex = cs.load_compustat_extract(CompustatSpec())
    rep = cex.info["report"]
    print(f"\n[compustat] {rep['gvkeys']} GVKEYs; link ambiguities {rep['link_ambiguities']}; "
          f"keyset-8 firm-quarters {rep['restated_keyset_firm_quarters']}")
    assert "DlyOpen" in ex.stock.columns and rep["gvkeys"] > 400
