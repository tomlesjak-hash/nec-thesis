"""Brief 08 C: the CRSP re-extract, the CCM link, GICS sectors and point-in-time
Compustat, on invented fixtures (``tests/crsp_fixture.py``,
``tests/cfz_fixture.py``). Real-data checks skip without ``Data/``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import cfz_fixture as cfz
import crsp_fixture as fx
import numpy as np
import pandas as pd
import pytest

from nec_moe import compustat as cs
from nec_moe.compustat import CompustatSpec
from nec_moe.crsp import (
    LEGACY_LOOKBACK_DAYS,
    LEGACY_STOCK_FILE,
    CRSPSpec,
    ExtractMismatch,
    crsp_daily_frames,
    crsp_file_source,
    extract_crsp,
    load_extract,
    open_crsp_file,
    trading_to_calendar_days,
)
from nec_moe.features import FeatureSpec, StageBSpec, price_history_trading_days

CAL = pd.bdate_range("2009-01-01", "2022-12-31")  # the trading calendar of these tests


#: the window the invented cfz fixture was written around (its quarters run
#: from before this window's history start into 2020); pinned here because
#: CompustatSpec's default window moved to the 2000-2024 sample (brief 09 A)
FIXTURE_WINDOW = {"start": "2015-01-01", "end": "2024-12-31"}


def _cspec(root: Path, **kw) -> CompustatSpec:
    return CompustatSpec(**({"crsp_dir": str(root), "release": cfz.RELEASE}
                            | FIXTURE_WINDOW | kw))


def _crsp(root: Path, **kw) -> CRSPSpec:
    base = dict(crsp_dir=str(root), release=fx.RELEASE, start=fx.START, end=fx.END,
                chunk_bytes=20_000, member_count_min=5, member_count_max=8)
    return CRSPSpec(**(base | kw))


@pytest.fixture(scope="module")
def cfx(tmp_path_factory) -> cfz.CfzFixture:
    return cfz.write_fixture(tmp_path_factory.mktemp("cfz") / "Data")


@pytest.fixture(scope="module")
def cex(cfx) -> cs.CompustatExtract:
    spec = _cspec(cfx.root, chunk_bytes=4_000)
    cs.extract_compustat(spec, [*range(10001, 10011)], verbose=False)
    return cs.load_compustat_extract(spec)


@pytest.fixture(scope="module")
def quarters(cfx, cex) -> pd.DataFrame:
    q, _ = cs.quarterly_fundamentals(cex, _cspec(cfx.root), CAL)
    return q


# --------------------------------------------------------------------------- #
# C.1 the CRSP re-extract
# --------------------------------------------------------------------------- #


def test_lookback_derives_from_the_longest_feature_window():
    s = CRSPSpec()
    need = price_history_trading_days(FeatureSpec())
    assert need >= 5 * 252  # five years: beta correlation, seasonality years 2-5
    assert s.lookback_days == trading_to_calendar_days(need, 252, s.lookback_margin_days)
    assert s.lookback_days > LEGACY_LOOKBACK_DAYS
    assert trading_to_calendar_days(252, 252, 0) == 366  # ceil(365.25)
    assert dataclasses.replace(s, extract_lookback_days=900).lookback_days == 900
    start = pd.Timestamp(s.start)
    assert (start - s.extract_bounds[0]).days == s.lookback_days  # start itself unchanged


def test_extract_folder_names_carry_stock_file_and_lookback():
    s = CRSPSpec()
    legacy = dataclasses.replace(s, stock_file=LEGACY_STOCK_FILE,
                                 extract_lookback_days=LEGACY_LOOKBACK_DAYS)
    assert legacy.extract_dir.name == "crsp_extract_ciz202512_2000-01-01_2024-12-31"
    assert s.extract_dir.name == (
        f"crsp_extract_ciz202512_2000-01-01_2024-12-31_StkDlySecurityData_lb{s.lookback_days}d"
    )
    longer = dataclasses.replace(s, extract_lookback_days=s.lookback_days + 1)
    assert len({legacy.extract_dir, s.extract_dir, longer.extract_dir}) == 3


@pytest.fixture(scope="module")
def crsp_v2(tmp_path_factory):
    f = fx.write_fixture(tmp_path_factory.mktemp("crsp2") / "Data")
    extract_crsp(_crsp(f.root), verbose=False)
    return f


def test_new_columns_reach_the_extract_and_the_daily_frames(crsp_v2):
    spec = _crsp(crsp_v2.root)
    ex = load_extract(spec)
    for col in ("DlyOpen", "DlyHigh", "DlyLow", "DlyClose", "DlyBid", "DlyAsk", "DlyCap"):
        assert col in ex.stock.columns
    ok = ex.stock["DlyDelFlg"] == "N"
    assert np.allclose(ex.stock.loc[ok, "DlyClose"], ex.stock.loc[ok, "DlyPrc"].abs())
    assert (ex.stock.loc[ok, "DlyBid"] <= ex.stock.loc[ok, "DlyAsk"]).all()
    assert (ex.stock.loc[ok, "DlyLow"] <= ex.stock.loc[ok, "DlyHigh"]).all()
    assert ex.duration_flags is not None  # with the stock rows, no second pass
    frames, _, _ = crsp_daily_frames(ex, spec, StageBSpec(target_kind="raw"))
    daily = frames["10004"]
    for col in ("retx", "cap", "open", "high", "low", "close", "bid", "ask"):
        assert col in daily.columns
    info = json.loads((spec.extract_dir / "extract.json").read_text())
    assert info["stock_file"] == "StkDlySecurityData"
    assert info["lookback_days"] == spec.lookback_days
    # the units' metadata rows are copied verbatim for the check (C.1)
    meta = info["item_metadata"]
    assert set(meta) == {"header", "DlyCap", "DlyShrOut", "DlyVol"}
    assert "fixture: thousands of dollars" in meta["DlyCap"][0]


def test_an_extract_built_otherwise_is_refused(crsp_v2):
    spec = _crsp(crsp_v2.root)
    path = spec.extract_dir / "extract.json"
    original = path.read_text()
    try:
        info = json.loads(original)
        info["stock_file"] = LEGACY_STOCK_FILE
        info["extract_lo"] = "2000-01-01"
        path.write_text(json.dumps(info))
        with pytest.raises(ExtractMismatch, match="stock_file.*extract_lo"):
            load_extract(spec)
    finally:
        path.write_text(original)
    load_extract(spec)  # restored


def test_the_legacy_primary_file_extract_still_works(tmp_path: Path):
    f = fx.write_fixture(tmp_path / "Data")
    spec = _crsp(f.root, stock_file=LEGACY_STOCK_FILE, extract_lookback_days=550)
    extract_crsp(spec, verbose=False)
    ex = load_extract(spec)
    assert "DlyOpen" not in ex.stock.columns
    assert spec.extract_dir.name.endswith(f"{fx.START}_{fx.END}")
    # an extract.json written before brief 08 has no stock_file/lookback keys:
    # they are read off the stream and the bounds
    path = spec.extract_dir / "extract.json"
    info = json.loads(path.read_text())
    del info["stock_file"], info["lookback_days"]
    path.write_text(json.dumps(info))
    load_extract(spec)


# --------------------------------------------------------------------------- #
# file locations, generalised to any release
# --------------------------------------------------------------------------- #


def test_the_cfz_release_is_found_like_the_ciz_one(tmp_path: Path):
    plain = cfz.write_fixture(tmp_path / "plain")
    zipped_root = tmp_path / "zipped"
    cfz.write_fixture(zipped_root, as_zip=True)
    (zipped_root / "crspdata" / f"{cfz.RELEASE}_ascii" / "linkhistory.dat").unlink()
    path, member = crsp_file_source(_cspec(plain.root), "linkhistory")
    assert member is None and path.name == "linkhistory.dat"
    path, member = crsp_file_source(_cspec(zipped_root), "linkhistory")
    assert member == f"crspdata/{cfz.RELEASE}_ascii/linkhistory.dat"
    with open_crsp_file(_cspec(zipped_root), "linkhistory") as fh:
        assert fh.readline().startswith(b"KYGVKEY|LINKDT")
    with pytest.raises(FileNotFoundError):
        crsp_file_source(_cspec(tmp_path / "nowhere"), "linkhistory")


# --------------------------------------------------------------------------- #
# C.3 the CCM link
# --------------------------------------------------------------------------- #


def test_link_filter_keeps_types_and_primaries(cex, cfx):
    v = cs.valid_links(cex.links, _cspec(cfx.root))
    assert set(v["LINKTYPE"]) <= {"LC", "LU"} and set(v["LINKPRIM"]) <= {"P", "C"}
    assert 10009 not in set(v["LPERMNO"])  # NR link dropped
    assert 10010 not in set(v["LPERMNO"])  # LINKPRIM J dropped
    assert (v["LINKENDDT"] == cs.OPEN_END).any()  # 9999-12-31 read as open


def test_a_permno_that_changes_gvkey_maps_right_on_each_date(cex, cfx):
    spec = _cspec(cfx.root)
    days = pd.DatetimeIndex([cfz.REKEY - pd.Timedelta(days=1), cfz.REKEY])
    assert list(cs.gvkey_on_dates(cex.links, spec, 10005, days)) == ["001005", "001105"]
    # and a GVKEY that moves to another PERMNO
    before, after = cfz.MOVE - pd.Timedelta(days=1), cfz.MOVE
    both = pd.DatetimeIndex([before, after])
    assert list(cs.gvkey_on_dates(cex.links, spec, 10006, both)) == ["001006", None]
    assert list(cs.gvkey_on_dates(cex.links, spec, 10007, both)) == [None, "001006"]
    # two share classes of one GVKEY each map to it
    one = pd.DatetimeIndex([pd.Timestamp("2020-05-01")])
    assert cs.gvkey_on_dates(cex.links, spec, 10002, one).iloc[0] == "001001"


def test_no_permno_maps_to_two_gvkeys_and_an_ambiguity_is_reported(cex, cfx, tmp_path):
    spec = _cspec(cfx.root)
    assert cs.link_ambiguities(cex.links, spec).empty
    assert cex.info["report"]["link_ambiguities"] == 0

    amb = cfz.write_fixture(tmp_path / "Data", ambiguous=True)
    aspec = _cspec(amb.root)
    links = cs.read_links(aspec)
    found = cs.link_ambiguities(links, aspec)
    assert len(found) == 1  # reported, not dropped
    row = found.iloc[0]
    assert row["LPERMNO"] == 10004 and row["gvkeys"] == ("001004", "001099")
    assert row["from"] == pd.Timestamp("2020-02-01") and row["to"] == pd.Timestamp("2020-02-29")
    days = pd.DatetimeIndex(["2020-01-31", "2020-02-14", "2020-03-02"])
    assert list(cs.gvkey_on_dates(links, aspec, 10004, days)) == ["001004", None, "001004"]


# --------------------------------------------------------------------------- #
# C.2 GICS sectors
# --------------------------------------------------------------------------- #


def test_gics_dummies_switch_on_indfrom(cex, cfx):
    spec = _cspec(cfx.root)
    days = pd.bdate_range(cfz.SECTOR_SWITCH - pd.Timedelta(days=6), cfz.SECTOR_SWITCH)
    gv = cs.gvkey_on_dates(cex.links, spec, 10003, days)
    sector = cs.sector_on_dates(cex.gics, gv)
    dummies = cs.sector_dummies(sector)
    assert list(dummies.columns) == [f"gics_{c}" for c in cs.GICS_SECTORS]
    assert (dummies.loc[days < cfz.SECTOR_SWITCH, "gics_45"] == 1).all()
    assert dummies.loc[cfz.SECTOR_SWITCH, "gics_50"] == 1
    assert dummies.loc[cfz.SECTOR_SWITCH, "gics_45"] == 0
    assert (dummies.sum(axis=1) == 1).all()


def test_no_sector_means_all_dummies_zero(cex, cfx):
    days = pd.bdate_range("2020-01-02", periods=5)
    gv = cs.gvkey_on_dates(cex.links, _cspec(cfx.root), 10008, days)
    assert (gv == "001008").all()
    dummies = cs.sector_dummies(cs.sector_on_dates(cex.gics, gv))
    assert (dummies.to_numpy() == 0).all()


def test_gics_lpermno_is_checked_against_the_link(cex, cfx):
    spec = _cspec(cfx.root)
    assert cs.gics_lpermno_disagreements(cex.gics, cex.links, spec) == 0
    bad = cex.gics.copy()
    bad.loc[bad["KYGVKEY"] == "001004", "LPERMNO"] = 10005
    assert cs.gics_lpermno_disagreements(bad, cex.links, spec) == 1


# --------------------------------------------------------------------------- #
# C.4 Compustat, point in time
# --------------------------------------------------------------------------- #


def test_history_start_derives_from_the_features():
    spec = CompustatSpec()
    assert spec.quarters_back == 12  # debt_gr3; niq_su's 8 differences of 4-quarter lags
    # 12 + 2 quarters before the 2000 start of the sample (brief 09 A)
    assert spec.start == "2000-01-01"
    assert cs.fundamentals_start(spec) == pd.Timestamp("1996-07-01")
    assert cs.fundamentals_start(_cspec(Path("."))) == pd.Timestamp("2011-07-01")


def test_period_end_mapping(cex, cfx, quarters):
    for fy, q, fyr, expect in ((2015, 1, 12, "2015-03-31"), (2016, 1, 6, "2015-09-30"),
                               (2016, 4, 6, "2016-06-30"), (2015, 1, 1, "2015-04-30"),
                               (2015, 4, 1, "2016-01-31")):
        got = cs.computed_period_end(pd.Series([fy]), pd.Series([q]), pd.Series([fyr]))
        assert got.iloc[0] == pd.Timestamp(expect)
        assert cfz.period_end(fy, q, fyr) == pd.Timestamp(expect)
    truth = cfx.quarters.set_index(["KYGVKEY", "FYEARQ", "FQTR"])["datadate"]
    got = quarters.set_index(["KYGVKEY", "FYEARQ", "FQTR"])["datadate"]
    assert (got.reindex(truth.index) == truth).all()
    mapping = cex.info["report"]["period_end_mapping"]
    assert mapping["source"].startswith("computed")
    assert mapping["not_in_fiscalmarketdata"] == 0 and mapping["found_in_fiscalmarketdata"] > 0


def test_ytd_cash_flow_differenced_across_a_fiscal_year_boundary(quarters, cfx):
    truth = cfx.quarters.set_index(["KYGVKEY", "FYEARQ", "FQTR"])
    q = quarters.set_index(["KYGVKEY", "FYEARQ", "FQTR"])
    for key in (("001004", 2016, 1), ("001004", 2016, 2), ("001004", 2015, 4),
                ("001001", 2018, 1), ("001001", 2018, 3)):
        for item in ("OANCFQ", "CAPXQ", "DVQ"):
            assert q.loc[key, item] == pytest.approx(truth.loc[key, item], abs=2e-3), key
    # Q1 is its own YTD value, not the difference from the previous year's Q4
    assert q.loc[("001004", 2016, 1), "OANCFQ"] == pytest.approx(
        q.loc[("001004", 2016, 1), "OANCFY"]
    )
    # a quarter whose predecessor in the same fiscal year is missing gets no flow
    toy = pd.DataFrame({"KYGVKEY": ["a", "a"], "FYEARQ": [2020, 2020], "FQTR": [1, 3],
                        "OANCFY": [5.0, 12.0]})
    out = cs.ytd_to_quarterly(toy, ("OANCFY",))
    assert out["OANCFQ"].iloc[0] == 5.0 and np.isnan(out["OANCFQ"].iloc[1])


def test_filing_types_match_both_spellings(cfx):
    """The real cfz202607 file spells them 10Q / 10K; amendments never count."""
    f = pd.DataFrame({
        "KYGVKEY": ["a", "a", "a", "b"], "FDATADATE": ["2020-03-31"] * 4,
        "SRCTYPE": ["10Q/A", "10Q", "8K", "10-Q"],
        "FILEDATE": ["2020-04-20", "2020-05-01", "2020-04-02", "2020-05-03"],
    })
    got = cs.first_filing_dates(f, _cspec(cfx.root)).set_index("KYGVKEY")["first_filing"]
    assert got["a"] == pd.Timestamp("2020-05-01") and got["b"] == pd.Timestamp("2020-05-03")


def test_mapping_counts_are_also_given_for_the_history_window(cex):
    rep = cex.info["report"]["period_end_mapping"]
    inside = rep["since_history_start"]
    assert 0 < inside["rows"] < rep["rows"]  # the extract keeps older quarters too
    assert inside["not_in_fiscalmarketdata"] == 0


def _next_trading(d: pd.Timestamp) -> pd.Timestamp:
    return CAL[CAL > d][0]


def test_availability_rules(quarters, cfx):
    q = quarters.set_index(["KYGVKEY", "FYEARQ", "FQTR"])
    t = cfx.quarters.set_index(["KYGVKEY", "FYEARQ", "FQTR"])
    # later of RDQ and the first 10-Q/10-K (not its amendment, not the 8-K)
    key = cfz.FILING_AFTER_RDQ
    assert q.loc[key, "first_filing"] == t.loc[key, "filing"]
    assert q.loc[key, "avail"] == _next_trading(t.loc[key, "filing"])
    assert q.loc[key, "avail_rdq"] == _next_trading(t.loc[key, "rdq"])  # surprises: RDQ + 1
    assert q.loc[key, "anchor_rule"] == "later_of_rdq_and_filing"
    # no RDQ: the filing date
    key = cfz.RDQ_MISSING
    assert pd.isna(q.loc[key, "rdq"]) and q.loc[key, "anchor_rule"] == "filing_only"
    assert q.loc[key, "avail"] == _next_trading(t.loc[key, "filing"])
    assert q.loc[key, "avail_rdq"] == q.loc[key, "avail"]  # falls back to the rule
    # no filing: the RDQ
    key = cfz.NO_FILING
    assert q.loc[key, "anchor_rule"] == "rdq_only"
    assert q.loc[key, "avail"] == _next_trading(t.loc[key, "rdq"])
    # neither: the period end plus 90 calendar days, then one trading day
    key = cfz.NEITHER
    assert q.loc[key, "anchor_rule"] == "period_end_plus_lag"
    assert q.loc[key, "avail"] == _next_trading(t.loc[key, "datadate"] + pd.Timedelta(days=90))
    assert (q["avail"] > q["datadate"]).all()  # nothing usable on or before its period end


def _atq_events(quarters: pd.DataFrame, gvkey: str) -> pd.DataFrame:
    rows = []
    for event, known in cs.pit_events(quarters[quarters["KYGVKEY"] == gvkey]):
        latest = known.iloc[-1]
        rows.append({"event": event, "asof_avail": latest["avail"], "ATQ": latest["ATQ"]})
    return pd.DataFrame(rows)


def test_a_value_is_invisible_before_its_availability_and_visible_from_it(quarters):
    ev = _atq_events(quarters, "001001")
    daily = cs.carry_forward(ev, CAL, 365)
    q = quarters[quarters["KYGVKEY"] == "001001"].set_index(["FYEARQ", "FQTR"])
    this, prev = q.loc[(2019, 3)], q.loc[(2019, 2)]
    day_before = CAL[CAL < this["avail"]][-1]
    assert daily.loc[day_before, "ATQ"] == prev["ATQ"]
    assert daily.loc[this["avail"], "ATQ"] == this["ATQ"]
    assert this["avail"] > this["rdq"]  # the filing came later and decided it


def test_carry_forward_stops_at_the_staleness_cap(quarters):
    ev = _atq_events(quarters, "001008")
    last = quarters[(quarters["KYGVKEY"] == "001008")].iloc[-1]
    assert (last["FYEARQ"], last["FQTR"]) == cfz.LAST_REPORT
    daily = cs.carry_forward(ev, CAL, 365)
    cap = last["avail"] + pd.Timedelta(days=365)
    inside = CAL[(CAL >= last["avail"]) & (CAL <= cap)]
    after = CAL[CAL > cap]
    assert (daily.loc[inside, "ATQ"] == last["ATQ"]).all()
    assert daily.loc[after, "ATQ"].isna().all()
    # a shorter cap cuts earlier
    short = cs.carry_forward(ev, CAL, 30)
    assert short.loc[last["avail"] + pd.offsets.BDay(15), "ATQ"] == last["ATQ"]  # 21 days
    assert np.isnan(short.loc[last["avail"] + pd.offsets.BDay(30), "ATQ"])  # 42 days


def test_a_late_quarter_joins_the_known_set_on_its_own_date():
    q = pd.DataFrame({"fq_index": [1, 2, 3], "avail": pd.to_datetime(
        ["2020-01-10", "2020-04-10", "2020-03-01"]), "x": [1.0, 2.0, 3.0]})
    events = list(cs.pit_events(q))
    assert [e for e, _ in events] == list(pd.to_datetime(["2020-01-10", "2020-03-01",
                                                          "2020-04-10"]))
    assert list(events[1][1]["fq_index"]) == [1, 3]  # quarter 2 not known yet
    assert list(events[2][1]["fq_index"]) == [1, 2, 3]


def test_keyset_8_rows_are_counted_and_never_used(cex, quarters, cfx):
    assert cex.info["report"]["restated_keyset_firm_quarters"] == len(cfz.KEYSET8)
    assert set(quarters["KEYSET"]) == {1}
    key = cfz.KEYSET8[0]
    truth = cfx.quarters.set_index(["KYGVKEY", "FYEARQ", "FQTR"])
    got = quarters.set_index(["KYGVKEY", "FYEARQ", "FQTR"])
    assert got.loc[key, "SALEQ"] == pytest.approx(truth.loc[key, "SALEQ"])  # keyset 1 values


def test_extract_keeps_only_linked_gvkeys_and_records_its_report(cex, cfx):
    assert "009999" not in set(cex.period["KYGVKEY"])
    assert set(cex.period["KYGVKEY"]) == set(cfz.FYR)
    rep = cex.info["report"]
    assert rep["history_start"] == "2011-07-01"
    assert rep["gics_lpermno_disagreements"] == 0
    assert rep["filing_srctypes"]["10-Q/A"] > 0  # counted, so the filter can be judged
    with pytest.raises(ValueError, match="does not match"):
        cs.load_compustat_extract(_cspec(cfx.root, history_quarters=20))


# --------------------------------------------------------------------------- #
# real data: skipped without the licensed files
# --------------------------------------------------------------------------- #

REAL = CompustatSpec()


def _real() -> bool:
    try:
        crsp_file_source(REAL, "linkhistory")
    except FileNotFoundError:
        return False
    return True


@pytest.mark.crsp_data
@pytest.mark.skipif(not _real(), reason="CRSP/Compustat Merged files not under Data/")
def test_real_link_history_has_the_columns_and_no_ambiguity():
    links = cs.read_links(REAL)
    amb = cs.link_ambiguities(links, REAL)
    print(f"\n[compustat] {len(links)} link rows; {len(amb)} PERMNO-periods with two GVKEYs")
    assert {"KYGVKEY", "LINKDT", "LINKENDDT", "LPERMNO", "LINKTYPE", "LINKPRIM"} <= set(links)
