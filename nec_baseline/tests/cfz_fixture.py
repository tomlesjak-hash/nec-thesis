"""A small CRSP/Compustat Merged (cfz) fixture: real column names, **invented numbers**.

Licence rule (briefs 06 to 08): no real Compustat or CRSP row appears here.
GVKEYs, PERMNOs, dates, sectors and every amount are made up to exercise the
link, sector and point-in-time rules; they resemble no real company. The
PERMNOs are the CRSP fixture's (``tests/crsp_fixture.py``).

The cast:

- ``001001``: linked to 10001 (LC/P) and 10002 (LC/C), two share classes; a
  third link to 10009 is type NR (dropped by the filter). Three of its
  firm-quarters also carry a keyset-8 ("PRE") row.
- ``001003``: linked to 10003 (LU/P); its sibling class 10010 is linked with
  LINKPRIM J (dropped). Its GICS sector changes from 45 to 50 on
  ``SECTOR_SWITCH`` (``INDFROM``).
- ``001004``: linked to 10004; fiscal year ends in June (``fyrq`` 6), the
  case for period ends and for YTD differencing across a fiscal year.
- ``001005`` then ``001105``: PERMNO 10005 changes GVKEY on ``REKEY``.
- ``001006``: linked to 10006 until ``MOVE`` - 1 day, then to 10007.
- ``001008``: linked to 10008; no GICS row (sector missing); stops reporting
  after ``LAST_REPORT`` (for the staleness cap).

Report dates (RDQ) and filings are invented around each period end, with:
one quarter whose RDQ is 0 (missing), one with no filing, one with neither,
one whose filing comes after its RDQ, and filings of other types (10-Q/A,
8-K) that must be ignored. ``ambiguous=True`` adds a GVKEY linked to 10004
for February 2020 on top of ``001004``, an ambiguity the link check must
report.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RELEASE = "cfzfixture"
SECTOR_SWITCH = pd.Timestamp("2020-03-16")
REKEY = pd.Timestamp("2020-01-01")
MOVE = pd.Timestamp("2019-07-01")
LAST_REPORT = (2019, 2)  # 001008's last fiscal quarter
FAR = "9999-12-31"
FYR = {"001001": 12, "001003": 12, "001004": 6, "001005": 12, "001105": 12,
       "001006": 12, "001008": 12}
RDQ_MISSING = ("001003", 2018, 2)
NO_FILING = ("001003", 2018, 3)
NEITHER = ("001003", 2018, 4)
FILING_AFTER_RDQ = ("001001", 2019, 3)
KEYSET8 = (("001001", 2016, 1), ("001001", 2016, 2), ("001001", 2017, 4))

HEADERS = {
    "linkhistory": "KYGVKEY|LINKDT|LINKENDDT|LPERMNO|LPERMCO|LIID|LINKTYPE|LINKPRIM",
    "gicshistory": "KYGVKEY|KEYSET|INDFROM|INDTHRU|lpermno|lpermco|LinkRangeTypeCd|"
                   "GGROUPH|GINDH|GSECTORH|GSUBINDH",
    "perioddescriptorquarterly": "KYGVKEY|KEYSET|FYYYYQ|fyrq|lpermno|lpermco|"
                                 "LinkRangeTypeCd|DATACQTR|DATAFQTR|FQTR|FYEARQ|RDQ|"
                                 "FDATEQ|PDATEQ",
    "incomestatementquarterly": "KYGVKEY|KEYSET|FYYYYQ|lpermno|lpermco|LinkRangeTypeCd|"
                                "SALEQ|SALEQ_DC|SALEQ_FN|REVTQ|COGSQ|XSGAQ|IBQ|IBQ_DC|NIQ|"
                                "OIADPQ|EPSPXQ|DPQ|XINTQ|TXTQ|PIQ",
    "balancesheetquarterly": "KYGVKEY|KEYSET|FYYYYQ|lpermno|lpermco|LinkRangeTypeCd|"
                             "ATQ|ATQ_DC|ACTQ|CHEQ|LCTQ|DLCQ|DLTTQ|LTQ|CEQQ|SEQQ|PSTKQ|"
                             "TXDITCQ|IVAOQ|IVSTQ|MIBQ|INVTQ|RECTQ|PPENTQ|CSHOQ",
    "cashflowyeartodate": "KYGVKEY|KEYSET|FYYYYQ|lpermno|lpermco|LinkRangeTypeCd|"
                          "OANCFY|OANCFY_DC|IVNCFY|FINCFY|CAPXY|SSTKY|PRSTKCY|DVY",
    "filingdates": "KYGVKEY|FDATADATE|LPERMNO|LPERMCO|LinkRangeTypeCd|FCONSOL|FPOPSRC|"
                   "SRCTYPE|FILEDATE|FILEDATETIME",
    "fiscalmarketdataquarterly": "KYGVKEY|DATADATE|lpermno|lpermco|LinkRangeTypeCd|"
                                 "PRCCQ|CSHOQ|KEYSET",
}

INCOME = ("SALEQ", "REVTQ", "COGSQ", "XSGAQ", "IBQ", "NIQ", "OIADPQ", "EPSPXQ", "DPQ",
          "XINTQ", "TXTQ", "PIQ")
BALANCE = ("ATQ", "ACTQ", "CHEQ", "LCTQ", "DLCQ", "DLTTQ", "LTQ", "CEQQ", "SEQQ", "PSTKQ",
           "TXDITCQ", "IVAOQ", "IVSTQ", "MIBQ", "INVTQ", "RECTQ", "PPENTQ", "CSHOQ")
FLOWS = ("OANCF", "IVNCF", "FINCF", "CAPX", "SSTK", "PRSTKC", "DV")


@dataclass
class CfzFixture:
    root: Path  # plays the role of Quant Model/Data/
    quarters: pd.DataFrame  # the truth: one row per (gvkey, fyearq, fqtr), keyset 1


def _row(cols: str, **values: object) -> str:
    return "|".join("" if values.get(c) is None else str(values[c]) for c in cols.split("|"))


def period_end(fyearq: int, fqtr: int, fyr: int) -> pd.Timestamp:
    """Compustat's DATADATE convention, written out independently for the fixture."""
    end_year = fyearq + (1 if fyr <= 5 else 0)
    month = fyr - 3 * (4 - fqtr)
    year = end_year
    while month <= 0:
        month += 12
        year -= 1
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def write_fixture(root: Path, *, ambiguous: bool = False, as_zip: bool = False) -> CfzFixture:
    g = np.random.default_rng(20261005)
    files: dict[str, list[str]] = {name: [h] for name, h in HEADERS.items()}

    def link(gv, permno, lo, hi, ltype="LC", prim="P"):
        files["linkhistory"].append(_row(
            HEADERS["linkhistory"], KYGVKEY=gv, LINKDT=lo, LINKENDDT=hi, LPERMNO=permno,
            LPERMCO=50000 + permno, LIID="01", LINKTYPE=ltype, LINKPRIM=prim,
        ))

    link("001001", 10001, "2000-01-01", FAR)
    link("001001", 10002, "2000-01-01", FAR, prim="C")
    link("001001", 10009, "2000-01-01", FAR, ltype="NR")
    link("001003", 10003, "2000-01-01", FAR, ltype="LU")
    link("001003", 10010, "2000-01-01", FAR, ltype="LU", prim="J")
    link("001004", 10004, "2000-01-01", FAR)
    link("001005", 10005, "2000-01-01", (REKEY - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    link("001105", 10005, REKEY.strftime("%Y-%m-%d"), FAR)
    link("001006", 10006, "2000-01-01", (MOVE - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    link("001006", 10007, MOVE.strftime("%Y-%m-%d"), FAR)
    link("001008", 10008, "2000-01-01", FAR)
    link("009999", 99999, "2000-01-01", FAR)  # a firm outside the universe
    if ambiguous:
        link("001099", 10004, "2020-02-01", "2020-02-29")

    sectors = {"001001": [("2000-01-01", FAR, 45)],
               "001003": [("2000-01-01", (SECTOR_SWITCH - pd.Timedelta(days=1))
                           .strftime("%Y-%m-%d"), 45),
                          (SECTOR_SWITCH.strftime("%Y-%m-%d"), FAR, 50)],
               "001004": [("2000-01-01", FAR, 20)], "001005": [("2000-01-01", FAR, 35)],
               "001105": [(REKEY.strftime("%Y-%m-%d"), FAR, 35)],
               "001006": [("2000-01-01", FAR, 10)],
               "009999": [("2000-01-01", FAR, 55)]}
    first_permno = {"001001": 10001, "001003": 10003, "001004": 10004, "001005": 10005,
                    "001105": 10005, "001006": 10006, "009999": 99999}
    for gv, rows in sectors.items():
        for lo, hi, sec in rows:
            files["gicshistory"].append(_row(
                HEADERS["gicshistory"], KYGVKEY=gv, KEYSET=1, INDFROM=lo, INDTHRU=hi,
                lpermno=first_permno[gv], GSECTORH=sec, GGROUPH=sec * 100,
                GINDH=sec * 10000, GSUBINDH=sec * 1000000,
            ))

    truth = []
    for gv, fyr in FYR.items():
        years = range(2010, 2022)
        level = 1000.0 * (1 + g.uniform())
        for fy in years:
            ytd = dict.fromkeys(FLOWS, 0.0)
            for q in (1, 2, 3, 4):
                if gv == "001008" and (fy, q) > LAST_REPORT:
                    continue
                if gv == "001105" and fy < 2019:
                    continue
                end = period_end(fy, q, fyr)
                level *= float(np.exp(g.normal(0.01, 0.03)))
                vals = {k: round(level * g.uniform(0.05, 1.0), 3) for k in (*INCOME, *BALANCE)}
                vals["ATQ"] = round(level * 3, 3)
                vals["IBQ"] = round(level * g.normal(0.05, 0.03), 3)
                flows = {k: round(level * g.normal(0.08, 0.04), 3) for k in FLOWS}
                for k in FLOWS:
                    ytd[k] = round(ytd[k] + flows[k], 3)
                key = (gv, fy, q)
                rdq = None if key in (RDQ_MISSING, NEITHER) else \
                    end + pd.Timedelta(days=int(g.integers(25, 40)))
                filing = None if key in (NO_FILING, NEITHER) else \
                    end + pd.Timedelta(days=int(g.integers(30, 50) if q < 4 else 60))
                if key == FILING_AFTER_RDQ:
                    rdq = end + pd.Timedelta(days=20)
                    filing = end + pd.Timedelta(days=55)
                truth.append({"KYGVKEY": gv, "FYEARQ": fy, "FQTR": q, "datadate": end,
                              "rdq": rdq, "filing": filing, **vals,
                              **{f"{k}Q": flows[k] for k in FLOWS}})
                fyyyyq = fy * 10 + q
                cal_q = (end.month - 1) // 3 + 1
                for keyset in (1, 8) if key in KEYSET8 else (1,):
                    bump = 1.0 if keyset == 1 else 1.1  # pre-amendment values differ
                    common = dict(KYGVKEY=gv, KEYSET=keyset, FYYYYQ=fyyyyq, lpermno=1,
                                  lpermco=1, LinkRangeTypeCd="X")
                    files["perioddescriptorquarterly"].append(_row(
                        HEADERS["perioddescriptorquarterly"], **common, fyrq=fyr,
                        DATACQTR=f"{end.year}Q{cal_q}", DATAFQTR=f"{fy}Q{q}", FQTR=q,
                        FYEARQ=fy, RDQ=0 if rdq is None else rdq.strftime("%Y%m%d"),
                        FDATEQ=end.strftime("%Y-%m-%d"), PDATEQ=end.strftime("%Y-%m-%d"),
                    ))
                    files["incomestatementquarterly"].append(_row(
                        HEADERS["incomestatementquarterly"], **common,
                        **{k: round(vals[k] * bump, 3) for k in INCOME},
                        SALEQ_DC="", SALEQ_FN="", IBQ_DC="",
                    ))
                    files["balancesheetquarterly"].append(_row(
                        HEADERS["balancesheetquarterly"], **common,
                        **{k: round(vals[k] * bump, 3) for k in BALANCE}, ATQ_DC="",
                    ))
                    files["cashflowyeartodate"].append(_row(
                        HEADERS["cashflowyeartodate"], **common,
                        **{f"{k}Y": round(ytd[k] * bump, 3) for k in FLOWS}, OANCFY_DC="",
                    ))
                files["fiscalmarketdataquarterly"].append(_row(
                    HEADERS["fiscalmarketdataquarterly"], KYGVKEY=gv,
                    DATADATE=end.strftime("%Y-%m-%d"), PRCCQ=round(20 + 10 * g.uniform(), 2),
                    CSHOQ=100, KEYSET=1, lpermno=1, lpermco=1, LinkRangeTypeCd="X",
                ))
                if filing is not None:
                    form = "10-K" if q == 4 else "10-Q"
                    for srctype, when in ((form, filing),
                                          (f"{form}/A", filing + pd.Timedelta(days=90)),
                                          ("8-K", end + pd.Timedelta(days=3))):
                        files["filingdates"].append(_row(
                            HEADERS["filingdates"], KYGVKEY=gv,
                            FDATADATE=end.strftime("%Y-%m-%d"), LPERMNO=1, LPERMCO=1,
                            LinkRangeTypeCd="X", FCONSOL="C", FPOPSRC="D", SRCTYPE=srctype,
                            FILEDATE=when.strftime("%Y-%m-%d"),
                            FILEDATETIME=when.strftime("%Y-%m-%d") + " 16:05:00",
                        ))

    folder = root / "crspdata" / f"{RELEASE}_ascii"
    texts = {name: "\n".join(lines) + "\n" for name, lines in files.items()}
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (folder / f"{name}.dat").write_text(text)
    if as_zip:
        with zipfile.ZipFile(root / f"{RELEASE}_ascii.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            for name, text in texts.items():
                zf.writestr(f"crspdata/{RELEASE}_ascii/{name}.dat", text)
    return CfzFixture(root=root, quarters=pd.DataFrame(truth))
