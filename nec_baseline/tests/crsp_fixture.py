"""A small CRSP CIZ-format fixture: the real headers, **invented numbers**.

Licence rule (brief 06): no real CRSP row may appear in a fixture, docstring,
test or report. Everything below is generated from a seeded random walk; the
PERMNOs, index numbers' returns, dates of events and codes are made up to
exercise the construction rules, and resemble no real security.

The cast, on a business-day calendar from 2018-06-01 to 2020-12-31, window
2020-01-02 .. 2020-06-30:

- 10001..10006: ordinary members over the whole window;
- 10007: joins on ``JOIN`` (inclusive start), was trading long before;
- 10008: leaves on ``LEAVE`` (inclusive end), keeps trading after;
- 10003: a missing return on ``MISSING_DAY``;
- 10004: a 2-for-1 split on ``SPLIT_DAY`` (shares double, price halves);
- 10005: delists: last trade on ``DELIST_LAST``, delisting row the next
  trading day with a delisting return ``DELIST_RET``;
- 10006: delists on ``DELIST2_LAST`` and its delisting return is missing;
- 10001 and 10002: two share classes of one company (same PERMCO);
- 10009: never a member of the membership index (a member of another one).
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RELEASE = "cizfixture"
MEMBER_INDNO = 1000500
OTHER_INDNO = 1000200  # a second index series with different returns
START, END = "2020-01-02", "2020-06-30"
CAL = pd.bdate_range("2018-06-01", "2020-12-31")
JOIN = pd.Timestamp("2020-03-02")
LEAVE = pd.Timestamp("2020-04-15")
MISSING_DAY = pd.Timestamp("2020-02-12")
SPLIT_DAY = pd.Timestamp("2020-05-11")
DELIST_LAST = pd.Timestamp("2020-06-10")
DELIST_RET = -0.35
DELIST2_LAST = pd.Timestamp("2020-06-17")

HEADERS = {
    "StkDlySecurityPrimaryData": "|".join((
        "PERMNO", "DlyCalDt", "DlyDelFlg", "DlyPrc", "DlyPrcFlg", "DlyCap", "DlyCapFlg",
        "DlyRet", "DlyRetx", "DlyRetMissFlg", "DlyDistRetFlg", "DlyVol"
    )),
    "StkDlyCumulativeAdjFactor": "|".join((
        "PERMNO", "DlyCalDt", "DlyShrOut", "DlyCumFacPr", "DlyCumFacShr"
    )),
    "StkDelists": "|".join((
        "PERMNO", "DelistingDt", "DelDtPrc", "DelDtPrcFlg", "DelActionType", "DelStatusType",
        "DelReasonType", "DelPaymentType", "DelPERMNO", "DelPERMCO", "DelRet", "DelRetMissType",
        "DelNextDt", "DelNextPrc", "DelNextPrcFlg", "DelAmtDt", "DelDivAmt", "DelDisType",
        "DelDlyDt"
    )),
    "IndDlySeriesData": "|".join((
        "INDNO", "YYYYMMDD", "DlyCalDt", "DlyTotRet", "DlyTotInd", "DlyPrcRet", "DlyPrcInd",
        "DlyIncRet", "DlyIncInd", "DlyUsdCnt", "DlyUsdVal", "DlyTotCnt", "DlyTotVal",
        "DlyEligCnt", "DlyWgtAmt"
    )),
    "StkIndMembership": "|".join((
        "PERMNO", "INDNO", "MbrStartDt", "MbrEndDt", "MbrFlg", "INDFAM"
    )),
    "StkSecurityInfoHist": "|".join((
        "PERMNO", "SecInfoStartDt", "SecInfoEndDt", "SecurityBegDt", "SecurityEndDt",
        "SecurityHdrFlg", "HdrCUSIP", "HdrCUSIP9", "CUSIP", "CUSIP9", "PrimaryExch",
        "ConditionalType", "ExchangeTier", "TradingStatusFlg", "SecurityNm", "ShareClass",
        "USIncFlg", "IssuerType", "SecurityType", "SecuritySubType", "ShareType",
        "SecurityActiveFlg", "DelActionType", "DelStatusType", "DelReasonType",
        "DelPaymentType", "Ticker", "TradingSymbol", "PERMCO", "SICCD", "NAICS", "ICBIndustry",
        "UESIndustry", "NASDCompno", "NASDIssuno", "IssuerNm"
    )),
}


@dataclass
class Fixture:
    root: Path  # plays the role of Quant Model/Data/
    returns: dict[int, pd.Series]  # simple daily returns as written (NaN = missing)
    market: dict[int, pd.Series]  # DlyTotRet per INDNO


def _d(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def _row(cols: str, **values: object) -> str:
    return "|".join("" if values.get(c) is None else str(values[c]) for c in cols.split("|"))


def _next_day(ts: pd.Timestamp) -> pd.Timestamp:
    return CAL[CAL.get_loc(ts) + 1]


def write_fixture(root: Path, *, as_zip: bool = False, extracted: bool = True) -> Fixture:
    """Write the fixture's ``.dat`` files under ``root`` (and/or a zip of them)."""
    g = np.random.default_rng(20260926)
    folder = root / "crspdata" / f"{RELEASE}_ascii"
    files: dict[str, list[str]] = {name: [header] for name, header in HEADERS.items()}

    market = {
        MEMBER_INDNO: pd.Series(g.normal(0.0004, 0.01, len(CAL)), index=CAL),
        OTHER_INDNO: pd.Series(g.normal(-0.0002, 0.013, len(CAL)), index=CAL),
    }
    for indno, s in market.items():
        for day, r in s.items():
            files["IndDlySeriesData"].append(_row(
                HEADERS["IndDlySeriesData"], INDNO=indno, YYYYMMDD=day.strftime("%Y%m%d"),
                DlyCalDt=_d(day), DlyTotRet=f"{r:.8f}", DlyTotCnt=500,
            ))

    life = {p: (CAL[0], CAL[-1]) for p in range(10001, 10010)}
    life[10005] = (CAL[0], _next_day(DELIST_LAST))
    life[10006] = (CAL[0], _next_day(DELIST2_LAST))
    returns: dict[int, pd.Series] = {}
    for permno, (first, last) in life.items():
        days = CAL[(CAL >= first) & (CAL <= last)]
        r = pd.Series(g.normal(0.0003, 0.02, len(days)), index=days)
        vol = pd.Series(np.round(1e6 * np.exp(g.normal(0, 0.3, len(days)))), index=days)
        fac = pd.Series(1.0, index=days)
        price = 50.0 * np.exp(np.cumsum(np.log1p(r)))
        if permno == 10003:
            r[MISSING_DAY] = np.nan
        if permno == 10004:
            split = days >= SPLIT_DAY
            price[split] /= 2.0
            vol[split] *= 2.0
            fac[~split] = 2.0  # anchored at the end: 1 on the last row
        delisting_day = None
        if permno in (10005, 10006):
            delisting_day = days[-1]
            r[delisting_day] = DELIST_RET if permno == 10005 else np.nan
            vol[delisting_day] = 0.0
        returns[permno] = r
        for day in days:
            is_del = day == delisting_day
            ret = r[day]
            files["StkDlySecurityPrimaryData"].append(_row(
                HEADERS["StkDlySecurityPrimaryData"], PERMNO=permno, DlyCalDt=_d(day),
                DlyDelFlg="Y" if is_del else "N",
                DlyPrc=None if is_del else f"{price[day]:.4f}",
                DlyPrcFlg="DA" if is_del else "TR",
                DlyCap=None if is_del else f"{price[day] * 1000:.2f}",
                DlyCapFlg="NA",
                DlyRet=None if np.isnan(ret) else f"{ret:.8f}",
                DlyRetx=None if np.isnan(ret) else f"{ret:.8f}",
                DlyRetMissFlg=("DM" if is_del else "NT") if np.isnan(ret) else "NA",
                DlyDistRetFlg="NA",
                DlyVol=f"{vol[day]:.0f}",
            ))
            files["StkDlyCumulativeAdjFactor"].append(_row(
                HEADERS["StkDlyCumulativeAdjFactor"], PERMNO=permno, DlyCalDt=_d(day),
                DlyShrOut=f"{1000 * fac[day]:.0f}", DlyCumFacPr=f"{fac[day]:.1f}",
                DlyCumFacShr=f"{fac[day]:.1f}",
            ))

    for permno, last, ret, miss in (
        (10005, DELIST_LAST, DELIST_RET, None),
        (10006, DELIST2_LAST, None, "DM"),
    ):
        files["StkDelists"].append(_row(
            HEADERS["StkDelists"], PERMNO=permno, DelistingDt=_d(last),
            DelActionType="GDR" if permno == 10005 else "MER",
            DelStatusType="VCL", DelReasonType="XYZ" if permno == 10005 else "ABC",
            DelPaymentType="CASH", DelRet=None if ret is None else f"{ret:.6f}",
            DelRetMissType=miss or "NA", DelDlyDt=_d(_next_day(last)),
        ))

    far = "2030-12-31"
    spells = [(p, "2010-01-04", far) for p in (10001, 10002, 10003, 10004)]
    spells += [
        (10005, "2010-01-04", _d(DELIST_LAST)),
        (10006, "2010-01-04", _d(DELIST2_LAST)),
        (10007, _d(JOIN), far),
        (10008, "2010-01-04", _d(LEAVE)),
    ]
    for permno, s, e in spells:
        files["StkIndMembership"].append(_row(
            HEADERS["StkIndMembership"], PERMNO=permno, INDNO=MEMBER_INDNO,
            MbrStartDt=s, MbrEndDt=e, MbrFlg="NORM", INDFAM=1100500,
        ))
    # the same spells under a second INDNO, plus a PERMNO only it has
    for permno, s, e in [*spells, (10009, "2010-01-04", far)]:
        files["StkIndMembership"].append(_row(
            HEADERS["StkIndMembership"], PERMNO=permno, INDNO=OTHER_INDNO,
            MbrStartDt=s, MbrEndDt=e, MbrFlg="NORM", INDFAM=1100200,
        ))

    ticker_change = pd.Timestamp("2020-03-16")
    for permno in life:
        permco = 50001 if permno in (10001, 10002) else 50000 + permno
        periods = [(CAL[0], CAL[-1], f"T{permno % 100:02d}")]
        if permno == 10007:
            periods = [
                (CAL[0], ticker_change - pd.Timedelta(days=1), "OLDT"),
                (ticker_change, CAL[-1], "NEWT"),
            ]
        for s, e, ticker in periods:
            files["StkSecurityInfoHist"].append(_row(
                HEADERS["StkSecurityInfoHist"], PERMNO=permno, SecInfoStartDt=_d(s),
                SecInfoEndDt=_d(e), ShareClass="B" if permno == 10002 else "A",
                Ticker=ticker, PERMCO=permco, IssuerNm="INVENTED CO",
            ))

    texts = {name: "\n".join(lines) + "\n" for name, lines in files.items()}
    if extracted:
        folder.mkdir(parents=True, exist_ok=True)
        for name, text in texts.items():
            (folder / f"{name}.dat").write_text(text)
    if as_zip:
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(root / f"{RELEASE}_ascii.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            for name, text in texts.items():
                zf.writestr(f"crspdata/{RELEASE}_ascii/{name}.dat", text)
    return Fixture(root=root, returns=returns, market=market)
