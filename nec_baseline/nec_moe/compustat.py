"""CRSP/Compustat Merged: the CCM link, GICS sectors, and point-in-time fundamentals.

Brief 08 part C (sections C.2 to C.4); decisions Q26 parts 4 and 5.

LICENCE — the same rules as :mod:`nec_moe.crsp`
-----------------------------------------------
Compustat is licensed with CRSP. Everything read or derived here stays inside
``Quant Model/Data/`` (extracts in ``Data/derived/``); tests use an invented
fixture with the real column names; reports quote aggregate counts only.

Files (release ``cfz202607``, pipe-delimited with a header row)
---------------------------------------------------------------
Found exactly like the CIZ files (:func:`nec_moe.crsp.crsp_file_source`):
``Data/crspdata/cfz202607_ascii/<file>.dat``, else the member
``crspdata/cfz202607_ascii/<file>.dat`` of ``Data/cfz202607_ascii.zip``.
Copies unpacked under other folder names (``Data/crspdata 3/...``) are not
looked for. Columns are read **by name** (matched case-insensitively, since
the brief writes ``fyrq`` in lower case beside upper-case names); a missing
column raises. Numeric fields are empty when missing.

- ``linkhistory``: ``KYGVKEY, LINKDT, LINKENDDT, LPERMNO, LPERMCO, LIID,
  LINKTYPE, LINKPRIM``. A link is valid on date d when
  ``LINKDT <= d <= LINKENDDT``; kept when ``LINKTYPE`` is in
  ``CompustatSpec.link_types`` (LC, LU) and ``LINKPRIM`` in ``link_prims``
  (P, C). An open end is a far-future date; any date from
  :data:`OPEN_END_YEAR` on is read as open.
- ``gicshistory``: ``KYGVKEY, KEYSET, INDFROM, INDTHRU, lpermno, ...,
  GSECTORH`` (the 2-digit sector). Dated, so point in time; ``INDTHRU`` of
  9999-12-31 means current. Linked to PERMNOs through ``linkhistory`` on the
  date, never through its own ``lpermno`` alone (which is only checked).
- ``perioddescriptorquarterly``: ``KYGVKEY, KEYSET, FYYYYQ, fyrq, ...,
  DATACQTR, DATAFQTR, FQTR, FYEARQ, RDQ, FDATEQ, PDATEQ``. ``RDQ`` (the
  report date) is ``YYYYMMDD`` or ``0`` when missing.
- ``incomestatementquarterly``, ``balancesheetquarterly``: quarterly items
  keyed by ``KYGVKEY, KEYSET, FYYYYQ``.
- ``cashflowyeartodate``: year-to-date items, differenced within the fiscal
  year into quarterly flows (:func:`ytd_to_quarterly`; Q1 is its YTD value).
- ``filingdates``: ``KYGVKEY, FDATADATE`` (period end), ``SRCTYPE`` (10-Q,
  10-K, ...), ``FILEDATE``.
- ``fiscalmarketdataquarterly``: ``KYGVKEY, DATADATE, KEYSET``; used to
  check the period-end mapping (below).

``KEYSET`` 1 is the industrial, consolidated, standardised format
(``d_keysetinfo``) and is the one used. Keysets 3 ("PRES") and 8 ("PRE"),
summary data collected before a company amendment, are **not** used; the
extract counts the universe firm-quarters with a keyset-8 row so the
restatement question can be judged later.

The fiscal period end (``DATADATE``) of a ``FYYYYQ``
---------------------------------------------------
The quarterly item files are keyed by fiscal quarter, the filing dates by
period end. The mapping (:func:`period_end_dates`), in order of preference:

1. a ``DATADATE`` column of ``perioddescriptorquarterly``, if the release
   has one;
2. ``fiscalmarketdataquarterly`` joined on ``KYGVKEY, KEYSET, FYYYYQ``, if
   that file carries ``FYYYYQ``;
3. computed from ``FYEARQ``, ``FQTR`` and the fiscal year-end month
   ``fyrq`` by Compustat's convention: a fiscal year ending in months 6 to
   12 is labelled with the calendar year it ends in, one ending in months 1
   to 5 with the year before; quarter ``q`` ends ``3 (4 - q)`` months before
   the fiscal year does; ``DATADATE`` is that month's last day.

Whatever the source, the computed date is compared with the file's and with
the ``(KYGVKEY, DATADATE)`` pairs of ``fiscalmarketdataquarterly``; the
agreement counts go to the extract report. **Not verified on the real
release** (this code has not seen it): the coverage report prints them.
``d_fiscalperiod`` is not read.

Availability (Q26 part 5, as proposed in the brief)
---------------------------------------------------
A quarter's numbers become usable ``availability_lag_trading_days`` (1)
trading days after the later of its ``RDQ`` and its first ``filing_types``
filing for that period end; with no ``RDQ``, the filing date; with neither,
the period end plus ``fallback_lag_days`` (90) calendar days, then the same
trading-day lag. The earnings and revenue surprises use ``RDQ`` plus the lag
(``avail_rdq``), falling back to the rule above without an ``RDQ``. After
availability a value is carried forward daily until the next quarter
arrives, for at most ``fundamental_max_staleness_days`` (365) calendar days
after its availability date (:func:`carry_forward`).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .crsp import (
    REPO_DATA_DIR,
    _arrow,
    _csv_options,
    _read_parts,
    _stream_extract,
    _write_json,
    open_crsp_file,
)
from .features import FeatureSpec, fundamental_history_quarters

__all__ = [
    "OPEN_END_YEAR",
    "GICS_SECTORS",
    "CompustatSpec",
    "CompustatExtract",
    "fundamentals_start",
    "parse_dates",
    "read_links",
    "valid_links",
    "link_ambiguities",
    "gvkey_on_dates",
    "read_gics",
    "sector_on_dates",
    "sector_dummies",
    "gics_lpermno_disagreements",
    "computed_period_end",
    "period_end_dates",
    "ytd_to_quarterly",
    "first_filing_dates",
    "trading_days_after",
    "availability_dates",
    "quarterly_fundamentals",
    "pit_events",
    "carry_forward",
    "keyset_count",
    "extract_compustat",
    "load_compustat_extract",
]

#: A date in this year or later is an open end ("current"), e.g. 9999-12-31.
OPEN_END_YEAR = 2200
#: The open end, as a timestamp pandas can hold.
OPEN_END = pd.Timestamp(f"{OPEN_END_YEAR}-12-31")
#: The 11 two-digit GICS sectors (``GSECTORH``), in code order.
GICS_SECTORS: tuple[int, ...] = (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)

# Columns read from each file: canonical (upper-case) name -> type. Dates are
# read as strings and parsed here, because 9999-12-31 does not fit pandas'
# nanosecond timestamps; GVKEYs are strings (their leading zeros matter).
LINK_COLUMNS = {
    "KYGVKEY": "string", "LINKDT": "string", "LINKENDDT": "string", "LPERMNO": "float64",
    "LPERMCO": "float64", "LIID": "string", "LINKTYPE": "string", "LINKPRIM": "string",
}
GICS_COLUMNS = {
    "KYGVKEY": "string", "KEYSET": "int64", "INDFROM": "string", "INDTHRU": "string",
    "LPERMNO": "float64", "GSECTORH": "string",
}
PERIOD_COLUMNS = {
    "KYGVKEY": "string", "KEYSET": "int64", "FYYYYQ": "int64", "FYRQ": "float64",
    "DATACQTR": "string", "DATAFQTR": "string", "FQTR": "float64", "FYEARQ": "float64",
    "RDQ": "float64", "FDATEQ": "string", "PDATEQ": "string",
}
INCOME_ITEMS: tuple[str, ...] = (
    "SALEQ", "REVTQ", "COGSQ", "XSGAQ", "IBQ", "NIQ", "OIADPQ", "EPSPXQ", "DPQ", "XINTQ",
    "TXTQ", "PIQ",
)
BALANCE_ITEMS: tuple[str, ...] = (
    "ATQ", "ACTQ", "CHEQ", "LCTQ", "DLCQ", "DLTTQ", "LTQ", "CEQQ", "SEQQ", "PSTKQ",
    "TXDITCQ", "IVAOQ", "IVSTQ", "MIBQ", "INVTQ", "RECTQ", "PPENTQ", "CSHOQ",
)
CASHFLOW_YTD_ITEMS: tuple[str, ...] = (
    "OANCFY", "IVNCFY", "FINCFY", "CAPXY", "SSTKY", "PRSTKCY", "DVY",
)
_KEYS = {"KYGVKEY": "string", "KEYSET": "int64", "FYYYYQ": "int64"}
FILING_COLUMNS = {
    "KYGVKEY": "string", "FDATADATE": "string", "SRCTYPE": "string", "FILEDATE": "string",
}
FISCAL_MARKET_COLUMNS = {"KYGVKEY": "string", "DATADATE": "string", "KEYSET": "int64"}

#: file name -> (required columns, optional columns)
FILES: dict[str, tuple[dict[str, str], dict[str, str]]] = {
    "linkhistory": (LINK_COLUMNS, {}),
    "gicshistory": (GICS_COLUMNS, {}),
    "perioddescriptorquarterly": (PERIOD_COLUMNS, {"DATADATE": "string"}),
    "incomestatementquarterly": (_KEYS | dict.fromkeys(INCOME_ITEMS, "float64"), {}),
    "balancesheetquarterly": (_KEYS | dict.fromkeys(BALANCE_ITEMS, "float64"), {}),
    "cashflowyeartodate": (_KEYS | dict.fromkeys(CASHFLOW_YTD_ITEMS, "float64"), {}),
    "filingdates": (FILING_COLUMNS, {}),
    "fiscalmarketdataquarterly": (FISCAL_MARKET_COLUMNS, {"FYYYYQ": "int64"}),
}


@dataclass(frozen=True)
class CompustatSpec:
    """Where the CRSP/Compustat Merged files are, and the point-in-time rules.

    Separate from :class:`nec_moe.crsp.CRSPSpec`: another release
    (``cfz202607``), another licence file set. ``start``/``end`` are the
    panel window (the CRSP spec's); the fundamentals reach back to
    :func:`fundamentals_start`, derived from the feature definitions.

    - ``keyset``: the Compustat data format used (1: industrial,
      consolidated, standardised); ``restated_keyset`` (8, "PRE") is only
      counted, never used.
    - ``link_types`` / ``link_prims``: the CCM link filter (LC, LU; P, C).
    - ``filing_types``: ``SRCTYPE`` values that count as a quarter's
      filing (10-Q, 10-K).
    - ``availability_lag_trading_days``: trading days after the anchor date
      (later of RDQ and filing) a quarter becomes usable (1).
    - ``fallback_lag_days``: calendar days after the period end used when a
      quarter has neither RDQ nor filing date (90).
    - ``history_quarters``: fiscal quarters the features reach back from the
      latest one; ``None`` derives it from the default
      :class:`~nec_moe.features.FeatureSpec`
      (:func:`~nec_moe.features.fundamental_history_quarters`, 12).
      ``history_margin_quarters`` (2) covers the reporting lag and the
      quarter in progress at ``start``.
    """

    crsp_dir: str = str(REPO_DATA_DIR)
    release: str = "cfz202607"
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    keyset: int = 1
    restated_keyset: int = 8
    link_types: tuple[str, ...] = ("LC", "LU")
    link_prims: tuple[str, ...] = ("P", "C")
    filing_types: tuple[str, ...] = ("10-Q", "10-K")
    availability_lag_trading_days: int = 1
    fallback_lag_days: int = 90
    history_quarters: int | None = None
    history_margin_quarters: int = 2
    chunk_bytes: int = 256 * 2**20

    def validate(self) -> CompustatSpec:
        if pd.Timestamp(self.end) < pd.Timestamp(self.start):
            raise ValueError(f"end {self.end} is before start {self.start}")
        if not self.link_types or not self.link_prims or not self.filing_types:
            raise ValueError("link_types, link_prims and filing_types must be non-empty")
        if self.availability_lag_trading_days < 1:
            raise ValueError("availability_lag_trading_days must be >= 1 (no same-day use)")
        if self.fallback_lag_days < 0 or self.history_margin_quarters < 0:
            raise ValueError("fallback_lag_days and history_margin_quarters must be >= 0")
        if self.history_quarters is not None and self.history_quarters < 1:
            raise ValueError("history_quarters must be >= 1")
        if self.chunk_bytes < 1:
            raise ValueError("chunk_bytes must be >= 1")
        return self

    @property
    def quarters_back(self) -> int:
        if self.history_quarters is not None:
            return self.history_quarters
        return fundamental_history_quarters(FeatureSpec())

    @property
    def derived_dir(self) -> Path:
        return Path(self.crsp_dir) / "derived"

    @property
    def extract_dir(self) -> Path:
        return self.derived_dir / f"compustat_{self.release}_{self.start}_{self.end}"


def fundamentals_start(spec: CompustatSpec) -> pd.Timestamp:
    """The earliest period end the extract keeps: ``start`` moved back by
    ``quarters_back + history_margin_quarters`` quarters (about mid-2011 for
    the defaults), so the 12-quarter growth and the 8-quarter surprise
    volatility exist on the first panel date."""
    months = 3 * (spec.quarters_back + spec.history_margin_quarters)
    return (pd.Timestamp(spec.start) - pd.DateOffset(months=months)).normalize()


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def parse_dates(values: pd.Series, *, fmt: str | None = None) -> pd.Series:
    """Parse ISO (or ``fmt``) date strings; empty, ``"0"`` and ``0`` are NaT,
    and a date from :data:`OPEN_END_YEAR` on is :data:`OPEN_END`."""
    s = values.astype("string").str.strip()
    s = s.mask(s.isin(["", "0", "0.0", "<NA>"]) | s.isna())
    if fmt is not None:
        s = s.str.replace(r"\.0$", "", regex=True)
    year = pd.to_numeric(s.str.slice(0, 4), errors="coerce").astype(float)
    far = pd.Series(np.asarray(year >= OPEN_END_YEAR, dtype=bool), index=s.index)
    out = pd.to_datetime(s.mask(far), format=fmt, errors="raise").astype("datetime64[ns]")
    return out.mask(far, OPEN_END)


def _resolve(header: list[str], wanted: dict[str, str], optional: dict[str, str], name: str):
    """Map canonical names to the header's spelling (case-insensitive)."""
    by_upper = {h.upper(): h for h in header}
    missing = [c for c in wanted if c.upper() not in by_upper]
    if missing:
        raise ValueError(f"{name}: header lacks columns {missing}")
    found = {by_upper[c.upper()]: t for c, t in {**wanted, **optional}.items()
             if c.upper() in by_upper}
    rename = {by_upper[c.upper()]: c for c in {**wanted, **optional} if c.upper() in by_upper}
    return found, rename


def _header(spec: CompustatSpec, name: str) -> list[str]:
    with open_crsp_file(spec, name) as fh:
        return fh.readline().decode().rstrip("\r\n").split("|")


def _read_file(spec: CompustatSpec, name: str) -> pd.DataFrame:
    """Read a whole (small) file's listed columns, typed, canonically named."""
    _, _, pacsv, _ = _arrow()
    wanted, optional = FILES[name]
    types, rename = _resolve(_header(spec, name), wanted, optional, name)
    read, parse, convert = _csv_options(types)
    with open_crsp_file(spec, name) as fh:
        table = pacsv.read_csv(fh, read_options=read, parse_options=parse,
                               convert_options=convert)
    return table.to_pandas().rename(columns=rename)


def read_links(spec: CompustatSpec) -> pd.DataFrame:
    return _typed_links(_read_file(spec, "linkhistory"))


def _typed_links(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["LINKDT"] = parse_dates(out["LINKDT"])
    out["LINKENDDT"] = parse_dates(out["LINKENDDT"])
    out["LINKENDDT"] = out["LINKENDDT"].fillna(OPEN_END)  # blank end: open (counted)
    return out


def read_gics(spec: CompustatSpec) -> pd.DataFrame:
    return _typed_gics(_read_file(spec, "gicshistory"))


def _typed_gics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["INDFROM"] = parse_dates(out["INDFROM"])
    out["INDTHRU"] = parse_dates(out["INDTHRU"]).fillna(OPEN_END)
    out["GSECTORH"] = pd.to_numeric(out["GSECTORH"].astype("string").str.strip(),
                                    errors="coerce")
    return out


# --------------------------------------------------------------------------- #
# The CCM link (C.3)
# --------------------------------------------------------------------------- #


def valid_links(links: pd.DataFrame, spec: CompustatSpec) -> pd.DataFrame:
    """The link rows kept by ``link_types`` and ``link_prims``, with a PERMNO."""
    keep = (
        links["LINKTYPE"].isin(spec.link_types)
        & links["LINKPRIM"].isin(spec.link_prims)
        & links["LPERMNO"].notna()
    )
    out = links.loc[keep].copy()
    out["LPERMNO"] = out["LPERMNO"].astype("int64")
    return out.reset_index(drop=True)


def link_ambiguities(links: pd.DataFrame, spec: CompustatSpec) -> pd.DataFrame:
    """Every (PERMNO, period) where two GVKEYs are linked at once, under the
    filters: ``LPERMNO``, the overlap's first and last day, and both GVKEYs.

    Reported, never dropped: :func:`gvkey_on_dates` gives such dates no
    GVKEY (so no fundamentals and no sector, which the flags then show) and
    the coverage report counts them.
    """
    v = valid_links(links, spec)
    rows = []
    for permno, g in v.groupby("LPERMNO"):
        g = g.sort_values("LINKDT")
        recs = g[["KYGVKEY", "LINKDT", "LINKENDDT"]].to_numpy()
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                if a[0] == b[0]:
                    continue
                lo, hi = max(a[1], b[1]), min(a[2], b[2])
                if lo <= hi:
                    rows.append({"LPERMNO": int(permno), "from": lo, "to": hi,
                                 "gvkeys": tuple(sorted((a[0], b[0])))})
    return pd.DataFrame(rows, columns=["LPERMNO", "from", "to", "gvkeys"])


def gvkey_on_dates(
    links: pd.DataFrame, spec: CompustatSpec, permno: int, dates: pd.DatetimeIndex
) -> pd.Series:
    """The GVKEY linked to ``permno`` on each of ``dates`` (``None`` if none,
    or if two are — an ambiguity, see :func:`link_ambiguities`)."""
    v = valid_links(links, spec)
    v = v[v["LPERMNO"] == permno]
    d = dates.to_numpy(dtype="datetime64[ns]")
    count = np.zeros(len(d), dtype=int)
    gvkey = np.full(len(d), None, dtype=object)
    for gv, g in v.groupby("KYGVKEY"):
        on = np.zeros(len(d), dtype=bool)
        for lo, hi in g[["LINKDT", "LINKENDDT"]].itertuples(index=False):
            on |= (d >= np.datetime64(lo, "ns")) & (d <= np.datetime64(hi, "ns"))
        count += on
        gvkey[on] = gv
    gvkey[count != 1] = None  # no link, or two GVKEYs at once (an ambiguity)
    return pd.Series(gvkey, index=dates, dtype=object)


# --------------------------------------------------------------------------- #
# GICS sectors (C.2)
# --------------------------------------------------------------------------- #


def sector_on_dates(gics: pd.DataFrame, gvkeys: pd.Series) -> pd.Series:
    """The GICS sector (``GSECTORH``) of each date's GVKEY, ``NaN`` without one.

    ``gvkeys`` is :func:`gvkey_on_dates` output (date-indexed). A row is in
    force on d when ``INDFROM <= d <= INDTHRU``. Two different sectors in
    force on one date (overlapping rows, e.g. from different keysets) are
    an ambiguity and give no sector rather than a guess.
    """
    out = pd.Series(np.nan, index=gvkeys.index, dtype=float)
    by_gv = {gv: g for gv, g in gics.groupby("KYGVKEY")}
    for gv in {g for g in gvkeys.dropna().unique()}:
        g = by_gv.get(gv)
        if g is None:
            continue
        mask = (gvkeys == gv).to_numpy()
        d = gvkeys.index[mask].to_numpy()
        found: list[set[float]] = [set() for _ in range(len(d))]
        for lo, hi, sec in g[["INDFROM", "INDTHRU", "GSECTORH"]].itertuples(index=False):
            if pd.isna(lo) or pd.isna(sec):
                continue
            for i in np.flatnonzero((d >= np.datetime64(lo)) & (d <= np.datetime64(hi))):
                found[i].add(float(sec))
        vals = [next(iter(s)) if len(s) == 1 else np.nan for s in found]
        out.iloc[np.flatnonzero(mask)] = vals
    return out


def sector_dummies(sector: pd.Series) -> pd.DataFrame:
    """11 columns ``gics_<code>``, 0/1; all 0 on a date without a sector."""
    return pd.DataFrame(
        {f"gics_{code}": (sector == code).astype(np.float32) for code in GICS_SECTORS},
        index=sector.index,
    )


def gics_lpermno_disagreements(
    gics: pd.DataFrame, links: pd.DataFrame, spec: CompustatSpec
) -> int:
    """GICS rows whose own ``lpermno`` is not linked to their GVKEY by the CCM
    link history at the row's ``INDFROM`` (a check, never used to map)."""
    v = valid_links(links, spec)
    bad = 0
    for gv, lo, lp in gics[["KYGVKEY", "INDFROM", "LPERMNO"]].itertuples(index=False):
        if pd.isna(lp) or pd.isna(lo):
            continue
        hit = v[(v["KYGVKEY"] == gv) & (v["LINKDT"] <= lo) & (lo <= v["LINKENDDT"])]
        bad += int(int(lp) not in set(hit["LPERMNO"]))
    return bad


# --------------------------------------------------------------------------- #
# Quarterly fundamentals, point in time (C.4)
# --------------------------------------------------------------------------- #


def computed_period_end(fyearq: pd.Series, fqtr: pd.Series, fyr: pd.Series) -> pd.Series:
    """``DATADATE`` from fiscal year, quarter and year-end month (module
    docstring, rule 3). NaT where any input is missing."""
    ok = fyearq.notna() & fqtr.notna() & fyr.notna()
    y = fyearq.where(ok, 0).astype(int)
    q = fqtr.where(ok, 1).astype(int)
    m = fyr.where(ok, 12).astype(int)
    end_year = y + (m <= 5).astype(int)
    idx = end_year * 12 + (m - 1) - 3 * (4 - q)
    first = pd.to_datetime(
        {"year": idx // 12, "month": idx % 12 + 1, "day": 1}, errors="coerce"
    )
    out = first + pd.offsets.MonthEnd(0)
    return out.where(ok)


def period_end_dates(
    period: pd.DataFrame, fiscal_market: pd.DataFrame | None
) -> tuple[pd.Series, dict[str, Any]]:
    """``DATADATE`` for each row of ``period`` and how it was found.

    Returns the dates (aligned to ``period``'s index) and a report: the
    source used and agreement counts against the computed rule and the
    ``(KYGVKEY, DATADATE)`` pairs of ``fiscalmarketdataquarterly``.
    """
    computed = computed_period_end(period["FYEARQ"], period["FQTR"], period["FYRQ"])
    report: dict[str, Any] = {}
    if "DATADATE" in period.columns:
        dates = parse_dates(period["DATADATE"])
        report["source"] = "perioddescriptorquarterly.DATADATE"
    elif fiscal_market is not None and "FYYYYQ" in fiscal_market.columns:
        fm = fiscal_market.assign(DATADATE=parse_dates(fiscal_market["DATADATE"]))
        fm = fm.drop_duplicates(["KYGVKEY", "KEYSET", "FYYYYQ"])
        merged = period[["KYGVKEY", "KEYSET", "FYYYYQ"]].merge(
            fm[["KYGVKEY", "KEYSET", "FYYYYQ", "DATADATE"]],
            on=["KYGVKEY", "KEYSET", "FYYYYQ"], how="left",
        )
        dates = pd.Series(merged["DATADATE"].to_numpy(), index=period.index)
        report["source"] = "fiscalmarketdataquarterly"
    else:
        dates = computed
        report["source"] = "computed (FYEARQ, FQTR, fyrq)"
    both = dates.notna() & computed.notna()
    report["rows"] = int(len(period))
    report["missing"] = int(dates.isna().sum())
    report["agree_with_computed"] = int((dates[both] == computed[both]).sum())
    report["disagree_with_computed"] = int((dates[both] != computed[both]).sum())
    if fiscal_market is not None and len(fiscal_market):
        pairs = set(zip(fiscal_market["KYGVKEY"], parse_dates(fiscal_market["DATADATE"]),
                        strict=True))
        known = [(g, d) in pairs for g, d in zip(period["KYGVKEY"], dates, strict=True)
                 if pd.notna(d)]
        report["found_in_fiscalmarketdata"] = int(sum(known))
        report["not_in_fiscalmarketdata"] = int(len(known) - sum(known))
    return dates, report


def ytd_to_quarterly(
    df: pd.DataFrame, items: tuple[str, ...], *, suffix_in: str = "Y", suffix_out: str = "Q"
) -> pd.DataFrame:
    """Year-to-date items differenced within the fiscal year into quarterly flows.

    ``df`` holds one row per (``KYGVKEY``, ``FYEARQ``, ``FQTR``) with the YTD
    columns. Q1's flow is its YTD value; quarter ``k > 1`` is ``YTD_k -
    YTD_{k-1}`` of the **same** fiscal year, and missing when that earlier
    quarter (or either value) is missing — never the YTD value itself, which
    would overstate the quarter. Output columns replace ``suffix_in`` by
    ``suffix_out`` (``OANCFY`` -> ``OANCFQ``).
    """
    out = df.copy()
    key = ["KYGVKEY", "FYEARQ"]
    prev = df[[*key, "FQTR", *items]].copy()
    prev["FQTR"] = prev["FQTR"] + 1
    joined = out[[*key, "FQTR"]].merge(prev, on=[*key, "FQTR"], how="left",
                                       suffixes=("", "_prev"))
    for item in items:
        name = item[: -len(suffix_in)] + suffix_out if item.endswith(suffix_in) else item + "_q"
        cur = df[item].to_numpy(dtype=float)
        before = joined[item].to_numpy(dtype=float)
        q1 = (df["FQTR"] == 1).to_numpy()
        out[name] = np.where(q1, cur, cur - before)
    return out


def first_filing_dates(filings: pd.DataFrame, spec: CompustatSpec) -> pd.DataFrame:
    """``KYGVKEY, period_end, first_filing``: the earliest ``FILEDATE`` of a
    ``filing_types`` filing for each period end."""
    f = filings[filings["SRCTYPE"].astype("string").str.strip().isin(spec.filing_types)].copy()
    f["period_end"] = parse_dates(f["FDATADATE"])
    f["filed"] = parse_dates(f["FILEDATE"])
    f = f.dropna(subset=["period_end", "filed"])
    out = f.groupby(["KYGVKEY", "period_end"], as_index=False)["filed"].min()
    return out.rename(columns={"filed": "first_filing"})


def trading_days_after(
    anchors: pd.Series, calendar: pd.DatetimeIndex, lag: int
) -> pd.Series:
    """The ``lag``-th trading day strictly after each anchor (NaT if none)."""
    cal = calendar.to_numpy()
    out = np.full(len(anchors), np.datetime64("NaT"), dtype="datetime64[ns]")
    a = anchors.to_numpy(dtype="datetime64[ns]")
    ok = ~np.isnat(a)
    pos = np.searchsorted(cal, a[ok], side="right") + (lag - 1)
    inside = pos < len(cal)
    vals = np.full(int(ok.sum()), np.datetime64("NaT"), dtype="datetime64[ns]")
    vals[inside] = cal[pos[inside]]
    out[ok] = vals
    return pd.Series(out, index=anchors.index)


def availability_dates(
    quarters: pd.DataFrame, calendar: pd.DatetimeIndex, spec: CompustatSpec
) -> pd.DataFrame:
    """Add ``anchor``, ``anchor_rule``, ``avail`` and ``avail_rdq`` (module docstring).

    ``quarters`` needs ``rdq``, ``first_filing`` and ``datadate``.
    """
    q = quarters.copy()
    rdq, filing, end = q["rdq"], q["first_filing"], q["datadate"]
    later = pd.concat([rdq, filing], axis=1).max(axis=1)
    fallback = end + pd.Timedelta(days=spec.fallback_lag_days)
    anchor = later.where(rdq.notna() | filing.notna(), fallback)
    q["anchor_rule"] = np.select(
        [rdq.notna() & filing.notna(), rdq.notna(), filing.notna(), end.notna()],
        ["later_of_rdq_and_filing", "rdq_only", "filing_only", "period_end_plus_lag"],
        default="none",
    )
    q["anchor"] = anchor
    lag = spec.availability_lag_trading_days
    q["avail"] = trading_days_after(anchor, calendar, lag)
    q["avail_rdq"] = trading_days_after(rdq, calendar, lag).where(rdq.notna(), q["avail"])
    return q


def keyset_count(period: pd.DataFrame, keyset: int, gvkeys: set[str] | None = None,
                 since: pd.Timestamp | None = None) -> int:
    """Distinct firm-quarters (``KYGVKEY``, ``FYYYYQ``) with a row in ``keyset``."""
    p = period[period["KEYSET"] == keyset]
    if gvkeys is not None:
        p = p[p["KYGVKEY"].isin(gvkeys)]
    if since is not None and "datadate" in p.columns:
        p = p[p["datadate"] >= since]
    return int(p[["KYGVKEY", "FYYYYQ"]].drop_duplicates().shape[0])


def quarterly_fundamentals(
    ex: CompustatExtract, spec: CompustatSpec, calendar: pd.DatetimeIndex
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One row per (``KYGVKEY``, ``FYYYYQ``) of ``spec.keyset``: the items,
    the quarterly cash flows, ``datadate``, ``rdq``, ``first_filing``,
    ``avail``, ``avail_rdq`` and ``fq_index`` (fiscal quarters counted, so
    consecutive quarters differ by 1). Plus an aggregate report."""
    period = ex.period[ex.period["KEYSET"] == spec.keyset].copy()
    period = period.drop_duplicates(["KYGVKEY", "FYYYYQ"])
    dates, mapping = period_end_dates(period, ex.fiscal_market)
    period["datadate"] = dates.to_numpy()
    period["rdq"] = parse_dates(period["RDQ"], fmt="%Y%m%d")
    keys = ["KYGVKEY", "KEYSET", "FYYYYQ"]
    q = period
    for table in (ex.income, ex.balance, ex.cashflow):
        t = table[table["KEYSET"] == spec.keyset].drop_duplicates(keys)
        q = q.merge(t, on=keys, how="left")
    q = ytd_to_quarterly(q, CASHFLOW_YTD_ITEMS)
    filings = first_filing_dates(ex.filings, spec)
    q = q.merge(filings, left_on=["KYGVKEY", "datadate"],
                right_on=["KYGVKEY", "period_end"], how="left").drop(columns="period_end")
    q = availability_dates(q, calendar, spec)
    q["fq_index"] = q["FYEARQ"] * 4 + q["FQTR"] - 1
    report = {
        "period_end_mapping": mapping,
        "firm_quarters": int(len(q)),
        "anchor_rules": {str(k): int(v) for k, v in q["anchor_rule"].value_counts().items()},
        "with_first_filing": int(q["first_filing"].notna().sum()),
        "filing_srctypes": {
            str(k): int(v)
            for k, v in ex.filings["SRCTYPE"].astype("string").value_counts().items()
        },
    }
    return q.sort_values(["KYGVKEY", "fq_index"]).reset_index(drop=True), report


def pit_events(
    quarters: pd.DataFrame, avail_col: str = "avail"
) -> Iterator[tuple[pd.Timestamp, pd.DataFrame]]:
    """For one GVKEY's quarters: each availability date, with every quarter
    known by then (``avail_col <= date``), in fiscal order.

    The building block of point-in-time characteristics: a value computed at
    an event from the quarters known then is what the market could compute
    that day. A quarter that arrives late joins the known set on its own
    date, never earlier.
    """
    q = quarters.dropna(subset=[avail_col]).sort_values("fq_index")
    for date in sorted(q[avail_col].unique()):
        yield pd.Timestamp(date), q[q[avail_col] <= date]


def carry_forward(
    events: pd.DataFrame,
    dates: pd.DatetimeIndex,
    max_staleness_days: int,
    *,
    event_col: str = "event",
    age_col: str = "asof_avail",
) -> pd.DataFrame:
    """Daily values from event rows, carried forward with a staleness cap.

    ``events`` has one row per event (sorted by ``event_col``): the values
    known from that date on, and ``age_col``, the availability date of the
    newest quarter they rest on. On each of ``dates`` the latest event on or
    before it applies, unless the date is more than ``max_staleness_days``
    calendar days after ``age_col``: then every value is missing. Never uses
    an event after the date.
    """
    value_cols = [c for c in events.columns if c not in (event_col, age_col)]
    if events.empty:
        return pd.DataFrame(np.nan, index=dates, columns=value_cols)
    ev = events.sort_values(event_col)
    pos = np.searchsorted(ev[event_col].to_numpy(dtype="datetime64[ns]"),
                          dates.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    out = pd.DataFrame(np.nan, index=dates, columns=value_cols)
    has = pos >= 0
    if has.any():
        rows = ev.iloc[pos[has]]
        age = (dates[has] - pd.DatetimeIndex(rows[age_col])).days.to_numpy()
        fresh = age <= max_staleness_days
        vals = rows[value_cols].to_numpy(dtype=float, copy=True)
        vals[~fresh] = np.nan
        out.iloc[np.flatnonzero(has)] = vals
    return out


# --------------------------------------------------------------------------- #
# Extraction (resumable) and loading
# --------------------------------------------------------------------------- #


@dataclass
class CompustatExtract:
    """The loaded Compustat extract: the tables the fundamentals are built from."""

    info: dict[str, Any]
    links: pd.DataFrame
    gics: pd.DataFrame
    period: pd.DataFrame
    income: pd.DataFrame
    balance: pd.DataFrame
    cashflow: pd.DataFrame
    filings: pd.DataFrame
    fiscal_market: pd.DataFrame | None = field(default=None)


_STREAMED: dict[str, str] = {
    "perioddescriptorquarterly": "period",
    "incomestatementquarterly": "income",
    "balancesheetquarterly": "balance",
    "cashflowyeartodate": "cashflow",
    "filingdates": "filings",
    "fiscalmarketdataquarterly": "fiscal_market",
}


def _stream(spec: CompustatSpec, name: str, out_dir: Path, gvkeys: list[str],
            verbose: bool) -> dict[str, Any]:
    wanted, optional = FILES[name]
    types, rename = _resolve(_header(spec, name), wanted, optional, name)
    key_col = next(h for h, c in rename.items() if c == "KYGVKEY")
    state = _stream_extract(spec, name, out_dir, types, gvkeys, None, None, verbose,
                            key_col=key_col, date_col=None)
    _write_json(out_dir / "columns.json", {"rename": rename, "types": types})
    return state


def _load_stream(out_dir: Path) -> pd.DataFrame:
    meta = json.loads((out_dir / "columns.json").read_text())
    df = _read_parts(out_dir, meta["types"])
    return df.rename(columns=meta["rename"])


def extract_compustat(
    spec: CompustatSpec,
    permnos: list[int],
    *,
    verbose: bool = True,
    period_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> Path:
    """Write the Compustat extract for the GVKEYs linked to ``permnos``.

    ``permnos`` are the universe's (and their companies') PERMNOs, from the
    CRSP extract. A GVKEY is kept when a link of ``link_types``/``link_prims``
    to one of them is in force at some point from :func:`fundamentals_start`
    to ``end``. Writes, inside ``Data/derived/compustat_<release>_<window>/``,
    the link and GICS rows of those GVKEYs and their rows of every quarterly
    file (streamed, resumable, every keyset kept so the keyset-8 count can
    be taken), plus ``extract.json``: provenance, the history start, and the
    aggregate report (link ambiguities, the period-end mapping check, the
    keyset-8 firm-quarters, filing types). ``period_filter`` is for tests.
    """
    spec.validate()
    root = spec.extract_dir
    root.mkdir(parents=True, exist_ok=True)
    since, end = fundamentals_start(spec), pd.Timestamp(spec.end)
    links = read_links(spec)
    v = valid_links(links, spec)
    v = v[v["LPERMNO"].isin(permnos) & (v["LINKDT"] <= end) & (v["LINKENDDT"] >= since)]
    gvkeys = sorted(set(v["KYGVKEY"]))
    if verbose:
        print(f"[compustat] {len(gvkeys)} GVKEYs linked to {len(set(permnos))} PERMNOs "
              f"from {since.date()} to {spec.end}")
    raw_links = links[links["KYGVKEY"].isin(gvkeys)]
    _parquet(raw_links, root / "links.parquet")
    gics = read_gics(spec)
    _parquet(gics[gics["KYGVKEY"].isin(gvkeys)], root / "gics.parquet")
    streams = {}
    for name, sub in _STREAMED.items():
        state = _stream(spec, name, root / sub, gvkeys, verbose)
        streams[sub] = {"file": name, "rows_read": state["rows_read"],
                        "rows_kept": state["rows_kept"]}

    ex = load_compustat_extract(spec, check=False)
    period = ex.period if period_filter is None else period_filter(ex.period)
    dates, mapping = period_end_dates(period, ex.fiscal_market)
    period = period.assign(datadate=dates.to_numpy())
    ambiguous = link_ambiguities(raw_links, spec)
    report = {
        "gvkeys": len(gvkeys),
        "history_start": str(since.date()),
        "link_ambiguities": int(len(ambiguous)),
        "link_ambiguous_permnos": int(ambiguous["LPERMNO"].nunique()) if len(ambiguous) else 0,
        "gics_lpermno_disagreements": gics_lpermno_disagreements(ex.gics, raw_links, spec),
        "period_end_mapping": mapping,
        "keyset_firm_quarters": {
            str(k): keyset_count(period, int(k), set(gvkeys), since)
            for k in sorted(period["KEYSET"].unique())
        },
        "restated_keyset": spec.restated_keyset,
        "restated_keyset_firm_quarters": keyset_count(
            period, spec.restated_keyset, set(gvkeys), since
        ),
        "filing_srctypes": {
            str(k): int(n) for k, n in ex.filings["SRCTYPE"].astype("string")
            .value_counts().items()
        },
    }
    _write_json(root / "extract.json", {
        "release": spec.release,
        "start": spec.start,
        "end": spec.end,
        "history_start": str(since.date()),
        "keyset": spec.keyset,
        "link_types": list(spec.link_types),
        "link_prims": list(spec.link_prims),
        "streams": streams,
        "report": report,
        "complete": True,
    })
    return root


def _parquet(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype).startswith("datetime64"):
            out[c] = out[c].astype("datetime64[ns]")
    out.to_parquet(path, index=False)


def load_compustat_extract(spec: CompustatSpec, *, check: bool = True) -> CompustatExtract:
    """Load the extract of ``spec``; refuse one built for another history start
    or link filter."""
    root = spec.extract_dir
    info_path = root / "extract.json"
    if check:
        if not info_path.exists():
            raise FileNotFoundError(
                f"no complete Compustat extract at {root}: run scripts/extract_compustat.py"
            )
        info = json.loads(info_path.read_text())
        want = {
            "history_start": str(fundamentals_start(spec).date()),
            "keyset": spec.keyset,
            "link_types": list(spec.link_types),
            "link_prims": list(spec.link_prims),
        }
        diff = {k: (info.get(k), v) for k, v in want.items() if info.get(k) != v}
        if diff or not info.get("complete"):
            raise ValueError(
                f"the Compustat extract at {root} does not match the spec or is "
                f"incomplete: {diff or 'incomplete'}"
            )
    else:
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
    links = _typed_links(pd.read_parquet(root / "links.parquet"))
    gics = _typed_gics(pd.read_parquet(root / "gics.parquet"))
    tables = {sub: _load_stream(root / sub) for sub in _STREAMED.values()}
    return CompustatExtract(
        info=info, links=links, gics=gics, period=tables["period"],
        income=tables["income"], balance=tables["balance"], cashflow=tables["cashflow"],
        filings=tables["filings"], fiscal_market=tables["fiscal_market"],
    )

