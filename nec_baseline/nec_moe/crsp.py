"""CRSP daily data (CIZ format): the source of the Stage B panel (brief 06 section A).

LICENCE — read before touching anything this module reads or writes
-------------------------------------------------------------------
CRSP is licensed to the university and redistribution is prohibited.
``Quant Model/Data/`` is gitignored. Everything derived from it (extracts,
parquet files, built panels, coverage tables) is written **inside**
``Data/derived/`` and nowhere else: never ``data_cache/``, ``results/``,
``figs/`` or ``Master Thesis/``. Tests use CIZ-format fixtures with the real
headers and invented numbers. Committed reports quote aggregate statistics
only: counts, date ranges, means, rates.

Source files (release ``ciz202512``, pipe-delimited with a header row)
-----------------------------------------------------------------------
Read from the extracted copy ``Data/crspdata/<release>_ascii/<file>.dat`` when
it exists, else streamed from ``Data/<release>_ascii.zip``; the archive is
never unzipped whole. Only the smaller primary stock file is used
(``StkDlySecurityPrimaryData``); every column this module needs is in it or in
the files below, so the 42 GB ``StkDlySecurityData`` is not read.

- ``StkDlySecurityPrimaryData``: ``DlyRet``, ``DlyPrc``, ``DlyVol`` and flags;
- ``StkDlyCumulativeAdjFactor``: ``DlyCumFacShr`` for split-invariant volume;
- ``StkDelists``: delisting records (for reporting; see rule 2);
- ``IndDlySeriesData``: ``DlyTotRet`` of the market index;
- ``StkIndMembership``: index membership spells;
- ``StkSecurityInfoHist``: tickers and ``PERMCO``, for report labels only.

Construction rules (brief 06 A.4), checked against the release's metadata
--------------------------------------------------------------------------
1. **Returns.** The daily log return is ``log(1 + DlyRet)``; ``DlyRet`` is the
   total return, dividends included. A return without a value is missing,
   never zero, and invalidates every row whose features or target need it.
   ``DlyRetMissFlg`` carries the reason (flag type ``RM``; ``NA`` means not
   missing). On the real extract an ordinary row has a value exactly when
   its flag is ``NA`` (the missing ones are ``NT``, ``NS``, ``MP``, ``RA``);
   the only rows with a value under another flag are two delisting rows
   flagged ``MV`` ("missing corporate action value"), whose value is CRSP's
   own delisting return, ``DelRet``, and is used.
2. **Delisting returns: already in ``DlyRet``.** ``MetaSIZtoCIZ`` maps the
   legacy ``DLRET`` to both ``StkDelists.DelRet`` and ``DlyRet``, and on the
   real data every delisting dated 2015-2024 has a stock row on ``DelDlyDt``
   flagged ``DlyDelFlg = "Y"`` whose ``DlyRet`` equals ``DelRet`` (5,376 of
   5,376 with a ``DelRet``, largest difference 1e-16; the 142 without one are
   missing in both, under the same missing codes). So nothing is compounded
   in: the delisting row is kept as the stock's final return, and it is never
   a panel row (it is the day after the last trade, with zero volume).
3. **Price.** ``DlyPrc`` is "the closing trade or the average of the closing
   bid and ask" (flag type ``PC``; ``BA`` marks the bid/ask average). Its
   absolute value is used, which guards against the legacy negative-price
   convention for bid/ask averages. Price only enters dollar volume.
4. **Drawdown** compounds ``DlyRet`` inside its own 60-day window
   (:func:`nec_moe.features.stock_features`); no price level is an input.
5. **Dollar volume** is ``|DlyPrc| x DlyVol`` of the same day, the dollars
   actually traded; no adjustment.
6. **Share volume across splits.** ``DlyCumFacShr`` is anchored at the end of
   the sample (it is 1 on every security's last row), so only its ratios
   inside a trailing window are used: shares times the factor is continuous
   across a split, checked on the real file.
7. **Market series.** ``log(1 + DlyTotRet)`` of ``CRSPSpec.market_indno``
   feeds market features, the rolling beta and the residual target, and its
   dates are the trading calendar.
8. **Universe.** A PERMNO is a member on date t iff a spell of
   ``CRSPSpec.membership_indno`` covers t, bounds inclusive
   (:class:`nec_moe.universe.SpellUniverse`). Features may use history from
   before the stock joined; the point-in-time filter and its re-rank (audit
   D-1) run exactly as before. Dual-class companies keep both share classes.
9. **Labels.** Entities are PERMNOs; :class:`TickerLookup` gives the ticker
   on a date, for reports only. Nothing keys on tickers.

Post-delisting fill (decided 2026-09-26)
----------------------------------------
A stock that delists inside a row's forward window has no returns after its
delisting row. Leaving that target missing would drop the row because of a
future event, which is look-ahead selection, and it would lose the delisting
return for all but one date. So a forward window that runs past a stock's
final CRSP return is **completed** with the post-delisting return of
``CRSPSpec.post_delisting_return`` and the row is never dropped for that
reason. The residual target is computed on the completed window. A delisting
row whose own return is missing stays missing: nothing is imputed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Literal

import numpy as np
import pandas as pd

from .data import Panel
from .features import (
    StageBSpec,
    _valid_rows,
    assemble_panel,
    feature_warmup,
    market_frame,
    post_fill_used,
    stock_features,
)
from .universe import SpellUniverse, filter_point_in_time

__all__ = [
    "REPO_DATA_DIR",
    "CRSPSpec",
    "CRSPExtract",
    "CRSPBuild",
    "TickerLookup",
    "crsp_file_source",
    "open_crsp_file",
    "read_membership",
    "read_index_returns",
    "read_delists",
    "read_security_info",
    "membership_universe",
    "flag_meanings",
    "extract_crsp",
    "extract_return_duration_flags",
    "load_extract",
    "missing_return_split",
    "crsp_daily_frames",
    "build_crsp_panel",
]

#: ``Quant Model/Data/``: the licensed files, gitignored. Derived files go to
#: its ``derived/`` subfolder only.
REPO_DATA_DIR = Path(__file__).resolve().parents[2] / "Data"

STOCK_COLUMNS: dict[str, str] = {
    "PERMNO": "int64",
    "DlyCalDt": "date32",
    "DlyDelFlg": "string",
    "DlyPrc": "float64",
    "DlyPrcFlg": "string",
    "DlyCap": "float64",
    "DlyCapFlg": "string",
    "DlyRet": "float64",
    "DlyRetx": "float64",
    "DlyRetMissFlg": "string",
    "DlyDistRetFlg": "string",
    "DlyVol": "float64",
}
ADJ_FACTOR_COLUMNS: dict[str, str] = {
    "PERMNO": "int64",
    "DlyCalDt": "date32",
    "DlyShrOut": "float64",
    "DlyCumFacPr": "float64",
    "DlyCumFacShr": "float64",
}
MEMBERSHIP_COLUMNS: dict[str, str] = {
    "PERMNO": "int64",
    "INDNO": "int64",
    "MbrStartDt": "date32",
    "MbrEndDt": "date32",
}
INDEX_COLUMNS: dict[str, str] = {
    "INDNO": "int64",
    "DlyCalDt": "date32",
    "DlyTotRet": "float64",
}
DELIST_COLUMNS: dict[str, str] = {
    "PERMNO": "int64",
    "DelistingDt": "date32",
    "DelActionType": "string",
    "DelStatusType": "string",
    "DelReasonType": "string",
    "DelPaymentType": "string",
    "DelRet": "float64",
    "DelRetMissType": "string",
    "DelDlyDt": "date32",
}
#: From the full daily security file, read only for the missing-return
#: diagnostic: ``DlyRetDurFlg`` (flag type ``RD``, how many trading days a
#: return spans) exists in ``StkDlySecurityData`` alone, not in the primary file.
RETURN_DURATION_COLUMNS: dict[str, str] = {
    "PERMNO": "int64",
    "DlyCalDt": "date32",
    "DlyRetMissFlg": "string",
    "DlyRetDurFlg": "string",
}
SECURITY_FILE = "StkDlySecurityData"
SECURITY_INFO_COLUMNS: dict[str, str] = {
    "PERMNO": "int64",
    "SecInfoStartDt": "date32",
    "SecInfoEndDt": "date32",
    "ShareClass": "string",
    "Ticker": "string",
    "PERMCO": "int64",
}

#: ``DlyRetMissFlg`` / ``DelRetMissType`` value meaning "not missing" (flag
#: type ``RM`` in ``MetaFlagInfo``).
RETURN_NOT_MISSING = "NA"
#: ``DlyDelFlg`` values (flag type ``YN``): the delisting-return row is "Y".
DELISTING_ROW, ORDINARY_ROW = "Y", "N"


@dataclass(frozen=True)
class CRSPSpec:
    """Where the CRSP files are and how the panel is built from them.

    ``market_indno`` is the market series, default ``1000500``, the "CRSP
    Value-Weighted Index of the S&P 500 Universe", total return
    (``DlyTotRet``). It is the like-for-like replacement of SPY for an S&P 500
    universe, not a new modelling choice. For a wider universe the
    alternative is ``1000200``, the CRSP NYSE/NYSE American/Nasdaq/Arca
    value-weighted market index.

    ``membership_indno`` is the index whose membership spells define the
    universe; ``universe="sp500"`` is the only value implemented. It is
    ``1000500``, not ``1000502``, on this evidence from ``ciz202512``
    (approved 2026-09-26; brief 06 A.3 erratum):

    - ``1000502`` ("S&P 500 Composite", family ``1100502``) has **0** rows in
      ``StkIndMembership``: it is S&P's index level series, with no
      constituents;
    - ``1000500`` (family ``1100500``, "CRSP Index of the S&P 500 Universe")
      has 2,084 membership spells over 1,956 PERMNOs, all flagged ``NORM``;
      ``1000501``, its equal-weighted twin, has identical spells;
    - its member count lies between 502 and 508 on every trading day of
      2015-2024 (median 505). The excess over 500 comes from companies with
      two share classes in the index, which are kept as two PERMNOs;
    - the bounds are inclusive: counting both bounds as inclusive, the
      per-date count changes 70 times over 2,516 trading days; treating
      either bound as exclusive, it changes 276 times, dipping by one on
      every index change. Of 239 spell starts in the window, 202 abut an end
      on the previous trading day (a replacement handed over overnight).

    ``member_count_min``/``member_count_max`` are the band the real-data test
    and the build check hold the per-date member count to: the observed 502
    to 508 with a margin of 2.

    ``post_delisting_return`` completes a forward window that runs past a
    stock's final CRSP return: ``"cash"`` fills those days with a 0 log
    return, ``"market"`` with the ``market_indno`` log return of the same
    days (module docstring). The default ``"cash"`` is **not a decision**; it
    is recorded in every trial so results under the two can't be mixed up.

    ``start``/``end`` are the panel window, unchanged from the free-data
    panel so the two are comparable; the window itself is part of the open
    question Q16. The extract reaches ``extract_lookback_days`` calendar days
    before ``start`` (550 days, about 378 trading days, cover the longest
    warm-up of the default ``StageBSpec``: a 250-day beta plus margin) and
    ``extract_lead_days`` after ``end`` (covers the forward horizon);
    :func:`build_crsp_panel` checks both against the actual calendar.
    ``chunk_bytes`` is the size of one streamed block of the large files, the
    unit the extraction resumes at.
    """

    crsp_dir: str = str(REPO_DATA_DIR)
    release: str = "ciz202512"
    stock_file: str = "StkDlySecurityPrimaryData"
    adj_factor_file: str = "StkDlyCumulativeAdjFactor"
    market_indno: int = 1000500
    universe: Literal["sp500"] = "sp500"
    membership_indno: int = 1000500
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    post_delisting_return: Literal["cash", "market"] = "cash"
    member_count_min: int = 500
    member_count_max: int = 510
    extract_lookback_days: int = 550
    extract_lead_days: int = 60
    chunk_bytes: int = 256 * 2**20

    def validate(self) -> CRSPSpec:
        if self.universe != "sp500":
            raise ValueError(f"universe={self.universe!r}: only 'sp500' is implemented")
        if self.post_delisting_return not in ("cash", "market"):
            raise ValueError(
                f"post_delisting_return={self.post_delisting_return!r}: "
                "must be 'cash' or 'market'"
            )
        if pd.Timestamp(self.end) < pd.Timestamp(self.start):
            raise ValueError(f"end {self.end} is before start {self.start}")
        if not 0 < self.member_count_min <= self.member_count_max:
            raise ValueError(
                f"member count band ({self.member_count_min}, "
                f"{self.member_count_max}) is not a valid band"
            )
        if self.extract_lookback_days < 0 or self.extract_lead_days < 0:
            raise ValueError("extract_lookback_days and extract_lead_days must be >= 0")
        if self.chunk_bytes < 1:
            raise ValueError("chunk_bytes must be >= 1")
        return self

    @property
    def data_source(self) -> str:
        """The provenance tag every trial on this panel carries (brief 06 A.6)."""
        return f"crsp_{self.release}"

    @property
    def derived_dir(self) -> Path:
        return Path(self.crsp_dir) / "derived"

    @property
    def extract_dir(self) -> Path:
        return self.derived_dir / f"crsp_extract_{self.release}_{self.start}_{self.end}"

    @property
    def panel_path(self) -> Path:
        return self.derived_dir / f"pit_panel_crsp_{self.start}_{self.end}.pt"

    @property
    def extract_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Calendar dates the extract covers: the window plus lookback and lead."""
        return (
            pd.Timestamp(self.start) - pd.Timedelta(days=self.extract_lookback_days),
            pd.Timestamp(self.end) + pd.Timedelta(days=self.extract_lead_days),
        )


# --------------------------------------------------------------------------- #
# File access
# --------------------------------------------------------------------------- #


def _arrow() -> Any:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.csv as pacsv
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the CRSP layer needs pyarrow: pip install -e '.[crsp]'"
        ) from exc
    return pa, pc, pacsv, pq


def crsp_file_source(spec: CRSPSpec, name: str) -> tuple[Path, str | None]:
    """``(path, None)`` for an extracted ``.dat`` file, else ``(zip, member)``."""
    plain = Path(spec.crsp_dir) / "crspdata" / f"{spec.release}_ascii" / f"{name}.dat"
    if plain.exists():
        return plain, None
    archive = Path(spec.crsp_dir) / f"{spec.release}_ascii.zip"
    if archive.exists():
        return archive, f"crspdata/{spec.release}_ascii/{name}.dat"
    raise FileNotFoundError(
        f"CRSP file {name}.dat of release {spec.release} not found: neither "
        f"{plain} nor {archive} exists (crsp_dir={spec.crsp_dir})"
    )


@contextlib.contextmanager
def open_crsp_file(spec: CRSPSpec, name: str) -> Iterator[IO[bytes]]:
    """A binary stream of one CRSP file, from the extracted copy or the zip."""
    path, member = crsp_file_source(spec, name)
    if member is None:
        with path.open("rb") as fh:
            yield fh
    else:
        with zipfile.ZipFile(path) as zf, zf.open(member) as fh:
            yield fh


def _file_size(spec: CRSPSpec, name: str) -> int:
    path, member = crsp_file_source(spec, name)
    if member is None:
        return path.stat().st_size
    with zipfile.ZipFile(path) as zf:
        return zf.getinfo(member).file_size


def _csv_options(column_types: dict[str, str], column_names: list[str] | None = None):
    pa, _, pacsv, _ = _arrow()
    types = {
        c: pa.date32() if t == "date32" else pa.string() if t == "string" else pa.type_for_alias(t)
        for c, t in column_types.items()
    }
    read = pacsv.ReadOptions(column_names=column_names) if column_names else pacsv.ReadOptions()
    parse = pacsv.ParseOptions(delimiter="|", quote_char=False)
    convert = pacsv.ConvertOptions(
        column_types=types,
        include_columns=list(column_types),
        null_values=[""],  # CRSP writes a missing value as an empty field
        strings_can_be_null=False,  # flag values such as "NA" are values, not nulls
    )
    return read, parse, convert


def _read_table(spec: CRSPSpec, name: str, column_types: dict[str, str]) -> pd.DataFrame:
    """Read a whole (small) CRSP file, only the listed columns, typed."""
    _, _, pacsv, _ = _arrow()
    read, parse, convert = _csv_options(column_types)
    with open_crsp_file(spec, name) as fh:
        table = pacsv.read_csv(fh, read_options=read, parse_options=parse, convert_options=convert)
    return _to_pandas(table)


def _to_pandas(table: Any) -> pd.DataFrame:
    df = table.to_pandas(date_as_object=False)
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime64"):
            df[c] = df[c].astype("datetime64[ns]")
    return df


def read_membership(spec: CRSPSpec) -> pd.DataFrame:
    """Membership spells of ``spec.membership_indno``: PERMNO, MbrStartDt, MbrEndDt."""
    df = _read_table(spec, "StkIndMembership", MEMBERSHIP_COLUMNS)
    df = df[df["INDNO"] == spec.membership_indno].drop(columns="INDNO")
    if df.empty:
        raise ValueError(
            f"no membership rows under INDNO {spec.membership_indno} in "
            f"StkIndMembership ({spec.release}); check membership_indno"
        )
    return df.sort_values(["PERMNO", "MbrStartDt"]).reset_index(drop=True)


def read_index_returns(spec: CRSPSpec, indno: int) -> pd.Series:
    """``DlyTotRet`` of one index series, indexed by date."""
    df = _read_table(spec, "IndDlySeriesData", INDEX_COLUMNS)
    df = df[df["INDNO"] == indno]
    if df.empty:
        raise ValueError(f"no rows for INDNO {indno} in IndDlySeriesData ({spec.release})")
    s = df.set_index("DlyCalDt")["DlyTotRet"].sort_index()
    s.index.name = "date"
    return s.rename(f"indno_{indno}")


def read_delists(spec: CRSPSpec) -> pd.DataFrame:
    return _read_table(spec, "StkDelists", DELIST_COLUMNS)


def read_security_info(spec: CRSPSpec) -> pd.DataFrame:
    return _read_table(spec, "StkSecurityInfoHist", SECURITY_INFO_COLUMNS)


def flag_meanings(spec: CRSPSpec, flag_type: str) -> dict[str, str]:
    """Code -> description of one flag type (e.g. ``RM``, ``RD``), from ``MetaFlagInfo``."""
    df = _read_table(spec, "MetaFlagInfo", {
        "FlagType": "string", "FlagValue": "string", "FlagDesc": "string",
    })
    rows = df[df["FlagType"] == flag_type]
    return dict(zip(rows["FlagValue"], rows["FlagDesc"], strict=True))


def membership_universe(spells: pd.DataFrame) -> SpellUniverse:
    """The spells as a :class:`SpellUniverse` keyed by PERMNO strings."""
    return SpellUniverse(
        pd.DataFrame(
            {
                "entity": spells["PERMNO"].astype("int64").astype(str),
                "start": spells["MbrStartDt"],
                "end": spells["MbrEndDt"],
            }
        )
    )


# --------------------------------------------------------------------------- #
# Extraction (resumable, streamed once per large file)
# --------------------------------------------------------------------------- #


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)  # atomic: a crash never leaves a half-written manifest


def _stream_extract(
    spec: CRSPSpec,
    name: str,
    out_dir: Path,
    column_types: dict[str, str],
    permnos: list[int],
    lo: pd.Timestamp,
    hi: pd.Timestamp,
    verbose: bool,
) -> dict[str, Any]:
    """Keep the rows of ``name`` for ``permnos`` dated in ``[lo, hi]``.

    The file is read in blocks of ``spec.chunk_bytes`` cut at a line end; each
    block's kept rows go to their own parquet part, and ``manifest.json``
    records the byte offset reached after every block. A rerun with the same
    inputs continues from that offset; changed inputs start over.
    """
    pa, pc, pacsv, pq = _arrow()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    size = _file_size(spec, name)
    key = hashlib.sha256(
        json.dumps(
            [spec.release, name, permnos, str(lo.date()), str(hi.date()), spec.chunk_bytes]
        ).encode()
    ).hexdigest()
    state: dict[str, Any] = {}
    if manifest_path.exists():
        state = json.loads(manifest_path.read_text())
        if state.get("key") != key or state.get("source_size") != size:
            state = {}
    if state.get("complete"):
        return state
    if not state:
        for old in out_dir.glob("part-*.parquet"):
            old.unlink()
        state = {"file": name, "key": key, "source_size": size, "offset": 0,
                 "parts": 0, "rows_read": 0, "rows_kept": 0, "complete": False}

    value_set = pa.array(permnos, type=pa.int64())
    lo32 = pa.scalar(lo.date(), type=pa.date32())
    hi32 = pa.scalar(hi.date(), type=pa.date32())
    with open_crsp_file(spec, name) as fh:
        header = fh.readline()
        names = header.decode().rstrip("\r\n").split("|")
        missing = set(column_types) - set(names)
        if missing:
            raise ValueError(f"{name}: header lacks columns {sorted(missing)}")
        read, parse, convert = _csv_options(column_types, names)
        if state["offset"] > len(header):
            fh.seek(state["offset"])
        else:
            state["offset"] = len(header)
        while True:
            block = fh.read(spec.chunk_bytes)
            if not block:
                break
            if not block.endswith(b"\n"):
                block += fh.readline()
            table = pacsv.read_csv(
                pa.BufferReader(block), read_options=read,
                parse_options=parse, convert_options=convert,
            )
            mask = pc.and_(
                pc.is_in(table["PERMNO"], value_set=value_set),
                pc.and_(
                    pc.greater_equal(table["DlyCalDt"], lo32),
                    pc.less_equal(table["DlyCalDt"], hi32),
                ),
            )
            kept = table.filter(mask)
            if kept.num_rows:
                pq.write_table(kept, out_dir / f"part-{state['parts']:05d}.parquet")
                state["parts"] += 1
            state["offset"] += len(block)
            state["rows_read"] += table.num_rows
            state["rows_kept"] += kept.num_rows
            _write_json(manifest_path, state)
            if verbose:
                print(f"[extract] {name}: {state['offset'] / size:6.1%} read, "
                      f"{state['rows_kept']} rows kept")
    state["complete"] = True
    _write_json(manifest_path, state)
    return state


def extract_crsp(spec: CRSPSpec, *, verbose: bool = True) -> Path:
    """Write the columnar extract the panel is built from, inside ``Data/derived/``.

    Keeps the PERMNOs that were members of ``membership_indno`` at any point in
    ``[start, end]``, from ``start - extract_lookback_days`` to ``end +
    extract_lead_days``: their stock rows and cumulative adjustment factors
    (streamed once, resumable), their membership spells, delisting records and
    security-info history, and the ``market_indno`` series. ``extract.json``
    records the provenance. Returns the extract directory.
    """
    spec.validate()
    root = spec.extract_dir
    root.mkdir(parents=True, exist_ok=True)
    lo, hi = spec.extract_bounds
    spells = read_membership(spec)
    universe = membership_universe(spells)
    members = sorted(int(p) for p in universe.members_union(spec.start, spec.end))
    if verbose:
        print(f"[extract] {len(members)} PERMNOs were members of INDNO "
              f"{spec.membership_indno} in {spec.start}..{spec.end}")
    spells[spells["PERMNO"].isin(members)].to_parquet(root / "membership.parquet", index=False)
    market = read_index_returns(spec, spec.market_indno)
    market = market[(market.index >= lo) & (market.index <= hi)]
    market.to_frame("DlyTotRet").to_parquet(root / "market.parquet")
    delists = read_delists(spec)
    delists[delists["PERMNO"].isin(members)].to_parquet(root / "delists.parquet", index=False)
    info = read_security_info(spec)
    info[info["PERMNO"].isin(members)].to_parquet(root / "security_info.parquet", index=False)

    streams = {}
    for name, sub, types in (
        (spec.stock_file, "stock", STOCK_COLUMNS),
        (spec.adj_factor_file, "adjfac", ADJ_FACTOR_COLUMNS),
    ):
        state = _stream_extract(spec, name, root / sub, types, members, lo, hi, verbose)
        streams[sub] = {"file": name, "rows_read": state["rows_read"],
                        "rows_kept": state["rows_kept"]}
    _write_json(
        root / "extract.json",
        {
            "release": spec.release,
            "data_source": spec.data_source,
            "market_indno": spec.market_indno,
            "membership_indno": spec.membership_indno,
            "universe": spec.universe,
            "start": spec.start,
            "end": spec.end,
            "extract_lo": str(lo.date()),
            "extract_hi": str(hi.date()),
            "n_permnos": len(members),
            "streams": streams,
            "complete": True,
        },
    )
    return root


def extract_return_duration_flags(spec: CRSPSpec, *, verbose: bool = True) -> Path:
    """Add ``DlyRetDurFlg`` for the extract's PERMNOs and dates, for diagnostics only.

    The flag says how many trading days a return spans (``D1``..``DU``: one
    trading day; ``P1``..``P9``: 2 to 10 trading days; ``MR``: missing). It is
    only in the 42 GB ``StkDlySecurityData``, so that file is streamed once,
    resumably, keeping four columns (``RETURN_DURATION_COLUMNS``) of the same
    PERMNOs and date bounds as the main extract, into ``retdur/``. The panel
    never reads it; the missing-return report does when it is present.
    """
    spec.validate()
    root = spec.extract_dir
    info_path = root / "extract.json"
    if not info_path.exists():
        raise FileNotFoundError(f"no CRSP extract at {root}: run extract_crsp first")
    info = json.loads(info_path.read_text())
    spells = pd.read_parquet(root / "membership.parquet")
    members = sorted(int(p) for p in spells["PERMNO"].unique())  # the main extract's PERMNOs
    lo, hi = pd.Timestamp(info["extract_lo"]), pd.Timestamp(info["extract_hi"])
    _stream_extract(spec, SECURITY_FILE, root / "retdur", RETURN_DURATION_COLUMNS,
                    members, lo, hi, verbose)
    return root / "retdur"


@dataclass
class CRSPExtract:
    """The loaded extract: everything the panel build reads.

    ``duration_flags`` (``DlyRetDurFlg`` by PERMNO and date) is present only
    after :func:`extract_return_duration_flags`; only the missing-return
    diagnostic uses it.
    """

    info: dict[str, Any]
    stock: pd.DataFrame
    adj: pd.DataFrame
    market: pd.Series  # DlyTotRet by date
    spells: pd.DataFrame
    delists: pd.DataFrame
    security_info: pd.DataFrame
    duration_flags: pd.DataFrame | None = None


def _read_parts(folder: Path, column_types: dict[str, str]) -> pd.DataFrame:
    pa, _, _, pq = _arrow()
    parts = sorted(folder.glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in column_types})
    return _to_pandas(pa.concat_tables([pq.read_table(p) for p in parts]))


def load_extract(spec: CRSPSpec) -> CRSPExtract:
    root = spec.extract_dir
    info_path = root / "extract.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"no complete CRSP extract at {root}: run scripts/extract_crsp.py first"
        )
    info = json.loads(info_path.read_text())
    if not info.get("complete"):
        raise RuntimeError(f"the extract at {root} is incomplete: rerun scripts/extract_crsp.py")
    market = pd.read_parquet(root / "market.parquet")["DlyTotRet"]
    market.index = pd.DatetimeIndex(market.index).astype("datetime64[ns]")
    retdur = root / "retdur" / "manifest.json"
    durations = (
        _read_parts(root / "retdur", RETURN_DURATION_COLUMNS)
        if retdur.exists() and json.loads(retdur.read_text()).get("complete")
        else None
    )
    return CRSPExtract(
        info=info,
        stock=_read_parts(root / "stock", STOCK_COLUMNS),
        adj=_read_parts(root / "adjfac", ADJ_FACTOR_COLUMNS),
        market=market.sort_index(),
        spells=pd.read_parquet(root / "membership.parquet"),
        delists=pd.read_parquet(root / "delists.parquet"),
        security_info=pd.read_parquet(root / "security_info.parquet"),
        duration_flags=durations,
    )


# --------------------------------------------------------------------------- #
# Daily frames and the panel
# --------------------------------------------------------------------------- #


class TickerLookup:
    """Date-aware PERMNO -> ticker, from ``StkSecurityInfoHist``. Report labels only."""

    def __init__(self, security_info: pd.DataFrame) -> None:
        self._by_permno = {
            int(p): g.sort_values("SecInfoStartDt")
            for p, g in security_info.groupby("PERMNO")
        }

    def __call__(self, permno: int | str, date: str | pd.Timestamp) -> str | None:
        g = self._by_permno.get(int(permno))
        if g is None:
            return None
        d = pd.Timestamp(date)
        hit = g[(g["SecInfoStartDt"] <= d) & (d <= g["SecInfoEndDt"])]
        if hit.empty:
            return None
        ticker = str(hit["Ticker"].iloc[-1])
        return ticker or None


def crsp_daily_frames(
    extract: CRSPExtract,
    spec: CRSPSpec,
    stage_b: StageBSpec,
    permnos: list[int] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.Series, dict[str, Any]]:
    """Per-PERMNO daily frames (:data:`nec_moe.features.DAILY_COLUMNS`) on the
    market calendar, the market log return, and aggregate counts.

    Each stock runs from its first to its last extracted row on the calendar
    of ``market_indno``; a calendar day without a row has a missing return. A
    stock whose last row is its delisting row (``DlyDelFlg = "Y"``) gets the
    next ``stage_b.horizon`` calendar days appended as post-delisting fill
    days: not tradable, no return of their own, ``fill_ret`` from
    ``spec.post_delisting_return``.
    """
    mkt_ret = np.log1p(extract.market.astype(float))
    calendar = pd.DatetimeIndex(mkt_ret.index)
    stock = extract.stock
    if permnos is not None:
        stock = stock[stock["PERMNO"].isin(permnos)]
    stock = stock.merge(
        extract.adj[["PERMNO", "DlyCalDt", "DlyCumFacShr"]],
        on=["PERMNO", "DlyCalDt"], how="left",
    )
    counts: dict[str, Any] = {
        "stock_rows": int(len(stock)),
        "rows_off_calendar": int((~stock["DlyCalDt"].isin(calendar)).sum()),
        "missing_returns": int(stock["DlyRet"].isna().sum()),
        "returns_with_missing_flag": int(
            (stock["DlyRet"].notna() & (stock["DlyRetMissFlg"] != RETURN_NOT_MISSING)).sum()
        ),
        "returns_at_or_below_minus_one": int((stock["DlyRet"] <= -1.0).sum()),
        "negative_prices": int((stock["DlyPrc"] < 0).sum()),
        "missing_share_factor": int(stock["DlyCumFacShr"].isna().sum()),
        "delisting_rows": int((stock["DlyDelFlg"] == DELISTING_ROW).sum()),
        "delisting_rows_not_last": 0,
        "stocks_filled_after_delisting": 0,
    }
    counts["price_flags"] = {
        str(k): int(v) for k, v in stock["DlyPrcFlg"].value_counts().items()
    }
    stock = stock[stock["DlyCalDt"].isin(calendar)]

    frames: dict[str, pd.DataFrame] = {}
    for permno, g in stock.groupby("PERMNO", sort=True):
        g = g.sort_values("DlyCalDt").set_index("DlyCalDt")
        idx = calendar[(calendar >= g.index[0]) & (calendar <= g.index[-1])]
        g = g.reindex(idx)
        present = g["PERMNO"].notna()
        flag = g["DlyDelFlg"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.log1p(g["DlyRet"].astype(float))
        daily = pd.DataFrame(
            {
                "ret": ret.where(np.isfinite(ret)),
                "volume": g["DlyVol"].astype(float),
                "dollar_volume": g["DlyPrc"].astype(float).abs() * g["DlyVol"].astype(float),
                "share_factor": g["DlyCumFacShr"].astype(float),
                "tradable": (present & (flag == ORDINARY_ROW)).astype(bool),
                "fill_ret": np.nan,
            },
            index=idx,
        )
        is_delisting = (flag == DELISTING_ROW).to_numpy()
        counts["delisting_rows_not_last"] += int(is_delisting[:-1].sum())
        if is_delisting[-1]:
            after = calendar[calendar > idx[-1]][: stage_b.horizon]
            fill = (
                np.zeros(len(after))
                if spec.post_delisting_return == "cash"
                else mkt_ret.reindex(after).to_numpy(dtype=float)
            )
            tail = pd.DataFrame(
                {
                    "ret": np.nan, "volume": np.nan, "dollar_volume": np.nan,
                    "share_factor": np.nan, "tradable": False, "fill_ret": fill,
                },
                index=after,
            )
            daily = pd.concat([daily, tail])
            counts["stocks_filled_after_delisting"] += 1
        frames[str(int(permno))] = daily
    return frames, mkt_ret, counts


@dataclass
class CRSPBuild:
    """A built CRSP panel and its aggregate build report (no per-security values)."""

    panel: Panel
    report: dict[str, Any]
    coverage: pd.DataFrame = field(repr=False)


def _summary(x: np.ndarray) -> dict[str, float]:
    return {"min": float(np.min(x)), "median": float(np.median(x)), "max": float(np.max(x))}


def build_crsp_panel(
    spec: CRSPSpec,
    stage_b: StageBSpec | None = None,
    *,
    extract: CRSPExtract | None = None,
    verbose: bool = True,
) -> CRSPBuild:
    """Build the point-in-time CRSP panel for ``[spec.start, spec.end]``.

    Reads the extract (``extract`` may be a wider one, e.g. the default
    window's, for a sub-window build), builds every member's daily frame,
    assembles the panel over all PERMNOs that were ever members in the window,
    then keeps a row only on dates its PERMNO was a member, re-ranking the
    snapshot features among the members (audit D-1). Raises if the extract
    does not reach back far enough for the features' warm-up or forward far
    enough for the horizon, or if the member count leaves the configured band.
    """
    spec = spec.validate()
    stage_b = (stage_b if stage_b is not None else StageBSpec()).validate()
    extract = extract if extract is not None else load_extract(spec)
    start, end = pd.Timestamp(spec.start), pd.Timestamp(spec.end)

    calendar = pd.DatetimeIndex(extract.market.index)
    before = int((calendar < start).sum())
    after = int((calendar > end).sum())
    if before < feature_warmup(stage_b):
        raise ValueError(
            f"the extract has {before} trading days before {spec.start}; the "
            f"features need {feature_warmup(stage_b)}: raise extract_lookback_days"
        )
    if after < stage_b.horizon:
        raise ValueError(
            f"the extract has {after} trading days after {spec.end}; the "
            f"target needs {stage_b.horizon}: raise extract_lead_days"
        )
    window = calendar[(calendar >= start) & (calendar <= end)]

    universe = membership_universe(extract.spells)
    member_counts = universe.count_by_date(window)
    in_band = bool(
        (member_counts >= spec.member_count_min).all()
        and (member_counts <= spec.member_count_max).all()
    )
    if not in_band:
        raise ValueError(
            f"members per date range {member_counts.min()}..{member_counts.max()}, "
            f"outside the band {spec.member_count_min}..{spec.member_count_max}"
        )
    ever = sorted(int(p) for p in universe.members_union(start, end))

    frames, mkt_ret, counts = crsp_daily_frames(extract, spec, stage_b, ever)
    metadata = {
        "crsp_release": spec.release,
        "market_indno": spec.market_indno,
        "membership_indno": spec.membership_indno,
        "universe": spec.universe,
        "start": spec.start,
        "end": spec.end,
        "post_delisting_return": spec.post_delisting_return,
        "stage_b": dataclasses.asdict(stage_b),
    }
    mkt = market_frame(mkt_ret, stage_b)
    candidates = assemble_panel(
        frames, mkt, stage_b,
        data_source=spec.data_source, metadata=metadata, window=(spec.start, spec.end),
    )
    panel = filter_point_in_time(candidates, universe)

    # rows whose forward window used the post-delisting fill
    assert panel.date_labels is not None and panel.entity_labels is not None
    ent = np.asarray(panel.entity_labels)[panel.entity.numpy()]
    day = np.asarray(panel.date_labels)[panel.date.numpy()]
    filled = {
        (p, str(d.date()))
        for p, daily in frames.items()
        for d in daily.index[post_fill_used(daily, stage_b).to_numpy()]
    }
    fill_rows = int(sum((p, d) in filled for p, d in zip(ent, day, strict=True)))

    report = _build_report(
        spec, extract, universe, window, member_counts, panel, counts, fill_rows, ever
    )
    split = missing_return_split(extract, frames, mkt, stage_b, window)
    if split["totals"]["panel_rows"] != len(panel):
        raise AssertionError(
            f"missing-return split counted {split['totals']['panel_rows']} panel rows, "
            f"the panel has {len(panel)}: the diagnostic does not see the panel's rows"
        )
    report["missing_return_split"] = split
    panel.metadata["build"] = report
    coverage = _coverage_by_year(universe, window, panel)
    if verbose:
        print(f"[crsp] {report['rows']} rows, {report['dates']} dates, "
              f"{report['entities']} PERMNOs; members/date "
              f"{report['members_per_date']}; post-delisting fill touched "
              f"{fill_rows} rows")
    return CRSPBuild(panel=panel, report=report, coverage=coverage)


def _window_any(mask: np.ndarray, back: int, fwd: int) -> tuple[np.ndarray, np.ndarray]:
    """Per position t: any(mask[t-back+1 .. t]) and any(mask[t+1 .. t+fwd]),
    both truncated at the ends of the series."""
    n = len(mask)
    c = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    t = np.arange(n)
    backward = (c[t + 1] - c[np.maximum(t - back + 1, 0)]) > 0
    forward = (c[np.minimum(t + fwd, n - 1) + 1] - c[t + 1]) > 0
    return backward, forward


def missing_return_split(
    extract: CRSPExtract,
    frames: dict[str, pd.DataFrame],
    mkt: pd.DataFrame,
    stage_b: StageBSpec,
    window: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Which rows rule A.4.1 drops, and whether the missing return is past or future.

    A diagnostic only: it changes nothing in the panel. A candidate row is a
    member on a tradable day inside the window. It is **dropped by a missing
    return** if it is invalid as built but valid once every missing return of
    its stock is set to 0 (so rows lost to the warm-up, a missing volume or a
    thin date are counted apart, as ``other_invalid``). Those rows split by
    where the missing return sits relative to date t:

    - (a) only in the backward feature window, the ``feature_warmup`` returns
      through t (past information);
    - (b) only in the forward target window ``(t, t + horizon]`` (a future
      event: the row is lost because of what happens after t);
    - (c) in both;
    - (d) the forward window holds a delisting row whose own return is
      missing, counted on its own line whatever the backward window holds.

    Post-delisting fill days are not missing. For the missing returns behind
    (b), the report gives their ``DlyRetMissFlg`` codes (``"no CRSP row"``
    for a calendar day without a row) and, when the duration flags were
    extracted, the ``DlyRetDurFlg`` of the first non-missing return after
    each gap (a run of consecutive missing days): ``P1``..``P9`` there mean
    CRSP's next return already spans the missing days. Aggregate counts only.
    """
    back_n, fwd_n = feature_warmup(stage_b), stage_b.horizon
    start, end = window[0], window[-1]
    stock = extract.stock
    delisting_day = (
        stock.loc[stock["DlyDelFlg"] == DELISTING_ROW].set_index("PERMNO")["DlyCalDt"]
    )
    flags = stock.set_index(["PERMNO", "DlyCalDt"])["DlyRetMissFlg"]
    durations = None
    if extract.duration_flags is not None:
        durations = extract.duration_flags.set_index(["PERMNO", "DlyCalDt"])["DlyRetDurFlg"]
    spells = {int(k): g for k, g in extract.spells.groupby("PERMNO")}
    keys = ("a_backward_only", "b_forward_only", "c_both", "d_missing_delisting_return")
    totals = dict.fromkeys((*keys, "other_invalid", "unexplained", "panel_rows"), 0)
    by_year: dict[int, dict[str, int]] = {}
    behind_b: list[tuple[int, pd.Timestamp]] = []
    next_after_gap: list[tuple[int, pd.Timestamp] | None] = []

    for label, daily in frames.items():
        permno = int(label)
        idx = pd.DatetimeIndex(daily.index)
        fill = daily["fill_ret"].notna().to_numpy()
        miss = daily["ret"].isna().to_numpy() & ~fill
        on_delisting = idx == delisting_day.get(permno, pd.NaT)
        miss_ord, miss_del = miss & ~on_delisting, miss & on_delisting
        back_any, _ = _window_any(miss, back_n, fwd_n)
        _, fwd_ord = _window_any(miss_ord, back_n, fwd_n)
        _, fwd_del = _window_any(miss_del, back_n, fwd_n)
        tradable = daily["tradable"].to_numpy(dtype=bool)
        valid = _valid_rows(stock_features(daily, mkt, stage_b), stage_b,
                            daily["tradable"]).to_numpy()
        if miss.any():
            filled = daily.copy()
            filled.loc[miss, "ret"] = 0.0  # counterfactual only: never enters the panel
            valid_cf = _valid_rows(stock_features(filled, mkt, stage_b), stage_b,
                                   filled["tradable"]).to_numpy()
        else:
            valid_cf = valid
        sp = spells.get(permno)
        member = np.zeros(len(idx), dtype=bool)
        if sp is not None:
            for s0, e0 in zip(sp["MbrStartDt"], sp["MbrEndDt"], strict=True):
                member |= (idx >= s0) & (idx <= e0)
        candidate = member & tradable & (idx >= start) & (idx <= end)
        totals["panel_rows"] += int((candidate & valid).sum())
        dropped = candidate & ~valid
        totals["other_invalid"] += int((dropped & ~valid_cf).sum())
        by_missing = dropped & valid_cf
        cats = {
            "d_missing_delisting_return": by_missing & fwd_del,
            "a_backward_only": by_missing & ~fwd_del & back_any & ~fwd_ord,
            "b_forward_only": by_missing & ~fwd_del & fwd_ord & ~back_any,
            "c_both": by_missing & ~fwd_del & back_any & fwd_ord,
        }
        totals["unexplained"] += int((by_missing & ~fwd_del & ~back_any & ~fwd_ord).sum())
        years = idx.year.to_numpy()
        for key, m in cats.items():
            totals[key] += int(m.sum())
            for y in np.unique(years[m]):
                row = by_year.setdefault(int(y), dict.fromkeys(keys, 0))
                row[key] += int((m & (years == y)).sum())

        # the missing returns behind (b), and the gaps they sit in
        behind = np.zeros(len(idx), dtype=bool)
        for t in np.flatnonzero(cats["b_forward_only"]):
            behind[t + 1 : t + 1 + fwd_n] |= miss_ord[t + 1 : t + 1 + fwd_n]
        behind_b += [(permno, idx[i]) for i in np.flatnonzero(behind)]
        i = 0
        while i < len(idx):
            if miss_ord[i]:
                j = i
                while j + 1 < len(idx) and miss_ord[j + 1]:
                    j += 1
                if behind[i : j + 1].any():
                    nxt = j + 1
                    ok = nxt < len(idx) and not miss[nxt] and not fill[nxt]
                    next_after_gap.append((permno, idx[nxt]) if ok else None)
                i = j + 1
            else:
                i += 1

    codes: dict[str, int] = {}
    for missing_day in behind_b:
        code = str(flags.get(missing_day, "no CRSP row"))
        codes[code] = codes.get(code, 0) + 1
    duration_counts: dict[str, int] | None = None
    if durations is not None:
        duration_counts = {}
        for next_day in next_after_gap:
            flag = (
                "no later return" if next_day is None
                else str(durations.get(next_day, "no row"))
            )
            duration_counts[flag] = duration_counts.get(flag, 0) + 1
    return {
        "backward_window_returns": back_n,
        "forward_window_returns": fwd_n,
        "totals": totals,
        "b_share_of_panel_rows": (
            totals["b_forward_only"] / totals["panel_rows"] if totals["panel_rows"] else 0.0
        ),
        "by_year": {str(y): by_year[y] for y in sorted(by_year)},
        "behind_b_missing_returns": len(behind_b),
        "behind_b_gaps": len(next_after_gap),
        "behind_b_missing_codes": dict(sorted(codes.items())),
        "next_return_duration_flags": (
            None if duration_counts is None else dict(sorted(duration_counts.items()))
        ),
    }


def _build_report(
    spec: CRSPSpec,
    extract: CRSPExtract,
    universe: SpellUniverse,
    window: pd.DatetimeIndex,
    member_counts: np.ndarray,
    panel: Panel,
    counts: dict[str, Any],
    fill_rows: int,
    ever: list[int],
) -> dict[str, Any]:
    """Aggregate statistics of one build: counts, ranges, rates. No per-security values."""
    start, end = window[0], window[-1]
    _, per_date = np.unique(panel.date.numpy(), return_counts=True)

    # delistings: PERMNOs that were members on their last trading day
    dl = extract.delists
    dl = dl[(dl["DelistingDt"] >= start) & (dl["DelistingDt"] <= end)]
    dl = dl[dl["PERMNO"].isin(ever)]
    sp = extract.spells.merge(dl[["PERMNO", "DelistingDt"]], on="PERMNO")
    covered = sp[(sp["MbrStartDt"] <= sp["DelistingDt"]) & (sp["DelistingDt"] <= sp["MbrEndDt"])]
    member_dl = dl[dl["PERMNO"].isin(covered["PERMNO"])]
    no_ret = member_dl[member_dl["DelRet"].isna()]

    # dual-class companies: one PERMCO with two or more member PERMNOs on a date
    info = extract.security_info
    permco = (
        info.sort_values("SecInfoEndDt").groupby("PERMNO")["PERMCO"].last()
        if len(info) else pd.Series(dtype="int64")
    )
    dual_by_date = []
    dual_companies: set[int] = set()
    for d in window:
        members = [int(p) for p in universe.members_asof(d)]
        cos = permco.reindex(members).dropna().astype(int)
        dup = cos[cos.duplicated(keep=False)]
        dual_companies |= set(dup.tolist())
        dual_by_date.append(dup.nunique())

    return {
        "release": spec.release,
        "data_source": spec.data_source,
        "market_indno": spec.market_indno,
        "membership_indno": spec.membership_indno,
        "window": [str(start.date()), str(end.date())],
        "post_delisting_return": spec.post_delisting_return,
        "rows": int(len(panel)),
        "dates": int(len(per_date)),
        "entities": int(np.unique(panel.entity.numpy()).size),
        "trading_days_in_window": int(len(window)),
        "members_per_date": _summary(member_counts),
        "member_count_band": [spec.member_count_min, spec.member_count_max],
        "panel_names_per_date": _summary(per_date),
        "ever_members": len(ever),
        "dual_class_companies": len(dual_companies),
        "dual_class_companies_per_date": _summary(np.asarray(dual_by_date)),
        "member_delistings": int(len(member_dl)),
        "member_delistings_by_action": {
            str(k): int(v) for k, v in member_dl["DelActionType"].value_counts().items()
        },
        "member_delistings_without_return": int(len(no_ret)),
        "without_return_codes": {
            f"{a}/{r}": int(n)
            for (a, r), n in no_ret.groupby(["DelActionType", "DelReasonType"]).size().items()
        },
        "post_delisting_fill_rows": fill_rows,
        "extract_counts": counts,
    }


def _coverage_by_year(
    universe: SpellUniverse, window: pd.DatetimeIndex, panel: Panel
) -> pd.DataFrame:
    """Per year: members, members with at least one panel row, and those without."""
    assert panel.date_labels is not None and panel.entity_labels is not None
    ent = np.asarray(panel.entity_labels)[panel.entity.numpy()]
    year = pd.DatetimeIndex(np.asarray(panel.date_labels)[panel.date.numpy()]).year
    rows = []
    for y in sorted(set(window.year)):
        days = window[window.year == y]
        members = universe.members_union(days[0], days[-1])
        with_rows = set(ent[year == y]) & members
        rows.append(
            {
                "year": int(y),
                "members": len(members),
                "with_rows": len(with_rows),
                "without_rows": len(members) - len(with_rows),
                "coverage": len(with_rows) / len(members) if members else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("year")
