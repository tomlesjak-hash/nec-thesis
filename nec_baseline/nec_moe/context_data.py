"""Stage C context data: VIX history and Kenneth French daily factors.

The syllabus's Stage C (§3): free regime/context series used for **diagnostics
and reporting only, never as a training signal** (Decision B — the gate stays
unsupervised; VIX enters only as the yardstick the learned regimes are compared
against, which is what keeps the thesis's secondary research question testable).

These are the only free downloads left in the pipeline. The price, return and
universe data now come from CRSP (:mod:`nec_moe.crsp`, brief 06); VIX and the
French factors stay because CRSP has no equivalent of either. VIX never
reaches the model. One exception to "diagnostics only": the Hamilton gate's
registered date-level series ``market_excess_return`` (brief 03 §2) is the
French daily Mkt-RF from :func:`load_french_factors`, so that series is the
gate's input. It is not a target and not an expert feature.

Sources, both official and free:

- **VIX** — CBOE's published history CSV
  (``https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv``,
  header ``DATE,OPEN,HIGH,LOW,CLOSE`` with MM/DD/YYYY dates).
- **Daily factors** — Kenneth French's data library zips
  (``F-F_Research_Data_Factors_daily_CSV.zip`` → Mkt-RF, SMB, HML, RF; plus
  ``F-F_Momentum_Factor_daily_CSV.zip`` → Mom). CSVs carry a text preamble and
  a copyright footer around ``YYYYMMDD`` rows with **percent** values —
  converted to decimal returns here.

Raw downloads land in the cache directory (``data_cache/``, public data only;
nothing derived from CRSP goes there) and every parser is a pure function of
file text, so all tests run offline against fixtures. FRED macro series / NBER
recession dates remain deferred (syllabus: "if needed").
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "VIX_URL",
    "FRENCH_FACTORS_URL",
    "FRENCH_MOMENTUM_URL",
    "parse_vix_csv",
    "parse_french_csv",
    "load_vix",
    "load_french_factors",
    "build_context",
]

VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
_FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FRENCH_FACTORS_URL = f"{_FRENCH_BASE}/F-F_Research_Data_Factors_daily_CSV.zip"
FRENCH_MOMENTUM_URL = f"{_FRENCH_BASE}/F-F_Momentum_Factor_daily_CSV.zip"


def _fetch_cached(url: str, path: Path, refresh: bool) -> bytes:
    if path.exists() and not refresh:
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (https)
        raw = resp.read()
    if len(raw) < 200:
        raise RuntimeError(f"suspiciously small response from {url}: {raw[:80]!r}")
    path.write_bytes(raw)
    return raw


# --------------------------------------------------------------------------- #
# Parsers (pure functions of file text — offline-testable)
# --------------------------------------------------------------------------- #


def parse_vix_csv(text: str) -> pd.Series:
    """CBOE VIX history -> close series with a DatetimeIndex, named 'vix'."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"unexpected VIX CSV columns: {list(df.columns)}")
    out = pd.Series(
        df["close"].to_numpy(dtype=float),
        index=pd.to_datetime(df["date"], format="%m/%d/%Y"),
        name="vix",
    ).sort_index()
    return out[~out.index.duplicated(keep="last")]


def parse_french_csv(text: str) -> pd.DataFrame:
    """A French-library daily CSV -> decimal-return DataFrame.

    Handles the text preamble (skips to the ``,Col1,Col2,...`` header), reads
    consecutive ``YYYYMMDD,...`` rows, stops at the footer, and converts the
    percent values to decimals. Column names are lowercased with ``-``/spaces
    mapped to ``_`` (``Mkt-RF`` -> ``mkt_rf``).
    """
    lines = text.splitlines()
    header_i = next(
        (
            i
            for i, ln in enumerate(lines)
            if ln.startswith(",")
            and re.match(r"^\s*\d{8},", lines[i + 1] if i + 1 < len(lines) else "")
        ),
        None,
    )
    if header_i is None:
        raise ValueError("no French data header found (',Col,...' line before YYYYMMDD rows)")
    cols = [
        c.strip().lower().replace("-", "_").replace(" ", "_")
        for c in lines[header_i].split(",")[1:]
    ]
    cols = [c for c in cols if c]  # trailing commas (e.g. ',Mom,') -> empty names
    dates, rows = [], []
    for ln in lines[header_i + 1 :]:
        if not re.match(r"^\s*\d{8},", ln):
            break  # footer / blank line: the data block is contiguous
        parts = ln.split(",")
        dates.append(parts[0].strip())
        rows.append(
            [
                float(p) if p.strip() else float("nan")
                for p in parts[1 : len(cols) + 1]
            ]
        )
    df = pd.DataFrame(rows, columns=cols, index=pd.to_datetime(dates, format="%Y%m%d"))
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # French marks missing values as -99.99 / -999 (percent); drop those rows
    df = df.mask(df <= -99.0)
    return df / 100.0  # percent -> decimal daily returns


def _csv_from_zip(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = z.namelist()[0]
        return z.read(name).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #


def load_vix(cache_dir: str | Path, *, refresh: bool = False) -> pd.Series:
    """VIX daily close (cache-first)."""
    raw = _fetch_cached(VIX_URL, Path(cache_dir) / "vix_history.csv", refresh)
    return parse_vix_csv(raw.decode("utf-8", errors="replace"))


def load_french_factors(
    cache_dir: str | Path, *, momentum: bool = True, refresh: bool = False
) -> pd.DataFrame:
    """Daily French factors (decimal returns): mkt_rf, smb, hml, rf [+ mom]."""
    cache_dir = Path(cache_dir)
    raw = _fetch_cached(FRENCH_FACTORS_URL, cache_dir / "ff_factors_daily.zip", refresh)
    factors = parse_french_csv(_csv_from_zip(raw))
    if momentum:
        raw_m = _fetch_cached(
            FRENCH_MOMENTUM_URL, cache_dir / "ff_momentum_daily.zip", refresh
        )
        mom = parse_french_csv(_csv_from_zip(raw_m))
        factors = factors.join(mom, how="left")
    return factors


def build_context(
    vix: pd.Series | None = None, factors: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Join context series into one date-indexed frame for alignment reports.

    Adds the derived columns the syllabus's diagnostics list asks about:
    ``abs_mkt`` (magnitude of the market move) and ``mkt_vol_20d`` (trailing
    realized market volatility) when factors are present.
    """
    parts: list[pd.DataFrame] = []
    if vix is not None:
        parts.append(vix.to_frame())
    if factors is not None:
        f = factors.copy()
        if "mkt_rf" in f.columns:
            f["abs_mkt"] = f["mkt_rf"].abs()
            f["mkt_vol_20d"] = f["mkt_rf"].rolling(20).std() * np.sqrt(252.0)
        parts.append(f)
    if not parts:
        raise ValueError("provide at least one of vix / factors")
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, how="outer")
    return out.sort_index()
