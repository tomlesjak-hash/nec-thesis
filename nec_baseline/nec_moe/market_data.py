"""Stage B market data: free daily OHLCV, cached to disk as Stooq-format CSVs.

The syllabus's staged data plan (§3): Stage A was the synthetic generator;
Stage B is a free daily-price source for U.S. equities, with SPY as the
broad-market symbol for market-relative features. The download layer is
strictly separated from the feature layer: **the cache CSV is the interface**
(header ``Date,Open,High,Low,Close,Volume``), so all feature and panel code is
source-agnostic and testable offline against fixture CSVs; tests never touch
the network.

Sources (``source=`` on the loaders):

- ``"stooq"`` (syllabus primary, ``https://stooq.com``) — CAVEAT: as of
  mid-2026 Stooq gates its CSV endpoint behind a JavaScript anti-bot
  challenge, which this code deliberately does **not** circumvent. The fetch
  detects the challenge and raises with guidance. Two legitimate paths remain:
  download the CSVs in a browser (the URL format is printed) and drop them in
  the cache under the documented filename — the pipeline then runs fully
  offline — or use the fallback source.
- ``"yfinance"`` (the syllabus's sanctioned *prototyping fallback*; unofficial
  Yahoo endpoints via the ``yfinance`` package, imported lazily). Fetched
  **without** dividend back-adjustment, with the adjusted close alongside
  (see *Price adjustment* below), into a cache file of its own.

Price adjustment (audit finding D-2)
------------------------------------
A back-adjusted close at date ``t`` is scaled by every dividend and spin-off
paid **after** ``t``. Return features are unaffected, because a price ratio
only involves adjustments inside its own window, but a **level** built from
it is not: dollar volume at ``t`` would carry the size of later dividends.
The yfinance cache therefore holds two closes. ``Close`` is Yahoo's
split-adjusted, not dividend-adjusted close; with the split-adjusted
``Volume``, the split factors cancel in ``Close x Volume``, which is the
dollar volume actually traded that day. ``Adj Close`` adds dividends, and
returns, drawdown and the target use it. The files used to be auto-adjusted
(``Close`` dividend-adjusted), and those legacy files are never read for
``source="yfinance"``: the new contract lives under its own file name
(``*.raw.csv``). A source that supplies one close column (Stooq) is used for
both, and must not be dividend back-adjusted for dollar volume to be causal.

SURVIVORSHIP BIAS — read before believing any backtest on this data
--------------------------------------------------------------------
``DEFAULT_UNIVERSE`` is a static list of *currently* large, liquid U.S.
equities. Names that shrank, merged, or delisted are absent by construction,
which inflates apparent performance (Module 5's catalogued bias; quantified in
its focused exercises). This universe exists to verify the pipeline and
develop methodology. Thesis result claims need a point-in-time universe with
delisting handling — Module 12's data-engineering work, still deferred.
"""

from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path
from typing import Literal

import pandas as pd

__all__ = [
    "DEFAULT_UNIVERSE",
    "MARKET_SYMBOL",
    "DataSource",
    "stooq_url",
    "fetch_stooq_csv",
    "fetch_yfinance_csv",
    "load_ohlcv",
    "load_universe",
]

DataSource = Literal["stooq", "yfinance"]

#: Broad-market ETF used for market-relative features (syllabus §3).
MARKET_SYMBOL = "spy"

#: Static large-cap universe — SURVIVORSHIP-BIASED, pipeline-verification only
#: (see module docstring). ~30 liquid names across sectors.
DEFAULT_UNIVERSE: tuple[str, ...] = (
    "aapl", "msft", "googl", "amzn", "nvda", "meta", "brk-b", "jpm", "v",
    "unh", "xom", "jnj", "pg", "hd", "cvx", "ko", "pep", "mrk", "wmt",
    "csco", "mcd", "ibm", "cat", "ba", "ge", "mmm", "nke", "dis", "t", "intc",
)

_STOOQ_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")
#: optional column: the dividend-and-split adjusted close, for returns only
_ADJ_CLOSE = "Adj Close"
#: cache file suffix of the yfinance contract that carries both closes; the
#: legacy auto-adjusted files have no suffix and are never read for yfinance
_YF_RAW_SUFFIX = ".raw"


def stooq_url(symbol: str, start: str, end: str) -> str:
    """Daily-CSV endpoint for a symbol. Dates are ``YYYY-MM-DD``."""
    d1 = start.replace("-", "")
    d2 = end.replace("-", "")
    return f"https://stooq.com/q/d/l/?s={symbol}.us&d1={d1}&d2={d2}&i=d"


def _cache_path(
    cache_dir: Path, symbol: str, start: str, end: str, suffix: str = ""
) -> Path:
    return cache_dir / f"{symbol}.us.{start}.{end}{suffix}.csv"


def fetch_stooq_csv(
    symbol: str,
    start: str,
    end: str,
    cache_dir: str | Path,
    *,
    refresh: bool = False,
    pause_s: float = 0.5,
) -> Path:
    """Download one symbol's daily CSV to the cache (skipped if cached).

    Returns the cache path. Raises ``RuntimeError`` on an empty/'No data'
    response (Stooq's answer for unknown symbols or exceeded daily limits).
    ``pause_s`` is a politeness delay after each actual network hit.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, symbol, start, end)
    if path.exists() and not refresh:
        return path
    url = stooq_url(symbol, start, end)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (https)
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Stooq refused the request for {symbol!r} (HTTP {exc.code}). "
            "Stooq gates automated access; either download the CSV in a "
            f"browser from {url} and save it as {path.name!r} in the cache, "
            "or use source='yfinance' (the syllabus's prototyping fallback)."
        ) from exc
    if "<!DOCTYPE html" in raw[:200] or "JavaScript" in raw[:400]:
        raise RuntimeError(
            f"Stooq served its JavaScript anti-bot challenge for {symbol!r} — "
            "not circumvented by design. Either download the CSV in a browser "
            f"from {url} and save it as {path.name!r} in the cache (the "
            "pipeline then runs fully offline), or use source='yfinance'."
        )
    if not raw.startswith("Date,") or raw.count("\n") < 2:
        raise RuntimeError(
            f"Stooq returned no data for {symbol!r} ({url}): "
            f"{raw[:80]!r} — unknown symbol, empty range, or daily limit hit"
        )
    path.write_text(raw)
    time.sleep(pause_s)
    return path


def fetch_yfinance_csv(
    symbol: str,
    start: str,
    end: str,
    cache_dir: str | Path,
    *,
    refresh: bool = False,
    pause_s: float = 0.5,
) -> Path:
    """Fetch daily bars via ``yfinance``: raw close plus adjusted close.

    The syllabus's prototyping fallback (unofficial Yahoo endpoints). Written
    as ``Date,Open,High,Low,Close,Adj Close,Volume`` with ``auto_adjust=False``
    (see the module's *Price adjustment* note, audit finding D-2), to
    ``<symbol>.us.<start>.<end>.raw.csv``. ``yfinance`` is imported lazily;
    it is an optional dependency.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, symbol, start, end, _YF_RAW_SUFFIX)
    if path.exists() and not refresh:
        return path
    import yfinance as yf  # lazy: optional dependency

    df = yf.download(
        symbol.upper(), start=start, end=end, progress=False, auto_adjust=False
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol!r}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if _ADJ_CLOSE not in df.columns:
        raise RuntimeError(
            f"yfinance returned no {_ADJ_CLOSE!r} column for {symbol!r}; "
            "the cache contract needs both the raw and the adjusted close"
        )
    out = df[["Open", "High", "Low", "Close", _ADJ_CLOSE, "Volume"]].copy()
    out.insert(0, "Date", out.index.strftime("%Y-%m-%d"))
    path.write_text(out.to_csv(index=False))
    time.sleep(pause_s)
    return path


def _parse_stooq_csv(text: str, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    missing = [c for c in _STOOQ_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: CSV missing columns {missing}")
    df["Date"] = pd.to_datetime(df["Date"])
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if _ADJ_CLOSE in df.columns:
        cols.append(_ADJ_CLOSE)
    df = (
        df.set_index("Date")[cols]
        .rename(columns=lambda c: c.lower().replace(" ", "_"))
        .sort_index()
    )
    df = df[~df.index.duplicated(keep="last")]
    if (df["close"] <= 0).any():
        raise ValueError(f"{symbol}: non-positive close prices in CSV")
    if "adj_close" in df.columns and (df["adj_close"] <= 0).any():
        raise ValueError(f"{symbol}: non-positive adjusted close prices in CSV")
    return df.astype("float64")


_FETCHERS = {"stooq": fetch_stooq_csv, "yfinance": fetch_yfinance_csv}


def load_ohlcv(
    symbol: str,
    start: str,
    end: str,
    cache_dir: str | Path,
    *,
    source: DataSource = "stooq",
    refresh: bool = False,
) -> pd.DataFrame:
    """Cached daily OHLCV for one symbol: DatetimeIndex, columns o/h/l/c/v.

    Plus ``adj_close`` when the source provides it (yfinance): returns use it,
    dollar volume uses the raw ``close`` (audit finding D-2).

    Cache-first regardless of ``source`` — a present cache file short-circuits
    any network access (which is also how the offline tests and the
    manual-browser-download workflow operate).
    """
    path = _FETCHERS[source](symbol, start, end, cache_dir, refresh=refresh)
    return _parse_stooq_csv(path.read_text(), symbol)


def load_universe(
    tickers: tuple[str, ...],
    start: str,
    end: str,
    cache_dir: str | Path,
    *,
    market_symbol: str = MARKET_SYMBOL,
    source: DataSource = "stooq",
    min_tickers: int = 5,
    refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load the universe + the market symbol; skip-and-report failures.

    Returns ``(prices_by_ticker, market_prices)``. Individual ticker failures
    are collected and reported (sources occasionally reject symbols or hit
    rate limits); fewer than ``min_tickers`` successes — or a failed market
    symbol — is an error, since the panel would be meaningless.
    """
    market = load_ohlcv(
        market_symbol, start, end, cache_dir, source=source, refresh=refresh
    )
    prices: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    for t in tickers:
        if t == market_symbol:
            continue
        try:
            prices[t] = load_ohlcv(
                t, start, end, cache_dir, source=source, refresh=refresh
            )
        except Exception as exc:  # noqa: BLE001 — collected and re-raised below
            failures[t] = str(exc)
    if failures:
        detail = "; ".join(f"{k}: {v[:60]}" for k, v in failures.items())
        print(f"[market_data] skipped {len(failures)} ticker(s): {detail}")
    if len(prices) < min_tickers:
        raise RuntimeError(
            f"only {len(prices)}/{len(tickers)} tickers loaded "
            f"(need >= {min_tickers}); failures: {sorted(failures)}"
        )
    return prices, market
