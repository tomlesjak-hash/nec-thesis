"""Trexquant competition data pipeline — adapted COPY of nec_baseline loaders.

DO NOT EDIT nec_baseline. This module is self-contained: it borrows the design
(cache-first, source-agnostic, skip-and-report failures) but changes four things
that matter for this competition:

1. BATCHED DOWNLOAD. nec_baseline calls yf.download() once per ticker. For ~900
   tickers that is ~900 sequential network round-trips. Here we batch in chunks
   and let yfinance thread them: minutes instead of hours.

2. DATE-RANGE-AGNOSTIC CACHE. nec_baseline keys the cache on
   "{symbol}.us.{start}.{end}.csv", so changing the window invalidates every
   file. Here the key is the symbol alone; we store full history and slice at
   load time. `migrate_nec_cache()` seeds this cache from the 592 files already
   on disk so that work is not wasted.

3. UNADJUSTED + ADJUSTED. nec_baseline uses auto_adjust=True and keeps only
   OHLCV. We keep Close AND Adj Close so the split/dividend adjustment factor
   is recoverable — the tutorial warns that retro-adjustment is a real source
   of fast-vs-slow-mode divergence.

4. WIDE PARQUET PANEL. Output is one dates x tickers matrix per field, float32,
   matching the QIP pickle shape so genetic_factor.py ports with minimal change.

No torch dependency (nec_baseline.universe imports it; we do not need it).
"""

from __future__ import annotations

import io
import re
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

#: Below this date Wikipedia's constituent-change table is too sparse to
#: reconstruct membership (~40 events for the whole 2000s vs ~22/yr in the
#: 2010s). nec_baseline measured this; we inherit the constraint.
EARLIEST_RELIABLE = pd.Timestamp("2011-01-01")

#: Index / macro series. Not equities — downloaded separately, never ranked.
CONTEXT_TICKERS = ("^GSPC", "^VIX", "GLD")

#: Fields written to the panel. `close` is SPLIT+DIVIDEND adjusted, matching
#: Trexsim's CAX_ADJ_DIV_D1. `close_raw` is unadjusted, diagnostics only.
PANEL_FIELDS = ("open", "high", "low", "close", "volume", "close_raw")

#: TICKER REUSE — symbols whose original company died and whose ticker was
#: later assigned to a different security. Yahoo serves the NEW security's
#: prices under the OLD symbol, sometimes stitched onto fragments of the old,
#: producing price series that are not any single company's history.
#:
#: Identified by diagnose_outliers.py. The tell for CPWR: the pair
#: 1.153354 -> 3.460063 appears on BOTH 2010-05-05 and 2015-02-13, and the
#: values are exact multiples of one another — synthetic, not market data.
#:
#: NOT excluded, deliberately: GME (3 extreme bars = the real January 2021
#: squeeze) and HIG (1 bar = a real 2008 crisis move). Deleting genuine tail
#: events to tidy a distribution is itself a bias.
BAD_TICKERS: dict[str, str] = {
    "CPWR": "Compuware taken private Dec 2014; data continues to 2017, "
            "span 37,205x, repeated identical price pairs",
    "EP":   "El Paso acquired by Kinder Morgan 2012; data continues to 2017",
    "MI":   "Marshall & Ilsley acquired by BMO 2011; a 2021 bar shows "
            "299.50 -> 1130.00",
    "COL":  "Rockwell Collins acquired by UTC 2018; series runs $0.015-$1.80 "
            "against a real range of roughly $50-140",
    "SLE":  "Sara Lee; series runs $3,033-$97,344 against a real range of "
            "roughly $10-20",
}


# ---------------------------------------------------------------- universe --

def _is_ticker(t: object) -> bool:
    """True only for a real ticker string.

    `_norm()` returns None for unparseable cells; pandas stores that as NaN,
    and `if nan` is TRUE — so a bare truthiness check lets NaN into the ticker
    set and then sorted() raises "'<' not supported between float and str".
    """
    return isinstance(t, str) and bool(t)


def _month_anchors(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First trading day of each calendar month present in `dates`."""
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby([dates.year, dates.month]).min().values)


def _norm(t: object) -> str | None:
    if not isinstance(t, str):
        return None
    s = t.strip().upper().replace(".", "-")
    return s if re.fullmatch(r"[A-Z][A-Z0-9-]{0,6}", s) else None


def fetch_sp500_wiki(cache_dir: Path, *, refresh: bool = False) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "sp500_wiki.html"
    if path.exists() and not refresh:
        return path.read_text()
    req = urllib.request.Request(
        SP500_WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (research)"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
        html = r.read().decode("utf-8", errors="replace")
    path.write_text(html)
    return html


def parse_sp500_tables(html: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (current DataFrame[ticker, sector, industry], changes[date, added, removed]).

    Table discrimination, learned the hard way: the CURRENT-constituents table
    also contains a "Date added" column, so keying off the absence of "date"
    silently skipped it and left the universe with zero current members. The
    reliable discriminator is that only the CHANGES table has BOTH "added" and
    "removed" columns.

    The current table also carries GICS Sector / Sub-Industry, which we take
    here for free rather than making ~900 per-ticker yfinance info calls.
    """
    tables = pd.read_html(io.StringIO(html))
    cur_rows: list[dict[str, str]] = []
    changes: list[dict] = []

    for tb in tables:
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c)
                for c in tb.columns]
        low = [c.lower() for c in cols]
        has_add = any("added" in c for c in low)
        has_rem = any("removed" in c for c in low)

        if has_add and has_rem:                                   # changes
            di = next((i for i, c in enumerate(low) if "date" in c), None)
            add_i = next((i for i, c in enumerate(low)
                          if "added" in c and "ticker" in c), None)
            rem_i = next((i for i, c in enumerate(low)
                          if "removed" in c and "ticker" in c), None)
            if di is None or add_i is None or rem_i is None:
                continue
            for _, row in tb.iterrows():
                d = pd.to_datetime(row.iloc[di], errors="coerce")
                if pd.isna(d):
                    continue
                changes.append({"date": d,
                                "added": _norm(row.iloc[add_i]),
                                "removed": _norm(row.iloc[rem_i])})

        elif any(c.strip() in ("symbol", "ticker") or c.endswith("symbol")
                 for c in low):                                   # current
            si = next(i for i, c in enumerate(low)
                      if c.strip() in ("symbol", "ticker") or c.endswith("symbol"))
            sec_i = next((i for i, c in enumerate(low)
                          if "sector" in c), None)
            ind_i = next((i for i, c in enumerate(low)
                          if "sub-industry" in c or "sub industry" in c
                          or "industry" in c), None)
            for _, row in tb.iterrows():
                t = _norm(row.iloc[si])
                if not t:
                    continue
                cur_rows.append({
                    "ticker": t,
                    "sector": str(row.iloc[sec_i]) if sec_i is not None else "",
                    "industry": str(row.iloc[ind_i]) if ind_i is not None else "",
                })

    cur = (pd.DataFrame(cur_rows).drop_duplicates("ticker").set_index("ticker")
           if cur_rows else pd.DataFrame(columns=["sector", "industry"]))
    ch = (pd.DataFrame(changes).sort_values("date").reset_index(drop=True)
          if changes else pd.DataFrame(columns=["date", "added", "removed"]))
    return cur, ch


@dataclass
class PointInTimeUniverse:
    current: frozenset[str]
    changes: pd.DataFrame
    sectors: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["sector", "industry"]))
    _warned: bool = False

    def members_asof(self, date) -> frozenset[str]:
        """Roll membership *backwards* from today, undoing each change."""
        date = pd.Timestamp(date)
        if date < EARLIEST_RELIABLE and not self._warned:
            # warn ONCE, not once per monthly anchor
            object.__setattr__(self, "_warned", True)
            warnings.warn(
                f"{date.date()} precedes EARLIEST_RELIABLE "
                f"({EARLIEST_RELIABLE.date()}); membership is unreliable "
                "(Wikipedia's change table is too sparse before 2011). "
                "Treat pre-2011 as burn-in only — train from 2011.",
                stacklevel=2,
            )
        members = {t for t in self.current if _is_ticker(t)}
        future = self.changes[self.changes["date"] > date]
        for _, r in future.iloc[::-1].iterrows():
            a, rm = r["added"], r["removed"]
            if _is_ticker(a):
                members.discard(a)            # undo the later addition
            if _is_ticker(rm):
                members.add(rm)               # restore the later removal
        return frozenset(members)

    def members_union(self, start, end) -> frozenset[str]:
        """Every ticker that was a member at any point in [start, end]."""
        out = set(self.members_asof(start))
        w = self.changes[
            (self.changes["date"] >= pd.Timestamp(start))
            & (self.changes["date"] <= pd.Timestamp(end))
        ]
        out |= {t for t in w["added"] if _is_ticker(t)}
        out |= {t for t in w["removed"] if _is_ticker(t)}
        return frozenset(out)

    def mask(self, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
        """Boolean dates x tickers membership matrix. Rebuilt monthly."""
        m = pd.DataFrame(False, index=dates, columns=tickers)
        anchors = _month_anchors(dates)
        tset = set(tickers)
        for a, hi in zip(anchors, list(anchors[1:]) + [None]):
            cols = [t for t in self.members_asof(a) if t in tset]
            seg = (dates >= a) if hi is None else ((dates >= a) & (dates < hi))
            m.loc[dates[seg], cols] = True
        return m


def load_sp500_universe(cache_dir: Path, *, refresh: bool = False):
    cur, ch = parse_sp500_tables(fetch_sp500_wiki(cache_dir, refresh=refresh))
    if cur.empty:
        raise RuntimeError(
            "Parsed 0 current S&P 500 members — the Wikipedia table layout "
            f"probably changed. Delete {cache_dir / 'sp500_wiki.html'} and "
            "retry; if it persists, inspect the table headers."
        )
    return PointInTimeUniverse(
        current=frozenset(cur.index), changes=ch,
        sectors=cur[["sector", "industry"]],
    )


# ---------------------------------------------------------------- download --

def migrate_nec_cache(nec_cache: Path, cache: Path) -> int:
    """Seed our symbol-keyed cache from nec_baseline's {sym}.us.{s}.{e}.csv files.

    Saves re-downloading the 592 tickers already on disk. Those files are
    auto-adjusted OHLCV with no Adj Close, so `close_raw` is left NaN for them;
    a later refresh fills it in.
    """
    cache.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(nec_cache.glob("*.us.*.csv")):
        sym = src.name.split(".us.")[0].upper()
        dst = cache / f"{sym}.csv"
        if dst.exists():
            continue
        try:
            df = pd.read_csv(src, parse_dates=["Date"]).set_index("Date")
        except Exception:
            continue
        df.columns = [c.lower() for c in df.columns]
        if "close" not in df:
            continue
        df["close_raw"] = np.nan
        df[["open", "high", "low", "close", "volume", "close_raw"]].to_csv(dst)
        n += 1
    return n


def download_batch(
    tickers: list[str], start: str, end: str, cache: Path,
    *, chunk: int = 25, refresh: bool = False, pause_s: float = 1.0,
    threads: bool = False, retries: int = 2,
) -> tuple[list[str], dict[str, str]]:
    """Batched yfinance download -> per-symbol CSVs in `cache`.

    Returns (succeeded, failures). auto_adjust=False so we keep both Close
    (raw) and Adj Close (split+dividend adjusted).

    FILE-DESCRIPTOR EXHAUSTION — why chunk=25 and threads=False by default:
    macOS ships `ulimit -n` at 256. yfinance with threads=True opens a socket
    per ticker PLUS a sqlite handle for its timezone cache. At chunk=100 that
    exhausts the table, and the failures are silent and misleading — you get
    `OperationalError('unable to open database file')` on live megacaps like
    MSFT and NVDA, which reads as "delisted" but is really "out of fds".

    Sequential at chunk=25 costs a few extra minutes and is far more reliable.
    Raise the limit first for more speed:  ulimit -n 4096
    """
    import time
    import yfinance as yf

    try:  # best-effort: lift the soft limit ourselves
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 4096:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
            print(f"  raised open-file limit {soft} -> "
                  f"{resource.getrlimit(resource.RLIMIT_NOFILE)[0]}")
    except Exception:
        pass

    cache.mkdir(parents=True, exist_ok=True)

    def _covers(sym: str) -> bool:
        """True only if the cached file spans the requested window.

        Critical with migrated files: nec_baseline's cache is 2015-01-01 to
        2024-12-31. A plain existence check would silently accept those for a
        2010-start run and you'd lose five years without any error.
        """
        p = cache / f"{sym}.csv"
        if not p.exists():
            return False
        try:
            idx = pd.read_csv(p, usecols=["Date"], parse_dates=["Date"])["Date"]
        except Exception:
            return False
        if idx.empty:
            return False
        want_lo, want_hi = pd.Timestamp(start), pd.Timestamp(end)
        # 10 trading-day tolerance for listings/delistings inside the window
        return (idx.min() <= want_lo + pd.Timedelta(days=14)
                and idx.max() >= want_hi - pd.Timedelta(days=14))

    todo = [t for t in tickers if refresh or not _covers(t)]
    ok = [t for t in tickers if t not in todo]
    if ok:
        print(f"  {len(ok)} already cached with full coverage; "
              f"{len(todo)} to fetch")
    fail: dict[str, str] = {}

    def _fetch(batch: list[str]) -> None:
        """Download one batch, writing CSVs and recording failures."""
        try:
            raw = yf.download(
                batch, start=start, end=end, auto_adjust=False,
                progress=False, group_by="ticker", threads=threads,
            )
        except Exception as exc:
            for t in batch:
                fail[t] = f"batch error: {exc}"
            return

        for t in batch:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.dropna(how="all")
                if df.empty:
                    fail[t] = "no data"
                    continue
                # CRITICAL: with auto_adjust=False, yfinance returns Open/High/
                # Low/Close UNADJUSTED and only "Adj Close" adjusted. Storing
                # them side by side mixes conventions — adjusted close then sits
                # BELOW unadjusted low on ~77% of bars, silently corrupting any
                # alpha that combines close with high/low.
                #
                # Trexsim applies CAX_ADJ_DIV_D1 to open, high, low AND close,
                # so we apply the same factor to all four here.
                # Guard: a zero/NaN raw Close would make the factor inf/NaN and
                # poison open/high/low. Across 800+ tickers including delisted
                # and thinly-traded names, that WILL occur somewhere.
                factor = df["Adj Close"] / df["Close"].replace(0, np.nan)
                factor = factor.replace([np.inf, -np.inf], np.nan).ffill().bfill()
                if factor.isna().all():
                    factor = pd.Series(1.0, index=df.index)
                out = pd.DataFrame({
                    "open": df["Open"] * factor,
                    "high": df["High"] * factor,
                    "low": df["Low"] * factor,
                    "close": df["Adj Close"],      # == Close * factor
                    "volume": df["Volume"],        # unadjusted share count
                    "close_raw": df["Close"],      # unadjusted, for dollar volume
                })
                out.index.name = "Date"
                out.to_csv(cache / f"{t}.csv")
                ok.append(t)
            except Exception as exc:
                fail[t] = str(exc)
        time.sleep(pause_s)

    # pass 1
    n_chunks = -(-len(todo) // chunk)
    for i in range(0, len(todo), chunk):
        print(f"  [{i//chunk + 1}/{n_chunks}] {len(todo[i:i+chunk])} tickers…")
        _fetch(todo[i : i + chunk])

    # retry passes — transient fd/network errors are common and recoverable.
    # Genuine delistings fail identically every time, so they drop out fast.
    for attempt in range(1, retries + 1):
        again = [t for t in fail if "no data" not in fail[t].lower()]
        if not again:
            break
        print(f"  retry {attempt}/{retries}: {len(again)} transient failures…")
        for t in again:
            fail.pop(t, None)
        time.sleep(3.0)
        for i in range(0, len(again), chunk):
            _fetch(again[i : i + chunk])

    return sorted(set(ok)), fail


def load_cached(sym: str, cache: Path) -> pd.DataFrame | None:
    p = cache / f"{sym}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[~df.index.duplicated(keep="last")]


def download_shares(tickers: list[str], start: str, end: str, cache: Path,
                    *, refresh: bool = False, pause_s: float = 0.3
                    ) -> pd.DataFrame:
    """Download HISTORICAL shares outstanding via yfinance get_shares_full().

    Real download, not a derivation — but read the coverage caveat:

    Yahoo's share-count history is thin. In practice get_shares_full() returns
    roughly the last ~4 years for most names and nothing at all for some. There
    is no free source for 15 years of point-in-time share counts; FMP's
    historical-market-cap endpoint caps at ~3 months on the free tier (tested),
    and CRSP/Compustat are the paid answer.

    So: this fills what it can. `mktcap` is then close_raw x shares, forward-
    and back-filled within each ticker (share counts are step functions that
    change only on issuance/buyback, so holding them constant across a gap is
    far less wrong than it would be for a price). Where a ticker has no share
    data at all, mktcap is NaN and you should fall back to `adv60` rank as the
    size proxy for GP work — see 05_UNIVERSE_DECISION.md.
    """
    import time
    import yfinance as yf

    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "_shares.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    out: dict[str, pd.Series] = {}
    for i, t in enumerate(tickers):
        if i % 50 == 0:
            print(f"    shares {i}/{len(tickers)}…")
        try:
            s = yf.Ticker(t).get_shares_full(start=start, end=end)
            if s is None or len(s) == 0:
                continue
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            out[t] = s[~s.index.duplicated(keep="last")]
        except Exception:
            pass
        time.sleep(pause_s)

    df = pd.DataFrame(out).sort_index().astype("float32") if out else pd.DataFrame()
    df.to_parquet(path)
    print(f"    shares history for {df.shape[1]}/{len(tickers)} tickers")
    return df


def download_sectors(tickers: list[str], cache: Path, *,
                     wiki: pd.DataFrame | None = None,
                     refresh: bool = False, pause_s: float = 0.15
                     ) -> pd.DataFrame:
    """Sector + industry per ticker via yfinance Ticker.info.

    Trexsim neutralizes by `industry`, so your LOCAL fitness must too — an
    alpha that looks strong raw but is really an industry bet will score well
    locally and badly on the platform. This is the field that prevents that.

    CAVEAT — current, not point-in-time. yfinance exposes only today's GICS
    classification; there is no free history. Sector reassignments are rare
    (a few per hundred names per decade), so holding it constant is a mild
    approximation. It is also NaN for tickers Yahoo no longer serves.

    Cached to `_sectors.csv`; ~900 tickers takes a few minutes, once.
    """
    import time

    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "_sectors.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0).fillna("")

    rows: dict[str, dict[str, str]] = {}

    # Wikipedia's current-constituents table already carries GICS Sector and
    # Sub-Industry. Free and instant — use it, and only pay for yfinance calls
    # on the names it doesn't cover (mostly departed members).
    if wiki is not None and not wiki.empty:
        for t in tickers:
            if t in wiki.index:
                r = wiki.loc[t]
                rows[t] = {"sector": str(r.get("sector", "") or ""),
                           "industry": str(r.get("industry", "") or "")}
        print(f"    {len(rows)}/{len(tickers)} from Wikipedia GICS (free)")

    missing = [t for t in tickers if t not in rows]
    if missing:
        print(f"    {len(missing)} not in the current index — querying yfinance…")
        import yfinance as yf  # only needed for the fallback path

        for i, t in enumerate(missing):
            if i and i % 100 == 0:
                print(f"    sectors {i}/{len(missing)}…")
            try:
                info = yf.Ticker(t).info or {}
                rows[t] = {"sector": info.get("sector") or "",
                           "industry": info.get("industry") or ""}
            except Exception:
                rows[t] = {"sector": "", "industry": ""}
            time.sleep(pause_s)

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    df.to_csv(path)
    got = int((df["sector"] != "").sum())
    print(f"    sector for {got}/{len(tickers)} tickers, "
          f"{df['sector'].nunique()} sectors / {df['industry'].nunique()} industries")
    return df


def add_sectors(panel: dict[str, pd.DataFrame], sectors: pd.DataFrame
                ) -> dict[str, pd.DataFrame]:
    """Broadcast sector/industry to dates x tickers as INTEGER CODES.

    Trexsim stores these as ints (sector 0-14, industry 0-84). We match that so
    group operators (`cs_*_group`, industry demeaning) work identically. Empty
    strings -> NaN, so unclassified names drop out of group operations rather
    than forming a spurious "unknown" group.
    """
    cols = panel["close"].columns
    idx = panel["close"].index
    out = panel
    for field in ("sector", "industry"):
        if sectors.empty or field not in sectors.columns:
            out[field] = pd.DataFrame(np.nan, index=idx, columns=cols,
                                      dtype="float32")
            continue
        s = sectors[field].reindex(cols).fillna("")
        codes = {v: i for i, v in enumerate(sorted(x for x in s.unique() if x))}
        vals = s.map(lambda v: codes.get(v, np.nan)).astype("float32")
        out[field] = pd.DataFrame(
            np.tile(vals.values, (len(idx), 1)), index=idx, columns=cols,
            dtype="float32",
        )
    n_ind = int(np.nanmax(out["industry"].iloc[0].values)) + 1 if len(idx) else 0
    print(f"    industry codes 0..{n_ind - 1} broadcast across {len(idx)} dates")
    return out


def add_marketcap(panel: dict[str, pd.DataFrame], shares: pd.DataFrame
                  ) -> dict[str, pd.DataFrame]:
    """mktcap = unadjusted close x shares outstanding, aligned to the panel.

    Uses `close_raw` (NOT the adjusted close): market cap is a real quantity in
    dollars, and multiplying an adjusted price by a current share count would
    give a number that is neither historical nor current.
    """
    c = panel.get("close_raw")
    if c is None or shares.empty:
        panel["mktcap"] = pd.DataFrame(
            np.nan, index=panel["close"].index,
            columns=panel["close"].columns, dtype="float32"
        )
        panel["shsout"] = panel["mktcap"].copy()
        return panel

    sh = (shares.reindex(c.index).ffill().bfill()
                .reindex(columns=c.columns).astype("float32"))
    panel["shsout"] = sh
    panel["mktcap"] = (c * sh).astype("float32")
    cov = panel["mktcap"].notna().any().sum()
    print(f"    mktcap populated for {cov}/{c.shape[1]} tickers")
    return panel


# ------------------------------------------------------------------- panel --

def build_panel(tickers: list[str], cache: Path, start: str, end: str
                ) -> dict[str, pd.DataFrame]:
    """-> {field: wide DataFrame(dates x tickers, float32)}."""
    frames: dict[str, dict[str, pd.Series]] = {f: {} for f in PANEL_FIELDS}
    for t in tickers:
        df = load_cached(t, cache)
        if df is None or df.empty:
            continue
        df = df.loc[start:end]
        for f in PANEL_FIELDS:
            if f in df.columns:
                frames[f][t] = df[f]
    return {
        f: pd.DataFrame(d).sort_index().astype("float32")
        for f, d in frames.items() if d
    }


def add_derived(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """ret1/5/10/20, dollar volume, ADV. All Trexsim-derivable."""
    c = panel["close"]
    for n in (1, 5, 10, 20):
        panel[f"ret{n}"] = (c / c.shift(n) - 1.0).astype("float32")

    # Dollar volume must use the UNADJUSTED close: `volume` is a raw share
    # count, so pairing it with an adjusted price understates historical
    # turnover by the cumulative split factor (a 4:1 split makes pre-split
    # dollar volume look 4x too small). That would distort adv60 and therefore
    # the universe mask itself. Fall back to adjusted close only if close_raw
    # is unavailable (e.g. files migrated from nec_baseline's auto_adjust cache).
    px = panel.get("close_raw")
    px = c if px is None else px.where(px.notna(), c)
    panel["dollar_volume"] = (px * panel["volume"]).astype("float32")
    panel["adv60"] = (
        panel["dollar_volume"].rolling(60, min_periods=20).mean().astype("float32")
    )
    return panel


def top_n_mask(adv: pd.DataFrame, n: int = 1000,
               membership: pd.DataFrame | None = None,
               price: pd.DataFrame | None = None) -> pd.DataFrame:
    """Monthly-rebalanced top-N-by-ADV mask, optionally intersected with index
    membership. Ranks are computed on the month's FIRST day and held, so the
    mask never uses same-month information (no look-ahead)."""
    if membership is not None:
        adv = adv.where(membership.reindex_like(adv).astype(bool))
    mask = pd.DataFrame(False, index=adv.index, columns=adv.columns)
    anchors = _month_anchors(adv.index)
    for a, hi in zip(anchors, list(anchors[1:]) + [None]):
        row = adv.loc[a].dropna()
        if row.empty:
            continue
        keep = row.nlargest(min(n, len(row))).index
        seg = (adv.index >= a) if hi is None else ((adv.index >= a) & (adv.index < hi))
        mask.loc[adv.index[seg], keep] = True

    # Membership is chosen at the month anchor and held, so a name that stops
    # trading mid-month would otherwise stay flagged with no price behind it —
    # an "orphan" the backtester treats as a live holding.
    #
    # But distinguish a GAP from an END. A naive `mask &= price.notna()` ejects
    # a name during a trading halt or a missing Yahoo day and re-admits it
    # afterwards, which (a) is wrong — you hold through a halt — and (b) shows
    # up as a mid-month ADDITION, indistinguishable from real look-ahead.
    #
    # So: a name is live from its FIRST to its LAST real observation. Interior
    # gaps keep it in the universe; the series ending ejects it for good.
    if price is not None:
        v = price.reindex_like(mask).notna()
        after_first = v.cummax()                      # True from first obs on
        before_last = v[::-1].cummax()[::-1]          # True up to last obs
        mask &= (after_first & before_last)
    return mask


def save_panel(panel: dict[str, pd.DataFrame], out: Path) -> None:
    """Write the panel, clearing stale files first.

    The wipe matters: if Desktop is on iCloud Drive (or any sync client),
    re-running creates conflict copies named "close 2.parquet". Those load as
    extra fields with mismatched shapes and produce bogus validation failures
    — 76 fields instead of 38, "34 seasonality fields" instead of 17.
    """
    out.mkdir(parents=True, exist_ok=True)
    stale = list(out.glob("*.parquet"))
    for p in stale:
        p.unlink()
    if stale:
        print(f"  cleared {len(stale)} stale parquet files")
    for f, df in panel.items():
        df.to_parquet(out / f"{f}.parquet", compression="snappy")
    print(f"  wrote {len(panel)} fields -> {out}")


def load_panel(out: Path, *, exclude: "list[str] | dict[str, str] | None" = None
               ) -> dict[str, pd.DataFrame]:
    """Load the panel, ignoring sync-client conflict copies ("close 2.parquet").

    `exclude` drops columns at LOAD time rather than deleting them from disk.
    The parquet files stay a faithful record of what the vendor served, so an
    exclusion can be revisited without re-downloading. Pass `BAD_TICKERS`.
    """
    files = [p for p in sorted(out.glob("*.parquet"))
             if not re.search(r" \d+$", p.stem)]
    panel = {p.stem: pd.read_parquet(p) for p in files}
    if exclude:
        drop = list(exclude)
        panel = {k: v.drop(columns=drop, errors="ignore") for k, v in panel.items()}
    return panel
