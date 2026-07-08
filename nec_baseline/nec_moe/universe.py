"""Partial Module 12: an approximately point-in-time S&P 500 universe.

WHAT THIS FIXES AND WHAT IT CANNOT (read before citing any backtest)
--------------------------------------------------------------------
Survivorship bias has two components:

1. **Backward-looking selection** — building today's universe from today's
   membership ("pick the winners, then backtest them"). This module *fixes*
   that component: membership at each historical date is reconstructed from
   Wikipedia's S&P 500 constituent-change history, so a panel filtered with
   :func:`filter_point_in_time` contains a name on a date only if it was in
   the index on that date.
2. **Missing departed names** — companies that were members then but have no
   price data now (acquired, delisted, renamed) are still absent, because free
   price sources rarely serve them, and free sources carry no delisting
   returns at all. This component is *not fixed* — it is made **measurable**:
   :func:`universe_coverage_report` quantifies, per date, what fraction of
   true point-in-time members your data actually covers. Report that table;
   the residual bias is then documented instead of silent.

The defensible thesis posture this enables: "survivorship-mitigated with
documented residual coverage", not "survivorship-free" (that needs CRSP).

Source and reliability
----------------------
Wikipedia's "List of S&P 500 companies" page: the current-constituents table
plus the selected-changes table (date, added, removed). Cache-first like every
other loader; parsers are pure functions of the HTML, offline-testable.

Measured (2026-07): the live changes table carries ~400 events — ~complete for
the 2010s (~22/yr, matching real index turnover) and the 2020s, but only ~40
events for the whole 2000s and ~8 for the 1990s, i.e. those decades are mostly
missing. Hence ``EARLIEST_RELIABLE = 2011-01-01``:
:meth:`PointInTimeUniverse.members_asof` warns below it. Reconstruction is
best-effort (community-edited data; renames appear as remove+add) — good
enough to kill component 1 for a 2010s-2020s thesis window, and honestly
labeled.
"""

from __future__ import annotations

import io
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

from .data import Panel

__all__ = [
    "SP500_WIKI_URL",
    "EARLIEST_RELIABLE",
    "fetch_sp500_wiki",
    "parse_sp500_tables",
    "PointInTimeUniverse",
    "load_sp500_universe",
    "filter_point_in_time",
    "universe_coverage_report",
]

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

#: Below this date the Wikipedia changes table is too sparse to trust
#: (measured: ~complete event coverage only from the 2010s onward).
EARLIEST_RELIABLE = "2011-01-01"


def _norm_ticker(t: object) -> str | None:
    """Wikipedia 'BRK.B' -> our price-layer 'brk-b'; NaN/blank -> None."""
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return None
    s = str(t).strip().lower().replace(".", "-")
    return s or None


def fetch_sp500_wiki(cache_dir: str | Path, *, refresh: bool = False) -> str:
    """Raw page HTML, cache-first (``sp500_wiki.html`` in the cache dir)."""
    path = Path(cache_dir) / "sp500_wiki.html"
    if path.exists() and not refresh:
        return path.read_text()
    req = urllib.request.Request(  # noqa: S310 (https)
        SP500_WIKI_URL,
        headers={"User-Agent": "nec-thesis-research/0.1 (point-in-time universe)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        html = resp.read().decode("utf-8", errors="replace")
    if "constituents" not in html:
        raise RuntimeError("unexpected S&P 500 page content (no constituents table)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return html


def _flat_cols(df: pd.DataFrame) -> list[str]:
    """Flatten (possibly MultiIndex) columns to lowercase 'level0|level1' keys."""
    if isinstance(df.columns, pd.MultiIndex):
        return ["|".join(str(p).lower() for p in tup) for tup in df.columns]
    return [str(c).lower() for c in df.columns]


def parse_sp500_tables(html: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> ``(current, changes)``.

    ``current``: one row per current member, column ``ticker`` (normalized).
    ``changes``: one row per ticker event, columns ``date`` (Timestamp),
    ``added`` and ``removed`` (normalized ticker or None), sorted by date
    ascending. A rename is two half-filled rows sharing a date.
    """
    tables = pd.read_html(io.StringIO(html))
    current_df = changes_df = None
    for t in tables:
        cols = _flat_cols(t)
        # NB: the constituents table has a 'Date added' column, so the
        # discriminator between the two tables is 'removed', not 'added'
        if any("symbol" in c for c in cols) and not any("removed" in c for c in cols):
            current_df = t if current_df is None else current_df
        if any("added" in c and "ticker" in c for c in cols) and any(
            "removed" in c and "ticker" in c for c in cols
        ):
            changes_df = t if changes_df is None else changes_df
    if current_df is None or changes_df is None:
        raise ValueError(
            "could not locate the constituents and changes tables in the page"
        )

    cur_cols = _flat_cols(current_df)
    sym_i = next(i for i, c in enumerate(cur_cols) if "symbol" in c)
    current = pd.DataFrame(
        {"ticker": [_norm_ticker(v) for v in current_df.iloc[:, sym_i]]}
    ).dropna()

    ch_cols = _flat_cols(changes_df)

    def _col(pred: str) -> int:
        return next(i for i, c in enumerate(ch_cols) if pred in c)

    date_i = _col("date")
    added_i = next(
        i for i, c in enumerate(ch_cols) if "added" in c and "ticker" in c
    )
    removed_i = next(
        i for i, c in enumerate(ch_cols) if "removed" in c and "ticker" in c
    )
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(changes_df.iloc[:, date_i], errors="coerce"),
            "added": [_norm_ticker(v) for v in changes_df.iloc[:, added_i]],
            "removed": [_norm_ticker(v) for v in changes_df.iloc[:, removed_i]],
        }
    )
    changes = changes.dropna(subset=["date"])
    changes = changes[~(changes["added"].isna() & changes["removed"].isna())]
    return current, changes.sort_values("date").reset_index(drop=True)


@dataclass(frozen=True)
class PointInTimeUniverse:
    """Current membership + the event history that rolls it back in time."""

    current: frozenset[str]
    changes: pd.DataFrame = field(repr=False)  # date, added, removed (ascending)

    def members_asof(self, date: str | pd.Timestamp) -> frozenset[str]:
        """Membership at the end of ``date``, by undoing every later event.

        Events dated after ``date`` are reversed in reverse-chronological
        order: an addition is removed again, a removal is restored — so a
        ticker that left and rejoined resolves correctly.
        """
        asof = pd.Timestamp(date)
        if asof < pd.Timestamp(EARLIEST_RELIABLE):
            warnings.warn(
                f"S&P 500 change history is sparse before {EARLIEST_RELIABLE}; "
                f"membership as of {asof.date()} is unreliable",
                stacklevel=2,
            )
        members = set(self.current)
        later = self.changes[self.changes["date"] > asof]
        for _, ev in later.iloc[::-1].iterrows():  # newest first, undo each
            # isinstance guards: pandas may hold missing side as None OR NaN
            if isinstance(ev["added"], str):
                members.discard(ev["added"])
            if isinstance(ev["removed"], str):
                members.add(ev["removed"])
        return frozenset(members)

    def stable_members(
        self, start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> frozenset[str]:
        """Members on *every* day of ``[start, end]``: member at both ends and
        untouched by any change event inside the window."""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        inside = self.changes[(self.changes["date"] > s) & (self.changes["date"] <= e)]
        touched = {t for t in inside["added"] if isinstance(t, str)} | {
            t for t in inside["removed"] if isinstance(t, str)
        }
        return frozenset(
            (self.members_asof(s) & self.members_asof(e)) - touched
        )

    def members_union(
        self, start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> frozenset[str]:
        """Every ticker that was a member at any point in the window — the
        candidate list to *attempt* downloading (departed names often fail;
        that failure is what the coverage report measures)."""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        inside = self.changes[(self.changes["date"] > s) & (self.changes["date"] <= e)]
        union = set(self.members_asof(s)) | set(self.members_asof(e))
        union |= {t for t in inside["added"] if isinstance(t, str)}
        union |= {t for t in inside["removed"] if isinstance(t, str)}
        return frozenset(union)


def load_sp500_universe(
    cache_dir: str | Path, *, refresh: bool = False
) -> PointInTimeUniverse:
    """Cache-first end-to-end: fetch (or read) the page, parse, build."""
    current, changes = parse_sp500_tables(
        fetch_sp500_wiki(cache_dir, refresh=refresh)
    )
    return PointInTimeUniverse(
        current=frozenset(current["ticker"]), changes=changes
    )


def filter_point_in_time(panel: Panel, universe: PointInTimeUniverse) -> Panel:
    """Keep only rows whose entity was an index member on that row's date.

    Kills survivorship component 1 (backward-looking selection) on an already-
    built panel; requires the panel's ``date_labels``/``entity_labels`` (real
    panels carry them). Row order (date-major) is preserved.
    """
    if panel.date_labels is None or panel.entity_labels is None:
        raise ValueError(
            "filter_point_in_time needs date_labels and entity_labels "
            "(synthetic panels have no calendar to be point-in-time about)"
        )
    members_by_code = {
        code: universe.members_asof(label)
        for code, label in enumerate(panel.date_labels)
    }
    keep = torch.tensor(
        [
            panel.entity_labels[int(e)] in members_by_code[int(d)]
            for d, e in zip(panel.date, panel.entity)
        ],
        dtype=torch.bool,
    )
    dropped = int((~keep).sum())
    if dropped:
        print(
            f"[universe] point-in-time filter dropped {dropped}/{len(panel)} "
            "rows (names not in the index on those dates)"
        )
    return panel._take(keep)


def universe_coverage_report(
    universe: PointInTimeUniverse,
    dates: list[str],
    available: set[str],
) -> pd.DataFrame:
    """Per date: true members vs. members you actually have data for.

    ``available`` is the set of (normalized) tickers with usable price data —
    typically the keys of a loaded universe. The ``coverage`` column is the
    honest number to publish next to any backtest on this data: the fraction
    of the true point-in-time index the panel represents.
    """
    rows = []
    for d in dates:
        members = universe.members_asof(d)
        have = members & available
        rows.append(
            {
                "date": d,
                "members": len(members),
                "with_data": len(have),
                "coverage": len(have) / len(members) if members else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("date")
