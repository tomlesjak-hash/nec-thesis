"""The point-in-time universe: which entities were index members on each date.

A panel is built over every entity that was a member at some point in its
window (features may use history from before an entity joined); then
:func:`filter_point_in_time` keeps a row only if its entity was a member on
the row's date, and re-ranks rank-normalized snapshot features among the
members (audit finding D-1). That removes backward-looking selection
("pick today's winners, then backtest them").

The membership comes from CRSP's ``StkIndMembership`` spells
(:mod:`nec_moe.crsp`), loaded into a :class:`SpellUniverse`. CRSP carries the
departed names' prices and their delisting returns too, so the second
survivorship component the free-data universe could only measure (members
without data) is covered by the data itself; :func:`universe_coverage_report`
still reports it per date.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
import torch

from .data import Panel
from .features import _cross_sectional_rank

__all__ = [
    "Universe",
    "SpellUniverse",
    "filter_point_in_time",
    "universe_coverage_report",
]


class Universe(Protocol):
    """Anything that can say which entities were index members on a date."""

    def members_asof(self, date: str | pd.Timestamp) -> frozenset[str]: ...


@dataclass(frozen=True)
class SpellUniverse:
    """Index membership as spells: ``entity`` is a member on every date ``d``
    with ``start <= d <= end``. **Both bounds are inclusive.**

    This is the shape of CRSP's ``StkIndMembership`` (``MbrStartDt``,
    ``MbrEndDt``), which :mod:`nec_moe.crsp` loads into it with PERMNOs as the
    entity labels. The inclusive reading is CRSP's: the metadata classes the
    start as the first trading date and the end as the last daily date, and
    on the real S&P 500 spells only the inclusive reading keeps the member
    count free of a one-day dip at every index change (see ``CRSPSpec``).
    """

    spells: pd.DataFrame = field(repr=False)  # entity (str), start, end (Timestamp)

    def __post_init__(self) -> None:
        missing = {"entity", "start", "end"} - set(self.spells.columns)
        if missing:
            raise ValueError(f"spells need columns entity, start, end; missing {missing}")
        if (self.spells["end"] < self.spells["start"]).any():
            raise ValueError("a membership spell ends before it starts")

    def members_asof(self, date: str | pd.Timestamp) -> frozenset[str]:
        d = pd.Timestamp(date)
        s = self.spells
        return frozenset(s.loc[(s["start"] <= d) & (d <= s["end"]), "entity"])

    def members_union(
        self, start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> frozenset[str]:
        """Every entity that was a member on at least one date in ``[start, end]``."""
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        s = self.spells
        return frozenset(s.loc[(s["start"] <= hi) & (lo <= s["end"]), "entity"])

    def count_by_date(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Number of members on each date (one row per spell, so an entity
        with two spells covering the same date would count twice; CRSP spells
        of one PERMNO do not overlap)."""
        d = dates.to_numpy()[:, None]
        start = self.spells["start"].to_numpy()[None, :]
        end = self.spells["end"].to_numpy()[None, :]
        return ((start <= d) & (d <= end)).sum(axis=1)

def filter_point_in_time(panel: Panel, universe: Universe) -> Panel:
    """Keep only rows whose entity was an index member on that row's date.

    Kills survivorship component 1 (backward-looking selection) on an already-
    built panel; requires the panel's ``date_labels``/``entity_labels`` (real
    panels carry them). Row order (date-major) is preserved.

    **Re-ranks a rank-normalized panel** (``schema.rank_normalized``). Its
    snapshot ranks were computed over every candidate ticker, including names
    outside the index that day and names that join it later, so after the
    filter they would still depend on those names (audit finding D-1). The
    survivors are re-ranked per date among themselves. Ranks are distinct and
    re-ranking preserves their order, so the result equals ranking the raw
    features over the members alone, exactly.
    """
    if panel.date_labels is None or panel.entity_labels is None:
        raise ValueError(
            "filter_point_in_time needs date_labels and entity_labels "
            "(synthetic panels have no calendar to be point-in-time about)"
        )
    # member[d, e]: entity e was in the index on date code d
    ent_code = {label: i for i, label in enumerate(panel.entity_labels)}
    member = torch.zeros(len(panel.date_labels), len(ent_code), dtype=torch.bool)
    for code, label in enumerate(panel.date_labels):
        cols = [ent_code[m] for m in universe.members_asof(label) if m in ent_code]
        member[code, cols] = True
    keep = member[panel.date, panel.entity]
    dropped = int((~keep).sum())
    if dropped:
        print(
            f"[universe] point-in-time filter dropped {dropped}/{len(panel)} "
            "rows (names not in the index on those dates)"
        )
    filtered = panel._take(keep)
    if panel.schema.rank_normalized:
        filtered = dataclasses.replace(
            filtered, x_snap=_cross_sectional_rank(filtered.x_snap, filtered.date)
        )
    return filtered


def universe_coverage_report(
    universe: Universe,
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
