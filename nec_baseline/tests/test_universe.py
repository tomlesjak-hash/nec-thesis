"""Point-in-time universe: membership spells and the filter.

Membership comes from CRSP's inclusive spells (``SpellUniverse``); the filter
keeps a row only if its entity was a member on the row's date and re-ranks a
rank-normalized panel among the members (audit D-1). The spells below are
invented; the CRSP loading itself is tested in ``test_crsp.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from nec_moe import (
    SpellUniverse,
    SyntheticRegimePanel,
    SyntheticSpec,
    filter_point_in_time,
    universe_coverage_report,
)

DATES = ("2021-05-28", "2021-05-31", "2021-06-01", "2021-06-02")


def _universe() -> SpellUniverse:
    """10001 leaves on 2021-05-31 (inclusive); 10002 joins on 2021-06-01
    (inclusive); 10003 left, then rejoined; 10004 is a member throughout."""
    return SpellUniverse(pd.DataFrame({
        "entity": ["10001", "10002", "10003", "10003", "10004"],
        "start": pd.to_datetime(["2010-01-04", "2021-06-01", "2010-01-04", "2021-06-02",
                                 "2010-01-04"]),
        "end": pd.to_datetime(["2021-05-31", "2030-12-31", "2015-06-30", "2030-12-31",
                               "2030-12-31"]),
    }))


def test_spells_are_inclusive_and_resolve_leave_and_rejoin():
    u = _universe()
    assert u.members_asof("2021-05-31") == {"10001", "10004"}
    assert "10001" in u.members_asof("2021-05-31")  # its last day: inclusive end
    assert "10001" not in u.members_asof("2021-06-01")
    assert "10002" in u.members_asof("2021-06-01")  # its first day: inclusive start
    assert "10002" not in u.members_asof("2021-05-31")
    assert "10003" in u.members_asof("2014-01-02") and "10003" not in u.members_asof("2019-01-02")
    assert "10003" in u.members_asof("2021-06-02")  # rejoined
    assert u.members_union("2021-05-28", "2021-06-02") == {"10001", "10002", "10003", "10004"}
    counts = u.count_by_date(pd.DatetimeIndex(DATES))
    assert counts.tolist() == [2, 2, 2, 3]


def test_spells_are_validated():
    with pytest.raises(ValueError, match="columns"):
        SpellUniverse(pd.DataFrame({"entity": ["1"], "start": pd.to_datetime(["2020-01-01"])}))
    with pytest.raises(ValueError, match="ends before it starts"):
        SpellUniverse(pd.DataFrame({"entity": ["1"], "start": pd.to_datetime(["2020-01-02"]),
                                    "end": pd.to_datetime(["2020-01-01"])}))


def _panel():
    panel = SyntheticRegimePanel(SyntheticSpec(seed=0)).generate(n_dates=4, n_entities=2)
    panel.date_labels = DATES
    panel.entity_labels = ("10001", "10002")
    return panel


def test_filter_point_in_time_on_panel():
    """Rows for names not yet (or no longer) in the index are dropped."""
    panel = _panel()
    filtered = filter_point_in_time(panel, _universe())
    assert len(filtered) == 4  # each date keeps exactly one of the two
    kept = {
        (panel.date_labels[int(d)], panel.entity_labels[int(e)])
        for d, e in zip(filtered.date, filtered.entity, strict=True)
    }
    assert kept == {
        ("2021-05-28", "10001"), ("2021-05-31", "10001"),
        ("2021-06-01", "10002"), ("2021-06-02", "10002"),
    }
    assert torch.equal(filtered.date, filtered.date.sort().values)  # date-major kept


def test_filter_leaves_an_unranked_panel_untouched():
    """Re-ranking (audit D-1) applies only to rank-normalized panels."""
    panel = _panel()
    assert panel.schema.rank_normalized is False
    filtered = filter_point_in_time(panel, _universe())
    for row in range(len(filtered)):
        d, e = int(filtered.date[row]), int(filtered.entity[row])
        src = int(((panel.date == d) & (panel.entity == e)).nonzero()[0])
        assert torch.equal(filtered.x_snap[row], panel.x_snap[src])


def test_filter_requires_labels():
    panel = SyntheticRegimePanel(SyntheticSpec(seed=0)).generate(n_dates=3, n_entities=2)
    with pytest.raises(ValueError, match="date_labels"):
        filter_point_in_time(panel, _universe())


def test_coverage_report_math():
    rep = universe_coverage_report(
        _universe(), dates=["2014-12-31", "2021-06-02"], available={"10001", "10004"}
    )
    # 2014: members {10001, 10003, 10004}; data for two of them
    assert rep.loc["2014-12-31", "coverage"] == pytest.approx(2 / 3)
    # 2021-06-02: members {10002, 10003, 10004}; data for one
    assert rep.loc["2021-06-02", "coverage"] == pytest.approx(1 / 3)
    assert rep.loc["2021-06-02", "members"] == 3
