"""Point-in-time universe (partial Module 12): parsing, rollback, filtering.

The fixture reproduces the Wikipedia page's structure (a constituents table
and a two-level-header changes table). The rollback tests plant a full event
history — addition, removal, and a leave-and-rejoin — and assert membership
at every epoch. The live test (opt-in) pins a famous real event: TSLA joined
the index on 2020-12-21.
"""

from __future__ import annotations

import os
import warnings

import pandas as pd
import pytest
import torch

from nec_moe import (
    PointInTimeUniverse,
    SyntheticRegimePanel,
    SyntheticSpec,
    filter_point_in_time,
    load_sp500_universe,
    universe_coverage_report,
)
from nec_moe.universe import parse_sp500_tables

# A miniature of the real page: current members {AAA, BBB, CCC.D}; history:
# 2010: XXX removed, AAA added (rename-style pair of half-filled rows)
# 2015: YYY removed / BBB added (one row)
# 2020: ZZZ leaves; 2022: ZZZ rejoins... but ZZZ is not current -> instead:
# CCC.D added 2021 replacing WWW.
FIXTURE_HTML = """
<html><body>
<table id="constituents">
  <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
  <tr><td>AAA</td><td>Alpha Corp</td><td>Tech</td></tr>
  <tr><td>BBB</td><td>Beta Inc</td><td>Energy</td></tr>
  <tr><td>CCC.D</td><td>Gamma Class D</td><td>Financials</td></tr>
</table>
<table id="changes">
  <tr><th rowspan="2">Date</th><th colspan="2">Added</th>
      <th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>
  <tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
  <tr><td>June 1, 2021</td><td>CCC.D</td><td>Gamma Class D</td>
      <td>WWW</td><td>Omega Co</td><td>acquisition</td></tr>
  <tr><td>March 2, 2015</td><td>BBB</td><td>Beta Inc</td>
      <td>YYY</td><td>Ypsilon</td><td>market cap</td></tr>
  <tr><td>January 5, 2010</td><td>AAA</td><td>Alpha Corp</td>
      <td></td><td></td><td>addition</td></tr>
  <tr><td>January 5, 2010</td><td></td><td></td>
      <td>XXX</td><td>Chi Co</td><td>removal</td></tr>
</table>
</body></html>
"""


def _fixture_universe() -> PointInTimeUniverse:
    current, changes = parse_sp500_tables(FIXTURE_HTML)
    return PointInTimeUniverse(frozenset(current["ticker"]), changes)


def test_parse_tables_and_ticker_normalization():
    current, changes = parse_sp500_tables(FIXTURE_HTML)
    assert set(current["ticker"]) == {"aaa", "bbb", "ccc-d"}  # dots -> dashes
    assert len(changes) == 4
    assert changes["date"].is_monotonic_increasing
    # half-filled rows survive with None on the empty side
    add_only = changes[(changes["date"] == "2010-01-05") & changes["removed"].isna()]
    assert add_only["added"].tolist() == ["aaa"]


def test_members_asof_rollback_epochs():
    u = _fixture_universe()
    # after everything: the current set
    assert u.members_asof("2023-01-01") == {"aaa", "bbb", "ccc-d"}
    # before 2021 change: CCC.D not yet in, WWW still in
    assert u.members_asof("2021-05-31") == {"aaa", "bbb", "www"}
    # event dated exactly as-of counts as effective (end-of-day convention)
    assert u.members_asof("2021-06-01") == {"aaa", "bbb", "ccc-d"}
    # before 2015: BBB out, YYY in
    assert u.members_asof("2014-12-31") == {"aaa", "yyy", "www"}
    # before 2010: AAA out, XXX in (pre-2011 epoch: the sparse-history
    # warning fires by design — the rollback math itself is exact)
    with pytest.warns(UserWarning, match="sparse"):
        assert u.members_asof("2009-06-30") == {"xxx", "yyy", "www"}


def test_members_asof_warns_before_reliable_history():
    u = _fixture_universe()
    with pytest.warns(UserWarning, match="sparse"):
        u.members_asof("1998-01-01")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning in the reliable era
        u.members_asof("2015-01-01")


def test_leave_and_rejoin_resolves():
    """A ticker removed then re-added must be correctly absent in between."""
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-01-01", "2020-01-01"]),
            "added": [None, "rrr"],
            "removed": ["rrr", None],
        }
    )
    u = PointInTimeUniverse(frozenset({"rrr"}), changes)
    assert "rrr" in u.members_asof("2017-06-01")
    assert "rrr" not in u.members_asof("2019-06-01")
    assert "rrr" in u.members_asof("2021-01-01")


def test_stable_members_and_union():
    u = _fixture_universe()
    # window straddling the 2021 change: WWW and CCC.D both unstable
    assert u.stable_members("2020-01-01", "2022-01-01") == {"aaa", "bbb"}
    assert u.members_union("2020-01-01", "2022-01-01") == {
        "aaa", "bbb", "ccc-d", "www",
    }


def test_filter_point_in_time_on_panel():
    """Rows for names not yet (or no longer) in the index are dropped."""
    panel = SyntheticRegimePanel(SyntheticSpec(seed=0)).generate(
        n_dates=4, n_entities=2
    )
    # bolt calendar labels onto the synthetic panel: dates straddle the 2021
    # change; entities are WWW (leaves) and CCC.D (joins)
    panel.date_labels = ("2021-05-28", "2021-05-31", "2021-06-01", "2021-06-02")
    panel.entity_labels = ("www", "ccc-d")
    u = _fixture_universe()
    filtered = filter_point_in_time(panel, u)
    # 4 dates x 2 names = 8 rows; each date keeps exactly one of the two
    assert len(filtered) == 4
    kept = {
        (panel.date_labels[int(d)], panel.entity_labels[int(e)])
        for d, e in zip(filtered.date, filtered.entity, strict=True)
    }
    assert kept == {
        ("2021-05-28", "www"),
        ("2021-05-31", "www"),
        ("2021-06-01", "ccc-d"),
        ("2021-06-02", "ccc-d"),
    }
    # date-major order preserved
    assert torch.equal(filtered.date, filtered.date.sort().values)


def test_filter_requires_labels():
    panel = SyntheticRegimePanel(SyntheticSpec(seed=0)).generate(
        n_dates=3, n_entities=2
    )
    with pytest.raises(ValueError, match="date_labels"):
        filter_point_in_time(panel, _fixture_universe())


def test_coverage_report_math():
    u = _fixture_universe()
    rep = universe_coverage_report(
        u, dates=["2014-12-31", "2023-01-01"], available={"aaa", "bbb"}
    )
    # 2014: members {aaa, yyy, www}, we have data only for aaa -> 1/3
    assert rep.loc["2014-12-31", "coverage"] == pytest.approx(1 / 3)
    # 2023: members {aaa, bbb, ccc-d}, we have 2 -> 2/3
    assert rep.loc["2023-01-01", "coverage"] == pytest.approx(2 / 3)
    assert rep.loc["2023-01-01", "members"] == 3


# --------------------------------------------------------------------------- #
# Live network check — opt-in only (NEC_NETWORK_TESTS=1)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    os.environ.get("NEC_NETWORK_TESTS", "0") != "1",
    reason="network test; set NEC_NETWORK_TESTS=1 to run",
)
def test_live_sp500_universe(tmp_path):
    u = load_sp500_universe(tmp_path)
    now = u.members_asof("2024-12-31")
    assert 490 <= len(now) <= 520  # ~503 share classes
    assert len(u.changes) > 250
    assert {"aapl", "msft"} <= now
    # the famous one: TSLA joined 2020-12-21
    assert "tsla" not in u.members_asof("2020-01-01")
    assert "tsla" in u.members_asof("2021-01-01")
    # coverage of the survivorship-biased default universe, quantified
    from nec_moe import DEFAULT_UNIVERSE

    rep = universe_coverage_report(
        u, ["2015-01-02", "2024-12-31"], set(DEFAULT_UNIVERSE)
    )
    assert (rep["coverage"] < 0.12).all()  # ~30 names of ~500: the honest number
