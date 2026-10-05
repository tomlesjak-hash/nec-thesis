"""The CRSP data layer (brief 06 section A), one test per construction rule.

Fixture tests run on ``crsp_fixture``: CIZ-format files with the real headers
and **invented numbers** (licence rule: no real CRSP row in any test). Tests
marked ``crsp_data`` read the licensed files under ``Quant Model/Data/`` and
skip when they are absent, so CI passes without them; they assert and print
aggregate statistics only.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("pyarrow")

import crsp_fixture as fx  # noqa: E402
from conftest import small_config  # noqa: E402

import nec_moe.crsp as crsp  # noqa: E402
from nec_moe import (  # noqa: E402
    CRSPSpec,
    NECConfig,
    StageBSpec,
    SyntheticRegimePanel,
    SyntheticSpec,
    TickerLookup,
    TrialRegistry,
    build_crsp_panel,
    data_config_from_panel,
    extract_crsp,
    extract_return_duration_flags,
    flag_meanings,
    load_extract,
    membership_universe,
    nec_arm,
    read_index_returns,
    read_membership,
    run_sweep,
)
from nec_moe.config import EncoderConfig, ExpertConfig, TrainConfig  # noqa: E402
from nec_moe.crsp import crsp_daily_frames, crsp_file_source, open_crsp_file  # noqa: E402
from nec_moe.features import (  # noqa: E402
    _split_invariant_volume_z,
    market_frame,
    stock_features,
)
from nec_moe.train import DeadParameterWarning  # noqa: E402

# the raw target: the delisting and fill rules below are rules about the raw
# forward return; the market-neutral target built on it (Q25) is tested on
# its own, as builds["cash_mn"]
STAGE_B = StageBSpec(seq_len=10, horizon=5, min_names_per_date=4, target_kind="raw")
H = STAGE_B.horizon


def _spec(root: Path, **kw) -> CRSPSpec:
    base = dict(
        crsp_dir=str(root), release=fx.RELEASE, start=fx.START, end=fx.END,
        chunk_bytes=20_000, member_count_min=5, member_count_max=8,
    )
    return CRSPSpec(**(base | kw))


@pytest.fixture(scope="module")
def fixture(tmp_path_factory) -> fx.Fixture:
    return fx.write_fixture(tmp_path_factory.mktemp("crsp") / "Data")


@pytest.fixture(scope="module")
def extract(fixture: fx.Fixture) -> crsp.CRSPExtract:
    spec = _spec(fixture.root)
    extract_crsp(spec, verbose=False)
    return load_extract(spec)


@pytest.fixture(scope="module")
def builds(fixture: fx.Fixture, extract: crsp.CRSPExtract) -> dict[str, crsp.CRSPBuild]:
    """The fixture panel under both post-delisting fills, raw and residual
    target, and the market-neutral target under the cash fill."""
    out = {}
    res = dataclasses.replace(STAGE_B, target_kind="residual", beta_window=60)
    mn = dataclasses.replace(STAGE_B, target_kind="market_neutral")
    for fill in ("cash", "market"):
        spec = _spec(fixture.root, post_delisting_return=fill)
        out[fill] = build_crsp_panel(spec, STAGE_B, extract=extract, verbose=False)
        out[f"{fill}_residual"] = build_crsp_panel(spec, res, extract=extract, verbose=False)
    out["cash_mn"] = build_crsp_panel(
        _spec(fixture.root, post_delisting_return="cash"), mn, extract=extract, verbose=False
    )
    return out


def _targets(build: crsp.CRSPBuild, permno: int) -> pd.Series:
    """One PERMNO's panel targets by date."""
    p = build.panel
    assert p.entity_labels is not None and p.date_labels is not None
    code = p.entity_labels.index(str(permno))
    m = p.entity == code
    dates = pd.DatetimeIndex([p.date_labels[int(d)] for d in p.date[m]])
    return pd.Series(p.y[m].double().numpy(), index=dates)


def _log_ret(fixture: fx.Fixture, permno: int, days) -> float:
    return float(np.log1p(fixture.returns[permno].loc[days]).sum())


def _mkt_log_ret(fixture: fx.Fixture, days, indno: int = fx.MEMBER_INDNO) -> float:
    return float(np.log1p(fixture.market[indno].loc[days]).sum())


# --------------------------------------------------------------------------- #
# Config and file access
# --------------------------------------------------------------------------- #


def test_spec_defaults_are_the_brief_and_the_approved_indno():
    s = CRSPSpec()
    assert (s.stock_file, s.market_indno, s.membership_indno) == (
        "StkDlySecurityData", 1000500, 1000500,  # brief 08 C.1: OHLC and quotes
    )
    assert (s.universe, s.start, s.end) == ("sp500", "2015-01-01", "2024-12-31")
    assert s.post_delisting_return == "cash"
    assert (s.member_count_min, s.member_count_max) == (500, 510)  # observed 502-508, +-2
    assert s.data_source == "crsp_ciz202512"
    assert Path(s.crsp_dir).name == "Data"
    assert s.derived_dir == Path(s.crsp_dir) / "derived"  # derived data stays inside Data/
    assert s.panel_path.name == "pit_panel_crsp_2015-01-01_2024-12-31.pt"
    for bad in (dict(universe="r3000"), dict(post_delisting_return="zero"),
                dict(member_count_min=510, member_count_max=500),
                dict(start="2020-01-01", end="2019-01-01")):
        with pytest.raises(ValueError):
            dataclasses.replace(s, **bad).validate()


def test_zip_and_extracted_copy_yield_identical_bytes(tmp_path: Path):
    plain = fx.write_fixture(tmp_path / "plain")
    zipped = fx.write_fixture(tmp_path / "zipped", as_zip=True, extracted=False)
    for name in fx.HEADERS:
        a_src = crsp_file_source(_spec(plain.root), name)
        b_src = crsp_file_source(_spec(zipped.root), name)
        assert a_src[1] is None and b_src[1] == f"crspdata/{fx.RELEASE}_ascii/{name}.dat"
        with open_crsp_file(_spec(plain.root), name) as a, \
                open_crsp_file(_spec(zipped.root), name) as b:
            assert a.read() == b.read()
    with pytest.raises(FileNotFoundError):
        crsp_file_source(_spec(tmp_path / "nowhere"), "StkDelists")


def test_extract_keeps_member_rows_in_bounds_and_resumes(tmp_path: Path, monkeypatch):
    """The stock stream keeps only the window's members inside the extract
    bounds, and an interrupted extraction resumes to the same result."""
    clean_fx = fx.write_fixture(tmp_path / "clean")
    extract_crsp(_spec(clean_fx.root), verbose=False)
    clean = load_extract(_spec(clean_fx.root))
    lo, hi = _spec(clean_fx.root).extract_bounds
    # the members, plus 10010 (a non-member share class of a member's
    # company, kept for market equity); never 10009
    assert sorted(clean.stock["PERMNO"].unique()) == [*range(10001, 10009), fx.SIBLING]
    assert clean.info["n_company_permnos"] == 1
    assert sorted(clean.spells["PERMNO"].unique()) == list(range(10001, 10009))
    assert clean.stock["DlyCalDt"].between(lo, hi).all()
    assert len(clean.stock) == len(clean.adj)

    flaky_fx = fx.write_fixture(tmp_path / "flaky")
    real = crsp._arrow()
    calls = {"n": 0}

    class Interrupting:
        def __getattr__(self, name):
            return getattr(real[2], name)

        def read_csv(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 7:  # 4 small tables, then the 3rd stock block
                raise RuntimeError("interrupted")
            return real[2].read_csv(*args, **kwargs)

    monkeypatch.setattr(crsp, "_arrow", lambda: (real[0], real[1], Interrupting(), real[3]))
    with pytest.raises(RuntimeError, match="interrupted"):
        extract_crsp(_spec(flaky_fx.root), verbose=False)
    monkeypatch.undo()
    partial = crsp.json.loads(
        (_spec(flaky_fx.root).extract_dir / "stock" / "manifest.json").read_text()
    )
    assert not partial["complete"] and partial["parts"] == 2 and partial["offset"] > 0
    extract_crsp(_spec(flaky_fx.root), verbose=False)
    resumed = load_extract(_spec(flaky_fx.root))
    cols = ["PERMNO", "DlyCalDt", "DlyRet", "DlyVol"]
    pd.testing.assert_frame_equal(
        resumed.stock[cols].reset_index(drop=True), clean.stock[cols].reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# A.4 construction rules
# --------------------------------------------------------------------------- #


def test_market_series_comes_from_the_configured_indno(tmp_path: Path):
    f = fx.write_fixture(tmp_path / "Data")
    spec = _spec(f.root, market_indno=fx.OTHER_INDNO)
    assert np.allclose(
        read_index_returns(spec, fx.OTHER_INDNO).to_numpy(), f.market[fx.OTHER_INDNO].to_numpy()
    )
    extract_crsp(spec, verbose=False)
    ex = load_extract(spec)
    _, mkt_ret, _ = crsp_daily_frames(ex, spec, STAGE_B)
    expected = np.log1p(f.market[fx.OTHER_INDNO].reindex(mkt_ret.index))
    assert np.allclose(mkt_ret.to_numpy(), expected.to_numpy())
    assert not np.allclose(
        mkt_ret.to_numpy(), np.log1p(f.market[fx.MEMBER_INDNO].reindex(mkt_ret.index))
    )
    with pytest.raises(ValueError, match="INDNO 999"):
        read_index_returns(spec, 999)


def test_membership_bounds_are_inclusive(fixture: fx.Fixture, builds):
    u = membership_universe(read_membership(_spec(fixture.root)))
    before_join = fx.CAL[fx.CAL.get_loc(fx.JOIN) - 1]
    after_leave = fx.CAL[fx.CAL.get_loc(fx.LEAVE) + 1]
    assert "10007" in u.members_asof(fx.JOIN) and "10007" not in u.members_asof(before_join)
    assert "10008" in u.members_asof(fx.LEAVE) and "10008" not in u.members_asof(after_leave)
    assert "10009" not in u.members_union(fx.START, fx.END)  # a member of the other INDNO only
    joined, left = _targets(builds["cash"], 10007), _targets(builds["cash"], 10008)
    assert joined.index.min() == fx.JOIN  # features used its history before it joined
    assert left.index.max() == fx.LEAVE


def test_missing_return_is_missing_never_zero(fixture, extract, builds):
    frames, _, counts = crsp_daily_frames(extract, _spec(fixture.root), STAGE_B)
    daily = frames["10003"]
    assert math.isnan(daily.loc[fx.MISSING_DAY, "ret"])
    f = stock_features(daily, market_frame(np.log1p(extract.market), STAGE_B), STAGE_B)
    assert math.isnan(f.loc[fx.MISSING_DAY, "ret_1d"])
    assert math.isnan(f.loc[fx.MISSING_DAY, "ret_20d"])  # not the sum of 19 returns
    targets = _targets(builds["cash"], 10003)
    i = fx.CAL.get_loc(fx.MISSING_DAY)
    needs_it = fx.CAL[i - H : i + 1]  # targets of the h days before, and the day itself
    assert not targets.index.isin(needs_it).any()
    assert fx.CAL[i - H - 3 : i - H].isin(targets.index).all()  # rows just outside survive
    assert counts["missing_returns"] == 2  # the missing day and 10006's delisting row


def test_split_leaves_the_volume_zscore_unchanged():
    """Brief 06 A.4.6: a synthetic series with a 2-for-1 split gives the same
    z-score as the same series without the split."""
    g = np.random.default_rng(3)
    idx = pd.bdate_range("2021-01-04", periods=80)
    v = pd.Series(1e6 * np.exp(g.normal(0, 0.3, 80)), index=idx)
    split = idx >= idx[40]
    v_split = v.where(~split, 2.0 * v)  # twice the shares after the split
    factor = pd.Series(np.where(split, 1.0, 2.0), index=idx)  # anchored at the end
    z_plain = _split_invariant_volume_z(v, pd.Series(1.0, index=idx), 20)
    z_split = _split_invariant_volume_z(v_split, factor, 20)
    assert np.allclose(z_plain, z_split, equal_nan=True)
    assert z_plain.iloc[19:].notna().all()
    # without the factor the split shows up as a jump
    z_naive = _split_invariant_volume_z(v_split, pd.Series(1.0, index=idx), 20)
    assert not np.allclose(z_naive.iloc[40:60], z_plain.iloc[40:60])
    # and the factor's anchor does not matter: scale it and nothing moves
    assert np.allclose(_split_invariant_volume_z(v_split, 7.0 * factor, 20), z_plain,
                       equal_nan=True)


def test_split_on_the_fixture_leaves_volume_and_dollar_volume_continuous(fixture, extract):
    frames, mkt_ret, _ = crsp_daily_frames(extract, _spec(fixture.root), STAGE_B)
    daily = frames["10004"]
    unsplit = daily.copy()
    after = unsplit.index >= fx.SPLIT_DAY
    unsplit.loc[after, "volume"] /= 2.0
    unsplit["share_factor"] = 1.0
    mkt = market_frame(mkt_ret, STAGE_B)
    a, b = stock_features(daily, mkt, STAGE_B), stock_features(unsplit, mkt, STAGE_B)
    assert np.allclose(a["volume_z_20d"], b["volume_z_20d"], equal_nan=True)
    # price halves and shares double: the dollars traded do not jump
    i = daily.index.get_loc(fx.SPLIT_DAY)
    ratio = daily["dollar_volume"].iloc[i] / daily["dollar_volume"].iloc[i - 1]
    raw = daily["volume"].iloc[i] / daily["volume"].iloc[i - 1]
    assert ratio == pytest.approx(raw / 2.0 * math.exp(daily["ret"].iloc[i]), rel=1e-3)


def test_no_lookahead_on_the_crsp_fixture(fixture, extract):
    """Every feature at t is unchanged when all data after t is deleted."""
    frames, mkt_ret, _ = crsp_daily_frames(extract, _spec(fixture.root), STAGE_B)
    for permno in ("10003", "10004", "10005"):
        daily = frames[permno]
        full = stock_features(daily, market_frame(mkt_ret, STAGE_B), STAGE_B)
        # the forward-looking columns: the target and the market's forward
        # return that only the residual target uses; everything else is a feature
        cols = [c for c in full.columns if c not in (STAGE_B.target, "mkt_fwd_ret")]
        for cut in (fx.MISSING_DAY, fx.SPLIT_DAY, fx.DELIST_LAST):
            trunc = stock_features(
                daily.loc[:cut], market_frame(mkt_ret.loc[:cut], STAGE_B), STAGE_B
            )
            assert np.allclose(
                full.loc[cut, cols].to_numpy(float), trunc.loc[cut, cols].to_numpy(float),
                equal_nan=True,
            ), f"future data changed a feature of {permno} at {cut.date()}"


def test_the_delisting_row_is_never_a_panel_row(fixture, extract):
    frames, _, _ = crsp_daily_frames(extract, _spec(fixture.root), STAGE_B)
    d = fx.CAL[fx.CAL.get_loc(fx.DELIST_LAST) + 1]
    daily = frames["10005"]
    assert not daily.loc[d, "tradable"]
    assert daily.loc[d, "ret"] == pytest.approx(math.log1p(fx.DELIST_RET))
    assert not daily.loc[daily.index > d, "tradable"].any()  # the fill days


# --------------------------------------------------------------------------- #
# Delistings and the post-delisting fill (decided 2026-09-26)
# --------------------------------------------------------------------------- #


def test_delisting_two_days_into_the_window_keeps_its_row(fixture, builds):
    """Fill test 1: the target is the compounded return through the delisting
    row plus the post-delisting fill, and the row is kept."""
    d = fx.CAL.get_loc(fx.DELIST_LAST) + 1  # the delisting row
    t = fx.CAL[d - 2]  # the delisting row is day 2 of t's 5-day window
    own = fx.CAL[d - 1 : d + 1]  # the last trade and the delisting row
    filled = fx.CAL[d + 1 : d + 1 + (H - 2)]
    through_delisting = _log_ret(fixture, 10005, own)
    assert _targets(builds["cash"], 10005)[t] == pytest.approx(through_delisting, abs=1e-6)
    assert _targets(builds["market"], 10005)[t] == pytest.approx(
        through_delisting + _mkt_log_ret(fixture, filled), abs=1e-6
    )


@pytest.mark.parametrize("kind", ["", "_residual"])
def test_cash_and_market_fills_differ_by_the_market_return_over_filled_days(
    fixture, builds, kind
):
    """Fill test 2, for the raw and the residual target: the residual is
    computed on the completed window, so its beta term cancels."""
    cash, market = _targets(builds[f"cash{kind}"], 10005), _targets(builds[f"market{kind}"], 10005)
    assert cash.index.equals(market.index)
    d = fx.CAL.get_loc(fx.DELIST_LAST) + 1
    touched = 0
    for t in cash.index:
        i = fx.CAL.get_loc(t)
        window = fx.CAL[i + 1 : i + 1 + H]
        fill_days = window[window > fx.CAL[d]]
        diff = market[t] - cash[t]
        assert diff == pytest.approx(_mkt_log_ret(fixture, fill_days), abs=1e-6)
        touched += len(fill_days) > 0
    assert touched == 4  # t = the last 4 trading days before the delisting row


def test_no_row_is_dropped_for_forward_returns_after_the_final_return(fixture, builds):
    """Fill test 3: every row whose only missing forward returns lie after the
    final CRSP return is in the panel: the stock's last h member days."""
    d = fx.CAL.get_loc(fx.DELIST_LAST) + 1
    last_member_days = fx.CAL[d - H : d]  # through DELIST_LAST, inclusive
    for fill in ("cash", "market"):
        assert last_member_days.isin(_targets(builds[fill], 10005).index).all()
        assert builds[fill].report["post_delisting_fill_rows"] == 4


def test_a_delisting_without_a_return_stays_missing(fixture, builds):
    """10006's delisting return is missing: nothing is imputed, the targets
    whose window holds it stay missing, and the report counts it with codes."""
    d = fx.CAL.get_loc(fx.DELIST2_LAST) + 1
    needs_it = fx.CAL[d - H : d]
    assert not _targets(builds["cash"], 10006).index.isin(needs_it).any()
    assert _targets(builds["cash"], 10006).index.max() < needs_it[0]
    rep = builds["cash"].report
    assert rep["member_delistings"] == 2
    assert rep["member_delistings_without_return"] == 1
    assert rep["without_return_codes"] == {"MER/ABC": 1}


# --------------------------------------------------------------------------- #
# Diagnostic: rows dropped by a missing return, past versus future
# --------------------------------------------------------------------------- #

GAP1 = pd.Timestamp("2020-04-01")  # 10002: two missing returns three trading days apart
GAP2 = fx.CAL[fx.CAL.get_loc(GAP1) + 3]


@pytest.fixture(scope="module")
def gap_build(tmp_path_factory) -> crsp.CRSPBuild:
    f = fx.write_fixture(tmp_path_factory.mktemp("gaps") / "Data",
                         extra_missing={10002: (GAP1, GAP2)})
    spec = _spec(f.root)
    extract_crsp(spec, verbose=False)
    extract_return_duration_flags(spec, verbose=False)
    return build_crsp_panel(spec, STAGE_B, extract=load_extract(spec), verbose=False)


def test_missing_return_split_separates_past_and_future_windows(gap_build):
    """Exact counts. 10003 misses one return: the 5 rows before it lose their
    target (b), every row from it to the window's end loses its features (a).
    10002 misses two, 3 days apart: 5 rows (b) before the first, 3 rows (c)
    between them (the first in the past window, the second in the future
    one), (a) from the second on. 10006's missing delisting return costs its
    last 5 member rows (d), on their own line."""
    split = gap_build.report["missing_return_split"]
    end = pd.Timestamp(fx.END)

    def days(a: pd.Timestamp, b: pd.Timestamp) -> int:
        return int(((fx.CAL >= a) & (fx.CAL <= b)).sum())

    expected = {
        "a_backward_only": days(fx.MISSING_DAY, end) + days(GAP2, end),
        "b_forward_only": 2 * H,
        "c_both": fx.CAL.get_loc(GAP2) - fx.CAL.get_loc(GAP1),
        "d_missing_delisting_return": H,
    }
    totals = split["totals"]
    assert {k: totals[k] for k in expected} == expected
    assert totals["other_invalid"] == 0 and totals["unexplained"] == 0
    assert totals["panel_rows"] == len(gap_build.panel)
    assert split["by_year"] == {"2020": expected}
    assert split["b_share_of_panel_rows"] == pytest.approx(2 * H / len(gap_build.panel))
    assert (split["backward_window_returns"], split["forward_window_returns"]) == (120, H)
    # behind (b): 10003's missing day and both of 10002's (the second sits in
    # the forward window of the last (b) rows), three one-day gaps, and the
    # first return after each spans its gap
    assert split["behind_b_missing_returns"] == 3 and split["behind_b_gaps"] == 3
    assert split["behind_b_missing_codes"] == {"NT": 3}
    assert split["next_return_duration_flags"] == {"P1": 3}


def test_flag_meanings_come_from_meta_flag_info(fixture):
    rd = flag_meanings(_spec(fixture.root), "RD")
    assert rd == {"D1": "fixture: one day", "P1": "fixture: two trading days",
                  "MR": "fixture: missing"}
    assert flag_meanings(_spec(fixture.root), "RM")["NT"] == "fixture: not tracked"


def test_missing_return_split_changes_nothing_in_the_panel(fixture, builds):
    """The diagnostic reads the panel's rows and never alters them: the
    build without extra gaps still has only 10003's and 10006's losses."""
    split = builds["cash"].report["missing_return_split"]
    assert split["totals"]["b_forward_only"] == H
    assert split["totals"]["d_missing_delisting_return"] == H
    assert split["totals"]["c_both"] == 0
    # the duration flags now come with the stock rows (StkDlySecurityData,
    # brief 08 C.1): the gap after 10003's missing day spans one extra day
    assert split["next_return_duration_flags"] == {"P1": 1}


# --------------------------------------------------------------------------- #
# Labels, report, provenance
# --------------------------------------------------------------------------- #


def test_entity_labels_are_permnos_and_tickers_are_date_aware(extract, builds):
    labels = builds["cash"].panel.entity_labels
    assert labels is not None and set(labels) <= {str(p) for p in range(10001, 10009)}
    assert all(label.isdigit() for label in labels)
    lookup = TickerLookup(extract.security_info)
    assert lookup(10007, "2020-03-13") == "OLDT"
    assert lookup("10007", "2020-03-16") == "NEWT"
    assert lookup(99999, "2020-03-16") is None


def test_build_report_and_panel_metadata(builds):
    b = builds["cash"]
    rep = b.report
    assert rep["data_source"] == b.panel.data_source == f"crsp_{fx.RELEASE}"
    assert rep["dual_class_companies"] == 1  # 10001 and 10002 share a PERMCO: both kept
    assert rep["ever_members"] == 8
    assert rep["members_per_date"]["min"] >= 5 and rep["members_per_date"]["max"] <= 8
    meta = b.panel.metadata
    assert meta["crsp_release"] == fx.RELEASE and meta["post_delisting_return"] == "cash"
    assert (meta["market_indno"], meta["membership_indno"]) == (fx.MEMBER_INDNO, fx.MEMBER_INDNO)
    assert list(b.coverage.columns) == ["members", "with_rows", "without_rows", "coverage"]
    # the rank re-normalization among the members (audit D-1) still holds
    assert float(b.panel.x_snap.min()) >= -0.5 and float(b.panel.x_snap.max()) <= 0.5


@pytest.mark.parametrize("kind", ["cash", "market", "cash_residual"])
def test_daily_forward_returns_sum_to_the_target_through_the_fill(builds, kind):
    """``Panel.y_daily`` (audit E-3) carries the target's daily pieces, the
    post-delisting fill included, and they sum to the target."""
    panel = builds[kind].panel
    assert panel.y_daily is not None and panel.y_daily.shape == (len(panel), H)
    assert torch.allclose(panel.y_daily.sum(dim=1), panel.y, atol=1e-5)


def test_build_refuses_a_short_extract_and_an_out_of_band_count(fixture, extract):
    spec = _spec(fixture.root)
    market = extract.market
    short_back = dataclasses.replace(extract, market=market[market.index >= "2019-12-01"])
    with pytest.raises(ValueError, match="extract_lookback_days"):
        build_crsp_panel(spec, STAGE_B, extract=short_back, verbose=False)
    short_fwd = dataclasses.replace(extract, market=market[market.index <= fx.END])
    with pytest.raises(ValueError, match="extract_lead_days"):
        build_crsp_panel(spec, STAGE_B, extract=short_fwd, verbose=False)
    with pytest.raises(ValueError, match="outside the band"):
        build_crsp_panel(dataclasses.replace(spec, member_count_max=6), STAGE_B,
                         extract=extract, verbose=False)


def test_data_source_and_fill_reach_every_registry_row(builds, tmp_path: Path):
    """A.6: every sweep row names its source; the fill switch travels with it."""
    panel = builds["market"].panel
    cfg = NECConfig(
        data=data_config_from_panel(panel),
        encoder=EncoderConfig(hidden_dim=8),
        experts=ExpertConfig(hidden_dims=(8, 4), dropout=0.0),
        train=TrainConfig(sigma_init=0.1, sigma_freeze_steps=0, batch_size=128),
    )
    synth = SyntheticRegimePanel(SyntheticSpec(seed=1)).generate(60, 6)
    assert synth.data_source == "synthetic"
    for tag, p, c, source, fill in (
        ("crsp", panel, cfg, f"crsp_{fx.RELEASE}", "market"),
        ("synth", synth, small_config(), "synthetic", None),  # None: not applicable
    ):
        reg = TrialRegistry(tmp_path / f"{tag}.jsonl")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeadParameterWarning)
            run_sweep(p, [nec_arm("soft", c)], seeds=(0,), registry=reg, tag=tag,
                      steps=3, n_folds=2, test_dates_per_fold=10,
                      purge_dates=H if p is panel else 0, verbose=False)
        rows = reg.trials(tag)
        assert rows
        assert all(r.config["data_source"] == source for r in rows)
        assert all(r.config["post_delisting_return"] == fill for r in rows)


# --------------------------------------------------------------------------- #
# Real data (skipped without the licensed files); aggregate statistics only
# --------------------------------------------------------------------------- #

REAL = CRSPSpec()


def _real_files() -> bool:
    try:
        crsp_file_source(REAL, "StkIndMembership")
    except FileNotFoundError:
        return False
    return True


needs_crsp = pytest.mark.skipif(not _real_files(), reason="CRSP files not under Data/")
needs_extract = pytest.mark.skipif(
    not (REAL.extract_dir / "extract.json").exists(),
    reason="no CRSP extract: run scripts/extract_crsp.py",
)


@pytest.mark.crsp_data
@needs_crsp
def test_real_member_count_stays_in_the_band():
    spells = read_membership(REAL)
    calendar = read_index_returns(REAL, REAL.market_indno).index
    window = calendar[(calendar >= REAL.start) & (calendar <= REAL.end)]
    counts = membership_universe(spells).count_by_date(pd.DatetimeIndex(window))
    print(f"\n[crsp] members per date {counts.min()}..{counts.max()} over {len(window)} days")
    assert REAL.member_count_min <= counts.min() and counts.max() <= REAL.member_count_max
    with pytest.raises(ValueError, match="no membership rows"):  # the brief's first guess
        read_membership(dataclasses.replace(REAL, membership_indno=1000502))


@pytest.mark.crsp_data
@needs_extract
def test_real_delisting_returns_are_already_in_dlyret():
    """A.4.2: on every delisting-return row of the extract, ``DlyRet`` equals
    ``DelRet``, and the two are missing together."""
    ex = load_extract(REAL)
    rows = ex.stock[ex.stock["DlyDelFlg"] == "Y"][["PERMNO", "DlyCalDt", "DlyRet"]]
    joined = ex.delists.merge(rows, left_on=["PERMNO", "DelDlyDt"],
                              right_on=["PERMNO", "DlyCalDt"], how="inner")
    assert len(joined) == len(rows) > 0  # every delisting row has its record
    both = joined["DelRet"].notna() & joined["DlyRet"].notna()
    assert (joined["DelRet"].isna() == joined["DlyRet"].isna()).all()
    assert np.allclose(joined.loc[both, "DelRet"], joined.loc[both, "DlyRet"], atol=1e-12, rtol=0)
    print(f"\n[crsp] {int(both.sum())} delisting rows with DlyRet == DelRet, "
          f"{int((~both).sum())} missing in both")


@pytest.mark.crsp_data
@needs_extract
def test_real_one_year_panel_builds():
    spec = dataclasses.replace(REAL, start="2019-01-01", end="2019-12-31")
    b = build_crsp_panel(spec, StageBSpec(), extract=load_extract(REAL), verbose=False)
    rep = b.report
    assert rep["data_source"] == "crsp_ciz202512"
    assert 240 <= rep["dates"] <= 260 and rep["rows"] > 400 * rep["dates"]
    assert REAL.member_count_min <= rep["members_per_date"]["min"]
    assert rep["members_per_date"]["max"] <= REAL.member_count_max
