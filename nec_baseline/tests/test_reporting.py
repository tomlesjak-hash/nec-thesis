"""Brief 12: the reporting additions, on synthetic panels only.

The report runs after a run, from what it saved; nothing here touches the
real panel.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nec_moe import reporting as rp
from nec_moe.evaluation import long_short_book, rank_ic_by_date

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rr():
    mod = _load("report_run")
    yield mod
    sys.modules.pop("report_run_script", None)


def _cross_section(t_len=60, n=40, h=5, seed=0):
    """Dates x names: forecasts, an h-day target and its daily pieces."""
    rng = np.random.default_rng(seed)
    date = torch.arange(t_len).repeat_interleave(n)
    entity = torch.arange(n).repeat(t_len)
    pred = torch.from_numpy(rng.normal(size=t_len * n)).float()
    y_daily = torch.from_numpy(rng.normal(0, 0.01, (t_len * n, h))).float()
    y = y_daily.sum(dim=1)
    return pred, y, date, entity, y_daily


# --------------------------------------------------------------------------- #
# Part A: long leg against short leg
# --------------------------------------------------------------------------- #


def test_the_legs_sum_to_the_existing_spread():
    pred, y, date, entity, y_daily = _cross_section()
    for scheme in ("nonoverlapping", "staggered"):
        dates, gross, tno, long, short = rp.leg_returns(
            pred, y, date, entity, n_quantiles=5, horizon=5, scheme=scheme, y_daily=y_daily)
        d0, g0, t0 = long_short_book(pred, y, date, entity, n_quantiles=5, horizon=5,
                                     scheme=scheme, y_daily=y_daily)
        # the book itself is the existing one, bit for bit
        assert torch.equal(dates, d0) and torch.equal(gross, g0) and torch.equal(tno, t0)
        if scheme == "nonoverlapping":
            assert torch.equal(long + short, gross)  # exactly, in the book's float32
        else:
            torch.testing.assert_close(long + short, gross, atol=1e-7, rtol=1e-6)


def test_a_signal_that_only_ranks_the_losers_lives_in_the_short_leg():
    """Only the bottom decile by forecast underperforms, more so the lower
    the forecast; the rest is noise. Market-neutral (demeaned) target."""
    rng = np.random.default_rng(1)
    t_len, n = 400, 100
    date = torch.arange(t_len).repeat_interleave(n)
    entity = torch.arange(n).repeat(t_len)
    pred = rng.normal(size=(t_len, n))
    cut = np.quantile(pred, 0.1, axis=1, keepdims=True)
    effect = np.where(pred < cut, 0.02 * (pred - cut) - 0.01, 0.0)
    y = effect + rng.normal(0, 0.01, (t_len, n))
    y = y - y.mean(axis=1, keepdims=True)
    pred_t, y_t = torch.from_numpy(pred.ravel()).float(), torch.from_numpy(y.ravel()).float()
    _, _, _, long, short = rp.leg_returns(pred_t, y_t, date, entity, n_quantiles=10,
                                          horizon=1)
    s_long, s_short = rp.series_summary(long, 0), rp.series_summary(short, 0)
    assert s_short.t > 20 and s_short.mean > 0.01
    assert abs(s_long.mean) < 0.15 * s_short.mean  # only the demeaning offset
    (_, ic_lo), (_, ic_hi) = rp.half_sample_ics(pred_t, y_t, date)
    assert float(ic_lo.mean()) > 0.15 and abs(float(ic_hi.mean())) < 0.02


def test_half_sample_ics_rank_within_each_half():
    pred, y, date, _, _ = _cross_section(t_len=5, n=11)
    (dl, il), (du, iu) = rp.half_sample_ics(pred, y, date)
    for d in range(5):
        m = date == d
        order = torch.argsort(pred[m])
        lo, hi = order[:5], order[-5:]  # 11 names: the median one is dropped
        _, want_lo = rank_ic_by_date(pred[m][lo], y[m][lo], torch.zeros(5, dtype=torch.long))
        _, want_hi = rank_ic_by_date(pred[m][hi], y[m][hi], torch.zeros(5, dtype=torch.long))
        assert float(il[d]) == pytest.approx(float(want_lo[0]))
        assert float(iu[d]) == pytest.approx(float(want_hi[0]))


def test_series_summary_is_hansen_hodrick():
    from nec_moe.evaluation import ic_summary

    x = torch.from_numpy(np.random.default_rng(2).normal(0.01, 0.1, 500))
    s = rp.series_summary(x, 4)
    ref = ic_summary(x, 4, "uniform")
    assert s.mean == pytest.approx(ref.mean_ic) and s.t == pytest.approx(ref.t_stat)
    assert s.std == pytest.approx(ref.ic_std) and (s.n, s.lags, s.kernel) == (500, 4, "uniform")
    assert np.isnan(rp.series_summary(np.array([1.0]), 0).se)


# --------------------------------------------------------------------------- #
# The report on a finished synthetic run
# --------------------------------------------------------------------------- #


def _digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    """A finished evaluate run: a Hamilton-gated residual arm and a ridge
    baseline, two count folds, a quintile book, calendar dates in 2008."""
    import run_experiment as rx
    from nec_moe.train import DeadParameterWarning

    tmp = tmp_path_factory.mktemp("run")
    exp = rx.Experiment(
        tag="report", campaign="unit", mode="evaluate", out_dir=str(tmp / "results"),
        per_security_dir=str(tmp / "Data" / "derived" / "runs"), data="synthetic",
        synth_dates=260, synth_entities=20, synth_regime_process="markov",
        synth_calendar_start="2008-01-02", fold_scheme="count", n_folds=2,
        test_dates_per_fold=40, prior="markov", base_enabled=True, correction_mode=True,
        freeze_gate=True, gate_series="sequence_channel", gate_series_channel=0,
        gate_start_vol_quantiles=(0.6,), gate_start_vol_windows=(10,),
        gate_start_persistences=(0.95,), gate_start_draws_per_centre=1,
        gate_start_min_history=5, gate_start_min_group_size=5, base_steps=30,
        base_batch_size=64, expert_hidden_dims=(8, 4), encoder_hidden=8, steps=20,
        seeds=(0,), include_ridge=True, include_mlp=False, backtest_quantiles=5,
        figures=False, calibration=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        warnings.simplefilter("ignore", UserWarning)
        result = rx.main(exp)
    return {"exp": exp, "run_id": result["run_id"], "run_dir": Path(result["run_dir"]),
            "root": tmp / "results", "data": tmp / "Data"}


def test_the_report_reproduces_the_runs_own_metrics_and_changes_no_file(rr, synthetic_run):
    root, data = synthetic_run["root"], synthetic_run["data"]
    before = _digest(root) | {f"DATA/{k}": v for k, v in _digest(data).items()}
    inputs = rr.load_inputs(synthetic_run["run_id"], root)
    assert set(inputs.groups) == {"markov_seed0", "ridge_seed0"}
    frame = rr.report(inputs, rr.ReportSettings())
    rr.write_report(inputs, frame, rr.ReportSettings())
    after = _digest(root) | {f"DATA/{k}": v for k, v in _digest(data).items()}
    # nothing the run wrote has changed; only the report's own files are new
    assert {k: v for k, v in after.items() if k in before} == before
    new = sorted(set(after) - set(before))
    assert [Path(k).name for k in new if "metrics" in k] == [
        "reporting.csv", "reporting_across_seeds.csv", "reporting_settings.json"]
    assert all("/metrics/" in k for k in new)

    run_dir = synthetic_run["run_dir"]
    for group in ("markov_seed0", "ridge_seed0"):
        pooled = json.loads((run_dir / "metrics" / f"{group}_pooled.json").read_text())
        rows = frame[(frame["group"] == group) & (frame["predictor"] == "model")
                     & (frame["execution_lag"] == 0) & (frame["slice"] == "all")
                     ].set_index("metric")
        # the headline IC and the book are the run's own numbers, exactly
        assert rows.loc["ic", "mean"] == pooled["mean_ic"]
        assert rows.loc["ic", "t"] == pooled["t_stat"]
        assert rows.loc["ic", "icir"] == pooled["icir"]
        assert rows.loc["book_ir_net", "mean"] == pooled["net_ir"]
        assert rows.loc["book_mean_net", "mean"] == pooled["mean_net"]
        assert rows.loc["book_mean_turnover", "mean"] == pooled["mean_turnover"]
        assert {"ic_lower", "ic_upper", "spread", "long", "short"} <= set(rows.index)
        # per period long + short == spread exactly in float32 (tested above);
        # their float64 means agree to float32 rounding
        assert rows.loc["spread", "mean"] == pytest.approx(
            rows.loc["long", "mean"] + rows.loc["short", "mean"], rel=1e-6)
    # the frozen base of the residual arm is reported by the same code
    assert set(frame[frame["group"] == "markov_seed0"]["predictor"]) == {"model", "base"}
    assert set(frame[frame["group"] == "ridge_seed0"]["predictor"]) == {"model"}


def test_the_report_refuses_an_unfinished_run_or_changed_data(rr, synthetic_run):
    from nec_moe.runstore import RunStore

    root = synthetic_run["root"]
    run = RunStore(root).open(synthetic_run["run_id"])
    state = run.read_status()["state"]
    try:
        run.status(state="interrupted")
        with pytest.raises(ValueError, match="not completed"):
            rr.open_run(synthetic_run["run_id"], root)
    finally:
        run.status(state=state)
    run_obj, experiment, _ = rr.open_run(synthetic_run["run_id"], root)
    other = dict(experiment, synth_entities=21)
    with pytest.raises(ValueError, match="fingerprint"):
        rr.load_panel(other, run_obj)


# --------------------------------------------------------------------------- #
# Part B: decile monotonicity
# --------------------------------------------------------------------------- #


def test_sort_groups_split_each_date_by_forecast():
    pred = torch.tensor([5.0, 1.0, 4.0, 2.0, 3.0, 0.0, 9.0])
    y = torch.tensor([50.0, 10.0, 40.0, 20.0, 30.0, 0.0, 90.0])
    date = torch.zeros(7, dtype=torch.long)
    _, r = rp.sort_group_returns(pred, y, date, 3)
    # sorted 0, 1, 2 | 3, 4 | 5, 9: sizes 3, 2, 2 (extra names in the low groups)
    np.testing.assert_allclose(r[0], [10.0, 35.0, 70.0])
    with pytest.raises(ValueError, match="fewer than"):
        rp.sort_group_returns(pred[:2], y[:2], date[:2], 3)


def test_the_stationary_bootstrap_draws_blocks_of_the_mean_length():
    idx = rp.stationary_bootstrap_indices(5000, 10.0, 3, np.random.default_rng(0))
    assert idx.shape == (3, 5000) and idx.min() >= 0 and idx.max() < 5000
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % 5000).mean()
    assert cont == pytest.approx(0.9, abs=0.02)  # a new block with probability 1/10
    again = rp.stationary_bootstrap_indices(5000, 10.0, 3, np.random.default_rng(0))
    assert np.array_equal(idx, again)


def test_the_monotonic_relation_test_is_reproducible_under_a_seed():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.02, (400, 10)) + 0.001 * np.arange(10)
    a = rp.monotonic_relation_test(r, mean_block=10, reps=200, seed=7)
    b = rp.monotonic_relation_test(r, mean_block=10, reps=200, seed=7)
    c = rp.monotonic_relation_test(r, mean_block=10, reps=200, seed=8)
    assert a == b and a["p"] != c["p"]
    assert a["statistic"] == pytest.approx(min(np.diff(r.mean(axis=0))))


def test_increasing_means_reject_flat_ones_hold_size_one_extreme_does_not():
    flat = rp.mr_simulation(np.zeros(10), n_sims=300, seed=1)
    rising = rp.mr_simulation(0.006 * np.arange(10), n_sims=60, seed=2)
    extreme = rp.mr_simulation(np.r_[np.zeros(9), 0.03], n_sims=300, seed=3)
    print(f"\n[monotonic] flat {flat['rejection_rate']:.3f}, rising "
          f"{rising['rejection_rate']:.3f}, one extreme {extreme['rejection_rate']:.3f}")
    assert 0.01 <= flat["rejection_rate"] <= 0.10
    assert rising["rejection_rate"] >= 0.9
    assert extreme["rejection_rate"] <= 0.10


def test_the_shape_spearman():
    assert rp.shape_spearman(np.arange(10.0)) == pytest.approx(1.0)
    assert rp.shape_spearman(-np.arange(10.0)) == pytest.approx(-1.0)
    assert rp.shape_spearman(np.array([1.0, 1.0, 2.0])) == pytest.approx(np.sqrt(3) / 2)


def test_the_report_has_the_decile_tables(rr, synthetic_run):
    inputs = rr.load_inputs(synthetic_run["run_id"], synthetic_run["root"])
    s = rr.ReportSettings(mr_bootstrap_reps=50)
    frame = rr.report(inputs, s)
    rows = frame[(frame["group"] == "markov_seed0") & (frame["predictor"] == "model")
                 & (frame["slice"] == "all")].set_index("metric")
    assert [f"group_{j:02d}" for j in range(1, 11)] == [
        m for m in rows.index if m.startswith("group_") and m != "group_spearman"]
    assert rows.loc["group_01", "lags"] == inputs.horizon - 1
    assert 0.0 <= rows.loc["monotonic", "p"] <= 1.0
    assert "mean block 10" in rows.loc["monotonic", "kernel"]  # max(10, 2h), h = 1
    assert -1.0 <= rows.loc["group_spearman", "mean"] <= 1.0
    # the same seed, the same p
    again = rr.report(inputs, s).set_index(["group", "predictor", "slice", "metric"])
    assert again.loc[("markov_seed0", "model", "all", "monotonic"), "p"] == rows.loc[
        "monotonic", "p"]


# --------------------------------------------------------------------------- #
# Part C: by year, named episode and regime
# --------------------------------------------------------------------------- #


def test_the_episodes_are_the_ones_fixed_in_q9():
    assert rp.EPISODES == (
        ("euro_debt_flash_crash_2010", "2010-04-26", "2010-07-02"),
        ("us_downgrade_2011", "2011-07-25", "2011-10-31"),
        ("china_oil_2015_16", "2015-08-17", "2016-02-29"),
        ("q4_selloff_2018", "2018-10-01", "2018-12-31"),
        ("covid_crash_2020", "2020-02-20", "2020-04-30"),
        ("covid_rebound_2020", "2020-05-01", "2020-12-31"),
        ("bear_market_2022", "2022-01-03", "2022-10-31"),
        ("regional_banks_2023", "2023-03-08", "2023-05-05"),
    )


def test_episode_membership_is_by_formation_date():
    days = pd.DatetimeIndex(["2020-02-19", "2020-02-20", "2020-04-30", "2020-05-01"])
    np.testing.assert_array_equal(rp.in_window(days, "2020-02-20", "2020-04-30"),
                                  [False, True, True, False])


def _blocks(rr, t_len=500, n=30, start="2019-06-03", window=("2020-02-20", "2020-04-30"),
            seed=5, gate=None):
    """One fold whose forecasts predict the target only inside ``window``."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=t_len)
    inside = rp.in_window(days, *window)
    pred = rng.normal(size=(t_len, n))
    y = np.where(inside[:, None], 0.02 * pred, 0.0) + rng.normal(0, 0.01, (t_len, n))
    y = y - y.mean(axis=1, keepdims=True)
    date = torch.arange(t_len).repeat_interleave(n)
    block = rr.Block(0, date, torch.arange(n).repeat(t_len),
                     torch.from_numpy(y.ravel()).float(), None,
                     {"model": torch.from_numpy(pred.ravel()).float()},
                     tuple(str(d.date()) for d in days), gate)
    return [block], inside


def test_a_signal_working_inside_one_episode_is_found_there_only(rr):
    blocks, inside = _blocks(rr)
    series = rr.predictor_series(blocks, "model", 5, "nonoverlapping", 1, 10)
    rows = pd.DataFrame(rr.episode_rows(series, 0, rr.ReportSettings()))
    ic = rows[rows["metric"] == "ic"].set_index("slice")
    assert ic.loc["covid_crash_2020", "mean"] > 0.5
    assert ic.loc["covid_crash_2020", "formation_dates"] == inside.sum()
    assert abs(ic.loc["covid_rebound_2020", "mean"]) < 0.05
    assert ic.loc["covid_rebound_2020", "flag"] == ""
    # episodes outside the sample: no dates, flagged
    assert ic.loc["us_downgrade_2011", "formation_dates"] == 0
    assert "fewer than 20" in ic.loc["us_downgrade_2011", "flag"]
    spread = rows[rows["metric"] == "spread"].set_index("slice")
    assert spread.loc["covid_crash_2020", "mean"] > 5 * abs(spread.loc["q4_selloff_2018", "mean"]
                                                            if spread.loc["q4_selloff_2018", "n"]
                                                            else 0.001)


def test_the_year_tables_add_up_to_the_totals(rr):
    blocks, _ = _blocks(rr, t_len=700, start="2018-03-01")
    series = rr.predictor_series(blocks, "model", 5, "nonoverlapping", 1, 10)
    years = pd.DataFrame(rr.calendar_rows(series, 0))
    for metric, frame, col in (("ic", series["ic"], "value"),
                               ("spread", series["book"], "spread")):
        y = years[years["metric"] == metric]
        assert y["n"].sum() == len(frame)
        assert (y["mean"] * y["n"]).sum() / y["n"].sum() == pytest.approx(frame[col].mean())
    want = {str(y) for y in pd.bdate_range("2018-03-01", periods=700).year}
    assert set(years["slice"]) == want == {"2018", "2019", "2020"}


def test_the_regime_rows_use_the_gates_window_weights(rr, synthetic_run):
    from nec_moe.markov_gate import gate_weight_probs

    inputs = rr.load_inputs(synthetic_run["run_id"], synthetic_run["root"])
    blocks = inputs.groups["markov_seed0"]
    assert all(b.gate_file is not None for b in blocks)
    assert all(b.gate_file is None for b in inputs.groups["ridge_seed0"])
    # the weights are the gate's own filtered probabilities moved by its A
    state = torch.load(blocks[0].gate_file, weights_only=False)
    codes = sorted({int(c) for c in blocks[0].date.tolist()})
    fit = state["extra"]["fit_result"]
    xi = np.stack([state["extra"]["filtered"][c] for c in codes])
    np.testing.assert_allclose(rp.window_weights_from_gate_state(state, codes, 1),
                               gate_weight_probs(xi, fit.transition, "window", 1))
    frame = rr.report(inputs, rr.ReportSettings(mr_bootstrap_reps=20))
    reg = frame[(frame["slice_kind"] == "regime") & (frame["predictor"] == "model")]
    assert set(reg["group"]) == {"markov_seed0"}  # gated arms only
    assert set(reg["slice"]) == {"calm", "stress"} and set(reg["metric"]) == {
        "ic", "spread", "long", "short"}
    # the regime means recombine to the pooled mean (the weights sum to 1)
    series = rr.predictor_series(blocks, "model", 5, "nonoverlapping", 1, 10)
    w = series["ic"][["w_calm", "w_stress"]].to_numpy()
    ic = reg[reg["metric"] == "ic"].set_index("slice")["mean"]
    total = (w.sum(axis=0) * ic[["calm", "stress"]].to_numpy()).sum() / w.sum()
    assert total == pytest.approx(series["ic"]["value"].mean())


def test_across_seeds_summarises_each_arm(rr):
    frame = pd.DataFrame({
        "group": ["a_d2_seed0", "a_d2_seed1", "a_d2_seed0", "b_seed3"],
        "predictor": "model", "execution_lag": 0, "slice_kind": "all", "slice": "all",
        "metric": ["ic", "ic", "spread", "ic"], "mean": [0.02, 0.04, 0.01, 0.05],
    })
    out = rr.across_seeds(frame).set_index(["arm", "metric"])
    assert out.loc[("a_d2", "ic"), "mean"] == pytest.approx(0.03)
    assert out.loc[("a_d2", "ic"), "n_seeds"] == 2
    assert out.loc[("b", "ic"), "n_seeds"] == 1


# --------------------------------------------------------------------------- #
# Part D: the execution lag
# --------------------------------------------------------------------------- #


def _daily(t_len=40, n=6, seed=6):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2012-01-02", periods=t_len)
    return pd.DataFrame(rng.normal(0, 0.01, (t_len, n)), index=days,
                        columns=[str(100 + j) for j in range(n)])


def _rows(daily, dates):
    """Every entity on each formation date."""
    ents = list(daily.columns)
    row_dates = pd.DatetimeIndex([d for d in dates for _ in ents])
    codes = np.repeat(np.arange(len(dates)), len(ents))
    return row_dates, ents * len(dates), codes


def test_the_lagged_label_is_the_window_after_t_plus_l_demeaned_over_date_t():
    daily = _daily()
    dates = daily.index[:30]
    row_dates, ents, codes = _rows(daily, dates)
    h, lag = 5, 2
    label, pieces = rp.lagged_labels(daily, row_dates, ents, codes, lag, h)
    r = daily.to_numpy()
    raw = np.array([r[t + lag + 1: t + lag + 1 + h, j].sum()
                    for t in range(30) for j in range(6)])
    want = raw - np.repeat(raw.reshape(30, 6).mean(axis=1), 6)
    np.testing.assert_allclose(label, want, atol=1e-15)
    np.testing.assert_allclose(pieces.sum(axis=1), label, atol=1e-15)
    # only returns after t + L: changing every return up to t + L changes nothing
    t0 = 10
    moved = daily.copy()
    moved.iloc[: t0 + lag + 1] = 99.0
    again, _ = rp.lagged_labels(moved, row_dates, ents, codes, lag, h)
    sel = codes == t0
    np.testing.assert_allclose(again[sel], label[sel], atol=1e-12)


def test_the_last_l_formation_dates_have_no_lagged_label():
    daily = _daily(t_len=40)
    h = 5
    # the target's reach: the last formation date plus h days
    last = 40 - h - 1
    cut = daily.iloc[: last + h + 1]
    row_dates, ents, codes = _rows(cut, cut.index[: last + 1])
    for lag in (0, 1, 2):
        label, _ = rp.lagged_labels(cut, row_dates, ents, codes, lag, h)
        missing = sorted(set(codes[np.isnan(label)]))
        assert missing == list(range(last + 1 - lag, last + 1))


def test_a_next_day_signal_scores_at_l0_and_not_at_l1():
    rng = np.random.default_rng(7)
    t_len, n, h = 600, 60, 5
    days = pd.bdate_range("2013-01-02", periods=t_len)
    s = rng.normal(size=(t_len, n))
    r = np.full((t_len, n), np.nan)
    r[1:] = 0.01 * s[:-1] + rng.normal(0, 0.01, (t_len - 1, n))  # predicts t + 1 only
    daily = pd.DataFrame(r, index=days, columns=[str(j) for j in range(n)])
    row_dates, ents, codes = _rows(daily, days[: t_len - h - 2])
    pred = torch.from_numpy(s[: t_len - h - 2].ravel()).float()
    date = torch.from_numpy(codes)
    ics = {}
    for lag in (0, 1):
        label, _ = rp.lagged_labels(daily, row_dates, ents, codes, lag, h)
        keep = np.isfinite(label)
        _, ic = rank_ic_by_date(pred[keep], torch.from_numpy(label[keep]).float(), date[keep])
        ics[lag] = float(ic.mean())
    assert ics[0] > 0.15 and abs(ics[1]) < 0.02


def _synthetic_daily(synthetic_run):
    """Daily returns for the synthetic run: day t + 1's return is the panel's
    one-day target of t (enough to drive the lagged plumbing)."""
    import run_experiment as rx

    panel = rx._build_panel(synthetic_run["exp"])[0]
    labels = pd.DatetimeIndex(panel.date_labels)
    assert panel.entity_labels is None  # synthetic: entities are named by their codes
    frame = pd.DataFrame({"code": panel.date.numpy() + 1,
                          "e": [str(int(e)) for e in panel.entity.tolist()],
                          "r": panel.y.double().numpy()})
    # the calendar reaches one day past the last formation date, as the
    # one-day target does (the report cuts real data at the last date + h)
    days = labels.append(pd.DatetimeIndex([labels[-1] + pd.offsets.BDay()]))
    frame["day"] = days[frame["code"].to_numpy()]
    return frame.pivot(index="day", columns="e", values="r")


def test_lag_zero_is_unchanged_and_every_row_records_its_lag(rr, synthetic_run):
    inputs = rr.load_inputs(synthetic_run["run_id"], synthetic_run["root"])
    daily = _synthetic_daily(synthetic_run)
    only0 = rr.report(inputs, rr.ReportSettings(mr_bootstrap_reps=20, execution_lags=(0,)))
    both = rr.report(inputs, rr.ReportSettings(mr_bootstrap_reps=20), daily=daily)
    assert set(both["execution_lag"]) == {0, 1}
    pd.testing.assert_frame_equal(
        both[both["execution_lag"] == 0].reset_index(drop=True), only0.reset_index(drop=True))
    # L = 1 drops the last formation date of the sample (no lagged label)
    a = both[(both["group"] == "ridge_seed0") & (both["slice"] == "all")
             & (both["metric"] == "ic")].set_index("execution_lag")["n"]
    assert a[1] == a[0] - 1
    # without an extract (synthetic data) the L > 0 rows are skipped, not guessed
    skipped = rr.report(inputs, rr.ReportSettings(mr_bootstrap_reps=20))
    assert set(skipped["execution_lag"]) == {0}


def test_the_execution_lag_never_reaches_training(rr, synthetic_run):
    """Structural: the knob lives only in the reporting code; no training,
    model, harness or run setting carries it; and reporting at L = 0 and 1
    leaves every file of the run as it was."""
    import dataclasses

    from conftest import small_config

    import run_experiment as rx

    root = Path(__file__).resolve().parents[1]
    users = sorted(str(p.relative_to(root)) for p in [*root.glob("nec_moe/*.py"),
                                                      *root.glob("scripts/*.py"),
                                                      root / "run_experiment.py"]
                   if "execution_lag" in p.read_text())
    assert "scripts/report_run.py" in users
    assert set(users) <= {"nec_moe/reporting.py", "scripts/report_run.py"}
    assert "execution_lag" not in {f.name for f in dataclasses.fields(rx.Experiment)}
    assert "execution_lag" not in json.dumps(small_config(prior_kind="markov").to_dict())
    run_root = synthetic_run["root"]
    before = _digest(run_root)
    inputs = rr.load_inputs(synthetic_run["run_id"], run_root)
    rr.report(inputs, rr.ReportSettings(mr_bootstrap_reps=20),
              daily=_synthetic_daily(synthetic_run))
    assert _digest(run_root) == before


def test_the_lagged_returns_come_from_the_extract_with_the_fill(rr, tmp_path, monkeypatch):
    """L > 0 reads the target's own series from the CRSP extract (fixture,
    invented numbers): log(1 + DlyRet), the delisting return, then the cash
    fill for max lag + h days; cut at the panel's last date + h."""
    pytest.importorskip("pyarrow")
    import crsp_fixture as fx

    import nec_moe.crsp as crsp

    fixture = fx.write_fixture(tmp_path / "Data")
    spec = crsp.CRSPSpec(crsp_dir=str(fixture.root), release=fx.RELEASE, start=fx.START,
                         end=fx.END, chunk_bytes=20_000, member_count_min=5,
                         member_count_max=8)
    crsp.extract_crsp(spec, verbose=False)
    extract = crsp.load_extract(spec)
    monkeypatch.setattr(crsp, "load_extract", lambda _spec: extract)
    block = rr.Block(0, torch.tensor([0, 0]), torch.tensor([0, 1]), torch.zeros(2), None,
                     {"model": torch.zeros(2)}, ("2020-06-01",), None, ("10001", "10005"))
    inputs = rr.RunInputs(None, {}, "evaluate", {"g": [block]}, horizon=5,  # type: ignore
                          metadata={"crsp_release": fx.RELEASE,
                                    "crsp_stock_file": spec.stock_file,
                                    "post_delisting_return": "cash"},
                          last_date=fx.END)
    daily = rr.load_daily_returns(inputs, max_lag=1)
    cal = pd.DatetimeIndex(extract.market.index)
    pos = cal.get_loc(pd.Timestamp(fx.END))
    assert daily.index[-1] == cal[pos + 5]  # the target's reach, no further
    r = fixture.returns[10001].dropna()
    day = r.index[r.index <= daily.index[-1]][-1]
    # the fixture's files carry DlyRet to 8 decimals
    assert daily.loc[day, "10001"] == pytest.approx(np.log1p(round(r.loc[day], 8)), abs=1e-12)
    delisting_day = cal[cal.get_loc(fx.DELIST_LAST) + 1]
    assert daily.loc[delisting_day, "10005"] == pytest.approx(np.log1p(fx.DELIST_RET))
    after = daily["10005"].loc[delisting_day:].iloc[1:]
    assert (after.iloc[: 1 + 5] == 0.0).all()  # the cash fill, max lag + h days
