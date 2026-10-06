"""Brief 09 F: the pilot, its regime-balanced loss, and the selection lock.

Everything runs on a synthetic multi-year panel (business-day calendar
2004-2011). The pilot is never run on the real panel here (F.4).
"""

from __future__ import annotations

import dataclasses
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import run_experiment as rx
from nec_moe import (
    BaseCache,
    FoldConfig,
    PilotGrid,
    load_pilot_grid,
    load_selection,
    pilot_folds,
    pilot_slice,
    regime_balanced_loss,
    run_pilot,
    write_selection,
)
from nec_moe import pilot as pl
from nec_moe.train import DeadParameterWarning

statsmodels = pytest.importorskip("statsmodels")

ROOT = Path(__file__).resolve().parents[1]
CAL_START, CAL_END = "2004-01-01", "2011-12-30"


def test_the_regime_balanced_loss_gives_every_regime_equal_weight():
    """90 calm rows with loss 1 and 10 stress rows with loss 3, the gate
    certain of each: the plain mean is 1.2, the regime-balanced loss 2.0."""
    wbar = np.vstack([np.tile([1.0, 0.0], (90, 1)), np.tile([0.0, 1.0], (10, 1))])
    loss = np.r_[np.ones(90), 3 * np.ones(10)]
    assert loss.mean() == pytest.approx(1.2)
    assert regime_balanced_loss(loss, wbar) == pytest.approx(2.0)
    # soft weights: v(s) = sum_k wbar_k(s) / M_k
    w = np.array([[0.8, 0.2], [0.4, 0.6], [0.6, 0.4]])
    v = (w / w.mean(axis=0)).sum(axis=1)
    loss = np.array([1.0, 2.0, 4.0])
    assert regime_balanced_loss(torch.tensor(loss), w) == pytest.approx((v * loss).sum() / v.sum())
    with pytest.raises(ValueError, match="no gate weight"):
        regime_balanced_loss(loss, np.c_[np.ones(3), np.zeros(3)])


def test_the_default_grid_and_its_file():
    g = PilotGrid().validate()
    assert g.base_half_life_days == (None, 2520.0, 1260.0)
    # 10 and 5 calendar years of the stress regime at a 30% stress share
    assert g.expert_half_lives == (math.inf, 756.0, 378.0)
    assert g.gate_half_lives == g.expert_half_lives  # the same scale
    assert g.batch_size == (1024, 4096) and g.steps == (3000, 10000, 30000)
    assert len(g.settings()) == 3 * 3 * 2 * 2 * 2 * 1
    s = g.settings()[0]
    assert s["expert_decay"] == "none" and math.isinf(s["expert_decay_half_life"])
    from_file = load_pilot_grid(ROOT / "configs" / "pilot_grid.json")
    assert from_file == g  # the shipped config is the documented default
    for bad in (dict(search="random"), dict(steps=(10, 5)), dict(lr=())):
        with pytest.raises(ValueError):
            dataclasses.replace(g, **bad).validate()


def test_unknown_grid_keys_are_refused(tmp_path: Path):
    path = tmp_path / "grid.json"
    path.write_text(json.dumps({"grid": {"lr": [0.1], "learning_rate": [0.1]}}))
    with pytest.raises(ValueError, match="learning_rate"):
        load_pilot_grid(path)


# --------------------------------------------------------------------------- #
# End to end on a synthetic multi-year panel
# --------------------------------------------------------------------------- #

GRID = PilotGrid(
    base_half_life_days=(None, 250.0), expert_half_life_years=(None, 2.0),
    gate_half_life_years=(None, 2.0), lr=(3e-3,), weight_decay=(1e-4,),
    batch_size=(32, 64), steps=(2, 4), base_steps=(10,), seeds=(0,),
)


def _experiment(tmp_path: Path, **kw) -> rx.Experiment:
    base = dict(
        tag="pilot", campaign="unit", out_dir=str(tmp_path / "results"),
        per_security_dir=str(tmp_path / "Data" / "derived" / "runs"), data="synthetic",
        synth_dates=len(pd.bdate_range(CAL_START, CAL_END)), synth_entities=4,
        synth_calendar_start=CAL_START, synth_regime_process="markov", prior="markov",
        freeze_gate=True, base_enabled=True, correction_mode=True,
        gate_series="sequence_channel", gate_series_channel=0,
        gate_start_vol_quantiles=(0.6,), gate_start_vol_windows=(10,),
        gate_start_persistences=(0.95,), gate_start_draws_per_centre=1,
        gate_start_min_history=5, gate_start_min_group_size=5, base_batch_size=32,
        expert_hidden_dims=(8, 4), encoder_hidden=8, expert_dropout=0.0,
        sigma_init=1.0, first_test_year=2010, last_test_year=2011, include_ridge=False,
        include_mlp=False, backtest_quantiles=None, figures=False, calibration=False,
    )
    return rx.Experiment(**(base | kw))


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pilot")
    exp = _experiment(tmp)
    panel, purge = rx._build_panel(exp)
    seen_max: list[int] = []
    original = pl._train_and_score

    def spy(sliced, fold, *a, **kw):
        seen_max.append(int(sliced.date.max()))
        return original(sliced, fold, *a, **kw)

    def make_config(setting):
        return rx._nec_config(dataclasses.replace(exp, **setting), panel, 1.0)

    with pytest.MonkeyPatch.context() as mp, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        mp.setattr(pl, "_train_and_score", spy)
        selection = run_pilot(panel, make_config=make_config, grid=GRID,
                              fold_config=rx.fold_config(exp), purge_dates=purge,
                              verbose=False)
    path, digest = write_selection(selection, tmp / "results" / "pilot")
    return {"exp": exp, "panel": panel, "purge": purge, "selection": selection,
            "path": path, "digest": digest, "seen_max": seen_max, "tmp": tmp}


def test_the_pilot_never_touches_the_test_period(pilot):
    panel = pilot["panel"]
    last = max(pilot["seen_max"])
    assert panel.date_labels[last] < "2010-01-01"
    # and the last purge dates of 2009 are cut too (their labels reach 2010)
    dates_2009 = [d for d in panel.date_labels if d.startswith("2009")]
    assert panel.date_labels[last] == dates_2009[-pilot["purge"] - 1]
    assert pilot["selection"]["pilot_dates"][1] == dates_2009[-pilot["purge"] - 1]


def test_the_pilot_chooses_and_reports(pilot):
    sel = pilot["selection"]
    gate = sel["gate_memory"]
    assert [r["half_life"] for r in gate["results"]] == [math.inf, 151.2]
    best = max(gate["results"], key=lambda r: r["predictive_loglik"])
    assert gate["chosen_half_life"] == best["half_life"]
    for r in gate["results"]:
        assert len(r["per_fold"]) == 3 and r["worst_days"] == 20
        assert r["excluding_worst_days"] == pytest.approx(
            r["predictive_loglik"] - r["worst_days_loglik"])
    chosen = sel["chosen"]
    assert {"gate_fit_backend", "gate_memory", "gate_half_life", "base_decay_half_life_days",
            "expert_decay", "expert_decay_half_life", "lr", "weight_decay", "batch_size",
            "base_steps", "steps"} == set(chosen)
    assert len(sel["settings"]) == len(GRID.settings()) * len(GRID.steps)
    best_setting = min(sel["settings"], key=lambda r: r["mean_loss"])
    assert chosen["steps"] == best_setting["steps"]
    check = sel["step_budget_check"]
    assert set(check["best_steps_per_fold"]) == {"2007", "2008", "2009"}
    assert check["stable"] == (check["max_over_min"] <= 2.0)


def test_a_gate_memory_tie_goes_to_the_longer_memory(pilot):
    panel, purge = pilot["panel"], pilot["purge"]
    sliced = pilot_slice(panel, 2010, purge)
    folds = pilot_folds(sliced, validation_years=(2007, 2008, 2009), first_test_year=2010,
                        purge_dates=purge)
    exp = pilot["exp"]
    cfg = rx._nec_config(exp, panel, 1.0)
    tied = dataclasses.replace(GRID, gate_half_life_years=(2.0, None, 1.0), gate_tie_tol=1e9)
    report, _ = pl.gate_memory_selection(sliced, folds, cfg, tied)
    assert math.isinf(report["chosen_half_life"])


def test_step_milestones_equal_separate_budgets(pilot):
    """Reading budget 2 inside a run to 4 equals a run that stops at 2: the
    milestone evaluation runs in a forked RNG and training continues the
    same stream."""
    panel, purge, exp = pilot["panel"], pilot["purge"], pilot["exp"]
    sliced = pilot_slice(panel, 2010, purge)
    fold = pilot_folds(sliced, validation_years=(2008,), first_test_year=2010,
                       purge_dates=purge)[0]
    cfg = rx._nec_config(dataclasses.replace(exp, expert_dropout=0.1), panel, 1.0)
    from nec_moe.markov_gate import MarkovSwitchingRegimePrior

    gate = MarkovSwitchingRegimePrior(2, cfg.markov_gate, horizon=1)
    gate.fit(sliced.subset_dates(fold.train_dates))
    gate.apply_causal(sliced.subset_dates(torch.cat([fold.train_dates, fold.test_dates])))
    state = gate.gate_state()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        both = pl._train_and_score(sliced, fold, cfg, 0, state, BaseCache(), (2, 4), None)
        first = pl._train_and_score(sliced, fold, cfg, 0, state, BaseCache(), (2,), None)
    assert both[0] == first[0]


def test_the_selection_file_and_its_hash(pilot):
    selection, digest = load_selection(pilot["path"])
    assert digest == pilot["digest"]
    sidecar = pilot["path"].with_name("pilot_selection.json.sha256").read_text()
    assert sidecar.split()[0] == digest
    json.loads(pilot["path"].read_text())  # strict JSON (no Infinity)
    assert "Infinity" not in pilot["path"].read_text()
    assert selection["chosen"] == pilot["selection"]["chosen"]
    # aggregate only: no row-level or per-security key anywhere
    from nec_moe.coverage import assert_aggregate_only

    assert_aggregate_only(json.loads(pilot["path"].read_text()),
                          set(pilot["panel"].entity_labels or ()))


# --------------------------------------------------------------------------- #
# F.3 the lock in main mode
# --------------------------------------------------------------------------- #


def _main(exp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        return rx.main(exp)


def _chosen_experiment(pilot, **kw) -> rx.Experiment:
    chosen = pilot["selection"]["chosen"]
    return _experiment(pilot["tmp"], mode="evaluate",
                       pilot_selection_file=str(pilot["path"]), **(chosen | kw))


def test_main_mode_refuses_to_start_without_the_selection(pilot, tmp_path: Path):
    exp = _experiment(tmp_path, mode="evaluate", steps=2)
    with pytest.raises(rx.PreRegistrationRequired, match="pilot_selection_file"):
        _main(exp)
    assert not (tmp_path / "results" / "INDEX.csv").exists()  # nothing was started
    # quick mode on the same calendar scores 2010 dates too
    with pytest.raises(rx.PreRegistrationRequired):
        _main(dataclasses.replace(exp, mode="quick"))


def test_main_mode_records_the_hash_in_every_trial_row(pilot):
    result = _main(_chosen_experiment(pilot))
    info = json.loads((result["run_dir"] / "run.json").read_text())
    assert info["pilot_selection_hash"] == pilot["digest"] and info["pilot_deviations"] == {}
    rows = [json.loads(line) for line in (result["run_dir"] / "trials.jsonl").read_text()
            .splitlines()]
    assert rows and all(r["config"]["pilot_selection_hash"] == pilot["digest"] for r in rows)
    assert all(r["config"]["force_deviation"] is False for r in rows)
    # brief 09 I.1: every row records the training scheme it ran under
    chosen = pilot["selection"]["chosen"]
    labels = pilot["panel"].date_labels
    for r in rows:
        c = r["config"]
        assert (c["fold_scheme"], c["protocol"]) == ("calendar_year", "main")
        assert (c["sample_start"], c["sample_end"]) == (labels[0], labels[-1])
        assert r["seed"] == 0
    sweep = [r for r in rows if r["tag"] == "pilot"]
    starts = [r for r in rows if r["tag"] == "markov_gate_starts"]
    assert sweep and starts
    for r in sweep:
        c = r["config"]
        assert c["test_years"] == [2010, 2011]
        for key in ("gate_memory", "gate_fit_backend", "expert_decay",
                    "base_decay_half_life_days"):
            assert c[key] == chosen[key], key
        for key in ("gate_half_life", "expert_decay_half_life"):
            if c[key] is not None or chosen[key] != math.inf:
                assert c[key] == chosen[key], key
    assert {tuple(r["config"]["test_years"]) for r in starts} == {(2010,), (2011,)}


def test_main_mode_refuses_a_deviation_unless_forced(pilot):
    lr = pilot["selection"]["chosen"]["lr"] * 2
    with pytest.raises(ValueError, match="differ from the pilot selection"):
        _main(_chosen_experiment(pilot, lr=lr, tag="deviates"))
    result = _main(_chosen_experiment(pilot, lr=lr, tag="forced", force_deviation=True))
    info = json.loads((result["run_dir"] / "run.json").read_text())
    assert info["pilot_deviations"] == {"lr": [pilot["selection"]["chosen"]["lr"], lr]}
    rows = [json.loads(line) for line in (result["run_dir"] / "trials.jsonl").read_text()
            .splitlines()]
    assert all(r["config"]["force_deviation"] is True for r in rows)


def test_a_run_before_the_test_period_needs_no_selection(tmp_path: Path):
    """The lock guards the test period only: a run on a calendar that ends
    before first_test_year, or on a panel without a calendar, starts."""
    exp = _experiment(tmp_path, mode="evaluate", steps=2, fold_scheme="count",
                      synth_dates=len(pd.bdate_range("2004-01-01", "2008-12-31")),
                      n_folds=1, test_dates_per_fold=30)
    assert not rx.touches_test_period(exp, *rx._build_panel(exp))
    _main(exp)
    bare = dataclasses.replace(exp, synth_calendar_start=None, tag="bare")
    assert not rx.touches_test_period(bare, *rx._build_panel(bare))


def test_fold_config_matches_the_experiment():
    exp = rx.Experiment()
    assert rx.fold_config(exp) == FoldConfig()


def test_the_pilot_script_dry_run_reads_no_data():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "scripts/run_pilot.py", "--dry-run"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    assert "72 settings x 3 folds x 3 seeds = 648 expert trainings" in out
