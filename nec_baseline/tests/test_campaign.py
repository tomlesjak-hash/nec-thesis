"""Brief 09 G: the process-parallel campaign runner. Synthetic data only.

G.4: a 2-worker campaign of 4 tiny jobs gives results identical to running
them sequentially, and killing one job and resuming completes only that job.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import warnings
from pathlib import Path

import pytest
import torch

from nec_moe import (
    BaseCache,
    NECModel,
    Trainer,
    request_stop,
    walk_forward_evaluate,
)
from nec_moe import campaign as cp
from nec_moe.interrupt import clear_stop
from nec_moe.train import DeadParameterWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("statsmodels")

EXPERIMENT = {
    "data": "synthetic", "synth_dates": 160, "synth_entities": 6,
    "synth_regime_process": "markov", "prior": "markov", "freeze_gate": True,
    "base_enabled": True, "correction_mode": True, "gate_series": "sequence_channel",
    "gate_series_channel": 0, "gate_start_vol_quantiles": [0.6],
    "gate_start_vol_windows": [10], "gate_start_persistences": [0.95],
    "gate_start_draws_per_centre": 1, "gate_start_min_history": 5,
    "gate_start_min_group_size": 5, "base_steps": 30, "base_batch_size": 64,
    "encoder_hidden": 8, "expert_dropout": 0.05, "steps": 12, "sigma_init": 1.0,
    "fold_scheme": "count", "n_folds": 2, "test_dates_per_fold": 15,
    "backtest_quantiles": 3, "checkpoint_every": 3, "include_ridge": False,
    "include_mlp": False, "figures": False, "calibration": False,
}


def _config(tmp_path: Path, **kw) -> Path:
    cfg = {"campaign": "unit", "tag": "camp", "first_width": 8, "depths": [1, 2],
           "seeds": [0], "arms": {"hamilton": {"prior": "markov"}}, "n_workers": 2,
           "experiment": EXPERIMENT} | kw
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(cfg))
    return path


def _run(tmp_path: Path, config: Path, **kw):
    import run_campaign

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        return run_campaign.main(config, out_dir=str(tmp_path / "results"),
                                 per_security_dir=str(tmp_path / "Data" / "derived" / "runs"),
                                 **kw)


@pytest.fixture(scope="module")
def parallel(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("campaign")
    out = _run(tmp, _config(tmp), n_workers=2)
    return tmp, out


def test_the_campaign_has_four_jobs_run_by_two_one_thread_workers(parallel):
    _, out = parallel
    assert out["status"] == "completed"
    done = [r for r in out["records"] if r["status"] == "done"]
    assert len(done) == 4  # 1 arm x 2 depths x 1 seed x 2 folds
    assert {r["threads"] for r in done} == {1}  # torch.set_num_threads(1) per worker
    assert len({r["pid"] for r in done}) <= 2  # at most two worker processes
    assert all(r["pid"] != __import__("os").getpid() for r in done)


def test_parallel_equals_sequential(parallel):
    """Each (arm, depth, seed)'s folds, run by two workers, equal the same
    folds run one after another in this process by the plain harness."""
    import run_campaign

    import run_experiment as rx

    tmp, out = parallel
    cfg = run_campaign.load_config(_config(tmp))
    exp = run_campaign._experiment(cfg, str(tmp / "results"))
    panel, purge = rx._build_panel(exp)
    folds = rx.main_folds(exp, panel, purge)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for depth in (1, 2):
            arm_exp = run_campaign.arm_experiment(exp, {"prior": "markov"}, 8, depth)
            ncfg = rx._nec_config(arm_exp, panel, 1.0)

            seeded = dataclasses.replace(ncfg, train=dataclasses.replace(ncfg.train, seed=0))

            def make(cfg=seeded):
                torch.manual_seed(0)
                return Trainer(NECModel(cfg))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeadParameterWarning)
                ref = walk_forward_evaluate(
                    panel, make, folds=folds, purge_dates=purge, steps=ncfg.train.steps,
                    backtest_quantiles=3, cost_rate=exp.cost_rate, base_cache=BaseCache(),
                    seed=0,
                )
            got = out["results"][f"hamilton_d{depth}_seed0"]
            assert got.folds == ref.folds  # every field but the wall clock
            assert got.pooled_ic == ref.pooled_ic
            assert got.pooled_portfolio == ref.pooled_portfolio
    finally:
        torch.set_num_threads(threads)


def test_the_campaign_writes_aggregates_and_keeps_predictions_in_data(parallel):
    tmp, out = parallel
    run_dir = out["run_dir"]
    rows = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text().splitlines()]
    sweep_rows = [r for r in rows if r["tag"] == "camp"]
    assert sorted((r["config"]["arm"], r["config"]["depth"], r["seed"]) for r in sweep_rows) == [
        ("hamilton", 1, 0), ("hamilton", 2, 0)]
    assert any(r["tag"] == "markov_gate_starts" for r in rows)  # the gate's starts, too
    # brief 09 I.1: the scheme on every row (count folds: no test year)
    from nec_moe import PROVENANCE_KEYS

    for r in rows:
        assert set(PROVENANCE_KEYS) <= set(r["config"])
        assert (r["config"]["fold_scheme"], r["config"]["protocol"]) == ("count", "main")
        assert r["config"]["test_years"] is None and r["config"]["gate_memory"] == "full"
    assert (run_dir / "metrics" / "report.csv").exists()
    preds = list((tmp / "Data" / "derived" / "runs" / out["run_id"]).rglob("*_predictions.pt"))
    assert len(preds) == 4
    assert not list(run_dir.rglob("*_predictions.pt"))
    # the panel was built once, into the per-security area
    caches = list((tmp / "Data" / "derived" / "panel_cache").glob("*/tensors.pt"))
    assert len(caches) == 1


def test_killing_one_job_and_resuming_completes_only_that_job(parallel, tmp_path: Path):
    """A finished campaign with one job's result removed (as if its process
    had been killed before it finished) resumes that job only, and the
    combined results are unchanged."""
    _, reference = parallel
    out = _run(tmp_path, _config(tmp_path), n_workers=2)
    victim = (out["run_dir"] / "checkpoints" / "jobs" / "hamilton_d2_seed0" / "fold_1.pt")
    victim.unlink()
    resumed = _run(tmp_path, _config(tmp_path), n_workers=2, resume=out["run_id"])
    ran = [r["job"] for r in resumed["records"] if r["status"] == "done"]
    assert ran == ["hamilton_d2_s0_f1"]
    assert sorted(r["job"] for r in resumed["records"] if r["status"] == "already_done") == [
        "hamilton_d1_s0_f0", "hamilton_d1_s0_f1", "hamilton_d2_s0_f0"]
    for group, res in reference["results"].items():
        assert resumed["results"][group].folds == res.folds


def test_an_interrupted_job_continues_from_its_checkpoint(parallel, tmp_path: Path):
    """A stop request in the middle of a job's training checkpoints it and
    stops the campaign (the remaining jobs do not start); resuming finishes
    it from the checkpoint and the results equal the uninterrupted run."""
    _, out = parallel
    original = Trainer._after_step
    calls = {"n": 0}

    def stop_once(self, path):
        calls["n"] += 1
        if calls["n"] == 5:
            request_stop()
        original(self, path)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Trainer, "_after_step", stop_once)
        first = _run(tmp_path, _config(tmp_path), n_workers=0)
    clear_stop()
    statuses = {r["job"]: r["status"] for r in first["records"]}
    assert first["status"] == "interrupted"
    assert statuses["hamilton_d1_s0_f0"] == "interrupted"
    assert set(statuses.values()) == {"interrupted", "skipped"}
    ckpt = (first["run_dir"] / "checkpoints" / "jobs" / "hamilton_d1_seed0"
            / "fold_0_trainer.pt")
    assert ckpt.exists() and Trainer.load(ckpt).step_count == 5
    resumed = _run(tmp_path, _config(tmp_path), n_workers=0, resume=first["run_id"])
    assert resumed["status"] == "completed"
    for group, res in out["results"].items():
        assert resumed["results"][group].folds == res.folds


def test_the_disk_base_cache_is_shared_and_locked(tmp_path: Path):
    from conftest import small_base_config

    from nec_moe import SyntheticRegimePanel, SyntheticSpec

    panel = SyntheticRegimePanel(SyntheticSpec(seed=1)).generate(60, 4)
    cfg = small_base_config(steps=10)
    a = cp.DiskBaseCache(tmp_path / "bases")
    fit = a.get_or_fit(panel, cfg, window=(0, 59), seed=0)
    assert a.misses == 1 and len(list((tmp_path / "bases").glob("base_*.pt"))) == 1
    b = cp.DiskBaseCache(tmp_path / "bases")  # another process's view
    again = b.get_or_fit(panel, cfg, window=(0, 59), seed=0)
    assert b.misses == 0 and b.hits == 1
    for x, y in zip(fit.model.state_dict().values(), again.model.state_dict().values(),
                    strict=True):
        assert torch.equal(x, y)


def test_a_campaign_config_with_an_unknown_key_is_refused(tmp_path: Path):
    import run_campaign

    with pytest.raises(ValueError, match="worker_count"):
        run_campaign.load_config(_config(tmp_path, worker_count=4))


def test_a_stop_in_the_parent_reaches_every_worker(tmp_path: Path):
    """Ctrl+C's first effect (a stop request in the parent) is relayed to
    the workers through the shared event: no job runs to completion, none
    fails, and the campaign is marked interrupted."""
    import run_campaign

    original = run_campaign.run_jobs

    def stopped(*a, **kw):
        request_stop()
        try:
            return original(*a, **kw)
        finally:
            clear_stop()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run_campaign, "run_jobs", stopped)
        out = _run(tmp_path, _config(tmp_path), n_workers=2)
    statuses = {r["status"] for r in out["records"]}
    assert out["status"] == "interrupted"
    assert statuses <= {"skipped", "interrupted"}, statuses
    status = json.loads((out["run_dir"] / "status.json").read_text())
    assert status["state"] == "interrupted"


def test_the_compute_envelope_is_documented_defaults_not_limits():
    from nec_moe import COMPUTE_ENVELOPE, ComputeEnvelope

    assert COMPUTE_ENVELOPE == ComputeEnvelope(depths=(1, 2, 3), max_first_width=128,
                                               max_experts=4)
    assert COMPUTE_ENVELOPE.notes(first_width=64, depths=[1, 2, 3], n_experts=2) == []
    notes = COMPUTE_ENVELOPE.notes(first_width=256, depths=[1, 5], n_experts=6)
    assert len(notes) == 3  # reported, never refused
    # K and the widths are open (Q18, Q8/Q17): the defaults are unchanged
    import run_experiment as rx

    assert (rx.Experiment().n_experts, rx.Experiment().expert_hidden_dims) == (2, (64, 32))


def test_scheme_and_fold_provenance():
    from conftest import small_base_config, small_config

    from nec_moe import fold_provenance, scheme_provenance, walk_forward_folds

    assert set(scheme_provenance(None).values()) == {None}
    cfg = small_config(prior_kind="markov", base=small_base_config(decay_half_life_days=500.0),
                       expert_decay="regime_clock", expert_decay_half_life=300.0)
    cfg = dataclasses.replace(cfg, markov_gate=dataclasses.replace(
        cfg.markov_gate, fit_backend="native", memory="regime_clock", gate_half_life=200.0))
    assert scheme_provenance(cfg) == {
        "base_decay_half_life_days": 500.0, "expert_decay": "regime_clock",
        "expert_decay_half_life": 300.0, "gate_memory": "regime_clock",
        "gate_half_life": 200.0, "gate_fit_backend": "native"}
    soft = scheme_provenance(small_config())
    assert soft["expert_decay"] == "none" and soft["gate_memory"] is None
    count = walk_forward_folds(torch.arange(100), n_folds=2, test_dates_per_fold=10,
                               purge_dates=1)
    assert fold_provenance(count) == {"fold_scheme": "count", "protocol": "main",
                                      "test_years": None}
