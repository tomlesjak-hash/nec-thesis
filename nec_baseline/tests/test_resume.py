"""Resume after an interruption (brief 07 section C).

The central property: an interrupted and resumed run equals an uninterrupted
one **exactly** (CPU, fixed seeds), for the current design (residual base,
correction mode, frozen Markov gate) interrupted mid-fit inside a fold and
between folds, for a sweep interrupted mid-arm, and for a run whose newest
checkpoint was torn by a crash during the write. Plus: SIGINT leaves the run
``interrupted`` with a loadable checkpoint, a changed setting or changed data
refuses to resume, and ``force`` resumes and records the override. Expert
dropout is on (0.05) throughout, so the RNG state genuinely matters.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import signal
import warnings
from pathlib import Path

import pytest
import torch

pytest.importorskip("statsmodels")

import run_experiment as rx  # noqa: E402
from nec_moe import (  # noqa: E402
    RESUMABLE_STATES,
    ResumeRefused,
    RunInterrupted,
    RunStore,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    TrialRegistry,
    graceful_interrupts,
    request_stop,
    stop_requested,
)
from nec_moe.runstore import Run  # noqa: E402
from nec_moe.train import DeadParameterWarning  # noqa: E402

STEPS, FOLDS = 20, 2


def _exp(root: Path, **kw) -> rx.Experiment:
    base = dict(
        tag="resume", campaign="unit", mode="evaluate", out_dir=str(root),
        # explicit: the module-scoped reference is built before the autouse
        # store isolation applies, and must not touch the real Data/
        per_security_dir=str(root / "per_security"),
        data="synthetic", synth_dates=160, synth_entities=6, synth_regime_process="markov",
        prior="markov", base_enabled=True, correction_mode=True, zero_init_head=True,
        freeze_gate=True, gate_series="sequence_channel", gate_series_channel=0,
        gate_start_vol_quantiles=(0.6,), gate_start_vol_windows=(10,),
        gate_start_persistences=(0.95,), gate_start_draws_per_centre=1,
        gate_start_min_history=5, gate_start_min_group_size=5,
        base_steps=40, base_batch_size=64, expert_hidden_dims=(8, 4), encoder_hidden=8,
        expert_dropout=0.05, steps=STEPS, seeds=(0, 1), n_folds=FOLDS,
        test_dates_per_fold=15, include_ridge=False, include_mlp=False,
        backtest_quantiles=3, figures=False, calibration=False, checkpoint_every=5,
    )
    return rx.Experiment(**(base | kw))


def _main(exp: rx.Experiment) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        return rx.main(exp)


def _strategy_rows(run_dir: Path, tag: str = "resume") -> dict[tuple[str, int], dict]:
    rows = TrialRegistry(run_dir / "trials.jsonl").trials(tag)
    return {(r.config["arm"], r.seed): r.metrics for r in rows}


def _stop_at_step(monkeypatch, n: int, action=request_stop) -> None:
    """Act (by default: request a stop) after the n-th optimizer step of the process."""
    calls = {"n": 0}
    original = Trainer._optimize

    def counted(self, loss):
        original(self, loss)
        calls["n"] += 1
        if calls["n"] == n:
            action()

    monkeypatch.setattr(Trainer, "_optimize", counted)


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> dict[tuple[str, int], dict]:
    """The uninterrupted run every resumed run must reproduce exactly."""
    result = _main(_exp(tmp_path_factory.mktemp("ref")))
    return _strategy_rows(result["run_dir"])


def _interrupt_then_resume(tmp_path: Path, monkeypatch, n: int) -> tuple[Path, dict]:
    exp = _exp(tmp_path)
    _stop_at_step(monkeypatch, n)
    with pytest.raises(RunInterrupted):
        _main(exp)
    monkeypatch.undo()
    store = RunStore(tmp_path)
    run_id = store.latest(tag="resume", status=RESUMABLE_STATES)
    assert run_id is not None
    run = store.open(run_id)
    assert run.read_status()["state"] == "interrupted"
    return run.dir, _main(dataclasses.replace(exp, resume=True))


def test_resume_mid_fit_inside_a_fold_equals_the_uninterrupted_run(
    tmp_path: Path, monkeypatch, reference
):
    run_dir, resumed = _interrupt_then_resume(tmp_path, monkeypatch, n=7)
    assert resumed["run_dir"] == run_dir  # the same run was continued
    assert _strategy_rows(run_dir) == reference
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "completed"


def test_resume_between_folds_equals_the_uninterrupted_run(
    tmp_path: Path, monkeypatch, reference
):
    """The process dies as fold 1 starts: fold 0 is on disk, fold 1 reruns."""
    exp = _exp(tmp_path)
    original = Run.status
    fired = {"done": False}

    def dying(self, **fields):
        if fields.get("fold") == 1 and fields.get("step") == 0 and not fired["done"]:
            fired["done"] = True
            raise KeyboardInterrupt("killed between folds")
        return original(self, **fields)

    monkeypatch.setattr(Run, "status", dying)
    with pytest.raises(KeyboardInterrupt):
        _main(exp)
    monkeypatch.undo()
    resumed = _main(dataclasses.replace(exp, resume=True))
    assert _strategy_rows(resumed["run_dir"]) == reference


def test_a_sweep_interrupted_mid_arm_resumes_to_the_same_report(
    tmp_path: Path, monkeypatch, reference
):
    """Seed 0 completes, seed 1 is stopped 7 steps into its first fold."""
    run_dir, _ = _interrupt_then_resume(tmp_path, monkeypatch, n=STEPS * FOLDS + 7)
    assert _strategy_rows(run_dir) == reference


def test_a_torn_newest_checkpoint_falls_back_to_the_previous_one(
    tmp_path: Path, monkeypatch, reference, capsys
):
    """A crash during a checkpoint write: the newest file is truncated. Resume
    loads the one before it (step 10, not 12), replays, and still matches."""
    exp = _exp(tmp_path)
    _stop_at_step(monkeypatch, 12)
    with pytest.raises(RunInterrupted):
        _main(exp)
    monkeypatch.undo()
    run = RunStore(tmp_path).open(RunStore(tmp_path).latest(tag="resume"))  # type: ignore[arg-type]
    ckpt = run.checkpoints_dir / "markov_seed0" / "fold_0_trainer.pt"
    assert Trainer.load(ckpt).step_count == 12
    assert Trainer.load(ckpt.with_name(ckpt.name + ".1")).step_count == 10
    ckpt.write_bytes(ckpt.read_bytes()[:64])  # torn mid-write
    capsys.readouterr()
    _main(dataclasses.replace(exp, resume=True))
    assert "resumed from the older fold_0_trainer.pt.1 at step 10" in capsys.readouterr().out
    assert _strategy_rows(run.dir) == reference


def test_sigint_leaves_the_run_interrupted_with_a_loadable_checkpoint(
    tmp_path: Path, monkeypatch
):
    exp = _exp(tmp_path, seeds=(0,))
    _stop_at_step(monkeypatch, 6, action=lambda: os.kill(os.getpid(), signal.SIGINT))
    with pytest.raises(RunInterrupted):
        _main(exp)
    monkeypatch.undo()
    run = RunStore(tmp_path).open(RunStore(tmp_path).latest(tag="resume"))  # type: ignore[arg-type]
    assert run.read_status()["state"] == "interrupted"
    assert RunStore(tmp_path).index()[-1]["status"] == "interrupted"
    trainer = Trainer.load_latest(run.checkpoints_dir / "markov_seed0" / "fold_0_trainer.pt")
    assert trainer.step_count == 6  # the step in flight finished, then it stopped
    assert not stop_requested()  # the handler is gone and the flag cleared


def test_a_second_signal_stops_immediately():
    with graceful_interrupts():
        os.kill(os.getpid(), signal.SIGINT)
        assert stop_requested()  # the first only requests a stop
        with pytest.raises(KeyboardInterrupt, match="second signal"):
            os.kill(os.getpid(), signal.SIGINT)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    assert not stop_requested()


def _quick_panel_file(path: Path, seed: int) -> Path:
    panel = SyntheticRegimePanel(SyntheticSpec(regime_process="markov", seed=seed)).generate(
        160, 6
    )
    torch.save(panel, path)
    return path


def test_changed_settings_or_data_refuse_and_force_records_the_override(
    tmp_path: Path, monkeypatch
):
    panel_file = _quick_panel_file(tmp_path / "panel.pt", seed=1)
    exp = _exp(tmp_path / "store", mode="quick", data="panel_file",
               panel_file=str(panel_file), seeds=(0,))
    _stop_at_step(monkeypatch, 4)
    with pytest.raises(RunInterrupted):
        _main(exp)
    monkeypatch.undo()
    resume = dataclasses.replace(exp, resume=True)
    with pytest.raises(ResumeRefused, match="settings changed: lr: 0.001 -> 0.002"):
        _main(dataclasses.replace(resume, lr=2e-3))
    _quick_panel_file(panel_file, seed=2)  # same settings, different data
    with pytest.raises(ResumeRefused, match="data changed"):
        _main(resume)
    result = _main(dataclasses.replace(resume, force=True))
    info = json.loads((result["run_dir"] / "run.json").read_text())
    assert "data changed" in info["forced_resumes"][0]["changes"][0]
    assert json.loads((result["run_dir"] / "status.json").read_text())["state"] == "completed"


def test_a_stale_running_run_is_reported_crashed_and_is_resumable(tmp_path: Path):
    store = RunStore(tmp_path, stale_after=60)
    run = store.create(campaign="unit", tag="t", mode="quick", settings={"a": 1})
    assert store.effective_status(store.index()[0]) == "running"
    old = (dt.datetime.now().astimezone() - dt.timedelta(seconds=120)).isoformat()
    status = run.read_status() | {"heartbeat": old}
    (run.dir / "status.json").write_text(json.dumps(status))
    assert store.index_with_status()[0]["status"] == "crashed"
    assert store.latest(tag="t", status=RESUMABLE_STATES) == run.run_id


def test_runs_cli_resumes_a_run_with_its_own_settings(tmp_path: Path, monkeypatch):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import runs

    exp = _exp(tmp_path, mode="quick", seeds=(0,))
    _stop_at_step(monkeypatch, 3)
    with pytest.raises(RunInterrupted):
        _main(exp)
    monkeypatch.undo()
    run_id = RunStore(tmp_path).latest(tag="resume", status="interrupted")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        assert runs.main(["--root", str(tmp_path), "resume", str(run_id)]) == 0
    assert RunStore(tmp_path).index()[-1]["status"] == "completed"


def test_a_resumed_fold_reloads_its_gate_and_base_instead_of_refitting(
    tmp_path: Path, monkeypatch
):
    """C.1 and C.2 at the harness level: after a mid-fit interruption the
    resumed fold 0 takes its gate from fold_0_gate.pt and its base from
    fold_0_base.pt; only fold 1, which never started, fits either."""
    from conftest import small_base_config, small_config

    import nec_moe.base as base_mod
    from nec_moe import BaseCache, MarkovSwitchingRegimePrior, NECModel, walk_forward_evaluate

    cfg = small_config(prior_kind="markov", expert_dropout=0.05, correction_mode=True,
                       base=small_base_config(), freeze_gate=True, checkpoint_every=5,
                       batch_size=64)
    cfg = dataclasses.replace(cfg, markov_gate=dataclasses.replace(
        cfg.markov_gate, series="sequence_channel", start_vol_quantiles=(0.6,),
        start_vol_windows=(10,), start_persistences=(0.95,), start_draws_per_centre=1,
        start_min_history=5, start_min_group_size=5,
    ))
    panel = SyntheticRegimePanel(SyntheticSpec(regime_process="markov", seed=3)).generate(160, 6)

    def make() -> Trainer:
        torch.manual_seed(0)
        return Trainer(NECModel(cfg))

    kw = dict(n_folds=2, test_dates_per_fold=15, purge_dates=1, steps=20, seed=0,
              resume_dir=tmp_path / "wf")
    _stop_at_step(monkeypatch, 7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        with pytest.raises(RunInterrupted):
            walk_forward_evaluate(panel, make, base_cache=BaseCache(), **kw)
    monkeypatch.undo()
    from nec_moe.interrupt import clear_stop

    clear_stop()
    assert (tmp_path / "wf" / "fold_0_gate.pt").exists()
    assert (tmp_path / "wf" / "fold_0_base.pt").exists()
    fits = {"gate": 0, "base": 0}
    gate_fit, base_fit = MarkovSwitchingRegimePrior.fit, base_mod.fit_base

    def counted_gate(self, train_panel):
        fits["gate"] += 1
        return gate_fit(self, train_panel)

    def counted_base(*args, **kwargs):
        fits["base"] += 1
        return base_fit(*args, **kwargs)

    monkeypatch.setattr(MarkovSwitchingRegimePrior, "fit", counted_gate)
    monkeypatch.setattr(base_mod, "fit_base", counted_base)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        walk_forward_evaluate(panel, make, base_cache=BaseCache(), **kw)
    assert fits == {"gate": 1, "base": 1}  # fold 1 only
