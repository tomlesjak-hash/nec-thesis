"""Checkpoint/resume: an interrupted run equals the uninterrupted one, bit for bit.

Trainer level: save/load must carry everything a trajectory depends on — model,
optimizer, ``step_count`` (the sigma freeze and tau anneal key off it), the
load-balance buffer, the history, the minibatch sampler frozen mid-epoch, and
the *global* torch RNG (dropout draws from it; without it a resumed trajectory
silently diverges). Harness level: completed folds are persisted and skipped,
and a mid-fold interrupt composes with the trainer checkpoint. Sweep level:
completed (arm, seed) runs are skipped via the registry with no duplicate rows.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest
import torch
from conftest import small_config

os.environ.setdefault("MPLBACKEND", "Agg")

from nec_moe import (
    NECModel,
    RidgeBaseline,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    TrialRegistry,
    baseline_arm,
    nec_arm,
    run_sweep,
    walk_forward_evaluate,
    walk_forward_folds,
)


def _panel(n_dates: int = 120, n_entities: int = 8, **spec_overrides):
    kw = dict(vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=3)
    kw.update(spec_overrides)
    return SyntheticRegimePanel(SyntheticSpec(**kw)).generate(n_dates, n_entities)


def _vol_proxy(p):
    return p.x_seq[:, :, 0].std(dim=1)


# --------------------------------------------------------------------------- #
# Trainer level: fit / fit_sequence
# --------------------------------------------------------------------------- #


def _make_trainer() -> Trainer:
    # dropout > 0 on purpose: it draws from the global RNG, so this config
    # only resumes bit-exactly if the checkpoint restores that RNG state.
    # sigma_freeze_steps=60 spans the interruption at step 50, so the resumed
    # trainer must also keep the freeze schedule keyed off the restored count.
    torch.manual_seed(0)
    cfg = small_config(
        expert_dropout=0.05, sigma_init=1.5, lr=3e-3,
        batch_size=64, sigma_freeze_steps=60, checkpoint_every=20,
    )
    return Trainer(NECModel(cfg))


def test_fit_resume_matches_uninterrupted(tmp_path: Path):
    panel = _panel()
    ref = _make_trainer()
    ref.fit(panel, steps=80)

    ck = tmp_path / "trainer.pt"
    interrupted = _make_trainer()
    interrupted.fit(panel, steps=50, checkpoint_path=ck)

    resumed = Trainer.load(ck)
    assert resumed.step_count == 50  # continues, never resets to 0
    assert not resumed.model.experts.log_sigma.requires_grad  # freeze: 50 < 60
    resumed.fit(panel, steps=30)

    assert resumed.step_count == ref.step_count == 80
    assert resumed.history == ref.history  # the full trajectory, bitwise
    for k, v in ref.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[k], v), k
    assert torch.equal(resumed.lb_buffer.fractions(), ref.lb_buffer.fractions())
    assert not list(tmp_path.glob("*.tmp"))  # atomic writes leave no debris


def _make_hmm_trainer() -> Trainer:
    torch.manual_seed(0)
    cfg = small_config(
        prior_kind="hmm", sigma_init=2.0, lr=3e-3,
        sigma_freeze_steps=10, checkpoint_every=10,
    )
    return Trainer(NECModel(cfg))


def test_fit_sequence_resume_matches_uninterrupted(tmp_path: Path):
    panel = _panel(
        regime_process="markov", transition_stay=0.97,
        vol_levels=(1.0, 1.3), beta_scale=1.2, noise_std=0.8, seed=11,
    )
    seq = panel.time_sequence()
    ref = _make_hmm_trainer()
    ref.fit_sequence(seq, steps=40, chunk_len=25)

    # 120 dates / chunk_len 25 -> 5 chunks; interrupting at step 23 lands
    # mid-pass (cursor at chunk 3) with a live carried filter state — the
    # hard case for the resume, not a clean sequence-start boundary.
    ck = tmp_path / "hmm.pt"
    interrupted = _make_hmm_trainer()
    interrupted.fit_sequence(seq, steps=23, chunk_len=25, checkpoint_path=ck)

    resumed = Trainer.load(ck)
    assert resumed.step_count == 23
    resumed.fit_sequence(seq, steps=17, chunk_len=25)

    assert resumed.history == ref.history
    for k, v in ref.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[k], v), k

    # resuming with a different chunking is a loud error, not silent drift
    with pytest.raises(ValueError, match="chunking mismatch"):
        resumed.fit_sequence(seq, steps=1, chunk_len=10)


def test_fit_sequence_resume_is_exact_with_a_multi_day_target(tmp_path: Path):
    """With an h-period target the carried state is the last h posteriors
    (audit M-1); a checkpoint must carry all of them for the resume to stay
    bit-exact."""
    import dataclasses

    def make() -> Trainer:
        torch.manual_seed(0)
        cfg = small_config(
            prior_kind="hmm", sigma_init=2.0, lr=3e-3,
            sigma_freeze_steps=10, checkpoint_every=10,
        )
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, target="fwd_ret_3d"))
        return Trainer(NECModel(cfg))

    seq = _panel(
        regime_process="markov", transition_stay=0.97,
        vol_levels=(1.0, 1.3), beta_scale=1.2, noise_std=0.8, seed=11,
    ).time_sequence()
    ref = make()
    ref.fit_sequence(seq, steps=40, chunk_len=25)
    ck = tmp_path / "hmm3.pt"
    interrupted = make()
    interrupted.fit_sequence(seq, steps=23, chunk_len=25, checkpoint_path=ck)
    resumed = Trainer.load(ck)
    assert resumed._seq_state is not None and len(resumed._seq_state) == 3
    resumed.fit_sequence(seq, steps=17, chunk_len=25)
    assert resumed.history == ref.history
    for k, v in ref.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[k], v), k


# --------------------------------------------------------------------------- #
# Harness level: walk-forward fold resume
# --------------------------------------------------------------------------- #


def test_walk_forward_resume_skips_and_matches(tmp_path: Path):
    panel = _panel(n_dates=160)
    calls: list[int] = []

    def make_trainer() -> Trainer:
        calls.append(1)
        torch.manual_seed(0)
        return Trainer(NECModel(small_config(sigma_init=1.5, lr=3e-3,
                                             batch_size=256)))

    kw = dict(
        n_folds=3, test_dates_per_fold=10, purge_dates=5, steps=60,
        warmstart_key=_vol_proxy, backtest_quantiles=4, cost_rate=0.001,
    )
    ref = walk_forward_evaluate(panel, make_trainer, **kw)
    assert len(calls) == 3

    # first run with resume_dir: identical results + one file per fold
    rdir = tmp_path / "wf"
    calls.clear()
    first = walk_forward_evaluate(panel, make_trainer, resume_dir=rdir, **kw)
    assert first == ref  # the resume plumbing changes nothing
    assert sorted(p.name for p in rdir.glob("fold_*.pt")) == [
        "fold_0.pt", "fold_1.pt", "fold_2.pt",
    ]
    saved = torch.load(rdir / "fold_0.pt", weights_only=False)
    assert "model" in saved and "config" in saved  # trained state persisted

    # restart with fold 1's file gone: folds 0 and 2 are skipped (no trainer
    # built for them), only fold 1 retrains — and the pooled result is
    # identical to the never-interrupted run
    (rdir / "fold_1.pt").unlink()
    calls.clear()
    second = walk_forward_evaluate(panel, make_trainer, resume_dir=rdir, **kw)
    assert len(calls) == 1
    assert second == ref


def test_walk_forward_resume_mid_fold(tmp_path: Path):
    """Composition of the two levels: a fold interrupted mid-fit leaves a
    trainer checkpoint; the restarted harness picks it up for the remaining
    steps and lands on the uninterrupted run's exact result."""
    panel = _panel(n_dates=160)
    cfg = small_config(sigma_init=1.5, lr=3e-3, batch_size=256,
                       checkpoint_every=10)

    def make_trainer() -> Trainer:
        torch.manual_seed(0)
        return Trainer(NECModel(cfg))

    kw = dict(n_folds=2, test_dates_per_fold=10, purge_dates=5, steps=50,
              warmstart_key=_vol_proxy)
    ref = walk_forward_evaluate(panel, make_trainer, **kw)

    # simulate a crash 20 steps into fold 0: replay exactly what the harness
    # does up to that point, leaving only the trainer checkpoint behind
    rdir = tmp_path / "wf"
    rdir.mkdir()
    folds = walk_forward_folds(panel.date, n_folds=2, test_dates_per_fold=10,
                               purge_dates=5)
    train0 = panel.subset_dates(folds[0].train_dates)
    crashed = make_trainer()
    crashed.warmstart_experts(train0.full_batch(), _vol_proxy(train0))
    crashed.fit(train0, steps=20, checkpoint_path=rdir / "fold_0_trainer.pt")

    resumed = walk_forward_evaluate(panel, make_trainer, resume_dir=rdir, **kw)
    assert resumed == ref
    assert not (rdir / "fold_0_trainer.pt").exists()  # superseded, cleaned up


# --------------------------------------------------------------------------- #
# Sweep level: completed (arm, seed) runs skipped via the registry
# --------------------------------------------------------------------------- #


def test_sweep_resume_skips_completed(tmp_path: Path):
    panel = _panel(n_dates=160)
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    arms = [
        nec_arm("soft", small_config(sigma_init=1.5, lr=3e-3, batch_size=256),
                warmstart_key=_vol_proxy),
        baseline_arm("ridge", lambda seed: RidgeBaseline(l2=1.0)),
    ]
    kw = dict(
        seeds=(0, 1), registry=reg, tag="ck_sweep", steps=60, n_folds=2,
        test_dates_per_fold=10, purge_dates=5, verbose=False,
        resume_dir=tmp_path / "sweep",
    )
    first = run_sweep(panel, arms, **kw)
    assert reg.n_trials("ck_sweep") == 4
    assert (tmp_path / "sweep" / "soft_seed0" / "fold_0.pt").exists()

    # rerun: everything is already in the registry — nothing retrains, no
    # duplicate rows, and the report is reconstructed identically
    second = run_sweep(panel, arms, **kw)
    assert reg.n_trials("ck_sweep") == 4
    assert second == first


# --------------------------------------------------------------------------- #
# Control-panel wiring
# --------------------------------------------------------------------------- #


def test_run_experiment_quick_checkpoint_resume(tmp_path: Path):
    import run_experiment as rx

    exp = rx.Experiment(
        tag="ckq", mode="quick", data="synthetic", out_dir=str(tmp_path),
        synth_dates=120, synth_entities=8, encoder_hidden=16, expert_hidden_dims=(16, 8),
        expert_dropout=0.0, steps=40, lr=3e-3, sigma_freeze_steps=20,
        checkpoint_every=10, figures=False, calibration=False,
    )
    first = rx.main(exp)
    ck = first["run_dir"] / "checkpoints" / "trainer.pt"
    assert ck.exists()
    assert Trainer.load(ck).step_count == 40

    # relaunch with --resume and a raised budget: continues from 40, not 0
    resumed = rx._parse_cli(dataclasses.replace(exp, steps=60), ["--resume"])
    assert resumed.resume
    rx.main(resumed)
    assert Trainer.load(ck).step_count == 60


def test_parse_cli_resume_run_id():
    import run_experiment as rx

    got = rx._parse_cli(rx.Experiment(tag="x"), ["--resume", "20260930-120000_x_abcd1234"])
    assert got.resume and got.resume_run_id == "20260930-120000_x_abcd1234"
    latest = rx._parse_cli(rx.Experiment(tag="x"), ["--resume"])
    assert latest.resume and latest.resume_run_id is None
