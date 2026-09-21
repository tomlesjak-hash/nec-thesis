"""Chain persistence (brief 1.1), the dead-parameter audit (brief 2), and the
non-stub (brief 1.4).

Persistence is the property that separates the gate families, so it is
reported rather than inferred: expected duration ``1/(1-A_kk)`` and the
stationary distribution, both recovered on chains with a known stay
probability, logged every step and carried into the trial registry in
canonical state order.

The audit measures the parameter-count confound: under ``hmm`` + snapshot
experts the encoder and gate head have no autograd path from the loss and
never train, while under ``soft`` they do — so the arms differ in effective
model size, and ``live_param_count`` is what makes that visible.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest
import torch
from conftest import small_config

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
)
from nec_moe.diagnostics import (
    expected_durations,
    persistence_metrics,
    stationary_distribution,
    wasserstein_template_tracking,
)
from nec_moe.train import DeadParameterWarning


def _panel(n_dates: int = 120, n_entities: int = 8, **spec_overrides):
    kw = dict(vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=3)
    kw.update(spec_overrides)
    return SyntheticRegimePanel(SyntheticSpec(**kw)).generate(n_dates, n_entities)


def _vol_proxy(p):
    return p.x_seq[:, :, 0].std(dim=1)


# --------------------------------------------------------------------------- #
# 1.1 Expected duration and stationary distribution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stay", [0.5, 0.9, 0.97, 0.99])
def test_expected_duration_recovers_known_stay_probability(stay: float):
    """The headline claim: a chain built with a known stay probability reports
    expected duration 1/(1-stay)."""
    a = torch.tensor([[stay, 1 - stay], [1 - stay, stay]])
    got = expected_durations(a)
    assert got.tolist() == pytest.approx([1 / (1 - stay)] * 2, rel=1e-5)


def test_expected_duration_flags_a_degenerate_chain():
    """A chain that re-draws its state every period has duration ~1 — the
    degenerate regime process this diagnostic exists to make visible."""
    iid = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    assert expected_durations(iid).tolist() == pytest.approx([2.0, 2.0])
    almost_iid = torch.tensor([[0.02, 0.98], [0.98, 0.02]])
    assert float(expected_durations(almost_iid).max()) < 1.1


def test_stationary_distribution_solves_its_defining_equation():
    a = torch.tensor([[0.97, 0.03], [0.10, 0.90]])
    pi = stationary_distribution(a)
    assert float(pi.sum()) == pytest.approx(1.0)
    assert torch.allclose(pi @ a.double(), pi, atol=1e-8)  # pi A = pi
    # closed form for a two-state chain: pi ∝ (1-A22, 1-A11) reversed
    assert pi.tolist() == pytest.approx([0.10 / 0.13, 0.03 / 0.13], rel=1e-6)


def test_stationary_distribution_three_states():
    a = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.05, 0.15, 0.8]])
    pi = stationary_distribution(a)
    assert float(pi.sum()) == pytest.approx(1.0)
    assert torch.allclose(pi @ a.double(), pi, atol=1e-8)
    assert bool((pi > 0).all())


def test_symmetric_chain_has_uniform_stationary_distribution():
    a = torch.tensor([[0.93, 0.07], [0.07, 0.93]])
    assert stationary_distribution(a).tolist() == pytest.approx([0.5, 0.5])


def test_persistence_metrics_are_registry_safe_and_canonicalized():
    """Absorbing states make the duration infinite; the registry rejects
    non-finite metrics, so the key is dropped while the bounded stay
    probability always survives."""
    absorbing = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
    m = persistence_metrics(absorbing)
    assert math.isinf(float(expected_durations(absorbing)[0]))
    assert "expected_duration_0" not in m  # dropped, not fabricated
    assert m["stay_prob_0"] == pytest.approx(1.0)  # always present
    assert m["expected_duration_1"] == pytest.approx(5.0)
    assert all(math.isfinite(v) for v in m.values())

    # `order` relabels states, so a key means the same regime across refits
    a = torch.tensor([[0.97, 0.03], [0.90, 0.10]])
    swapped = persistence_metrics(a, torch.tensor([1, 0]))
    assert swapped["stay_prob_0"] == pytest.approx(0.10)
    assert swapped["stay_prob_1"] == pytest.approx(0.97)


def test_persistence_metrics_reject_a_non_transition_matrix():
    with pytest.raises(ValueError, match="rows must sum to 1"):
        persistence_metrics(torch.tensor([[0.5, 0.2], [0.1, 0.9]]))
    with pytest.raises(ValueError, match="square"):
        expected_durations(torch.rand(2, 3))
    with pytest.raises(ValueError, match="negative"):
        stationary_distribution(torch.tensor([[1.5, -0.5], [0.2, 0.8]]))


def test_persistence_logged_every_step_for_a_transition_prior():
    """Logged every eval interval for any prior exposing a transition matrix —
    and absent, rather than faked, for priors that have none."""
    panel = _panel(n_dates=60, n_entities=6)
    torch.manual_seed(0)
    hmm = Trainer(NECModel(small_config(prior_kind="hmm", sigma_init=1.5)))

    # at init the chain is persistence-biased (diagonal logit bias 2.0), so the
    # diagnostic reads a sticky chain before a single step is taken
    sticky = float(torch.softmax(torch.tensor([2.0, 0.0]), 0)[0])
    at_init = persistence_metrics(hmm.model.prior.transition_matrix.detach())
    assert at_init["stay_prob_0"] == pytest.approx(sticky, rel=1e-5)
    assert at_init["expected_duration_0"] == pytest.approx(1 / (1 - sticky), rel=1e-5)

    history = hmm.fit_sequence(panel.time_sequence(), steps=5, chunk_len=20)
    for row in history:
        assert {"stay_prob_0", "stationary_0", "expected_duration_0"} <= set(row)
        assert row["expected_duration_0"] > 1.0
        assert row["stationary_0"] + row["stationary_1"] == pytest.approx(1.0)
    # rows track the chain as it trains, starting from the biased init
    assert history[0]["stay_prob_0"] == pytest.approx(sticky, abs=0.01)

    torch.manual_seed(0)
    soft = Trainer(NECModel(small_config(sigma_init=1.5)))
    assert not any(k.startswith("stay_prob") for k in soft.fit(panel, steps=3)[0])


def test_persistence_carried_through_the_walk_forward_harness(tmp_path: Path):
    """Every fold records its own chain's persistence, and the sweep averages
    it into the registry — the per-window estimates the thesis reports."""
    panel = _panel(n_dates=90, n_entities=6)

    def make_trainer() -> Trainer:
        torch.manual_seed(0)
        return Trainer(NECModel(small_config(prior_kind="hmm", sigma_init=1.5)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        res = walk_forward_evaluate(
            panel, make_trainer, n_folds=2, test_dates_per_fold=10,
            purge_dates=5, steps=8, warmstart_key=_vol_proxy,
        )
    for fold in res.folds:
        keys = dict(fold.persistence)
        assert {"stay_prob_0", "stay_prob_1", "stationary_0"} <= set(keys)
        assert all(math.isfinite(v) for v in keys.values())
        assert keys["stationary_0"] + keys["stationary_1"] == pytest.approx(1.0)
        assert keys["expected_duration_0"] == pytest.approx(
            1.0 / (1.0 - keys["stay_prob_0"]), rel=1e-5
        )

    reg = TrialRegistry(tmp_path / "trials.jsonl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        run_sweep(
            panel,
            [nec_arm("hmm", small_config(prior_kind="hmm", sigma_init=1.5),
                     warmstart_key=_vol_proxy)],
            seeds=(0,), registry=reg, tag="persist", steps=8, n_folds=2,
            test_dates_per_fold=10, purge_dates=5, verbose=False,
        )
    logged = reg.trials("persist")[0].metrics
    assert logged["expected_duration_0"] > 1.0
    assert logged["stationary_0"] + logged["stationary_1"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 2. The parameter-count confound
# --------------------------------------------------------------------------- #


def _fit_one_step(prior_kind: str, input_mode: str = "snapshot") -> Trainer:
    panel = _panel(n_dates=40, n_entities=6)
    torch.manual_seed(0)
    cfg = small_config(prior_kind=prior_kind, input_mode=input_mode, sigma_init=1.5)
    trainer = Trainer(NECModel(cfg))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        if trainer.model.prior.stateful:
            trainer.fit_sequence(panel.time_sequence(), steps=1, chunk_len=10)
        else:
            trainer.fit(panel, steps=1)
    return trainer


def test_hmm_snapshot_leaves_encoder_and_gate_dead():
    """The confound, stated as a test: under the HMM prior the gate logits are
    used only for shape inference, so encoder and gate head get no gradient."""
    trainer = _fit_one_step("hmm")
    audit = trainer.grad_audit
    assert audit is not None and audit.has_dead_parameters
    dead = set(audit.disconnected)
    assert any(n.startswith("encoder.") for n in dead)
    assert any(n.startswith("gate.") for n in dead)
    assert set(audit.disconnected_by_group) == {"encoder", "gate"}
    assert audit.live_param_count < audit.total_param_count


def test_soft_prior_keeps_encoder_and_gate_live():
    """The other half of the confound: the same tensors train under `soft`, so
    the two arms differ in effective model size."""
    soft = _fit_one_step("soft")
    hmm = _fit_one_step("hmm")
    assert soft.grad_audit is not None and not soft.grad_audit.has_dead_parameters
    assert soft.live_param_count is not None and hmm.live_param_count is not None
    assert soft.live_param_count > hmm.live_param_count


def test_zero_initialized_gate_is_transient_not_dead():
    """A zero gradient is not deadness. GateConfig.zero_init makes the
    encoder's first-step gradient exactly zero while the autograd path is
    intact; counting that as dead would erase the confound above by declaring
    every arm equally dead."""
    soft = _fit_one_step("soft")
    audit = soft.grad_audit
    assert audit is not None
    assert any(n.startswith("encoder.") for n in audit.zero_grad)  # zero at step 0
    assert not any(n.startswith("encoder.") for n in audit.disconnected)  # connected
    # ... and nonzero once the gate weight has moved off zero
    panel = _panel(n_dates=40, n_entities=6)
    soft.fit(panel, steps=3)
    grad = soft.model.encoder.gru.weight_ih_l0.grad
    assert grad is not None and float(grad.abs().max()) > 0.0


def test_dead_parameters_emit_one_structured_warning():
    panel = _panel(n_dates=40, n_entities=6)
    torch.manual_seed(0)
    trainer = Trainer(NECModel(small_config(prior_kind="hmm", sigma_init=1.5)))
    with pytest.warns(DeadParameterWarning, match="live_param_count") as caught:
        trainer.fit_sequence(panel.time_sequence(), steps=3, chunk_len=10)
    assert len(caught) == 1  # once per fit, not once per step
    assert "encoder" in str(caught[0].message)


def test_sigma_freeze_is_reported_as_frozen_not_dead():
    """log_sigma is requires_grad=False during its warm-start freeze; that is a
    schedule, not a severed graph, so it must not be reported as dead."""
    panel = _panel(n_dates=40, n_entities=6)
    torch.manual_seed(0)
    cfg = small_config(sigma_init=1.5, sigma_freeze_steps=50)
    trainer = Trainer(NECModel(cfg))
    trainer.fit(panel, steps=1)
    audit = trainer.grad_audit
    assert audit is not None
    assert any("log_sigma" in n for n in audit.frozen)
    assert not any("log_sigma" in n for n in audit.disconnected)
    assert not audit.has_dead_parameters  # no warning-worthy deadness


def test_live_param_count_survives_checkpoint_resume(tmp_path: Path):
    """A fold resumed with zero remaining steps still reports the count."""
    panel = _panel(n_dates=40, n_entities=6)
    torch.manual_seed(0)
    trainer = Trainer(NECModel(small_config(sigma_init=1.5)))
    trainer.fit(panel, steps=2)
    ck = tmp_path / "t.pt"
    trainer.save(ck)
    restored = Trainer.load(ck)
    assert restored.live_param_count == trainer.live_param_count
    assert restored.grad_audit is not None


def test_live_param_count_reaches_the_registry(tmp_path: Path):
    panel = _panel(n_dates=90, n_entities=6)
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    arms = [
        nec_arm("soft", small_config(sigma_init=1.5, lr=3e-3, batch_size=256),
                warmstart_key=_vol_proxy),
        baseline_arm("ridge", lambda seed: RidgeBaseline(l2=1.0)),
    ]
    run_sweep(
        panel, arms, seeds=(0,), registry=reg, tag="confound", steps=10,
        n_folds=2, test_dates_per_fold=10, purge_dates=5, verbose=False,
    )
    by_arm = {r.config["arm"]: r.metrics for r in reg.trials("confound")}
    assert by_arm["soft"]["live_param_count"] > 0
    # baselines never run the audit: absent rather than a fabricated zero
    assert "live_param_count" not in by_arm["ridge"]


# --------------------------------------------------------------------------- #
# 1.4 No importable no-op
# --------------------------------------------------------------------------- #


def test_wasserstein_template_tracking_is_not_a_silent_stub():
    """Exported, so it must fail loudly rather than return None — and say
    both where the baseline's answer lives and what it is not."""
    with pytest.raises(NotImplementedError) as e:
        wasserstein_template_tracking(torch.zeros(2), torch.zeros(2))
    msg = str(e.value)
    assert "canonical_expert_order" in msg  # what to use instead
    assert "gate" in msg  # distinct from the Wasserstein gate
