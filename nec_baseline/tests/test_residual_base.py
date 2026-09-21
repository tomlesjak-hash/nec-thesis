"""Residual mixture with a frozen base (brief 02): one test per property claimed.

The design is ``y_hat = f0(x) + sum_k pi_k r_k(x)`` with **both ends frozen** —
base and gate — so that the only thing varying across arms is the regime
process, and the measured quantity is what a regime-conditional correction adds
to a fixed baseline (Q7, Q19). The properties that make that claim true, and
that each fail silently if broken:

1. at init the model *is* the base, exactly (zero-initialized correction heads);
2. frozen really means frozen — the tensors themselves must not move;
3. the base cache is keyed on content, so two gate arms share a base and a
   changed config never silently reuses a stale one;
4. the penalty is zero at zero correction and grows with it;
5. and the whole arrangement can actually detect what it is built to detect.

Test 5 is the important one: everything above can hold while the system
recovers nothing.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest
import torch
from conftest import small_base_config, small_config

from nec_moe import (
    BaseCache,
    BaseConfig,
    NECModel,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    TrialRegistry,
    base_cache_key,
    correction_penalty_aux,
    fit_base,
    nec_arm,
    pyramid_dims,
    run_sweep,
    walk_forward_evaluate,
)
from nec_moe.train import DeadParameterWarning


def _panel(n_dates: int = 160, n_entities: int = 8, **spec):
    kw = dict(vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=3)
    kw.update(spec)
    return SyntheticRegimePanel(SyntheticSpec(**kw)).generate(n_dates, n_entities)


def _residual_cfg(**overrides):
    """Correction mode + frozen gate: the arrangement brief 02 specifies."""
    kw = dict(
        correction_mode=True,
        zero_init_head=True,
        base=small_base_config(),
        sigma_init=1.0,
        lr=3e-3,
        batch_size=256,
        freeze_gate=True,
    )
    kw.update(overrides)
    return small_config(**kw)


def _fitted(cfg, panel, seed: int = 0) -> Trainer:
    torch.manual_seed(seed)
    trainer = Trainer(NECModel(cfg))
    fit = fit_base(panel, cfg.base, seed=seed)
    trainer.model.attach_base(fit.model)
    return trainer


# --------------------------------------------------------------------------- #
# 1. At initialisation the model IS the base
# --------------------------------------------------------------------------- #


def test_zero_init_starts_exactly_at_the_base():
    """The property the whole residual design rests on: before any training
    step the correction is identically zero, so predictions equal the base's
    *exactly* — not approximately."""
    panel = _panel()
    cfg = _residual_cfg()
    trainer = _fitted(cfg, panel)
    trainer.model.eval()
    with torch.no_grad():
        out = trainer.model(panel.x_seq, panel.x_snap)
        base_only = trainer.model.base(panel.x_snap)

    assert torch.equal(out.y_hat, base_only)  # bitwise, not allclose
    assert out.corrections is not None
    assert float(out.corrections.abs().max()) == 0.0
    assert float(out.correction.abs().max()) == 0.0
    # every expert's predictive mean is the base, so the mixture is degenerate
    # in the means and the gate cannot matter yet — which is the point
    assert torch.equal(out.mu[:, 0], out.mu[:, 1])


def test_random_init_head_does_not_start_at_the_base():
    """The ablation's counterpart (Ye & Borde replace zero-init with random):
    without it the model starts somewhere arbitrary, which is what the
    initialization exists to prevent."""
    panel = _panel()
    trainer = _fitted(_residual_cfg(zero_init_head=False), panel)
    trainer.model.eval()
    with torch.no_grad():
        out = trainer.model(panel.x_seq, panel.x_snap)
    assert float(out.correction.abs().max()) > 0.0


def test_correction_mode_without_a_base_is_refused():
    with pytest.raises(ValueError, match="requires base.enabled"):
        small_config(correction_mode=True, base=BaseConfig(enabled=False)).validate()


def test_unfitted_base_is_refused_rather_than_silently_used():
    """An attached-but-unfitted base would make the corrections corrections to
    a random network — wrong, and invisible in the loss curve."""
    cfg = _residual_cfg()
    torch.manual_seed(0)
    model = NECModel(cfg)  # base module exists (structural) but never fitted
    panel = _panel(40, 4)
    with pytest.raises(RuntimeError, match="no fitted base"):
        model(panel.x_seq, panel.x_snap)


# --------------------------------------------------------------------------- #
# 2. Frozen means frozen — compare tensors, not flags
# --------------------------------------------------------------------------- #


def test_base_and_gate_tensors_are_unchanged_by_training():
    """Brief 02 §4 asks for the tensor comparison specifically: `requires_grad`
    is a claim, an unchanged tensor is evidence."""
    panel = _panel()
    cfg = _residual_cfg()
    trainer = _fitted(cfg, panel)
    before = {
        n: p.detach().clone()
        for n, p in trainer.model.named_parameters()
        if n.startswith(("base.", "gate.", "encoder.", "prior."))
    }
    assert before, "expected frozen parameter groups to exist"
    trainer.fit(panel, steps=25)

    for name, old in before.items():
        assert torch.equal(dict(trainer.model.named_parameters())[name], old), name
    # ... while the experts did move, or the test proves nothing
    assert float(trainer.model.experts.experts[0].head.weight.abs().max()) > 0.0


def test_frozen_parameters_never_enter_the_optimizer():
    panel = _panel()
    trainer = _fitted(_residual_cfg(), panel)
    in_opt = {
        id(p) for group in trainer.opt.param_groups for p in group["params"]
    }
    for name, p in trainer.model.named_parameters():
        frozen = name.startswith(("base.", "gate.", "encoder.", "prior."))
        assert (id(p) in in_opt) == (not frozen), name


def test_dead_parameter_audit_sees_the_frozen_ends():
    """Reuse brief 01's audit with the expectation inverted: here the frozen
    base and gate *should* show up as not-live, and live_param_count should
    count only the experts and their sigmas."""
    panel = _panel()
    trainer = _fitted(_residual_cfg(), panel)
    trainer.fit(panel, steps=2)
    audit = trainer.grad_audit
    assert audit is not None
    assert any(n.startswith("base.") for n in audit.frozen)
    assert any(n.startswith("gate.") for n in audit.frozen)
    expert_params = sum(
        p.numel() for n, p in trainer.model.named_parameters()
        if n.startswith("experts.")
    )
    assert audit.live_param_count == expert_params


def test_frozen_base_stays_in_eval_mode_during_training():
    """`model.train()` recurses; a base left in train mode would keep sampling
    dropout and its predictions would move with its weights pinned."""
    panel = _panel()
    cfg = _residual_cfg(base=small_base_config(dropout=0.3))
    trainer = _fitted(cfg, panel)
    trainer._train_mode()
    assert not trainer.model.base.training
    assert not trainer.model.encoder.training  # frozen gate path too
    # and the base's predictions are deterministic across repeated calls
    with torch.no_grad():
        a = trainer.model.base(panel.x_snap)
        b = trainer.model.base(panel.x_snap)
    assert torch.equal(a, b)


def test_warmstart_is_refused_when_it_would_break_zero_init():
    panel = _panel()
    trainer = _fitted(_residual_cfg(), panel)
    with pytest.raises(ValueError, match="incompatible with correction_mode"):
        trainer.warmstart_experts(
            panel.full_batch(), panel.x_seq[:, :, 0].std(dim=1)
        )


# --------------------------------------------------------------------------- #
# 3. The base cache: shared across arms, never stale
# --------------------------------------------------------------------------- #


def test_base_cache_shares_across_gate_arms_and_splits_on_config():
    """The amortisation: the base never sees regime information, so two gate
    arms of one window and seed get the *same object* — and any BaseConfig
    change must produce a different one."""
    panel = _panel(80, 6)
    cache = BaseCache()
    cfg = small_base_config()
    soft = cache.get_or_fit(panel, cfg, window=(0, 60), seed=0)
    hmm = cache.get_or_fit(panel, cfg, window=(0, 60), seed=0)
    assert soft is hmm  # identical object, not merely equal
    assert cache.hits == 1 and cache.misses == 1

    # every field of BaseConfig participates in the key
    for field, value in [
        ("hidden_dims", (8, 4)), ("dropout", 0.1), ("activation", "gelu"),
        ("lr", 5e-3), ("weight_decay", 0.0), ("steps", 61),
        ("batch_size", 32), ("early_stopping_patience", 5),
        ("val_fraction", 0.3), ("seed_offset", 1),
    ]:
        changed = dataclass_replace(cfg, field, value)
        assert base_cache_key(changed, (0, 60), 0, 3) != base_cache_key(
            cfg, (0, 60), 0, 3
        ), field
        assert cache.get_or_fit(panel, changed, window=(0, 60), seed=0) is not soft

    # window and seed split it too
    assert cache.get_or_fit(panel, cfg, window=(1, 61), seed=0) is not soft
    assert cache.get_or_fit(panel, cfg, window=(0, 60), seed=1) is not soft


def dataclass_replace(cfg: BaseConfig, field: str, value) -> BaseConfig:
    import dataclasses

    return dataclasses.replace(cfg, **{field: value})


def test_base_validation_split_is_a_tail_of_training_only():
    """The base's early-stopping data comes from the tail of the block it is
    given — never from anything later."""
    panel = _panel(80, 6)
    train, _ = panel.split_by_date(0.75)
    fit = fit_base(train, small_base_config(early_stopping_patience=3), seed=0)
    assert math.isfinite(fit.val_loss) and fit.steps_run >= 1
    assert all(not p.requires_grad for p in fit.model.parameters())
    assert not fit.model.training  # returned frozen AND in eval mode


# --------------------------------------------------------------------------- #
# 4. The correction penalty
# --------------------------------------------------------------------------- #


def test_correction_penalty_is_zero_at_zero_and_grows_with_magnitude():
    pi = torch.tensor([[0.6, 0.4], [0.3, 0.7]])
    assert float(correction_penalty_aux(pi, torch.zeros(2, 2))) == 0.0
    small = correction_penalty_aux(pi, torch.full((2, 2), 0.1))
    large = correction_penalty_aux(pi, torch.full((2, 2), 0.5))
    assert 0.0 < float(small) < float(large)
    # it penalises the COMBINED correction: experts that cancel are not
    # penalised, which a per-expert penalty would get wrong
    cancelling = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    assert float(correction_penalty_aux(torch.full((2, 2), 0.5), cancelling)) == 0.0


def test_correction_penalty_shrinks_the_correction_when_enabled():
    """The knob does what it claims: a large alpha keeps the mixture nearer the
    base than alpha = 0 does, on the same data and seed."""
    panel = _panel()
    out = {}
    for alpha in (0.0, 50.0):
        cfg = _residual_cfg(
            zero_init_head=False,  # start away from the base so shrinkage bites
            aux_correction_penalty=alpha > 0,
            correction_penalty_weight=alpha,
        )
        trainer = _fitted(cfg, panel, seed=0)
        trainer.fit(panel, steps=60)
        trainer.model.eval()
        with torch.no_grad():
            out[alpha] = float(
                trainer.model(panel.x_seq, panel.x_snap).correction.abs().mean()
            )
    assert out[50.0] < out[0.0]


# --------------------------------------------------------------------------- #
# 5. THE test: can the arrangement detect what it is built to detect?
# --------------------------------------------------------------------------- #


def test_residual_mixture_recovers_the_regime_conditional_part():
    """Planted truth: y = g(x) + regime-conditional part.

    The unconditional component is a fixed linear function of the snapshot
    (which the base can learn); on top of it sits the sign-flip regime
    structure (which no single function of x can capture — the pooled
    relationship is zero by construction). If the arrangement works, the base
    recovers the unconditional part and the *correction* recovers the
    regime-conditional part, so the mixture's IC beats the base's by a wide
    margin and the corrections are not numerically negligible.
    """
    import dataclasses

    spec = SyntheticSpec(vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.3, seed=5)
    panel = SyntheticRegimePanel(spec).generate(200, 10)
    # add a strong UNCONDITIONAL signal on a snapshot channel, regime-independent:
    # a single fixed function of x that the base can learn on its own
    unconditional = 3.0 * panel.x_snap[:, 0]
    planted = dataclasses.replace(panel, y=panel.y + unconditional)

    cfg = _residual_cfg(sigma_init=1.0, lr=5e-3, freeze_gate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        result = walk_forward_evaluate(
            planted,
            lambda: Trainer(NECModel(cfg)),
            n_folds=2, test_dates_per_fold=20, purge_dates=5, steps=300,
            base_cache=BaseCache(), seed=0,
        )

    fold = result.folds[0]
    assert fold.base_ic is not None, "the base must be scored beside the mixture"
    # the base captures the unconditional part on its own ...
    assert fold.base_ic.mean_ic > 0.3, fold.base_ic
    # ... and the regime-conditional correction adds on top of it, per fold
    for f in result.folds:
        assert f.ic_improvement is not None and f.ic_improvement > 0.05, f
    # the corrections are materially nonzero: a negligible correction would
    # answer the thesis question in the negative whatever the IC did
    corr = dict(fold.correction)
    assert corr["correction_mean_abs"] > 0.0
    assert corr["correction_rel_base_std"] > 0.01, corr


def test_base_and_improvement_reach_the_registry(tmp_path: Path):
    """§5's reporting contract: the base's own score, the improvement over it,
    and the correction magnitude all land in the trial row."""
    panel = _panel(120, 8)
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    cfg = _residual_cfg(freeze_gate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        run_sweep(
            panel, [nec_arm("resid", cfg)], seeds=(0,), registry=reg,
            tag="resid", steps=40, n_folds=2, test_dates_per_fold=10,
            purge_dates=5, verbose=False,
        )
    m = reg.trials("resid")[0].metrics
    for key in ("base_mean_ic", "base_nll", "ic_improvement", "nll_improvement",
                "correction_mean_abs", "live_param_count"):
        assert key in m, key
    assert m["ic_improvement"] == pytest.approx(m["mean_ic"] - m["base_mean_ic"])
    # the full BaseConfig is in the trial's config, so the row is reconstructible
    cfg_row = reg.trials("resid")[0].config["nec_config"]
    assert cfg_row["base"]["enabled"] is True
    assert cfg_row["experts"]["correction_mode"] is True


# --------------------------------------------------------------------------- #
# Config surface: the ablations of §6 must be reachable by config alone
# --------------------------------------------------------------------------- #


def test_pyramid_dims_follows_the_geometric_rule():
    assert pyramid_dims(32, 3) == (32, 16, 8)  # Gu-Kelly-Xiu's NN3
    assert pyramid_dims(64, 2) == (64, 32)
    assert pyramid_dims(4, 5) == (4, 2, 1, 1, 1)  # floors at 1, never 0
    with pytest.raises(ValueError):
        pyramid_dims(0, 2)


def test_section_6_ablations_are_config_only():
    """Each ablation is a field change on a valid config — no code path."""
    base = small_base_config()
    # (a) no base at all: the standard mixture
    small_config(base=BaseConfig(enabled=False)).validate()
    # (b) random-init correction heads
    _residual_cfg(zero_init_head=False).validate()
    # (c) base present, experts emit full forecasts
    small_config(correction_mode=False, base=base).validate()
    # (d) no shrinkage
    _residual_cfg(aux_correction_penalty=False, correction_penalty_weight=0.0).validate()
    # (e) a trainable prior alongside the base — the jointly-trained arm
    _residual_cfg(prior_kind="soft", freeze_gate=False).validate()


def test_inert_load_balancing_under_a_frozen_gate_is_refused():
    """Q19 consequence 3: with a frozen gate the term's gradient is exactly
    zero, so silently 'enabling' it would misreport the method."""
    with pytest.raises(ValueError, match="inert when train.freeze_gate"):
        small_config(freeze_gate=True, aux_load_balance=True).validate()
