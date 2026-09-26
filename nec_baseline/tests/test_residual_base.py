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
    PriorContext,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    TrialRegistry,
    base_cache_key,
    base_single_gaussian_nll,
    correction_penalty_aux,
    expert_decorrelation_aux,
    expert_log_likelihood,
    fit_base,
    mixture_nll,
    nec_arm,
    pyramid_dims,
    run_sweep,
    walk_forward_evaluate,
    walk_forward_folds,
)
from nec_moe.diagnostics import expert_output_correlation, pairwise_expert_distance
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
    assert float(trainer.model.experts.experts[0].head.weight.detach().abs().max()) > 0.0


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


def test_base_is_fitted_on_each_folds_training_block_only(monkeypatch):
    """Audit B-5 / critical property P7. Within the walk-forward harness, the
    base must be fitted once per fold on exactly that fold's (purged) training
    dates: never on a purge-gap date, whose label overlaps the test block, and
    never on a test date. The spy records what fit_base actually receives,
    because the cache key alone (the window's end dates) would not notice a
    base fitted on a different panel under the same key.
    """
    import nec_moe.base as base_module

    seen: list[torch.Tensor] = []
    real_fit_base = base_module.fit_base

    def spy(panel, cfg, *, seed):
        seen.append(torch.unique(panel.date, sorted=True))
        return real_fit_base(panel, cfg, seed=seed)

    monkeypatch.setattr(base_module, "fit_base", spy)
    panel = _panel(160, 6)
    cfg = _residual_cfg(prior_kind="uniform")
    kw = dict(n_folds=3, test_dates_per_fold=10, purge_dates=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        walk_forward_evaluate(
            panel, lambda: Trainer(NECModel(cfg)), steps=3,
            base_cache=BaseCache(), seed=0, **kw,
        )

    folds = walk_forward_folds(panel.date, **kw)
    assert len(seen) == len(folds)  # once per fold
    for dates, fold in zip(seen, folds, strict=True):
        assert torch.equal(dates, torch.sort(fold.train_dates).values)
        assert int(dates.max()) < int(fold.purged_dates.min())
        assert int(dates.max()) < int(fold.test_dates.min())


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
# Base and experts are separately parameterised networks
# --------------------------------------------------------------------------- #


def test_base_and_experts_take_independent_architectures():
    """Both are MLPs, but nothing is shared between their configs: depth,
    width, dropout and activation are set per network, and the base carries
    its own optimiser knobs besides."""
    cfg = small_config(
        expert_hidden_dims=(12, 6, 3),
        expert_dropout=0.2,
        correction_mode=True,
        base=small_base_config(
            hidden_dims=(32, 8), dropout=0.05, activation="gelu",
            lr=7e-3, weight_decay=0.0, steps=25, batch_size=16,
        ),
    )
    cfg.validate()
    torch.manual_seed(0)
    model = NECModel(cfg)

    assert model.base.net.hidden_dims == (32, 8)
    assert model.experts.experts[0].hidden_dims == (12, 6, 3)
    # activations differ, and neither is hardcoded
    assert isinstance(model.base.net.hidden[1], torch.nn.GELU)
    assert isinstance(model.experts.experts[0].hidden[1], torch.nn.ReLU)
    # dropout rates differ
    assert model.base.net.hidden[2].p == 0.05
    assert model.experts.experts[0].hidden[2].p == 0.2
    # the base is deeper-per-layer but shallower; parameter counts differ
    base_n = sum(p.numel() for p in model.base.parameters())
    expert_n = sum(p.numel() for p in model.experts.experts[0].parameters())
    assert base_n != expert_n
    # and the base's own optimiser settings are its own
    assert cfg.base.lr == 7e-3 and cfg.train.lr != 7e-3


def test_control_panel_exposes_every_base_field_separately():
    """§2's constraint: each BaseConfig field is reachable from the settings
    block, under a base_* name distinct from the expert_* one."""
    import dataclasses

    import run_experiment as rx

    panel_fields = {f.name for f in dataclasses.fields(rx.Experiment)}
    for f in dataclasses.fields(BaseConfig):
        name = "base_enabled" if f.name == "enabled" else f"base_{f.name}"
        assert name in panel_fields, name
    # the expert-side twins exist and are distinct knobs
    for name in ("expert_hidden_dims", "expert_dropout", "expert_activation"):
        assert name in panel_fields, name

    exp = rx.Experiment(
        base_enabled=True, correction_mode=True,
        base_hidden_dims=(32, 8), base_dropout=0.05, base_activation="gelu",
        expert_hidden_dims=(12, 6), expert_dropout=0.2, expert_activation="tanh",
    )
    cfg = rx._nec_config(exp, _panel(40, 4), sigma_init=1.0)
    assert cfg.base.hidden_dims == (32, 8) and cfg.experts.hidden_dims == (12, 6)
    assert cfg.base.activation == "gelu" and cfg.experts.activation == "tanh"
    assert cfg.base.dropout == 0.05 and cfg.experts.dropout == 0.2


# --------------------------------------------------------------------------- #
# Correction 1: between-expert statistics must read r, never mu
# --------------------------------------------------------------------------- #


def test_corrections_are_materialised_not_recovered_by_subtraction():
    """`out.corrections` is the experts' raw output, captured before the base
    is added — so it is exact, and it does not depend on f0 being re-read."""
    panel = _panel(80, 6)
    trainer = _fitted(_residual_cfg(zero_init_head=False), panel)
    trainer.model.eval()
    with torch.no_grad():
        out = trainer.model(panel.x_seq, panel.x_snap)
        raw = torch.stack(
            [e(panel.x_snap) for e in trainer.model.experts.experts], dim=-1
        )
    assert torch.equal(out.corrections, raw)  # bitwise: it IS the expert output
    assert out.expert_signal is out.corrections
    # mu is the sum, and recovering r by subtraction is NOT bitwise equal —
    # which is exactly why the forward pass materialises it instead
    assert torch.allclose(out.mu - out.base_pred.unsqueeze(-1), out.corrections)


def test_between_expert_statistics_ignore_the_shared_base():
    """With mu_k = f0 + r_k the base is common to every column, so statistics
    computed on mu measure f0. A dominant base drives the correlation toward
    1 however the corrections behave; distances are immune (the base cancels
    in a difference) and the correlation must be taken on r."""
    b = 4096
    torch.manual_seed(0)
    r = torch.randn(b, 2) * 0.1  # INDEPENDENT corrections: the healthy state
    base = 20.0 * torch.randn(b, 1)  # a base that dominates them
    mu = base + r

    corr_on_r = expert_output_correlation(r)
    corr_on_mu = expert_output_correlation(mu)
    assert abs(float(corr_on_r[0, 1])) < 0.05  # truly uncorrelated
    assert float(corr_on_mu[0, 1]) > 0.99  # the base's correlation, not theirs

    # The aux loss inherits the distortion, and this is the damaging direction:
    # on r it correctly reports ~0 ("nothing to fix"), while on mu it reports
    # the maximum ~2 for K=2 and demands a decorrelation that no change in r
    # can deliver except by inflating r until it dominates f0. (Note the aux is
    # ||R - I||_F^2, which squares: it is blind to the SIGN of the correlation,
    # so anti-correlated corrections would score the same 2.0 as correlated
    # ones — which is why this test uses independent corrections to separate
    # the two readings.)
    assert float(expert_decorrelation_aux(r)) < 0.05
    assert float(expert_decorrelation_aux(mu)) > 1.9

    # distances are unaffected by the shared base: it cancels in a difference
    assert torch.allclose(
        pairwise_expert_distance(mu), pairwise_expert_distance(r), atol=1e-4
    )


def test_trainer_diagnostics_and_aux_use_corrections():
    """The wiring, end to end: the logged correlation and the decorrelation
    penalty both read r in correction mode."""
    panel = _panel(120, 8)
    cfg = _residual_cfg(
        zero_init_head=False, freeze_gate=False,
        aux_expert_decorrelation=True, aux_decorrelation_weight=1e-2,
    )
    trainer = _fitted(cfg, panel)
    history = trainer.fit(panel, steps=5)

    trainer.model.eval()
    with torch.no_grad():
        out = trainer.model(panel.x_seq, panel.x_snap)
    on_r = expert_output_correlation(out.corrections)
    on_mu = expert_output_correlation(out.mu)
    logged = history[-1]["max_offdiag_expert_corr"]
    # the logged value tracks r, not mu (they differ once the base is fitted)
    off = lambda c: float((c - torch.eye(2)).abs().max())  # noqa: E731
    assert abs(logged - off(on_r)) < abs(logged - off(on_mu)) or off(on_r) == pytest.approx(
        off(on_mu), abs=1e-3
    )
    # and the distance diagnostic is logged alongside it
    assert {"min_pairwise_expert_distance", "mean_pairwise_expert_distance"} <= set(
        history[-1]
    )


def test_pairwise_expert_distance_is_a_metric():
    v = torch.tensor([[0.0, 1.0, 5.0], [0.0, 1.0, 5.0], [0.0, 3.0, 5.0]])
    d = pairwise_expert_distance(v)
    assert torch.equal(d.diagonal(), torch.zeros(3))
    assert torch.allclose(d, d.T)
    assert float(d[0, 2]) == pytest.approx(5.0)
    # identical experts are at distance zero even when correlation is undefined
    same = torch.stack([v[:, 0], v[:, 0]], dim=-1)
    assert float(pairwise_expert_distance(same).max()) == 0.0
    with pytest.raises(ValueError, match=r"\(B, K\)"):
        pairwise_expert_distance(torch.zeros(3))


# --------------------------------------------------------------------------- #
# Correction 2: the prior must sum to one where f0's coefficient depends on it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("prior_kind", ["soft", "uniform", "hard", "topk", "gumbel"])
def test_every_prior_is_normalized_in_correction_mode(prior_kind: str):
    """f0's coefficient is sum_k pi_k; the identity y_hat = f0 + sum_k pi_k r_k
    holds only because that is exactly 1, so it is checked at the call site."""
    panel = _panel(60, 6)
    cfg = _residual_cfg(prior_kind=prior_kind, zero_init_head=False, freeze_gate=False)
    trainer = _fitted(cfg, panel)
    trainer.model.eval()
    with torch.no_grad():
        out = trainer.model(panel.x_seq, panel.x_snap)
    lse = torch.logsumexp(out.prior.log_prior, dim=-1)
    assert float(lse.abs().max()) < 1e-5
    # the identity itself, which is what the assertion protects
    assert torch.allclose(
        out.y_hat, out.base_pred + out.correction, atol=1e-5
    )


def test_unnormalized_prior_is_refused():
    """A prior whose rows do not sum to 1 would silently rescale the frozen
    base rather than fail; the guard turns that into an exception."""
    from nec_moe.model import _assert_normalized

    _assert_normalized(torch.log(torch.tensor([[0.3, 0.7], [0.5, 0.5]])))  # fine
    with pytest.raises(ValueError, match="rows must sum to 1"):
        _assert_normalized(torch.log(torch.tensor([[0.3, 0.3], [0.5, 0.5]])))


def test_posterior_equals_prior_at_initialisation():
    """At init every expert predicts the base, so the likelihood carries no
    information about which expert is responsible and Bayes returns the prior
    unchanged. The mixture is degenerate at step 0 by construction — which is
    what 'starts exactly at the base' means on the posterior side."""
    panel = _panel(120, 8)
    cfg = _residual_cfg(freeze_gate=False)  # a non-trivial (learned) prior
    trainer = _fitted(cfg, panel)
    trainer.model.eval()
    with torch.no_grad():
        out = trainer.model(panel.x_seq, panel.x_snap)
        log_lik = expert_log_likelihood(out.mu, out.log_sigma, panel.y)
        nll_out = mixture_nll(out.prior.log_prior, log_lik)

    # the per-expert likelihoods are identical across k ...
    assert torch.allclose(log_lik[:, 0], log_lik[:, 1], atol=1e-6)
    # ... so the posterior is the prior. Not bitwise: log_filtered computes
    # (log_prior + c) - logsumexp(log_prior + c), and float addition of the
    # common constant c is not exactly invertible.
    assert torch.allclose(
        nll_out.log_filtered, out.prior.log_prior, atol=1e-5
    )
    assert torch.allclose(
        nll_out.responsibilities, out.prior.log_prior.exp(), atol=1e-5
    )


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


def test_low_snr_residual_run_is_numerically_sane():
    """The same arrangement at a realistic signal size — and the honest
    assertion set that goes with it.

    Test 5 plants a signal a model can actually find. Real cross-sectional
    equity prediction is nowhere near that: Gu, Kelly and Xiu report a monthly
    out-of-sample R-squared of roughly 0.4% for the benchmark network class,
    and the base is expected to take most of even that, leaving the experts a
    residual that may be indistinguishable from noise. This test sizes the
    planted signal to R-squared ~= 0.005 and asserts **numerical sanity only**:
    that training completes, that nothing becomes NaN or infinite, that the
    corrections stay bounded rather than exploding against the frozen base,
    and that the reported improvement is a finite number.

    It deliberately does NOT assert recovery. A null is a real possible
    outcome here (Q7 consequence 5), and a test that demanded a positive
    improvement at this SNR would either be flaky or be silently tuned until
    it passed — which is the failure mode this whole codebase is built to
    avoid. What may be claimed at this SNR is 'the machinery ran and produced
    finite numbers', and that is what is claimed.
    """
    import dataclasses

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(11)
    spec = SyntheticSpec(vol_levels=(1.0, 1.0), beta_scale=1.0, noise_std=1.0, seed=7)
    panel = SyntheticRegimePanel(spec).generate(240, 12)

    # rebuild y as: tiny signal + dominant noise, with R^2 fixed by construction.
    # R^2 = var(signal) / (var(signal) + var(noise)) = 0.005  =>  s/n = sqrt(R2/(1-R2))
    r2_target = 0.005
    noise = torch.randn(len(panel.y), generator=g)
    raw_signal = panel.x_snap[:, 0]
    signal = raw_signal / raw_signal.std() * math.sqrt(r2_target / (1 - r2_target))
    low_snr = dataclasses.replace(panel, y=signal + noise)

    realized_r2 = float(signal.var() / low_snr.y.var())
    assert 0.002 < realized_r2 < 0.01, realized_r2  # the construction itself

    cfg = _residual_cfg(
        freeze_gate=False, sigma_init=1.0, lr=1e-3,
        base=small_base_config(steps=150, lr=3e-3),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        result = walk_forward_evaluate(
            low_snr,
            lambda: Trainer(NECModel(cfg)),
            n_folds=2, test_dates_per_fold=20, purge_dates=5, steps=150,
            base_cache=BaseCache(), seed=0,
        )

    for fold in result.folds:
        assert math.isfinite(fold.nll), fold
        assert fold.base_nll is not None and math.isfinite(fold.base_nll)
        assert math.isfinite(fold.ic.mean_ic)
        assert fold.ic_improvement is not None
        assert math.isfinite(fold.ic_improvement)  # finite, sign unconstrained
        corr = dict(fold.correction)
        assert all(math.isfinite(v) for v in corr.values()), corr
        # corrections must stay bounded: against an almost-pure-noise target
        # an unconstrained mixture can chase noise without limit, and that
        # would show up here long before it showed up in an IC
        assert corr["correction_max_abs"] < 100.0, corr
    assert math.isfinite(result.pooled_ic.mean_ic)


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
# Audit E-1: the base is scored by the same model with corrections at zero
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("prior_kind", ["uniform", "hmm"])
def test_nll_improvement_is_exactly_zero_before_any_training(prior_kind: str):
    """With zero-initialised heads and no training every correction is zero,
    so the mixture IS the base and the reported NLL improvement must be exactly
    zero, on the memoryless and on the stateful (warmed-up filter) path alike.
    The old base NLL (one Gaussian at an averaged sigma) failed this whenever
    the experts' noise scales differed."""
    panel = _panel(160, 8)
    cfg = _residual_cfg(prior_kind=prior_kind, sigma_init=0.7)

    def make() -> Trainer:
        torch.manual_seed(0)
        trainer = Trainer(NECModel(cfg))
        with torch.no_grad():  # experts disagree on noise; corrections stay 0
            trainer.model.experts.log_sigma.copy_(torch.tensor([0.5, 2.0]).log())
        return trainer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        result = walk_forward_evaluate(
            panel, make, n_folds=2, test_dates_per_fold=10, purge_dates=5,
            steps=0, base_cache=BaseCache(), seed=0,
        )
    for fold in result.folds:
        assert dict(fold.correction)["correction_max_abs"] == 0.0
        assert fold.nll_improvement == 0.0, fold


def test_base_nll_is_the_mixture_density_at_the_base():
    """With nonzero corrections the base NLL must be the mixture's own density
    evaluated at f0: -mean_i logsumexp_k(log pi_k + log N(y_i; f0_i, s_k^2)),
    checked here against that formula written out by hand."""
    from nec_moe import base_and_correction

    panel = _panel(80, 6)
    trainer = _fitted(_residual_cfg(zero_init_head=False, sigma_init=0.8), panel)
    with torch.no_grad():
        trainer.model.experts.log_sigma.copy_(torch.tensor([0.6, 1.3]).log())
    _, base_nll, corr = base_and_correction(trainer, panel)
    assert dict(corr)["correction_max_abs"] > 0.0  # the corrections are live

    trainer.model.eval()
    with torch.no_grad():
        out = trainer.model(panel.x_seq, panel.x_snap)
        f0 = trainer.model.base(panel.x_snap)
        log_lik = expert_log_likelihood(
            f0.unsqueeze(-1).expand_as(out.mu), out.log_sigma, panel.y
        )
        by_hand = -torch.logsumexp(out.prior.log_prior + log_lik, dim=-1).mean()
    assert base_nll == pytest.approx(float(by_hand), abs=1e-6)


def test_corrections_disabled_is_a_scoped_switch():
    """The switch zeroes the corrections inside the block only, restores itself
    even when the block raises, and is refused where there is no base."""
    panel = _panel(40, 4)
    trainer = _fitted(_residual_cfg(zero_init_head=False), panel)
    model = trainer.model.eval()
    with torch.no_grad():
        with model.corrections_disabled():
            off = model(panel.x_seq, panel.x_snap)
        on = model(panel.x_seq, panel.x_snap)
    assert float(off.corrections.abs().max()) == 0.0
    assert torch.equal(off.y_hat, off.base_pred)
    assert float(on.corrections.abs().max()) > 0.0  # restored after the block

    with pytest.raises(RuntimeError, match="boom"), model.corrections_disabled():
        raise RuntimeError("boom")
    assert model._corrections_off is False

    with pytest.raises(ValueError, match="correction_mode"):
        with NECModel(small_config()).corrections_disabled():
            pass


# --------------------------------------------------------------------------- #
# Brief 06 C: the old NLL improvement split into variance gain + improvement
# --------------------------------------------------------------------------- #


def _decomposed_folds(prior_kind: str, log_sigma, *, steps: int, zero_init_head: bool = True):
    panel = _panel(160, 8)
    cfg = _residual_cfg(prior_kind=prior_kind, sigma_init=0.7, zero_init_head=zero_init_head)

    def make() -> Trainer:
        torch.manual_seed(0)
        trainer = Trainer(NECModel(cfg))
        with torch.no_grad():
            trainer.model.experts.log_sigma.copy_(torch.tensor(log_sigma).log())
        return trainer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        return walk_forward_evaluate(
            panel, make, n_folds=2, test_dates_per_fold=10, purge_dates=5,
            steps=steps, base_cache=BaseCache(), seed=0,
        ).folds


@pytest.mark.parametrize("prior_kind", ["uniform", "hmm"])
def test_total_gain_is_variance_gain_plus_improvement(prior_kind: str):
    """``NLL_single - NLL_full == (NLL_single - NLL_base) + (NLL_base -
    NLL_full)`` on trained folds with live corrections. No sign is asserted
    for the variance gain: it has none in general."""
    folds = _decomposed_folds(prior_kind, [0.5, 2.0], steps=20, zero_init_head=False)
    for f in folds:
        assert dict(f.correction)["correction_max_abs"] > 0.0
        assert f.base_single_nll is not None and f.nll_total_gain is not None
        assert f.nll_total_gain == pytest.approx(
            f.nll_variance_gain + f.nll_improvement, abs=1e-9
        )
        assert f.nll_total_gain == pytest.approx(f.base_single_nll - f.nll, abs=1e-12)


@pytest.mark.parametrize("prior_kind", ["uniform", "soft", "hmm"])
def test_variance_gain_is_zero_when_every_sigma_is_equal(prior_kind: str):
    """With one noise scale for every expert the mixture density at the base
    is that single Gaussian whatever the gate says, so the variance gain is
    zero; it comes from the experts' noise scales, not the corrections."""
    for f in _decomposed_folds(prior_kind, [0.9, 0.9], steps=10, zero_init_head=False):
        assert f.nll_variance_gain == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("prior_kind", ["uniform", "hmm"])
def test_total_gain_is_the_variance_gain_when_every_correction_is_zero(prior_kind: str):
    """Zero corrections: NLL_full == NLL_base, so everything the old number
    reported was the variance gain."""
    for f in _decomposed_folds(prior_kind, [0.5, 2.0], steps=0):
        assert dict(f.correction)["correction_max_abs"] == 0.0
        assert f.nll_improvement == 0.0
        assert f.nll_total_gain == pytest.approx(f.nll_variance_gain, abs=1e-12)


def test_nll_single_is_the_pre_fix_base_nll():
    """``base_single_gaussian_nll`` is the base NLL as defined at commit
    9a29368: one Gaussian at the prior-weighted sigma, written out here."""
    panel = _panel(80, 6)
    trainer = _fitted(_residual_cfg(zero_init_head=False, sigma_init=0.8), panel)
    with torch.no_grad():
        trainer.model.experts.log_sigma.copy_(torch.tensor([0.6, 1.3]).log())
    trainer.model.eval()
    with torch.no_grad():
        out = trainer.model(panel.x_seq, panel.x_snap, PriorContext(date=panel.date))
        pi = out.prior.log_prior.exp().double()
        s = (pi * torch.tensor([0.6, 1.3], dtype=torch.float64)).sum(dim=-1)
        r = panel.y.double() - trainer.model.base(panel.x_snap).double()
        by_hand = float((0.5 * math.log(2 * math.pi) + s.log() + 0.5 * (r / s) ** 2).mean())
    assert base_single_gaussian_nll(trainer, panel) == pytest.approx(by_hand, rel=1e-5)
    assert base_single_gaussian_nll(Trainer(NECModel(small_config())), panel) is None


def test_sweep_rows_carry_all_three_nll_quantities(tmp_path):
    reg = TrialRegistry(tmp_path / "t.jsonl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        run_sweep(_panel(120, 6), [nec_arm("res", _residual_cfg(zero_init_head=False))],
                  seeds=(0,), registry=reg, tag="t", steps=5, n_folds=2,
                  test_dates_per_fold=10, purge_dates=5, verbose=False)
    m = reg.trials("t")[0].metrics
    for key in ("base_nll", "base_single_nll", "nll_improvement", "nll_variance_gain",
                "nll_total_gain"):
        assert key in m, key
    assert m["nll_total_gain"] == pytest.approx(
        m["nll_variance_gain"] + m["nll_improvement"], abs=1e-9
    )


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
