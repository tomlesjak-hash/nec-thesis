"""How the experts start (brief 06 D, audit X-1): ``ExpertConfig.hidden_init``.

``"diversified"`` (the default, today's behaviour) gives each expert its own
seeded draw; ``"identical"`` gives every expert the same hidden layers and the
same initial ``log_sigma``, so only the gate can separate them. The default
is not a decision; these tests pin what each setting does.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from conftest import small_base_config, small_config

from nec_moe import (
    MLPBlock,
    NECModel,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    fit_base,
    nec_arm,
)
from nec_moe.train import DeadParameterWarning


def _panel(n_dates: int = 60, n_entities: int = 8):
    spec = SyntheticSpec(vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=3)
    return SyntheticRegimePanel(spec).generate(n_dates, n_entities)


def _expert_params(model: NECModel) -> list[dict[str, torch.Tensor]]:
    return [
        {k: v.detach().clone() for k, v in e.state_dict().items()}
        for e in model.experts.experts
    ]


def _all_equal(params: list[dict[str, torch.Tensor]]) -> bool:
    return all(
        torch.equal(params[0][k], p[k]) for p in params[1:] for k in params[0]
    )


def _residual(**kw):
    return small_config(correction_mode=True, base=small_base_config(), **kw)


def _trained(cfg, panel, steps: int) -> Trainer:
    torch.manual_seed(0)
    trainer = Trainer(NECModel(cfg))
    if cfg.experts.correction_mode:
        trainer.model.attach_base(fit_base(panel, cfg.base, seed=0).model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        trainer.fit(panel, steps=steps)
    return trainer


@pytest.mark.parametrize("correction_mode", [True, False])
def test_identical_start_gives_every_expert_the_same_parameters(correction_mode: bool):
    """Test 1: at step 0 all experts are equal, log_sigma included; in
    correction mode the heads are zero."""
    cfg = _residual(hidden_init="identical") if correction_mode else small_config(
        hidden_init="identical", n_experts=3
    )
    model = NECModel(cfg)
    params = _expert_params(model)
    assert _all_equal(params)
    sigma = model.experts.log_sigma.detach()
    assert torch.equal(sigma, torch.full_like(sigma, float(sigma[0])))
    if correction_mode:
        for e in model.experts.experts:
            assert all(float(p.detach().abs().max()) == 0.0 for p in e.head.parameters())


def test_diversified_reproduces_the_previous_initialisation_bit_for_bit():
    """Test 2 (regression): the default draws expert k from seed
    ``train.seed + 7919 (k + 1)``, exactly the code before the switch."""
    for cfg in (_residual(seed=5), small_config(seed=5, n_experts=3)):
        assert cfg.experts.hidden_init == "diversified"
        model = NECModel(cfg)
        zero_head = cfg.experts.correction_mode and cfg.experts.zero_init_head
        for k, expert in enumerate(model.experts.experts):
            old = MLPBlock(
                cfg.expert_input_dim, cfg.experts.hidden_dims,
                dropout=cfg.experts.dropout, activation=cfg.experts.activation,
                zero_init_head=zero_head,
            )
            old.reset_parameters_seeded(5 + 7919 * (k + 1), zero_head=zero_head)
            for name, value in old.state_dict().items():
                assert torch.equal(expert.state_dict()[name], value), (k, name)
        assert not _all_equal(_expert_params(model))  # diversified means different


def test_identical_experts_stay_equal_under_a_uniform_gate_without_dropout():
    """Test 3: equal experts under equal weights receive equal gradients, so
    the symmetry survives training (dropout off)."""
    panel = _panel()
    cfg = _residual(hidden_init="identical", prior_kind="uniform", expert_dropout=0.0,
                    batch_size=64)
    trainer = _trained(cfg, panel, steps=25)
    params = _expert_params(trainer.model)
    assert any(float(p.abs().max()) > 0 for p in params[0].values())  # it trained
    assert _all_equal(params)
    sigma = trainer.model.experts.log_sigma.detach()
    assert torch.equal(sigma, torch.full_like(sigma, float(sigma[0])))


def test_identical_experts_separate_under_a_date_varying_fixed_prior():
    """Test 4: a frozen prior that weights the experts differently on
    different dates gives them different gradients; they separate."""
    panel = _panel()
    cfg = _residual(hidden_init="identical", prior_kind="markov", expert_dropout=0.0,
                    batch_size=64, freeze_gate=True, sequence_ordered=False)
    torch.manual_seed(0)
    trainer = Trainer(NECModel(cfg))
    trainer.model.attach_base(fit_base(panel, cfg.base, seed=0).model)
    dates = torch.unique(panel.date)
    first = torch.where(dates % 2 == 0, 0.9, 0.1)
    table = torch.stack([first, 1 - first], dim=1).log()
    trainer.model.prior.set_fitted_table(dates, table)
    assert _all_equal(_expert_params(trainer.model))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        trainer.fit(panel, steps=25)
    assert not _all_equal(_expert_params(trainer.model))


@pytest.mark.parametrize("hidden_init", ["diversified", "identical"])
def test_expert_start_is_bit_identical_across_gate_arms(hidden_init: str):
    """Test 5: for one seed, every gate arm starts its experts from the same
    weights, so arms differ only in the gate. The draws use a private
    generator, so the gate's own initialisation cannot shift them."""
    starts = []
    for prior_kind in ("uniform", "soft", "hmm", "markov"):
        cfg = _residual(hidden_init=hidden_init, prior_kind=prior_kind,
                        sequence_ordered=prior_kind == "hmm")
        trainer = nec_arm(prior_kind, cfg).build_trainer(7)  # type: ignore[misc]
        starts.append((_expert_params(trainer.model), trainer.model.experts.log_sigma.detach()))
    params0, sigma0 = starts[0]
    for params, sigma in starts[1:]:
        assert torch.equal(sigma, sigma0)
        for p, q in zip(params, params0, strict=True):
            assert all(torch.equal(p[k], q[k]) for k in q)


def test_dropout_masks_are_drawn_independently_per_expert():
    """The brief-06 D report item, pinned: each expert's dropout layer draws
    its own mask, so identical experts diverge under a uniform gate once
    dropout is on. The clean "no regime information" control needs dropout 0."""
    panel = _panel()
    cfg = _residual(hidden_init="identical", prior_kind="uniform", expert_dropout=0.3,
                    batch_size=64)
    trainer = _trained(cfg, panel, steps=25)
    assert not _all_equal(_expert_params(trainer.model))


def test_unknown_hidden_init_is_refused():
    with pytest.raises(ValueError, match="hidden_init"):
        small_config(hidden_init="random").validate()
