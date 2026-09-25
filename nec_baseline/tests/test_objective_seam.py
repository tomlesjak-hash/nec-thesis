"""The objective registry seam (brief 04 section A, specified in brief 02 3.0).

A seam and nothing more: Q20 (which error function to train against) is open,
so the registry exists to make answering it a registration rather than a
rewrite, and it holds exactly the objective the package already used. The
claims pinned here are therefore about *not changing anything*:

- the trainer's registry path and a direct ``mixture_nll`` call give
  bit-identical losses and gradients, on the memoryless and the stateful path;
- exactly one objective is registered;
- an unknown key is refused at validation, naming what is registered;
- every trial row records the objective beside the correction-penalty weight,
  because that weight is not comparable across objectives.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
from conftest import small_config

from nec_moe import (
    OBJECTIVE_REGISTRY,
    NECModel,
    PriorContext,
    RidgeBaseline,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    TrialRegistry,
    baseline_arm,
    expert_log_likelihood,
    mixture_nll,
    nec_arm,
    run_sweep,
)
from nec_moe.train import DeadParameterWarning


def _panel(**spec):
    kw = dict(vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=3)
    kw.update(spec)
    return SyntheticRegimePanel(SyntheticSpec(**kw)).generate(60, 6)


def _grads(model: NECModel) -> dict[str, torch.Tensor]:
    return {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def _twin(cfg) -> tuple[NECModel, NECModel]:
    """Two models with bit-identical parameters."""
    torch.manual_seed(0)
    a = NECModel(cfg)
    torch.manual_seed(0)
    b = NECModel(cfg)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert na == nb and torch.equal(pa, pb)
    return a, b


def test_registry_path_is_bit_identical_on_the_memoryless_path():
    """The trainer's routed loss vs. the objective called by hand, on the same
    batch and the same weights: equal to the last bit, and so are the
    gradients every parameter receives from it."""
    cfg = small_config(sigma_init=1.5)
    via_registry, direct = _twin(cfg)
    batch = _panel().full_batch()

    trainer = Trainer(via_registry)
    _, routed = trainer._forward_nll(batch, PriorContext(step=0))
    routed.nll.backward()

    out = direct(batch.x_seq, batch.x_snap, PriorContext(step=0, date=batch.date))
    log_lik = expert_log_likelihood(out.mu, out.log_sigma, batch.y)
    by_hand = mixture_nll(out.prior.log_prior, log_lik)
    by_hand.nll.backward()

    assert torch.equal(routed.nll, by_hand.nll)
    assert torch.equal(routed.log_filtered, by_hand.log_filtered)
    g_routed, g_hand = _grads(via_registry), _grads(direct)
    assert set(g_routed) == set(g_hand) and g_routed
    for name in g_routed:
        assert torch.equal(g_routed[name], g_hand[name]), name


def test_registry_path_is_bit_identical_on_the_stateful_path():
    """Same claim through the HMM recursion, where the objective's output is
    also the state threaded into the next date: two dates, the filtered
    posterior of the first feeding the prior of the second, gradients flowing
    back through both."""
    cfg = small_config(prior_kind="hmm", sigma_init=1.5)
    via_registry, direct = _twin(cfg)
    seq = _panel(regime_process="markov", transition_stay=0.95).time_sequence()[:2]

    trainer = Trainer(via_registry)
    state = None
    routed_losses = []
    for batch in seq:
        _, out_nll = trainer._forward_nll(batch, PriorContext(prev_filtered=state))
        routed_losses.append(out_nll.per_sample_nll)
        state = out_nll.log_filtered
    routed = torch.cat(routed_losses).mean()
    routed.backward()

    state = None
    hand_losses = []
    for batch in seq:
        out = direct(
            batch.x_seq, batch.x_snap,
            PriorContext(prev_filtered=state, date=batch.date),
        )
        log_lik = expert_log_likelihood(out.mu, out.log_sigma, batch.y)
        nll = mixture_nll(out.prior.log_prior, log_lik)
        hand_losses.append(nll.per_sample_nll)
        state = nll.log_filtered
    by_hand = torch.cat(hand_losses).mean()
    by_hand.backward()

    assert torch.equal(routed, by_hand)
    g_routed, g_hand = _grads(via_registry), _grads(direct)
    assert "prior.transition_logits" in g_routed  # gradient really crossed dates
    for name in g_routed:
        assert torch.equal(g_routed[name], g_hand[name]), name


def test_only_the_existing_objective_is_registered():
    """The scope fence as a test: Q20 is open, so nothing may be registered
    beyond what the package already trained against."""
    assert set(OBJECTIVE_REGISTRY) == {"mixture_nll"}
    assert OBJECTIVE_REGISTRY["mixture_nll"] is mixture_nll
    assert small_config().train.objective == "mixture_nll"


def test_unknown_objective_is_refused_with_the_registered_keys():
    import dataclasses

    cfg = small_config()
    bad = dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, objective="huber")
    )
    with pytest.raises(ValueError, match=r"unknown train\.objective 'huber'.*mixture_nll"):
        bad.validate()


def test_every_trial_row_records_objective_beside_penalty_weight(tmp_path: Path):
    """NEC rows carry both keys with their values; baseline rows carry both
    keys as not-applicable (a baseline is not trained against the NEC
    objective); a selection row inherits its winner's."""
    reg = TrialRegistry(tmp_path / "t.jsonl")
    cfg = small_config(sigma_init=1.5, lr=3e-3, batch_size=256)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        run_sweep(
            _panel(), [
                nec_arm("nec", cfg),
                baseline_arm("ridge", lambda seed: RidgeBaseline(l2=1.0)),
            ],
            seeds=(0,), registry=reg, tag="obj", steps=5, n_folds=2,
            test_dates_per_fold=8, purge_dates=2, verbose=False,
        )
    rows = {r.config["arm"]: r.config for r in reg.trials("obj")}
    assert rows["nec"]["objective"] == "mixture_nll"
    assert rows["nec"]["correction_penalty_weight"] == 0.0
    assert "objective" in rows["ridge"] and rows["ridge"]["objective"] is None
    assert "correction_penalty_weight" in rows["ridge"]

    pick = reg.best("obj", "mean_ic")
    selection = reg.selection_events("obj")[-1].config
    assert selection["objective"] == pick.config["objective"]
