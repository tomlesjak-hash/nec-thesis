"""Brief 09 H follow-up: the dead-encoder bypass and the vectorised date lookup
change no result.

With a precomputed (frozen Hamilton) gate and snapshot-only experts the GRU
encoder's output reaches neither the loss nor the prediction, so the model
skips it (``NECModel.encoder_is_dead``); the precomputed prior looks dates up
by binary search instead of a dict per row. Both must be bit-identical to the
full path: same parameters after training, same predictions, same RNG stream,
same gradient audit.
"""

from __future__ import annotations

import dataclasses
import warnings

import pytest
import torch
from conftest import small_config

from nec_moe import (
    NECModel,
    PriorContext,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
)
from nec_moe.priors import PrecomputedRegimePrior

pytest.importorskip("statsmodels")


def _panel():
    return SyntheticRegimePanel(SyntheticSpec(
        regime_process="markov", transition_stay=0.96, vol_levels=(0.5, 2.5),
        beta_scale=2.0, noise_std=0.4, seed=3,
    )).generate(240, 8)


def _cfg(**experts):
    cfg = small_config(prior_kind="markov", sigma_init=1.5, batch_size=64,
                       freeze_gate=True, expert_dropout=0.1)
    gate = dataclasses.replace(cfg.markov_gate, series="sequence_channel",
                               series_channel=0, search_reps=3)
    return dataclasses.replace(cfg, markov_gate=gate,
                               experts=dataclasses.replace(cfg.experts, **experts))


def _train(panel, cfg, *, skip: bool):
    torch.manual_seed(0)
    trainer = Trainer(NECModel(cfg))
    trainer.model.skip_dead_encoder = skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.model.prior.fit(panel)
        trainer.fit(panel, steps=40)
    rng_after = torch.get_rng_state()
    trainer.model.eval()
    with torch.no_grad():
        pred = trainer.model(panel.x_seq, panel.x_snap, PriorContext(date=panel.date)).y_hat
    return trainer, pred, rng_after


def test_frozen_hamilton_model_is_bit_identical_with_the_bypass():
    panel = _panel()
    cfg = _cfg()
    fast, p_fast, rng_fast = _train(panel, cfg, skip=True)
    slow, p_slow, rng_slow = _train(panel, cfg, skip=False)
    assert fast.model.encoder_is_dead
    for (name, a), (_, b) in zip(fast.model.state_dict().items(),
                                 slow.model.state_dict().items(), strict=True):
        assert torch.equal(a, b), name
    assert torch.equal(p_fast, p_slow)
    assert torch.equal(rng_fast, rng_slow)  # dropout draws unchanged
    assert fast.history == slow.history
    a, b = fast.audit_gradients(), slow.audit_gradients()
    assert fast.live_param_count == slow.live_param_count
    assert (a.disconnected, a.frozen) == (b.disconnected, b.frozen)


def test_the_bypass_applies_only_where_the_encoder_is_dead():
    assert NECModel(_cfg()).encoder_is_dead
    assert not NECModel(_cfg(input_mode="snapshot_plus_hidden")).encoder_is_dead
    assert not NECModel(small_config(prior_kind="soft")).encoder_is_dead
    cfg = _cfg()
    two_layer = dataclasses.replace(cfg, encoder=dataclasses.replace(
        cfg.encoder, num_layers=2, dropout=0.2))
    assert not NECModel(two_layer).encoder_is_dead  # its dropout would draw random numbers


def test_vectorised_lookup_matches_the_dict_and_refuses_unknown_dates():
    prior = PrecomputedRegimePrior(2)
    g = torch.Generator().manual_seed(1)
    dates = torch.tensor([3, 7, 8, 12, 20])
    table = torch.log_softmax(torch.randn(5, 2, generator=g), dim=-1)
    prior.set_fitted_table(dates[:3], table[:3])
    prior.extend_causal_table(dates[3:], table[3:])
    query = torch.tensor([20, 3, 3, 12, 8, 7, 20])
    got = prior(torch.zeros(len(query), 2), PriorContext(date=query)).log_prior
    want = prior._log_table[torch.tensor([prior._rows[int(d)] for d in query])]
    assert torch.equal(got, want)
    assert prior.covers(torch.tensor([3, 4, 20, 21])).tolist() == [True, False, True, False]
    for bad in ([4], [0], [21], [3, 99]):
        with pytest.raises(ValueError, match="outside the fitted information set"):
            prior(torch.zeros(len(bad), 2), PriorContext(date=torch.tensor(bad)))
    # a reloaded gate rebuilds its index
    clone = PrecomputedRegimePrior(2)
    clone.load_gate_state(prior.gate_state())
    assert torch.equal(clone(torch.zeros(len(query), 2), PriorContext(date=query)).log_prior,
                       got)
