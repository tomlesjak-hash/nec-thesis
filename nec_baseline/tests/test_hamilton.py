"""The emission axis and the classical Hamilton baseline (design doc §7).

The design's two orthogonal plug axes — emission (neural | classical) x prior
(memoryless | Markov) — must all be buildable from config, and the classical x
Markov corner must actually *be* a Hamilton / Gaussian-HMM model: the
parameter-recovery test fits it by backprop through the forward filter on data
generated from a known Gaussian HMM and checks the recovered
``(mu_k, sigma_k, A)`` against ground truth.
"""

from __future__ import annotations

import math

import pytest
import torch

from conftest import D_SEQ, D_SNAP, SEQ_LEN, small_config
from nec_moe import (
    Batch,
    ClassicalGaussianEmission,
    ExpertBank,
    NECModel,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
)
from nec_moe.diagnostics import canonical_expert_order, regime_recovery_auc


def _hamilton_sequence(
    n_dates: int,
    n_entities: int,
    mu: tuple[float, ...],
    sigma: tuple[float, ...],
    stay: float,
    seed: int,
):
    """Pure Gaussian-HMM data: y_t ~ N(mu_z, sigma_z^2), z a sticky date-level
    chain; features are pure noise (the classical emission ignores them and the
    static HMM prior ignores the gate — everything must come from y itself)."""
    g = torch.Generator().manual_seed(seed)
    k = len(mu)
    mu_t, sg_t = torch.tensor(mu), torch.tensor(sigma)
    trans = torch.full((k, k), (1.0 - stay) / (k - 1))
    trans.fill_diagonal_(stay)
    z = torch.empty(n_dates, dtype=torch.long)
    z[0] = torch.randint(k, (1,), generator=g)
    for t in range(1, n_dates):
        z[t] = torch.multinomial(trans[z[t - 1]], 1, generator=g)
    seq = []
    for t in range(n_dates):
        y = mu_t[z[t]] + sg_t[z[t]] * torch.randn(n_entities, generator=g)
        seq.append(
            Batch(
                torch.randn(n_entities, SEQ_LEN, D_SEQ, generator=g),
                torch.randn(n_entities, D_SNAP, generator=g),
                y,
            )
        )
    big = Batch(
        torch.cat([b.x_seq for b in seq]),
        torch.cat([b.x_snap for b in seq]),
        torch.cat([b.y for b in seq]),
    )
    return seq, big, z


def test_hamilton_parameter_recovery():
    """classical emission x HMM prior = a Gaussian HMM, and MLE-by-autograd
    through the log-domain forward filter recovers its true parameters.

    The strongest available validation of Decision C's 'differentiable forward
    algorithm' choice: the exact-inference classical model is estimated
    correctly by the same training loop the neural variant uses.
    """
    mu_true, sg_true, stay = (-1.0, 1.0), (0.4, 0.9), 0.9
    seq, big, z = _hamilton_sequence(500, 4, mu_true, sg_true, stay, seed=5)

    cfg = small_config(prior_kind="hmm", expert_kind="classical", lr=3e-2)
    model = NECModel(cfg)
    trainer = Trainer(model)
    # quantile moment init on y itself — the classical analogue of the expert
    # warm-start (k-means-style init for a GMM; training data only, no leakage)
    trainer.warmstart_experts(big, sort_key=big.y)
    trainer.fit_sequence(seq, steps=250, chunk_len=50)

    # canonical (sigma-sorted) order aligns learned components to truth,
    # which is itself sigma-sorted here
    order = canonical_expert_order(model.experts.log_sigma)
    mu_hat = model.experts.mu.detach()[order]
    sg_hat = model.experts.log_sigma.detach().exp()[order]
    a_hat = model.prior.transition_matrix.detach()[order][:, order]

    assert torch.allclose(mu_hat, torch.tensor(mu_true), atol=0.15), mu_hat
    assert torch.allclose(sg_hat, torch.tensor(sg_true), atol=0.15), sg_hat
    assert torch.allclose(
        a_hat.diagonal(), torch.tensor([stay, stay]), atol=0.05
    ), a_hat

    # and the filtered posterior tracks the true regime path
    ev = trainer.evaluate_sequence(seq)
    auc = regime_recovery_auc(
        ev.log_filtered[:, :, order[0]].exp().flatten(),
        z.repeat_interleave(4),
    )
    assert auc > 0.9, f"regime path AUC {auc:.3f}"


@pytest.mark.parametrize("expert_kind", ["mlp", "classical"])
@pytest.mark.parametrize("prior_kind", ["soft", "hmm"])
def test_four_corner_grid_trains(expert_kind: str, prior_kind: str):
    """All four corners of (emission x prior) build from config and train:
    neural x memoryless (the baseline NEC), neural x Markov (HMM-NEC),
    classical x memoryless (feature-gated classical mixture), and
    classical x Markov (Hamilton)."""
    panel = SyntheticRegimePanel(SyntheticSpec(seed=0)).generate(
        n_dates=40, n_entities=6
    )
    cfg = small_config(prior_kind=prior_kind, expert_kind=expert_kind, lr=3e-3)
    model = NECModel(cfg)
    expected = ClassicalGaussianEmission if expert_kind == "classical" else ExpertBank
    assert isinstance(model.experts, expected)
    trainer = Trainer(model)
    if prior_kind == "hmm":
        metrics = trainer.fit_sequence(panel.time_sequence(), steps=5, chunk_len=20)
    else:
        metrics = trainer.fit(panel, steps=5)
    assert all(math.isfinite(m["loss"]) for m in metrics)


def test_classical_moment_warmstart_is_exact():
    """The classical emission's warm-start is closed-form slice moments."""
    emission = ClassicalGaussianEmission(2, sigma_init=1.0)
    y = torch.tensor([-2.0, -1.0, -1.5, 3.0, 4.0, 3.5])
    x = torch.zeros(6, 3)
    slices = [torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])]
    emission.warmstart_slices(x, y, slices)
    assert torch.allclose(emission.mu.detach(), torch.tensor([-1.5, 3.5]))
    assert torch.allclose(
        emission.log_sigma.detach().exp(),
        torch.stack([y[:3].std(unbiased=True), y[3:].std(unbiased=True)]),
    )


def test_emission_registry_and_validation():
    cfg = small_config(expert_kind="classical")
    rebuilt = type(cfg).from_dict(cfg.to_dict())
    assert isinstance(NECModel(rebuilt).experts, ClassicalGaussianEmission)
    bad = type(cfg).from_dict(
        {**cfg.to_dict(), "experts": {**cfg.to_dict()["experts"], "kind": "nope"}}
    )
    with pytest.raises(ValueError, match="experts.kind"):
        bad.validate()
