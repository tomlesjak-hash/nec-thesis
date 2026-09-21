"""Memoryless routing variants (Variation 2's comparison grid).

Contract, semantics, and — the load-bearing part — gradient behavior of each
prior: the uniform ablation gets *no* gate gradient by construction, hard
routing gets the hard-EM ``pi - onehot`` self-training gradient (the trainable
replacement for the prototype's gradient-dead argmax), top-k flows through the
renormalized kept entries, and Gumbel is stochastic in train / deterministic in
eval with an annealed temperature.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from conftest import small_config

from nec_moe import (
    NECModel,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
)
from nec_moe.likelihood import expert_log_likelihood
from nec_moe.losses import mixture_nll
from nec_moe.priors import (
    PRIOR_REGISTRY,
    GumbelSoftmaxRegimePrior,
    HardRegimePrior,
    TopKRegimePrior,
    UniformRegimePrior,
)

MEMORYLESS_KINDS = ("soft", "uniform", "hard", "topk", "gumbel")


def test_registry_contains_all_kinds():
    # "hmm" is the backpropagated latent Markov mixture (a baseline arm);
    # "markov" is the separately-fitted, frozen Hamilton gate (brief 03).
    assert set(MEMORYLESS_KINDS) | {"hmm", "markov"} == set(PRIOR_REGISTRY)
    for kind in MEMORYLESS_KINDS:
        cfg = small_config(n_experts=3, prior_kind=kind)
        # config round-trip must also survive validation + serialization
        model = NECModel(type(cfg).from_dict(cfg.to_dict()))
        assert model.prior.n_experts == 3


# --------------------------------------------------------------------------- #
# Uniform: the frozen-uniform-gate ablation
# --------------------------------------------------------------------------- #


def test_uniform_prior_is_constant_and_gradient_free():
    k = 3
    prior = UniformRegimePrior(k)
    z = torch.randn(8, k, requires_grad=True)
    out = prior(z)
    assert torch.allclose(out.log_prior, torch.full((8, k), -math.log(k)))
    # constant in the logits: no gradient path to the gate, by construction
    # (log_lik carries grad so backward runs; none of it reaches z)
    log_lik = torch.randn(8, k, requires_grad=True)
    mixture_nll(out.log_prior, log_lik).nll.backward()
    assert z.grad is None or torch.all(z.grad == 0)
    assert log_lik.grad is not None  # the experts, by contrast, still train


# --------------------------------------------------------------------------- #
# Hard: top-1 with the hard-EM objective
# --------------------------------------------------------------------------- #


def test_hard_prior_prediction_weights_are_onehot():
    k = 4
    prior = HardRegimePrior(k)
    z = torch.randn(16, k)
    out = prior(z)
    sel = z.argmax(dim=-1)
    finite = torch.isfinite(out.log_prior)
    assert (finite.sum(dim=-1) == 1).all()
    assert (out.log_prior[torch.arange(16), sel] == 0).all()
    # train weights: log_softmax at the argmax, -inf elsewhere (sub-normalized)
    lt = out.log_train_weights
    assert lt is not None and (torch.isfinite(lt).sum(dim=-1) == 1).all()
    expected = F.log_softmax(z, dim=-1)[torch.arange(16), sel]
    assert torch.allclose(lt[torch.arange(16), sel], expected)
    assert (torch.logsumexp(lt, dim=-1) <= 1e-6).all()  # <= 0: sub-normalized


def test_hard_prior_predicts_selected_expert_and_trains_gate():
    """y_hat = mu_selected, and the gate gradient is (pi - onehot)/B —
    hard-EM self-training, in contrast to the prototype's exactly-zero."""
    model = NECModel(small_config(prior_kind="hard"))
    x_seq = torch.randn(32, 8, 2)
    x_snap = torch.randn(32, 3)
    y = torch.randn(32)
    # nudge the gate off zero-init so argmax is non-degenerate
    with torch.no_grad():
        model.gate.linear.weight.normal_(0, 0.5)
    out = model(x_seq, x_snap)
    sel = out.gate_logits.argmax(dim=-1)
    assert torch.allclose(out.y_hat, out.mu[torch.arange(32), sel])

    # analytic gate-logit gradient of the hard-EM objective
    z = out.gate_logits.detach().requires_grad_(True)
    keep = torch.zeros_like(z, dtype=torch.bool).scatter_(1, sel.unsqueeze(1), True)
    log_train = F.log_softmax(z, dim=-1).masked_fill(~keep, float("-inf"))
    log_lik = expert_log_likelihood(out.mu.detach(), out.log_sigma.detach(), y)
    mixture_nll(log_train, log_lik).nll.backward()
    pi = F.softmax(z.detach(), dim=-1)
    onehot = keep.float()
    assert torch.allclose(z.grad, (pi - onehot) / 32, atol=1e-6)
    assert z.grad.abs().max() > 1e-4  # nonzero: the gate actually trains


def test_hard_prior_eval_train_identical():
    prior = HardRegimePrior(3)
    z = torch.randn(8, 3)
    prior.train()
    a = prior(z).log_prior
    prior.eval()
    b = prior(z).log_prior
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# Top-k
# --------------------------------------------------------------------------- #


def test_topk_prior_sparsity_and_normalization():
    k, keep = 4, 2
    prior = TopKRegimePrior(k, keep)
    z = torch.randn(16, k, requires_grad=True)
    out = prior(z)
    finite = torch.isfinite(out.log_prior)
    assert (finite.sum(dim=-1) == keep).all()
    # normalized over the kept entries
    assert torch.allclose(
        torch.logsumexp(out.log_prior, dim=-1), torch.zeros(16), atol=1e-6
    )
    # the masked entries are exactly the smallest logits
    smallest = z.detach().topk(k - keep, dim=-1, largest=False).indices
    assert not finite.gather(1, smallest).any()
    # gradient flows to the gate through the kept entries
    log_lik = torch.randn(16, k)
    mixture_nll(out.log_prior, log_lik).nll.backward()
    assert z.grad is not None and z.grad.abs().max() > 0


def test_topk_equals_soft_when_k_is_n_experts():
    z = torch.randn(8, 3)
    full = TopKRegimePrior(3, 3)(z).log_prior
    assert torch.allclose(full, F.log_softmax(z, dim=-1), atol=1e-7)


def test_topk_bounds_validated():
    with pytest.raises(ValueError, match="top_k"):
        TopKRegimePrior(3, 4)
    # k = 1 is rejected loudly: a single renormalized entry is constant 1, so
    # the gate would be gradient-dead — ledger defect 1 reborn. The error
    # points at HardRegimePrior, whose hard-EM objective IS trainable top-1.
    with pytest.raises(ValueError, match="gradient-dead"):
        TopKRegimePrior(3, 1)
    cfg = small_config(n_experts=3, prior_kind="topk")
    bad = type(cfg).from_dict(
        {**cfg.to_dict(), "prior": {"kind": "topk", "top_k": 1}}
    )
    with pytest.raises(ValueError, match="gradient-dead"):
        bad.validate()


# --------------------------------------------------------------------------- #
# Gumbel-softmax
# --------------------------------------------------------------------------- #


def test_gumbel_stochastic_in_train_deterministic_in_eval():
    prior = GumbelSoftmaxRegimePrior(3, tau_init=1.0)
    z = torch.randn(8, 3)
    prior.train()
    torch.manual_seed(0)
    a = prior(z).log_prior
    b = prior(z).log_prior
    assert not torch.allclose(a, b)  # noise resampled per forward
    assert torch.allclose(torch.logsumexp(a, dim=-1), torch.zeros(8), atol=1e-6)
    prior.eval()
    c = prior(z).log_prior
    d = prior(z).log_prior
    assert torch.equal(c, d)
    assert torch.allclose(c, F.log_softmax(z, dim=-1))


def test_gumbel_temperature_anneal():
    prior = GumbelSoftmaxRegimePrior(2, tau_init=1.0, tau_min=0.1, tau_anneal_steps=100)
    taus = [prior.temperature(s) for s in (0, 50, 100, 500, None)]
    assert taus[0] == pytest.approx(1.0)
    assert taus[1] == pytest.approx((0.1) ** 0.5, rel=1e-6)  # geometric midpoint
    assert taus[2] == pytest.approx(0.1)
    assert taus[3] == pytest.approx(0.1)  # clamped past the schedule
    assert taus[4] == pytest.approx(1.0)  # no step info -> tau_init


def test_gumbel_max_trick_sampling_frequencies():
    """As tau -> 0, the perturbed argmax samples the categorical softmax(z)
    (the Gumbel-max trick, Foundations E)."""
    torch.manual_seed(0)
    prior = GumbelSoftmaxRegimePrior(3, tau_init=0.01)
    prior.train()
    logits = torch.tensor([1.0, 0.0, -1.0])
    z = logits.expand(4000, 3)
    sampled = prior(z).log_prior.argmax(dim=-1)
    freq = torch.bincount(sampled, minlength=3).float() / 4000
    assert torch.allclose(freq, F.softmax(logits, dim=0), atol=0.03)


# --------------------------------------------------------------------------- #
# End-to-end: every variant trains through the Trainer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", MEMORYLESS_KINDS)
def test_variant_trains_end_to_end(kind: str):
    torch.manual_seed(0)
    spec = SyntheticSpec(seed=0)
    panel = SyntheticRegimePanel(spec).generate(n_dates=40, n_entities=6)
    model = NECModel(small_config(prior_kind=kind, lr=3e-3))
    with torch.no_grad():  # off zero-init so hard/topk selections vary
        model.gate.linear.weight.normal_(0, 0.1)
    trainer = Trainer(model)
    metrics = trainer.fit(panel, steps=5)
    assert all(math.isfinite(m["loss"]) for m in metrics)
    assert math.isfinite(trainer.evaluate(panel.full_batch()))
    gate_grad = model.gate.linear.weight.grad
    if kind == "uniform":
        assert gate_grad is None or torch.all(gate_grad == 0)
    else:
        assert gate_grad is not None and gate_grad.abs().max() > 0
