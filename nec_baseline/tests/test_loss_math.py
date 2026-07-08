"""Loss/math verification tests (design doc §12.2).

The analytic gradient identities of §4.2, the log-domain numerics, and the
aux-loss properties (including the arXiv:2501.11873 batch-scope pitfall as an
executable test).
"""

from __future__ import annotations

import math

import pytest
import torch

from conftest import small_config
from nec_moe import (
    LoadBalanceBuffer,
    NECConfig,
    build_prior,
    expert_decorrelation_aux,
    expert_log_likelihood,
    load_balance_aux,
    mixture_nll,
)
from nec_moe.priors import HMMRegimePrior, SoftRegimePrior


def _benign(b=6, k=3, *, dtype=torch.float64, seed=1):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(b, k, generator=g, dtype=dtype)
    mu = torch.randn(b, k, generator=g, dtype=dtype)
    log_sigma = 0.3 * torch.randn(k, generator=g, dtype=dtype)
    y = torch.randn(b, generator=g, dtype=dtype)
    return z, mu, log_sigma, y


def test_nll_matches_hand_reference():
    z, mu, log_sigma, y = _benign()
    log_lik = expert_log_likelihood(mu, log_sigma, y)
    fused = mixture_nll(torch.log_softmax(z, -1), log_lik)

    # naive-but-safe float64 reference: explicit pdfs, explicit softmax
    pi = torch.softmax(z, -1)
    sigma = log_sigma.exp()
    pdf = torch.exp(-0.5 * ((y[:, None] - mu) / sigma) ** 2) / (
        sigma * math.sqrt(2 * math.pi)
    )
    ref = -(pi * pdf).sum(-1).log()
    assert torch.allclose(fused.per_sample_nll, ref, atol=1e-10)
    resp_ref = pi * pdf / (pi * pdf).sum(-1, keepdim=True)
    assert torch.allclose(fused.responsibilities, resp_ref, atol=1e-10)


def test_gate_gradient_is_pi_minus_r():
    """The defect-1/2 fix in one identity: dL/dz = (pi - r)/B (eq. 4.10.4)."""
    z, mu, log_sigma, y = _benign()
    z = z.clone().requires_grad_(True)
    log_lik = expert_log_likelihood(mu, log_sigma, y)
    mixture_nll(torch.log_softmax(z, -1), log_lik).nll.backward()

    pi = torch.softmax(z.detach(), -1)
    r = torch.softmax(torch.log_softmax(z.detach(), -1) + log_lik, -1)
    assert torch.allclose(z.grad, (pi - r) / z.shape[0], atol=1e-10)


def test_expert_gradient_responsibility_weighted():
    """dL/dmu_k = r_k (mu_k - y) / sigma_k^2 / B; dead component -> dead grad."""
    z, mu, log_sigma, y = _benign()
    log_prior = torch.log_softmax(z, -1)
    # kill component 0's prior: its responsibility (hence gradient) ~ 0
    log_prior = log_prior.clone()
    log_prior[:, 0] = -40.0
    log_prior = log_prior - torch.logsumexp(log_prior, -1, keepdim=True)

    mu = mu.clone().requires_grad_(True)
    out = mixture_nll(log_prior, expert_log_likelihood(mu, log_sigma, y))
    out.nll.backward()

    r = out.responsibilities
    var = (2 * log_sigma).exp()
    expected = r * (mu.detach() - y[:, None]) / var / mu.shape[0]
    assert torch.allclose(mu.grad, expected, atol=1e-10)
    assert mu.grad[:, 0].abs().max() < 1e-12  # the collapse twin, documented


def test_log_domain_stability():
    """Targets ~40 sigma out: fused loss finite; the naive pipeline underflows
    to -inf even in float64 (Module 8 Ex. 8.1's cliff)."""
    k = 2
    mu = torch.zeros(1, k, dtype=torch.float64)
    log_sigma = torch.zeros(k, dtype=torch.float64)
    y = torch.tensor([40.0], dtype=torch.float64)
    log_prior = torch.full((1, k), math.log(0.5), dtype=torch.float64)

    fused = mixture_nll(log_prior, expert_log_likelihood(mu, log_sigma, y))
    assert torch.isfinite(fused.nll)
    assert float(fused.nll) == pytest.approx(
        800 + 0.5 * math.log(2 * math.pi) + math.log(2) - math.log(2), rel=1e-6
    )  # = 40^2/2 + log Z; mixture of two identical components

    pdf = torch.exp(-0.5 * (y[:, None] - mu) ** 2) / math.sqrt(2 * math.pi)
    naive = -(log_prior.exp() * pdf).sum(-1).log()
    assert torch.isinf(naive).all(), "naive pipeline should underflow here"


def test_logit_gauge_invariance():
    z, mu, log_sigma, y = _benign()
    log_lik = expert_log_likelihood(mu, log_sigma, y)
    a = mixture_nll(torch.log_softmax(z, -1), log_lik)
    b = mixture_nll(torch.log_softmax(z + 7.3, -1), log_lik)
    assert torch.allclose(a.per_sample_nll, b.per_sample_nll, atol=1e-9)
    # and the point prediction is gauge-invariant too
    y_hat_a = (torch.softmax(z, -1) * mu).sum(-1)
    y_hat_b = (torch.softmax(z + 7.3, -1) * mu).sum(-1)
    assert torch.allclose(y_hat_a, y_hat_b, atol=1e-9)


def test_sigma_gradient_stationarity():
    """At sigma_k^2 = responsibility-weighted MSE, d(nll)/d(log sigma_k) = 0."""
    z, mu, _, y = _benign(b=64)
    log_prior = torch.log_softmax(z, -1)
    log_sigma = torch.zeros(3, dtype=torch.float64)
    for _ in range(60):  # fixed-point iteration to the weighted-MSE stationary point
        r = mixture_nll(
            log_prior, expert_log_likelihood(mu, log_sigma, y)
        ).responsibilities
        mse = (r * (y[:, None] - mu) ** 2).sum(0) / r.sum(0)
        log_sigma = 0.5 * mse.log()
    ls = log_sigma.clone().requires_grad_(True)
    mixture_nll(log_prior, expert_log_likelihood(mu, ls, y)).nll.backward()
    assert ls.grad.abs().max() < 1e-8


def test_prior_contract():
    cfg = small_config()
    prior = build_prior(cfg)
    assert isinstance(prior, SoftRegimePrior)
    out = prior(torch.randn(5, 2))
    assert torch.allclose(
        torch.logsumexp(out.log_prior, -1), torch.zeros(5), atol=1e-6
    )

    hmm_cfg = small_config(prior_kind="hmm")
    assert isinstance(build_prior(hmm_cfg), HMMRegimePrior)
    # config serialization round-trips (Module 5 registry requirement)
    assert NECConfig.from_dict(hmm_cfg.to_dict()) == hmm_cfg

    # -inf prior entries flow through the NLL finitely (sparse-prior future)
    log_prior = torch.tensor([[0.0, float("-inf")]], dtype=torch.float64)
    log_lik = torch.randn(1, 2, dtype=torch.float64)
    out = mixture_nll(log_prior, log_lik)
    assert torch.isfinite(out.nll)
    assert torch.allclose(out.responsibilities, torch.tensor([[1.0, 0.0]]).double())


def test_load_balance_scope():
    """The arXiv:2501.11873 pitfall, executable. A regime-homogeneous batch with
    sharp (correct) routing is unpenalized under buffered scope — and penalized
    by a per-batch-scope double."""
    k = 2
    buffer = LoadBalanceBuffer(n_experts=k, buffer_batches=8)
    # running buffer balanced: alternating one-regime batches
    for j in range(8):
        r = torch.zeros(32, k)
        r[:, j % k] = 1.0
        buffer.update(r)
    assert torch.allclose(buffer.fractions(), torch.full((k,), 0.5))

    # today's batch: one regime, routed toward expert 0 (moderate logits so the
    # softmax is not saturated — keeps the gradient comparison meaningful)
    bias = torch.tensor([2.0, 0.0])
    z = torch.zeros(32, k, requires_grad=True)
    probs = torch.softmax(z + bias, -1)

    buffered = load_balance_aux(probs, buffer.fractions())
    # uniform fractions => K * sum_k (1/K) p_k = sum_k p_k = 1 identically, so
    # the loss is constant in the routing and its gradient is exactly zero —
    # balanced buffered scope never fights sharp within-batch routing.
    assert float(buffered.detach()) == pytest.approx(1.0, abs=1e-6)
    buffered.backward()
    assert z.grad.abs().max() < 1e-7

    # per-batch scope double: fractions from THIS batch alone -> the same sharp
    # routing is penalized and actively pushed toward uniform.
    z2 = torch.zeros(32, k, requires_grad=True)
    probs2 = torch.softmax(z2 + bias, -1)
    batch_frac = torch.tensor([1.0, 0.0])  # argmax fractions of this batch
    per_batch = load_balance_aux(probs2, batch_frac)
    assert float(per_batch.detach()) > 1.4  # 2 * p0_bar > 1: sharp routing punished
    per_batch.backward()
    assert z2.grad.abs().max() > 1e-3  # and actively pushed toward uniform


def test_expert_decorrelation():
    k = 3
    g = torch.Generator().manual_seed(3)
    # duplicated experts: maximal penalty K^2 - K
    col = torch.randn(32, 1, generator=g)
    dup = col.expand(32, k).clone().requires_grad_(True)
    loss = expert_decorrelation_aux(dup)
    assert float(loss.detach()) == pytest.approx(k * k - k, rel=1e-4)
    loss.backward()
    assert dup.grad.abs().max() > 0  # pushes duplicated experts apart

    # decorrelated experts: ~0 (orthogonal centered columns)
    q, _ = torch.linalg.qr(torch.randn(32, k, generator=g, dtype=torch.float64))
    q = q - q.mean(0, keepdim=True)
    assert float(expert_decorrelation_aux(q)) < 0.05
