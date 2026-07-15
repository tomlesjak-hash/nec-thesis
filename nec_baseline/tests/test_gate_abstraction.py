"""Gate-abstraction & HMM-path tests (design doc §12.3).

These prove "config change, not rewrite": the shared likelihood core, the
stateful prior threading, and the filtering-not-smoothing invariant.
"""

from __future__ import annotations

import math

import pytest
import torch
from conftest import random_inputs, small_config

from nec_moe import (
    Batch,
    NECModel,
    PriorConfig,
    PriorContext,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    expert_log_likelihood,
    mixture_nll,
)
from nec_moe.diagnostics import regime_recovery_auc
from nec_moe.priors import HMMRegimePrior


# ------------------------------------------------------------- tiny reference
def _reference_forward(pi0, a, mu, sigma, ys):
    """Plain probability-domain HMM forward filter (float64, normalized)."""
    ll = 0.0
    alphas = []
    alpha = pi0.clone()
    for t, y in enumerate(ys):
        lik = torch.exp(-0.5 * ((y - mu) / sigma) ** 2) / (
            sigma * math.sqrt(2 * math.pi)
        )
        if t > 0:
            alpha = a.T @ alpha  # predict
        alpha = alpha * lik  # update
        c = alpha.sum()
        ll += float(c.log())
        alpha = alpha / c
        alphas.append(alpha.clone())
    return ll, torch.stack(alphas)


def _reference_smoothed(pi0, a, mu, sigma, ys):
    """Forward-backward smoothed posteriors gamma_t = p(z_t | y_{1:T})."""
    _, alphas = _reference_forward(pi0, a, mu, sigma, ys)
    n = len(ys)
    beta = torch.ones_like(pi0)
    gammas = [None] * n
    gammas[-1] = alphas[-1]
    for t in range(n - 2, -1, -1):
        lik = torch.exp(-0.5 * ((ys[t + 1] - mu) / sigma) ** 2) / (
            sigma * math.sqrt(2 * math.pi)
        )
        beta = a @ (lik * beta)
        beta = beta / beta.sum()  # rescale for stability
        g = alphas[t] * beta
        gammas[t] = g / g.sum()
    return torch.stack(gammas)


def _run_our_filter(prior, mu, log_sigma, ys):
    """Predict (prior) + update (mixture_nll) recursion over a scalar sequence."""
    state = None
    total = 0.0
    filtered = []
    zeros = torch.zeros(1, mu.shape[-1], dtype=torch.float64)
    total_t = None
    for y in ys:
        ctx = PriorContext(prev_filtered=state)
        p = prior(zeros, ctx)
        log_lik = expert_log_likelihood(mu, log_sigma, y.view(1))
        out = mixture_nll(p.log_prior, log_lik)
        total = total + out.per_sample_nll.sum()
        state = out.log_filtered
        filtered.append(out.log_filtered.exp().squeeze(0))
        total_t = total
    return total_t, torch.stack(filtered)


# --------------------------------------------------------------------- tests
def test_shared_likelihood_core():
    """Only the prior differs: with weights shared and both priors uniform at a
    sequence start, the two variations produce identical likelihoods, NLL and
    posteriors."""
    soft_model = NECModel(small_config()).eval()
    hmm_model = NECModel(small_config(prior_kind="hmm")).eval()
    missing, unexpected = hmm_model.load_state_dict(
        soft_model.state_dict(), strict=False
    )
    assert not unexpected
    assert all(k.startswith("prior.") for k in missing)  # prior params only

    x_seq, x_snap, y = random_inputs()
    out_s = soft_model(x_seq, x_snap)
    out_h = hmm_model(x_seq, x_snap)  # ctx None -> pi_0 (uniform)

    assert torch.allclose(out_s.mu, out_h.mu, atol=0)  # same emission, exactly
    assert torch.allclose(out_s.prior.log_prior, out_h.prior.log_prior, atol=1e-7)
    nll_s = mixture_nll(
        out_s.prior.log_prior, expert_log_likelihood(out_s.mu, out_s.log_sigma, y)
    )
    nll_h = mixture_nll(
        out_h.prior.log_prior, expert_log_likelihood(out_h.mu, out_h.log_sigma, y)
    )
    assert torch.allclose(nll_s.per_sample_nll, nll_h.per_sample_nll, atol=1e-7)
    assert torch.allclose(nll_s.responsibilities, nll_h.responsibilities, atol=1e-7)


def test_hmm_forward_filter_matches_reference():
    torch.manual_seed(4)
    a = torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float64)
    pi0 = torch.tensor([0.5, 0.5], dtype=torch.float64)
    mu_k = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    sigma_k = torch.tensor([0.8, 1.2], dtype=torch.float64)
    ys = torch.tensor([0.3, -1.4, 2.0, 0.9, -0.2, 1.7], dtype=torch.float64)

    prior = HMMRegimePrior(2, PriorConfig(learn_pi0=False)).double()
    with torch.no_grad():
        prior.transition_logits.copy_(a.log())  # softmax(log A) == A row-wise

    mu = mu_k.unsqueeze(0)  # (1, 2) constant emission means
    log_sigma = sigma_k.log()
    total_nll, filtered = _run_our_filter(prior, mu, log_sigma, ys)

    ref_ll, ref_alphas = _reference_forward(pi0, a, mu_k, sigma_k, ys)
    assert float(total_nll.detach()) == pytest.approx(-ref_ll, abs=1e-10)
    assert torch.allclose(filtered.detach(), ref_alphas, atol=1e-10)

    # gradient flows through the recursion into the transition matrix
    # (backprop through the forward algorithm == the recursive score,
    # arXiv:2205.01565; catches any accidental detach of the threaded state)
    total_nll.backward()
    assert prior.transition_logits.grad is not None
    assert prior.transition_logits.grad.abs().max() > 0


def test_filtering_is_causal_no_lookahead():
    """The leakage test, first-class: perturbing a FUTURE timestep leaves every
    earlier filtered posterior, prior, prediction and NLL bit-identical — while
    a smoothing double provably fails the same probe."""
    spec = SyntheticSpec(regime_process="markov", seed=11)
    panel = SyntheticRegimePanel(spec).generate(n_dates=6, n_entities=2)
    seq = panel.time_sequence()

    model = NECModel(small_config(prior_kind="hmm")).eval()
    trainer = Trainer(model)
    base = trainer.evaluate_sequence(seq)

    perturbed = list(seq)
    t_star = 4
    g = torch.Generator().manual_seed(99)
    perturbed[t_star] = Batch(
        torch.randn(seq[t_star].x_seq.shape, generator=g),
        torch.randn(seq[t_star].x_snap.shape, generator=g),
        torch.randn(seq[t_star].y.shape, generator=g),
    )
    alt = trainer.evaluate_sequence(perturbed)

    for t in range(t_star):
        assert torch.equal(base.log_filtered[t], alt.log_filtered[t])
        assert torch.equal(base.log_prior[t], alt.log_prior[t])
        assert torch.equal(base.y_hat[t], alt.y_hat[t])

    # smoothing conditions on the future: the same probe fails (why the
    # backward pass is prediction-time-illegal, design doc §3)
    a = torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float64)
    pi0 = torch.tensor([0.5, 0.5], dtype=torch.float64)
    mu_k = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    sigma_k = torch.tensor([1.0, 1.0], dtype=torch.float64)
    ys = torch.tensor([0.3, -1.4, 2.0, 0.9, -0.2, 1.7], dtype=torch.float64)
    ys_alt = ys.clone()
    ys_alt[4] = -3.0
    gamma = _reference_smoothed(pi0, a, mu_k, sigma_k, ys)
    gamma_alt = _reference_smoothed(pi0, a, mu_k, sigma_k, ys_alt)
    assert (gamma[2] - gamma_alt[2]).abs().max() > 1e-4


def test_transition_is_row_stochastic():
    prior = HMMRegimePrior(3, PriorConfig())
    with torch.no_grad():
        prior.transition_logits.normal_(std=3.0)
    rows = prior.transition_matrix.sum(dim=1)
    assert torch.allclose(rows, torch.ones(3), atol=1e-6)

    # still stochastic after an unconstrained gradient step — no projection
    opt = torch.optim.AdamW(prior.parameters(), lr=0.1)
    ctx = PriorContext(prev_filtered=torch.log(torch.tensor([[0.2, 0.5, 0.3]])))
    out = prior(torch.zeros(1, 3), ctx)
    (-out.log_prior[0, 1]).backward()
    opt.step()
    rows = prior.transition_matrix.sum(dim=1)
    assert torch.allclose(rows, torch.ones(3), atol=1e-6)


def test_transition_persistence_init():
    prior = HMMRegimePrior(2, PriorConfig(transition_diag_bias=2.0))
    a = prior.transition_matrix.detach()
    assert (a.diagonal() > a - torch.diag(a.diagonal())).all()
    expected = math.exp(2.0) / (math.exp(2.0) + 1.0)
    assert float(a[0, 0]) == pytest.approx(expected, abs=1e-4)


def test_stateful_threading_equivalence():
    """The trainer's two paths agree where they must: a memoryless prior run
    through the time-threaded path reproduces the one-shot loss."""
    model = NECModel(small_config())
    trainer = Trainer(model)
    x_seq, x_snap, y = random_inputs()
    batch = Batch(x_seq, x_snap, y)
    one_shot = trainer.evaluate(batch)
    threaded = trainer.evaluate_sequence([batch]).nll
    assert one_shot == pytest.approx(threaded, abs=1e-7)


def _vol_proxy(panel) -> torch.Tensor:
    return panel.x_seq[:, :, 0].std(dim=1)


def _train_hmm(panel, *, steps: int, seed: int = 0):
    cfg = small_config(
        prior_kind="hmm", sigma_init=2.0, seed=seed, lr=3e-3,
        sigma_freeze_steps=20,
    )
    model = NECModel(cfg)
    trainer = Trainer(model)
    # break expert symmetry first (design doc §4.3) — same device as the
    # memoryless path; the HMM's transition prior then learns persistence
    trainer.warmstart_experts(panel.full_batch(), sort_key=_vol_proxy(panel))
    trainer.fit_sequence(panel.time_sequence(), steps=steps, chunk_len=45)
    return model, trainer


def _persistence_spec(process: str, seed: int) -> SyntheticSpec:
    """Weak per-date vol signal + strong true persistence: the regime is
    ambiguous from a single date's window, so temporal evidence integration
    (filtering) is where the HMM prior earns its keep."""
    return SyntheticSpec(
        regime_process=process,
        transition_stay=0.97,
        vol_levels=(1.0, 1.3),  # weak: the memoryless gate is a poor per-date classifier
        beta_scale=1.2,
        noise_std=0.8,
        seed=seed,
    )


def test_hmm_recovers_persistent_regime():
    """The payoff test: when persistence is real and the per-date signal is
    weak, the HMM filter recovers the regime path and beats the memoryless
    prior on held-out NLL by integrating evidence over time."""
    panel = SyntheticRegimePanel(_persistence_spec("markov", seed=11)).generate(
        n_dates=360, n_entities=6
    )
    train, test = panel.split_by_date(0.75)

    hmm_model, hmm_trainer = _train_hmm(train, steps=150)

    soft_cfg = small_config(sigma_init=2.0, lr=3e-3, sigma_freeze_steps=20, batch_size=256)
    soft_trainer = Trainer(NECModel(soft_cfg))
    soft_trainer.warmstart_experts(train.full_batch(), sort_key=_vol_proxy(train))
    soft_trainer.fit(train, steps=150)

    hmm_eval = hmm_trainer.evaluate_sequence(test.time_sequence())
    soft_nll = soft_trainer.evaluate(test.full_batch())
    assert hmm_eval.nll < soft_nll - 0.03, (
        f"HMM should exploit persistence: hmm={hmm_eval.nll:.4f} soft={soft_nll:.4f}"
    )

    # regime recovery from the filtered posterior (label-switching-proof AUC)
    truth = torch.stack([b.regime for b in test.time_sequence()])  # (L, B)
    auc = regime_recovery_auc(hmm_eval.log_filtered[:, :, 0].exp().flatten(), truth.flatten())
    assert auc > 0.8, f"regime recovery AUC {auc:.3f}"

    # and the learned chain is sticky
    a = hmm_model.prior.transition_matrix.detach()
    assert float(a.diagonal().mean()) > 0.6, f"learned A not sticky:\n{a}"


def test_hmm_transition_reflects_true_persistence():
    """The transition matrix tracks reality: trained on genuinely persistent
    (markov) data it learns a stickier chain than on iid data — same model,
    same warm-start, same other parameters, differing only in the true regime
    process. (The 'does no harm on iid' framing is deliberately avoided: the
    baseline HMM prior is persistence-only and ignores the encoder, so with a
    useful per-date signal and no real persistence a memoryless gate can win —
    an honest limitation, resolved only by the deferred TVTP extension.)"""
    markov_diag = _learned_transition_diag("markov", seed=11)
    iid_diag = _learned_transition_diag("iid", seed=11)
    assert markov_diag > iid_diag + 0.03, (
        f"transition matrix should track true persistence: "
        f"markov={markov_diag:.3f} iid={iid_diag:.3f}"
    )


def _learned_transition_diag(process: str, seed: int) -> float:
    panel = SyntheticRegimePanel(_persistence_spec(process, seed)).generate(
        n_dates=360, n_entities=6
    )
    train, _ = panel.split_by_date(0.75)
    model, _ = _train_hmm(train, steps=150, seed=1)
    return float(model.prior.transition_matrix.detach().diagonal().mean())
