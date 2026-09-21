"""Training objective (design doc §4) and config-gated auxiliary losses.

The mixture NLL is computed entirely in the log domain with one fused
``logsumexp`` — never as a log of a sum of densities (Module 8 Ex. 8.1's
float64 cliff; Module 4 §4.10's fused-op lesson):

    a_k  = log pi_k + log N(y | mu_k, sigma_k^2)
    L    = -logsumexp_k(a_k)
    r_k  = softmax_k(a_k)          (Bayes posterior over experts)

Gradients (§4.2): gate logits receive ``pi - r`` (cross-entropy toward the
soft labels the experts negotiate — the fix for ledger defects 1–2); expert
means receive responsibility-weighted regression gradients; ``log sigma_k``
is stationary at the responsibility-weighted MSE.

Filtered posterior vs responsibilities
--------------------------------------
``log_filtered`` is returned **attached** because for the HMM prior it is the
recursion's state: gradient must flow through it to earlier timesteps (BPTT
over the regime posterior — Decision C's differentiable forward algorithm).
``responsibilities`` is a **detached** probability-domain copy for diagnostics
and monitoring only — never a second training signal (§4.1).

Auxiliary losses (both default OFF, §4.3)
-----------------------------------------
- :func:`load_balance_aux` with :class:`LoadBalanceBuffer` — Shazeer-form
  ``K * sum_k f_k * P_k`` where ``P_k`` (mean gate probability) carries the
  gradient and ``f_k`` (argmax-assignment fraction) is a stop-grad scale
  estimated over a **running buffer of many batches**, never a single batch:
  our batches are date-sliced and regime is date-level, so per-batch balancing
  is exactly the specialization-killing pitfall of arXiv:2501.11873.
- :func:`expert_decorrelation_aux` — RAVEN-style ``||R - I||_F^2`` on the batch
  correlation of expert outputs; targets expert *homogenization* (all experts
  learn the pooled regression), the failure mode load balancing does not.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "MixtureNLLOutput",
    "mixture_nll",
    "LoadBalanceBuffer",
    "load_balance_aux",
    "expert_decorrelation_aux",
    "correction_penalty_aux",
]


@dataclass
class MixtureNLLOutput:
    """Everything the fused loss produces in one pass.

    - ``nll``: scalar, mean per-sample negative log-likelihood (the objective).
    - ``per_sample_nll``: ``(B,)``.
    - ``log_filtered``: ``(B, K)`` log-domain Bayes posterior, **attached**
      (the HMM recursion's state; see module docstring).
    - ``responsibilities``: ``(B, K)`` probability-domain, **detached**
      (diagnostics only).
    """

    nll: Tensor
    per_sample_nll: Tensor
    log_filtered: Tensor
    responsibilities: Tensor


def mixture_nll(log_prior: Tensor, log_lik: Tensor) -> MixtureNLLOutput:
    """Fused mixture NLL + Bayes update, log domain end to end.

    ``log_prior`` may contain ``-inf`` (sparse priors): those components drop
    out of the ``logsumexp`` naturally and get exactly zero responsibility.
    """
    if log_prior.shape != log_lik.shape:
        raise ValueError(
            f"log_prior {tuple(log_prior.shape)} and log_lik "
            f"{tuple(log_lik.shape)} must match"
        )
    a = log_prior + log_lik  # (B, K)
    lse = torch.logsumexp(a, dim=-1)  # (B,)
    per_sample = -lse
    log_filtered = a - lse.unsqueeze(-1)
    return MixtureNLLOutput(
        nll=per_sample.mean(),
        per_sample_nll=per_sample,
        log_filtered=log_filtered,
        responsibilities=log_filtered.detach().exp(),
    )


class LoadBalanceBuffer:
    """Running argmax-assignment counts over the last N batches.

    The scope fix from arXiv:2501.11873: the assignment fraction ``f_k`` must
    be estimated over a pool spanning many dates so the penalty pushes the
    *marginal* utilization toward uniform while leaving any single
    (regime-homogeneous) date free to route sharply. Always updated — it also
    feeds the running-utilization dashboard even when the aux loss is off.
    """

    def __init__(self, n_experts: int, buffer_batches: int) -> None:
        self.n_experts = n_experts
        self._counts: deque[Tensor] = deque(maxlen=buffer_batches)

    def update(self, responsibilities: Tensor) -> None:
        """Record argmax assignments of a (detached) responsibility matrix."""
        assign = responsibilities.argmax(dim=-1)
        self._counts.append(
            torch.bincount(assign, minlength=self.n_experts).to(torch.float64)
        )

    def fractions(self) -> Tensor:
        """``f (K,)`` float32, uniform if the buffer is empty. No gradient."""
        if not self._counts:
            return torch.full((self.n_experts,), 1.0 / self.n_experts)
        total = torch.stack(tuple(self._counts)).sum(dim=0)
        return (total / total.sum()).to(torch.float32)

    # ---------------------------------------------------- checkpoint plumbing
    def get_state(self) -> list[Tensor]:
        """Buffer contents for a trainer checkpoint (oldest first)."""
        return list(self._counts)

    def set_state(self, counts: list[Tensor]) -> None:
        """Restore :meth:`get_state` output (maxlen re-truncates if it shrank)."""
        self._counts.clear()
        self._counts.extend(counts)


def load_balance_aux(gate_probs: Tensor, fractions: Tensor) -> Tensor:
    """``K * sum_k f_k * mean_b P_bk`` — gradient flows only through ``gate_probs``.

    With balanced ``f`` (uniform), the loss is identically 1 whatever the
    routing — sharp within-batch routing is unpenalized (the buffered-scope
    property §12.2 tests). ``fractions`` comes from :class:`LoadBalanceBuffer`
    and carries no gradient (the stop-grad scale of the Shazeer form).
    """
    k = gate_probs.shape[-1]
    p_bar = gate_probs.mean(dim=0)  # (K,)
    return k * (fractions.detach() * p_bar).sum()


def correction_penalty_aux(gate_probs: Tensor, corrections: Tensor) -> Tensor:
    """``mean_b ( sum_k pi_bk r_bk )^2`` — shrinkage toward the frozen base.

    The correction-magnitude penalty of the residual design (Ye & Borde,
    arXiv:2608.12251). It penalises the magnitude of the **combined**
    correction, not each expert separately: experts are free to disagree
    sharply as long as the routed result stays near the base, which is the
    behaviour wanted from a regime-conditional correction and is *not* what a
    per-expert penalty ``sum_k pi_k r_k^2`` would produce.

    ``gate_probs`` ``(B, K)`` and ``corrections`` ``(B, K)``; returns a scalar.
    Weighted by ``TrainConfig.correction_penalty_weight``, whose value is
    unknown — Ye & Borde do not report theirs — and is therefore selected on
    the training block, logged as a trial, and reported as a sensitivity
    curve. A result that exists only at one value of an unreported knob is
    not a result.
    """
    if gate_probs.shape != corrections.shape:
        raise ValueError(
            f"gate_probs {tuple(gate_probs.shape)} and corrections "
            f"{tuple(corrections.shape)} must match"
        )
    return (gate_probs * corrections).sum(dim=-1).pow(2).mean()


def expert_decorrelation_aux(mu: Tensor, eps: float = 1e-8) -> Tensor:
    """``||R - I||_F^2`` with ``R`` the batch correlation matrix of expert outputs.

    Caveat (kept from the design doc): this biases the likelihood objective —
    regimes may genuinely agree on part of the cross-section — which is why it
    is an ablation lever, not part of the baseline objective.
    """
    if mu.shape[0] < 2:
        return mu.new_zeros(())
    centered = mu - mu.mean(dim=0, keepdim=True)  # (B, K)
    norms = centered.norm(dim=0).clamp_min(eps)  # (K,)
    unit = centered / norms
    r = unit.T @ unit  # (K, K) cosine of centered columns == correlation
    k = mu.shape[-1]
    return (r - torch.eye(k, device=mu.device, dtype=mu.dtype)).pow(2).sum()
