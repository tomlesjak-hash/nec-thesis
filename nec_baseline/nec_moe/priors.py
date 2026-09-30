"""RegimePrior: the pluggable half of the gate (design doc §7B).

A ``RegimePrior`` forms the **log prior over experts** ``log pi (B, K)``,
normalized on the simplex (``logsumexp_k = 0``; ``-inf`` entries allowed for
future sparse priors). The shared likelihood core then combines it with the
per-expert log-likelihoods via Bayes' rule in :func:`nec_moe.losses.mixture_nll`.

Two families:

- **Memoryless (Variation 2)** — prior is a function of the current encoder
  output only: :class:`SoftRegimePrior` (the baseline), plus the comparison
  grid — :class:`UniformRegimePrior` (frozen uniform gate, the routing-value
  ablation), :class:`HardRegimePrior` (top-1 with the hard-EM training
  objective; the *trainable* version of the prototype's argmax),
  :class:`TopKRegimePrior` (masked + renormalized in the log domain), and
  :class:`GumbelSoftmaxRegimePrior` (Gumbel-perturbed logits with an annealed
  temperature). Hard routing genuinely changes the effective objective — that
  is the thesis comparison, and it is made explicit via
  ``PriorOutput.log_train_weights`` (see below).
- **Recursive (Variation 3)** — :class:`HMMRegimePrior`: the forward filter's
  *predict step* ``log pi_t[k] = logsumexp_j(log A_jk + log r_{t-1, j})`` with a
  learned row-stochastic transition matrix. **Filtering only, never smoothing**
  (design doc §3): the prior at t conditions only on data up to t-1; the
  update step (in the loss) conditions on data up to t. Anything conditioning
  on the future is lookahead leakage.

The name is deliberate: its job is to form the *prior*, which is what makes the
HMM gate a strict generalization of soft routing (a memoryless prior is a
stateful prior that ignores its state). ``Router`` is retained as an alias for
MoE-literature familiarity.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import NECConfig, PriorConfig
from .utils import assert_shape

if TYPE_CHECKING:  # pragma: no cover - import cycle: data imports nothing here
    from .data import Panel

__all__ = [
    "PriorOutput",
    "PriorContext",
    "RegimePrior",
    "SoftRegimePrior",
    "UniformRegimePrior",
    "HardRegimePrior",
    "TopKRegimePrior",
    "GumbelSoftmaxRegimePrior",
    "HMMRegimePrior",
    "PrecomputedRegimePrior",
    "PRIOR_REGISTRY",
    "build_prior",
    "Router",
]

_NEG_INF = float("-inf")


@dataclass
class PriorOutput:
    """Result of a prior evaluation.

    - ``log_prior``: ``(B, K)``, normalized (``logsumexp_k = 0``); ``-inf``
      entries allowed (sparse priors) — they drop out of the mixture's
      ``logsumexp`` naturally. These are the *prediction/combination* weights:
      ``y_hat = sum_k exp(log_prior_k) * mu_k``.
    - ``log_train_weights``: optional weights the **training objective** should
      use when they differ from ``log_prior``. Only :class:`HardRegimePrior`
      sets this: its predictive weights are one-hot (0/-inf) but its objective
      is the hard-EM surrogate ``-(log pi_sel + log N_sel)``, whose weights
      ``log pi_sel`` at the argmax (``-inf`` elsewhere) are deliberately
      *sub*-normalized — that is what routes a ``pi - onehot`` self-training
      gradient into the gate (the trainable replacement for the prototype's
      gradient-dead argmax).
    - ``aux_loss``: optional auxiliary term contributed by the prior itself.
    - ``info``: diagnostics (selected indices, temperatures, transition matrix…).
    """

    log_prior: Tensor
    log_train_weights: Tensor | None = None
    aux_loss: Tensor | None = None
    info: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class PriorContext:
    """Optional context threaded by the trainer.

    - ``prev_filtered``: ``(B, K)`` **log-domain** filtered posterior — the
      HMM recursion's state. ``None`` at a sequence start. With an ``h``-period
      forward target it is the posterior from ``h`` dates back, ``log r_{t-h}``
      (the latest one whose targets are all realised by ``t``), and
      ``predict_steps = h`` carries it forward to ``t``.
    - ``predict_steps``: how many predict steps separate ``prev_filtered``
      from the date being predicted; ``1`` for a one-period target.
    - ``step``: global training step, for temperature/annealing schedules.
    - ``date``: ``(B,)`` int64 date codes — the lookup key for a
      :class:`PrecomputedRegimePrior`. Ignored by every other prior.
    - ``prev_mask``: ``(B,)`` bool, which rows of ``prev_filtered`` hold a
      real previous posterior. ``False`` marks an entity with no state ``h``
      dates back (a new entrant, audit M-5): its prior is ``pi_0``, as at a
      sequence start. ``None`` means every row has one.
    """

    prev_filtered: Tensor | None = None
    predict_steps: int = 1
    step: int | None = None
    date: Tensor | None = None
    prev_mask: Tensor | None = None


class RegimePrior(nn.Module, ABC):
    """Maps gate logits (and optional recursive state) to a log prior."""

    #: True => the trainer must iterate timesteps chronologically, threading
    #: each step's filtered posterior into the next step's context.
    stateful: ClassVar[bool] = False

    #: True => the prior is fitted separately (``fit``) and frozen, rather
    #: than trained jointly by backpropagation. Under Q19 this is how all four
    #: compared gates work; the backpropagated priors are baseline arms.
    precomputed: ClassVar[bool] = False

    def __init__(self, n_experts: int) -> None:
        super().__init__()
        self.n_experts = n_experts

    def fit(self, train_panel: Panel) -> None:
        """Fit the prior on a **training block**. Default: a no-op.

        The walk-forward harness calls this once per fold, on that fold's
        training panel only, before the base and the experts are trained — so
        it inherits the purge discipline the harness already enforces. Every
        existing (backpropagated) prior is unaffected by the default.

        Implementations must treat ``train_panel`` as the entire information
        set they are permitted to see. Applying the fitted prior to later
        dates is a separate, causal operation (see
        :meth:`PrecomputedRegimePrior.apply_causal`).
        """

    @abstractmethod
    def forward(
        self, gate_logits: Tensor, ctx: PriorContext | None = None
    ) -> PriorOutput: ...


class PrecomputedRegimePrior(RegimePrior):
    """A regime prior computed **outside** the optimizer and looked up by date.

    The shape of every gate under Q19: the regime process is fitted on a
    training block, frozen, and then handed to the experts as an input they
    cannot influence. The table holds one row of **log** priors per date code;
    ``forward`` looks rows up and expands them to the batch, and contributes
    no gradient to anything.

    Two date populations live in the table, and the distinction is the whole
    point of the class:

    - dates written by :meth:`fit`, which the fitting procedure *saw*;
    - dates written by :meth:`apply_causal`, produced by running the frozen
      fitted model forward over later dates.

    A date beyond the training block is legal for *application* — a filter at
    date ``t`` conditions only on the series through ``t`` — but it must never
    arrive from a fit that saw it. ``apply_causal`` therefore refuses to
    overwrite a fitted row, and lookup refuses a date in neither population,
    so "the table quietly contained a date it should not have" is not a
    reachable state.

    ``stateful`` is False: any recursion has already been run at fit/apply
    time, so the trainer does not need to thread state.
    """

    stateful: ClassVar[bool] = False
    precomputed: ClassVar[bool] = True

    def __init__(self, n_experts: int) -> None:
        super().__init__(n_experts)
        self.fitted = False
        #: date code -> row index into ``_log_table``
        self._rows: dict[int, int] = {}
        self._log_table: Tensor = torch.zeros(0, n_experts)
        #: largest date the *fit* was allowed to see; apply_causal may only
        #: add dates strictly after it
        self.fit_max_date: int | None = None

    # ----------------------------------------------------------- table I/O
    def _write(self, dates: Tensor, log_prior: Tensor, *, source: str) -> None:
        assert_shape(log_prior, (dates.shape[0], self.n_experts), f"{source} log prior")
        lse = torch.logsumexp(log_prior, dim=-1)
        if float(lse.abs().max()) > 1e-4:
            raise ValueError(
                f"{source} rows must be normalized log probabilities "
                f"(max |logsumexp| = {float(lse.abs().max()):.3e})"
            )
        rows, table = dict(self._rows), [self._log_table]
        nxt = self._log_table.shape[0]
        keep: list[int] = []
        for i, d in enumerate(int(v) for v in dates):
            if d in rows:
                if source == "apply":
                    continue  # never overwrite a fitted row
                raise ValueError(f"duplicate date {d} written by {source}")
            rows[d] = nxt + len(keep)
            keep.append(i)
        if keep:
            table.append(log_prior[torch.tensor(keep, dtype=torch.long)].detach())
        self._rows = rows
        self._log_table = torch.cat(table, dim=0)

    def covers(self, dates: Tensor) -> Tensor:
        """Which of ``dates`` have a table row (a fitted or applied prior)."""
        return torch.tensor([int(d) in self._rows for d in dates], dtype=torch.bool)

    def set_fitted_table(self, dates: Tensor, log_prior: Tensor) -> None:
        """Install the rows produced by :meth:`fit` (the training block)."""
        self._write(dates, log_prior, source="fit")
        self.fit_max_date = int(dates.max())
        self.fitted = True

    def extend_causal_table(self, dates: Tensor, log_prior: Tensor) -> None:
        """Add rows produced by the causal application of the frozen fit."""
        if not self.fitted:
            raise ValueError("apply before fit: there is nothing frozen to apply")
        beyond = dates[dates > int(self.fit_max_date or 0)]
        if beyond.numel() == 0:
            return
        mask = dates > int(self.fit_max_date or 0)
        self._write(dates[mask], log_prior[mask], source="apply")

    def apply_causal(self, panel: Panel) -> None:
        """Extend the table to ``panel``'s later dates with **frozen** params.

        Subclasses implement the mechanics; the contract is that no parameter
        may be re-estimated here, because maximum likelihood over the extended
        series would use every observation including the future.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement causal application"
        )

    # ------------------------------------------------------------- forward
    def forward(
        self, gate_logits: Tensor, ctx: PriorContext | None = None
    ) -> PriorOutput:
        assert_shape(gate_logits, (None, self.n_experts), "gate logits (prior input)")
        if not self.fitted:
            raise ValueError(
                f"{type(self).__name__}.forward before fit(): a precomputed "
                "prior has no table until it is fitted on a training block. "
                "The walk-forward harness calls fit() once per fold"
            )
        if ctx is None or ctx.date is None:
            raise ValueError(
                f"{type(self).__name__} is looked up by date, but no dates "
                "were supplied (PriorContext.date is None). Batches carry "
                "their date codes; pass them through"
            )
        missing = sorted({int(d) for d in ctx.date} - set(self._rows))
        if missing:
            raise ValueError(
                f"{type(self).__name__}: {len(missing)} date(s) outside the "
                f"fitted information set, first {missing[:5]}. A date is only "
                "servable if fit() saw it or apply_causal() produced it by "
                "running the frozen model forward"
            )
        idx = torch.tensor(
            [self._rows[int(d)] for d in ctx.date], dtype=torch.long
        )
        return PriorOutput(log_prior=self._log_table[idx])


class SoftRegimePrior(RegimePrior):
    """Memoryless soft prior: ``log pi = log_softmax(gate_logits)``.

    The baseline's active prior (Variation 2's soft routing). Ignores ``ctx``.
    """

    stateful: ClassVar[bool] = False

    def forward(
        self, gate_logits: Tensor, ctx: PriorContext | None = None
    ) -> PriorOutput:
        assert_shape(gate_logits, (None, self.n_experts), "gate logits (prior input)")
        return PriorOutput(log_prior=F.log_softmax(gate_logits, dim=-1))


class UniformRegimePrior(RegimePrior):
    """Frozen uniform gate: ``log pi = -log K`` always.

    The routing-value ablation (design doc §10, ablation (a), after
    arXiv:2603.19136): experts + mixture prediction with **no learned routing**.
    Constant in the gate logits, so the gate/encoder receive no gradient —
    by construction, which is the point of the ablation.
    """

    stateful: ClassVar[bool] = False

    def forward(
        self, gate_logits: Tensor, ctx: PriorContext | None = None
    ) -> PriorOutput:
        assert_shape(gate_logits, (None, self.n_experts), "gate logits (prior input)")
        log_prior = torch.full_like(
            gate_logits.detach(), -math.log(self.n_experts)
        )
        return PriorOutput(log_prior=log_prior)


class HardRegimePrior(RegimePrior):
    """Top-1 routing with the hard-EM training objective (Viterbi/Switch-style).

    Prediction: ``log_prior`` is one-hot (0 at the argmax logit, ``-inf``
    elsewhere), so ``y_hat = mu_selected`` — genuine hard routing.

    Training: ``log_train_weights_k = log_softmax(z)_k`` at the argmax and
    ``-inf`` elsewhere, making the objective ``-(log pi_sel + log N_sel)`` —
    the complete-data / hard-EM surrogate (responsibilities hardened to the
    argmax indicator; an upper bound on the true mixture NLL). Its gate-logit
    gradient is ``pi - onehot(sel)``: cross-entropy toward the gate's own
    argmax, i.e. self-training. This is the standard trick (Switch Transformer
    scales the selected expert by its gate probability) that makes hard routing
    *trainable* — in sharp contrast to the prototype's bare ``argmax``, whose
    gradient to the gate is exactly zero (ledger defect 1).

    Deterministic; identical in train and eval modes.
    """

    stateful: ClassVar[bool] = False

    def forward(
        self, gate_logits: Tensor, ctx: PriorContext | None = None
    ) -> PriorOutput:
        assert_shape(gate_logits, (None, self.n_experts), "gate logits (prior input)")
        selected = gate_logits.argmax(dim=-1, keepdim=True)  # (B, 1), no grad
        keep = torch.zeros_like(gate_logits, dtype=torch.bool).scatter_(
            1, selected, True
        )
        log_prior = torch.zeros_like(gate_logits.detach()).masked_fill_(
            ~keep, _NEG_INF
        )
        log_train = F.log_softmax(gate_logits, dim=-1).masked_fill(~keep, _NEG_INF)
        return PriorOutput(
            log_prior=log_prior,
            log_train_weights=log_train,
            info={"selected": selected.squeeze(1)},
        )


class TopKRegimePrior(RegimePrior):
    """Sparse top-k routing: keep the k largest logits, renormalize in log domain.

    ``log_prior = log_softmax(z masked to top-k)``: exactly ``k`` finite
    entries per row, normalized over them (``logsumexp_k = 0``); the ``-inf``
    entries drop out of the mixture's ``logsumexp`` naturally, and gradient
    flows to the kept logits through the renormalized softmax. ``k = K``
    reduces exactly to :class:`SoftRegimePrior`. Deterministic.

    ``top_k >= 2`` is required: gradient reaches the gate only through the
    *competition among kept entries* — with a single kept entry the
    renormalized weight is constant 1 and the gate is gradient-dead, which is
    ledger defect 1 reborn. Trainable top-1 routing is
    :class:`HardRegimePrior` (whose hard-EM objective supplies the gradient
    the renormalization cannot).
    """

    stateful: ClassVar[bool] = False

    def __init__(self, n_experts: int, top_k: int) -> None:
        super().__init__(n_experts)
        if not 2 <= top_k <= n_experts:
            raise ValueError(
                f"top_k must be in [2, n_experts={n_experts}], got {top_k}: "
                "a single renormalized entry leaves the gate gradient-dead "
                "(ledger defect 1) — use HardRegimePrior for top-1"
            )
        self.top_k = top_k

    def forward(
        self, gate_logits: Tensor, ctx: PriorContext | None = None
    ) -> PriorOutput:
        assert_shape(gate_logits, (None, self.n_experts), "gate logits (prior input)")
        _, top_idx = gate_logits.topk(self.top_k, dim=-1)
        keep = torch.zeros_like(gate_logits, dtype=torch.bool).scatter_(
            1, top_idx, True
        )
        log_prior = F.log_softmax(
            gate_logits.masked_fill(~keep, _NEG_INF), dim=-1
        )
        return PriorOutput(log_prior=log_prior, info={"selected": top_idx})


class GumbelSoftmaxRegimePrior(RegimePrior):
    """Stochastic routing via Gumbel-perturbed logits with annealed temperature.

    Train mode: ``log pi = log_softmax((z + g) / tau)`` with ``g ~ Gumbel(0,1)``
    i.i.d. — by the Gumbel-max trick, as ``tau -> 0`` the perturbed argmax
    samples the categorical ``softmax(z)`` exactly; finite ``tau`` trades that
    bias for differentiability (Foundations E). ``tau`` anneals exponentially
    from ``tau_init`` to ``tau_min`` over ``tau_anneal_steps`` training steps
    (driven by ``ctx.step``; constant ``tau_init`` when unscheduled).

    Eval mode: deterministic ``log_softmax(z)`` (standard practice — noise and
    temperature are training devices).
    """

    stateful: ClassVar[bool] = False

    def __init__(
        self,
        n_experts: int,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        tau_anneal_steps: int = 0,
    ) -> None:
        super().__init__(n_experts)
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_anneal_steps = tau_anneal_steps

    def temperature(self, step: int | None) -> float:
        if self.tau_anneal_steps <= 0 or step is None:
            return self.tau_init
        frac = min(step / self.tau_anneal_steps, 1.0)
        return self.tau_init * (self.tau_min / self.tau_init) ** frac

    def forward(
        self, gate_logits: Tensor, ctx: PriorContext | None = None
    ) -> PriorOutput:
        assert_shape(gate_logits, (None, self.n_experts), "gate logits (prior input)")
        if not self.training:
            return PriorOutput(log_prior=F.log_softmax(gate_logits, dim=-1))
        u = torch.rand_like(gate_logits).clamp_(1e-9, 1.0 - 1e-9)
        gumbel = -torch.log(-torch.log(u))
        tau = self.temperature(ctx.step if ctx is not None else None)
        log_prior = F.log_softmax((gate_logits + gumbel) / tau, dim=-1)
        return PriorOutput(
            log_prior=log_prior, info={"tau": torch.tensor(tau)}
        )


class HMMRegimePrior(RegimePrior):
    """A **jointly fitted latent Markov mixture** — a baseline arm, not Hamilton.

    Named accurately, because the difference matters for the write-up: this is
    a latent Markov mixture with homogeneous, covariate-independent
    transitions, estimated **by gradient descent through the forward
    recursion** rather than by maximum likelihood. It is not Hamilton's Markov
    switching model as the econometrics literature estimates it, and the
    thesis must not describe it as though it were. The encoder and gate head
    are dormant under this prior: it ignores the gate logits and uses them for
    shape inference only (the dead-parameter audit reports exactly that).

    Under Q19 this is a **baseline arm**, not one of the four compared regime
    mechanisms. The Hamilton arm is
    :class:`~nec_moe.markov_gate.MarkovSwitchingRegimePrior`, fitted separately
    by maximum likelihood on the training block and frozen.

    Comparing the two is itself worth reporting: if the backpropagated filter
    recovers transition matrices and expected durations close to the
    maximum-likelihood ones, that is a defensible sentence in the methodology
    chapter; if it does not, better to find out now than at the defence.

    The mechanics below are unchanged — this docstring is a correction of what
    the object is *called*, not of what it does.

    Markov-transition prior: the forward filter's predict step (Variation 3).

    - Transition matrix ``A = row_softmax(L)`` over an unconstrained ``(K, K)``
      logit matrix ``L`` — a valid stochastic matrix under unconstrained
      optimization, no projection needed. Initialized persistence-biased
      (diagonal logits ``+transition_diag_bias``): financial regimes are
      assumed sticky, so training *starts* sticky.
    - Initial distribution ``pi_0 = softmax(pi0_logits)`` (learnable when
      ``learn_pi0``; else fixed uniform), used when ``ctx.prev_filtered`` is
      ``None`` (a sequence start) and for rows ``ctx.prev_mask`` marks as
      having no previous state (a new entrant).
    - **The state is per entity** (audit M-5). The regime posterior is a
      per-row quantity (each row is updated with its own target), and the
      trainer carries it keyed by entity: each row's previous posterior is
      looked up by its entity, new entrants start from ``pi_0``, exits are
      dropped. So a point-in-time panel whose membership changes runs, and
      the row order within a date is irrelevant. (A date-level regime would
      be a different model; that is not decided.)
    - Predict step: ``log pi_t[b, k] = logsumexp_j(log r_{t-1}[b, j] + log A[j, k])``.
    - **Multi-period targets** (audit finding M-1). The posterior ``r_s`` is
      updated with the target dated ``s``, a forward return over ``(s, s+h]``
      that is realised only at ``s+h``. The prior at ``t`` may therefore
      condition on posteriors up to ``r_{t-h}`` only, carried forward ``h``
      steps: ``pi_t = r_{t-h} A^h``. The trainer supplies ``r_{t-h}`` and
      ``ctx.predict_steps = h``; for ``h = 1`` this is the one-step
      recursion above, unchanged.

    ``gate_logits`` is used only for batch-size/shape inference: with a static
    transition matrix the encoder/gate path is dormant in this variant (their
    output re-enters only via the deferred TVTP extension, ``PriorConfig.tvtp``).
    Gradients flow through the recursion into ``L`` and back through earlier
    timesteps — BPTT over the regime posterior; backprop through this log-domain
    recursion is exactly the recursive score computation of arXiv:2205.01565,
    obtained automatically (Decision C).
    """

    stateful: ClassVar[bool] = True

    def __init__(self, n_experts: int, cfg: PriorConfig) -> None:
        super().__init__(n_experts)
        logits = torch.zeros(n_experts, n_experts)
        logits += cfg.transition_diag_bias * torch.eye(n_experts)
        self.transition_logits = nn.Parameter(logits)
        pi0 = torch.zeros(n_experts)
        if cfg.learn_pi0:
            self.pi0_logits: Tensor = nn.Parameter(pi0)
        else:
            self.register_buffer("pi0_logits", pi0)

    def log_transition(self) -> Tensor:
        """``log A`` with rows normalized on the simplex; ``A[j, k] = P(z_t=k | z_{t-1}=j)``."""
        return F.log_softmax(self.transition_logits, dim=1)

    @property
    def transition_matrix(self) -> Tensor:
        """``A`` in probability domain (diagnostics/tests)."""
        return F.softmax(self.transition_logits, dim=1)

    def forward(
        self, gate_logits: Tensor, ctx: PriorContext | None = None
    ) -> PriorOutput:
        assert_shape(gate_logits, (None, self.n_experts), "gate logits (prior input)")
        b = gate_logits.shape[0]
        log_a = self.log_transition()
        if ctx is None or ctx.prev_filtered is None:
            log_prior = F.log_softmax(self.pi0_logits, dim=0).unsqueeze(0).expand(b, -1)
        else:
            prev = ctx.prev_filtered
            assert_shape(prev, (b, self.n_experts), "prev_filtered")
            if ctx.predict_steps < 1:
                raise ValueError(
                    f"predict_steps must be >= 1, got {ctx.predict_steps}"
                )
            log_prior = prev
            for _ in range(ctx.predict_steps):
                # (B, K_from, 1) + (1, K_from, K_to) -> logsumexp over K_from
                log_prior = torch.logsumexp(
                    log_prior.unsqueeze(2) + log_a.unsqueeze(0), dim=1
                )
            if ctx.prev_mask is not None:
                assert_shape(ctx.prev_mask, (b,), "prev_mask")
                log_pi0 = F.log_softmax(self.pi0_logits, dim=0).unsqueeze(0).expand(b, -1)
                log_prior = torch.where(ctx.prev_mask.unsqueeze(1), log_prior, log_pi0)
        return PriorOutput(log_prior=log_prior, info={"log_transition": log_a})


# --------------------------------------------------------------------------- #
# Registry: a config string selects the mechanism (the ablation grid is a
# config sweep — "config change, not rewrite").
# --------------------------------------------------------------------------- #

PRIOR_REGISTRY: dict[str, Callable[[NECConfig], RegimePrior]] = {
    "soft": lambda cfg: SoftRegimePrior(cfg.experts.n_experts),
    "uniform": lambda cfg: UniformRegimePrior(cfg.experts.n_experts),
    "hard": lambda cfg: HardRegimePrior(cfg.experts.n_experts),
    "topk": lambda cfg: TopKRegimePrior(cfg.experts.n_experts, cfg.prior.top_k),
    "gumbel": lambda cfg: GumbelSoftmaxRegimePrior(
        cfg.experts.n_experts,
        tau_init=cfg.prior.tau_init,
        tau_min=cfg.prior.tau_min,
        tau_anneal_steps=cfg.prior.tau_anneal_steps,
    ),
    "hmm": lambda cfg: HMMRegimePrior(cfg.experts.n_experts, cfg.prior),
    # Hamilton's Markov switching model, fitted by ML on the training block
    # and frozen (brief 03). Late import: markov_gate imports from this module.
    "markov": lambda cfg: _build_markov(cfg),
}


def _build_markov(cfg: NECConfig) -> RegimePrior:
    from .markov_gate import MarkovSwitchingRegimePrior

    return MarkovSwitchingRegimePrior(cfg.experts.n_experts, cfg.markov_gate)


def build_prior(cfg: NECConfig) -> RegimePrior:
    try:
        return PRIOR_REGISTRY[cfg.prior.kind](cfg)
    except KeyError:
        raise ValueError(
            f"unknown prior.kind {cfg.prior.kind!r}; registered: "
            f"{sorted(PRIOR_REGISTRY)}"
        ) from None


#: Alias for MoE-literature familiarity.
Router = RegimePrior
