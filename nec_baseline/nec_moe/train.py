"""Trainer: AdamW on the mixture NLL, understood as generalized EM (§4.3).

Two execution paths, selected by ``model.prior.stateful``:

- **Memoryless (one-shot)** — Variation 2's soft routing: each batch is
  processed in a single forward; auxiliary losses (load balancing with
  buffered scope, expert decorrelation) compose additively when enabled.
- **Stateful (time-threaded)** — Variation 3's HMM prior: the trainer owns the
  recursion, iterating per-date batches chronologically and threading each
  step's **attached** filtered posterior into the next step's context
  (backprop through the forward algorithm — Decision C). Long sequences are
  trained in chunks with the carried state **detached** at chunk boundaries
  (truncated BPTT over the regime posterior).

Optimizer hygiene:

- Parameter groups: the gate's linear weight gets its own weight decay —
  structural, not cosmetic (separation-proofing, Module 4); biases, LayerNorm,
  ``log_sigma`` and prior parameters (transition logits, pi0) get **no** decay
  (decaying transition logits would pull transitions toward uniform);
  everything else gets the default decay.
- ``log_sigma`` is frozen for the first ``sigma_freeze_steps`` steps
  (warm-start: letting sigma move first lets the model explain everything as
  noise).

Checkpointing (resume-exact):

- :meth:`Trainer.save` / :meth:`Trainer.load` persist the *complete* training
  state — model, optimizer, ``step_count`` (every schedule keys off it: sigma
  freeze, Gumbel tau anneal), the load-balance buffer, the metrics history,
  the minibatch sampler (generator state + current permutation + position),
  the stateful path's chunk cursor + carried filter state, and the **global**
  torch RNG (dropout draws from it). Interrupt a run, ``Trainer.load`` it, and
  the remaining steps reproduce the uninterrupted trajectory bit for bit —
  given the same panel/sequence, which the checkpoint does *not* store.
- ``fit``/``fit_sequence`` take a ``checkpoint_path``: saved every
  ``TrainConfig.checkpoint_every`` steps plus once at the end of the call.
  Writes are atomic (tmp + rename), so a crash mid-write can't corrupt the
  last good checkpoint.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from .config import NECConfig
from .data import Batch, Panel
from .decay import weighted_mean
from .diagnostics import (
    expert_output_correlation,
    gate_entropy,
    pairwise_expert_distance,
    persistence_metrics,
    sharpness,
    utilization,
)
from .interrupt import RunInterrupted, stop_requested
from .likelihood import expert_log_likelihood
from .losses import (
    OBJECTIVE_REGISTRY,
    LoadBalanceBuffer,
    MixtureNLLOutput,
    correction_penalty_aux,
    expert_decorrelation_aux,
    load_balance_aux,
)
from .model import NECModel, NECOutput
from .priors import PriorContext
from .utils import atomic_torch_save

__all__ = [
    "Trainer",
    "SequenceEval",
    "GradientAudit",
    "DeadParameterWarning",
    "FilterState",
]

#: Bumped when the checkpoint payload layout changes incompatibly.
CHECKPOINT_FORMAT = 1


@dataclass(frozen=True)
class FilterState:
    """One date's filtered posteriors, keyed by entity.

    ``entity`` is ``(B,)`` entity codes (``None`` for batches that carry none:
    then rows are matched by position, the pre-M-5 behaviour, which needs the
    same rows in the same order on every date); ``log_filtered`` is ``(B, K)``.
    """

    entity: Tensor | None
    log_filtered: Tensor

    def detach(self) -> FilterState:
        return FilterState(self.entity, self.log_filtered.detach())


FilterHistory = list[FilterState]
"""The recursion's state: the latest filtered posteriors, oldest first.

The HMM prior at date ``t`` may use only targets realised by ``t``. With an
``h``-period forward target that is the posterior from ``h`` dates back, so
the state carried between chunks and passed to :meth:`Trainer.evaluate_sequence`
is the last ``h`` dates' posteriors, not only the last one (audit finding M-1).
Each is keyed by entity (audit M-5). A bare ``(B, K)`` tensor, a stacked
``(L, B, K)`` tensor or a list of tensors is accepted and read as a
positional history.
"""


def _as_history(
    state: Tensor | Sequence[Tensor] | Sequence[FilterState] | None,
) -> FilterHistory:
    if state is None:
        return []
    if isinstance(state, Tensor):
        if state.ndim == 2:
            return [FilterState(None, state)]
        if state.ndim == 3:
            return [FilterState(None, s) for s in state.unbind(0)]
        raise ValueError(
            f"filter state must be (B, K) or (L, B, K), got {tuple(state.shape)}"
        )
    return [s if isinstance(s, FilterState) else FilterState(None, s) for s in state]


def _lookup_rows(entity: Tensor, prev_entity: Tensor) -> tuple[Tensor, Tensor]:
    """For each of ``entity``: its row in ``prev_entity`` and whether it exists."""
    sorted_prev, order = torch.sort(prev_entity)
    pos = torch.searchsorted(sorted_prev, entity).clamp(max=len(sorted_prev) - 1)
    found = sorted_prev[pos] == entity
    return order[pos], found


class DeadParameterWarning(UserWarning):
    """Some parameters receive no gradient — see :class:`GradientAudit`."""


@dataclass
class SequenceEval:
    """Result of a no-grad filtering pass over a chronological sequence."""

    nll: float  # mean per-sample NLL over all timesteps
    # log-domain filtered posteriors, priors (the predict step) and causal
    # point predictions: stacked (L, B, K) / (L, B, K) / (L, B) when every
    # date has the same number of rows, else one tensor per date
    log_filtered: Tensor | list[Tensor]
    log_prior: Tensor | list[Tensor]
    y_hat: Tensor | list[Tensor]
    # the last h dates' posteriors, keyed by entity: the state to continue
    # from (e.g. the training block's, into its test block)
    final_state: FilterHistory = field(default_factory=list)


@dataclass(frozen=True)
class GradientAudit:
    """Which parameters actually train — the parameter-count confound, measured.

    Under ``prior.kind="hmm"`` the gate logits are used only for shape
    inference, so with ``experts.input_mode="snapshot"`` the GRU encoder and
    the gate head are **structurally disconnected** from the loss and never
    train; under ``prior.kind="soft"`` the same tensors are live. Comparing
    those two arms therefore compares different effective model sizes unless
    the difference is measured and reported — which is what this is for.

    Three disjoint categories, by cause (the causes need different readings):

    - ``disconnected`` — ``requires_grad`` but ``grad is None`` after backward:
      no autograd path from the loss exists at all. This is the structural
      deadness above, and it is what ``live_param_count`` excludes.
    - ``zero_grad`` — connected, but the gradient was all-zero on the audited
      batch. This is **not** evidence of structural deadness and may be purely
      transient: with the default zero-initialized gate head
      (``GateConfig.zero_init``) the encoder's gradient is exactly zero on the
      first backward pass and nonzero forever after, because the chain rule
      runs through a weight that is still zero. Counted as live.
    - ``frozen`` — ``requires_grad=False``: not trainable at this instant.
      Includes ``log_sigma`` while the ``sigma_freeze_steps`` warm-start holds
      it (it becomes live later), so read this list against the schedule.

    ``live_param_count`` is the headline number: elements that are trainable
    and connected to the loss. It is a snapshot taken at the first backward
    pass of a fit; the categories above say what that snapshot can and cannot
    be read to mean.
    """

    live_param_count: int
    total_param_count: int
    disconnected: tuple[str, ...] = ()
    zero_grad: tuple[str, ...] = ()
    frozen: tuple[str, ...] = ()
    step: int = 0
    #: numel by top-level module, for the disconnected tensors only
    disconnected_by_group: dict[str, int] = field(default_factory=dict)

    @property
    def has_dead_parameters(self) -> bool:
        return bool(self.disconnected)

    def summary(self) -> str:
        if not self.disconnected:
            return (
                f"all {self.total_param_count:,} parameters connected to the "
                f"loss; live_param_count={self.live_param_count:,}"
            )
        groups = ", ".join(
            f"{g} ({n:,} params)" for g, n in sorted(self.disconnected_by_group.items())
        )
        return (
            f"{len(self.disconnected)} parameter tensor(s) receive no gradient: "
            f"{groups}. live_param_count={self.live_param_count:,} of "
            f"{self.total_param_count:,}"
        )


class Trainer:
    """Owns the optimizer, the aux-loss composition, and the time threading."""

    def __init__(self, model: NECModel, cfg: NECConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg if cfg is not None else model.cfg
        t = self.cfg.train
        if model.prior.stateful and (t.aux_load_balance or t.aux_expert_decorrelation):
            raise ValueError(
                "auxiliary losses are supported on the memoryless path only "
                "(they are Variation-2 ablation levers); disable them for the "
                "HMM prior"
            )
        # the primary objective, resolved once through the registry (Q20 seam)
        self._objective = OBJECTIVE_REGISTRY[t.objective]
        if t.freeze_gate:
            self.freeze_gate()
        # after any freezing: _param_groups filters on requires_grad, so a
        # frozen regime path never reaches the optimizer in the first place
        self.opt = torch.optim.AdamW(self._param_groups(), lr=t.lr)
        self.lb_buffer = LoadBalanceBuffer(
            self.cfg.experts.n_experts, t.load_balance_buffer_batches
        )
        self.step_count = 0
        self.history: list[dict[str, float]] = []
        #: Filled on the first backward pass (see :meth:`audit_gradients`).
        self.grad_audit: GradientAudit | None = None
        # fit()'s sampler state — instance-owned (not fit-local) so a
        # checkpoint can freeze it mid-epoch and resume the exact stream
        self._fit_gen: torch.Generator | None = None
        self._fit_perm: Tensor | None = None
        self._fit_pos: int = 0
        #: called with the step count every TrainConfig.heartbeat_every steps
        #: (the run store's status heartbeat, brief 07 C.5); not checkpointed
        self.on_heartbeat: Callable[[int], None] | None = None
        # fit_sequence()'s cursor: which chunk is next + the carried
        # (detached) filter state, plus the chunking it was built under
        self._seq_ci: int = 0
        self._seq_state: FilterHistory | None = None
        self._seq_meta: dict[str, int] | None = None

    # ------------------------------------------------------------ internals
    def _param_groups(self) -> list[dict]:
        t = self.cfg.train
        gate_decay, no_decay, default = [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name == "gate.linear.weight":
                gate_decay.append(p)
            elif p.ndim <= 1 or name.startswith("prior."):
                # 1-D parameters are statistical/normalization quantities, not
                # weight matrices: biases, LayerNorm scales, log_sigma, the
                # classical emission's mu, pi0. Prior parameters (incl. the 2-D
                # transition logits) are also undecayed — decay would pull
                # transitions toward uniform.
                no_decay.append(p)
            else:
                default.append(p)
        return [
            {"params": gate_decay, "weight_decay": t.gate_weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": default, "weight_decay": t.weight_decay},
        ]

    def freeze_gate(self) -> None:
        """Freeze the whole regime path: encoder, gate head, prior parameters.

        Q19's design commitment. With the gate frozen every arm receives the
        same regime assignment as an input it cannot influence, so the only
        thing varying across arms is the regime process that produced it —
        which is what makes the comparison attributable. The cost is accepted
        deliberately: a jointly optimized gate does better on the fitted
        objective (the DeepSeekMoE observation), but then a gain cannot be
        credited to regime structure rather than to the freedom to move the
        boundaries.

        Called from ``__init__`` when ``TrainConfig.freeze_gate`` is set, i.e.
        *before* the optimizer is constructed. Calling it afterwards would
        leave the already-built parameter groups holding frozen tensors.
        """
        self.model.encoder.requires_grad_(False)
        self.model.gate.requires_grad_(False)
        self.model.prior.requires_grad_(False)
        self._train_mode()

    def _train_mode(self) -> None:
        """``model.train()``, then put every frozen submodule back in ``eval()``.

        ``nn.Module.train()`` recurses into children, so a plain
        ``model.train()`` would switch the frozen base's dropout back on and
        make its predictions a moving target with its weights pinned — a
        silent failure, since nothing raises and the numbers merely become
        wrong. Frozen means frozen in both senses: no gradient *and* no
        train-mode stochasticity. (For a frozen Gumbel prior this is also
        what makes the assignment deterministic, as a fixed gate must be.)
        """
        self.model.train()
        if self.cfg.train.freeze_gate:
            self.model.encoder.eval()
            self.model.gate.eval()
            self.model.prior.eval()
        if self.model.base is not None:
            self.model.base.eval()

    def _lagged_context(
        self, history: FilterHistory, entity: Tensor | None = None
    ) -> PriorContext:
        """Prior context for the next date: the posterior ``h`` dates back.

        ``h`` is the target's forward horizon (``DataConfig.horizon_periods``).
        The posterior from ``h`` dates back is the latest one updated only with
        targets realised by the date being predicted; it is carried forward
        ``h`` predict steps. Fewer than ``h`` posteriors available (a sequence
        start) means no legal state yet: the prior's initial distribution.
        With gaps in the sequence (the purge between a training block and its
        test block) ``h`` positions back is at least ``h`` dates back, so the
        rule errs toward older information, never newer.
        """
        h = self.cfg.data.horizon_periods
        if len(history) < h:
            return PriorContext(step=self.step_count)
        prev = history[-h]
        if prev.entity is None or entity is None:  # positional (pre-M-5) history
            return PriorContext(
                prev_filtered=prev.log_filtered, predict_steps=h, step=self.step_count
            )
        # keyed by entity (audit M-5): each row's own posterior h dates back;
        # an entity absent then (a new entrant) starts from pi_0
        rows, found = _lookup_rows(entity, prev.entity)
        return PriorContext(
            prev_filtered=prev.log_filtered[rows], prev_mask=found,
            predict_steps=h, step=self.step_count,
        )

    def _apply_sigma_schedule(self) -> None:
        frozen = self.step_count < self.cfg.train.sigma_freeze_steps
        self.model.experts.log_sigma.requires_grad_(not frozen)

    @property
    def live_param_count(self) -> int | None:
        """Trainable elements connected to the loss; ``None`` before the audit."""
        return None if self.grad_audit is None else self.grad_audit.live_param_count

    # -------------------------------------------- the parameter-count confound
    def audit_gradients(self) -> GradientAudit:
        """Classify every parameter by whether it received gradient.

        Call **after** a ``backward()`` and before the next ``zero_grad()``:
        the trainer zeroes with ``set_to_none=True``, so ``grad is None`` after
        a backward pass means precisely "no autograd path from the loss
        reached this tensor". See :class:`GradientAudit` for how to read the
        three categories.
        """
        disconnected, zero_grad, frozen = [], [], []
        live = total = 0
        by_group: dict[str, int] = {}
        for name, p in self.model.named_parameters():
            total += p.numel()
            if not p.requires_grad:
                frozen.append(name)
                continue
            if p.grad is None:
                disconnected.append(name)
                group = name.split(".")[0]
                by_group[group] = by_group.get(group, 0) + p.numel()
                continue
            # connected: counts as live even if this batch's gradient is zero
            live += p.numel()
            if float(p.grad.abs().max()) == 0.0:
                zero_grad.append(name)
        return GradientAudit(
            live_param_count=live,
            total_param_count=total,
            disconnected=tuple(disconnected),
            zero_grad=tuple(zero_grad),
            frozen=tuple(frozen),
            step=self.step_count,
            disconnected_by_group=by_group,
        )

    def _audit_once(self) -> None:
        """Run the audit on the first backward pass and warn once if dead."""
        if self.grad_audit is not None:
            return
        audit = self.grad_audit = self.audit_gradients()
        if audit.has_dead_parameters:
            warnings.warn(
                f"prior.kind={self.cfg.prior.kind!r}, "
                f"experts.input_mode={self.cfg.experts.input_mode!r}: "
                f"{audit.summary()}. Those parameters have no autograd path "
                "from the loss and will never train; any comparison across "
                "prior kinds compares different effective model sizes unless "
                "live_param_count is reported beside the result (it is logged "
                "to the trial registry).",
                DeadParameterWarning,
                # _audit_once <- _optimize <- train_step[_sequence] <- fit[_sequence]
                # <- caller: point at the user's fit() call, not at trainer internals
                stacklevel=5,
            )

    # ------------------------------------------------- expert warm-start
    def warmstart_experts(
        self,
        batch: Batch,
        sort_key: Tensor,
        *,
        steps: int = 150,
        lr: float = 1e-2,
    ) -> None:
        """Break expert symmetry on regime/vol-sorted slices (design doc §4.3).

        Seed diversity alone cannot escape the symmetric-mixture local optimum
        (both experts collapse to the pooled regression) on a genuinely
        symmetric task. This is the doc's prescribed stronger device: sort
        samples by an observable regime proxy (``sort_key`` — realized/window
        volatility or VIX on real data; here whatever the caller supplies),
        partition into K quantile slices, and fit emission component k to
        slice k (each emission breaks symmetry its own way — MSE pre-training
        for neural experts, closed-form moments for classical Gaussians; see
        :meth:`nec_moe.experts.Emission.warmstart_slices`). Only the emission
        is touched — the gate stays unsupervised (Decision B intact); it still
        has to learn to route via ``pi - r`` afterwards. Deliberate and
        interpretable symmetry breaking, not a supervised gate.
        """
        if sort_key.shape[0] != len(batch):
            raise ValueError(
                f"sort_key length {sort_key.shape[0]} != batch size {len(batch)}"
            )
        x = self.cfg.experts
        if x.correction_mode and x.zero_init_head:
            raise ValueError(
                "warmstart_experts is incompatible with correction_mode + "
                "zero_init_head: pre-training the experts as regressors makes "
                "their heads nonzero, so the model would no longer start "
                "exactly at the base. The frozen gate already breaks expert "
                "symmetry (its per-date weights differ), so the warm-start's "
                "job is done for it — the walk-forward harness skips it "
                "automatically in this mode. Set zero_init_head=False to "
                "warm-start against the residual instead (Ye & Borde's "
                "random-init ablation)"
            )
        self._train_mode()
        with torch.no_grad():
            h_t = self.model.encoder(batch.x_seq)
            x_exp = self.model.expert_input(batch.x_snap, h_t)
            # In correction mode the experts model y - f0(x), not y: warm-
            # starting them on the raw target would teach them the base's job.
            target = batch.y
            if x.correction_mode:
                assert self.model.base is not None
                target = batch.y - self.model.base(batch.x_snap)
        order = torch.argsort(sort_key)
        slices = torch.chunk(order, self.cfg.experts.n_experts)
        self.model.experts.warmstart_slices(
            x_exp, target, slices, steps=steps, lr=lr
        )

    def _forward_nll(
        self,
        batch: Batch,
        ctx: PriorContext | None = None,
        *,
        train_objective: bool = True,
    ) -> tuple[NECOutput, MixtureNLLOutput]:
        """Forward + fused loss.

        ``train_objective=True`` uses ``prior.log_train_weights`` when the
        prior provides them (hard routing's hard-EM surrogate, which is what
        trains its gate); evaluation uses ``log_prior`` — the prior's honest
        *predictive* weights (for hard routing: the selected expert's density).
        Priors that don't distinguish the two are unaffected.
        """
        # date codes reach the prior here, once, for every execution path:
        # a precomputed (date-keyed) gate is looked up by them
        if batch.date is not None and (ctx is None or ctx.date is None):
            ctx = dataclasses.replace(ctx or PriorContext(), date=batch.date)
        out = self.model(batch.x_seq, batch.x_snap, ctx)
        log_lik = expert_log_likelihood(out.mu, out.log_sigma, batch.y)
        log_w = out.prior.log_prior
        if train_objective and out.prior.log_train_weights is not None:
            log_w = out.prior.log_train_weights
        return out, self._objective(log_w, log_lik)

    def _optimize(self, loss: Tensor) -> None:
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self._audit_once()  # after backward, before the next zero_grad
        clip = self.cfg.train.grad_clip
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        self.opt.step()
        self.step_count += 1

    def _metrics(
        self, loss: Tensor, nll_out: MixtureNLLOutput, out: NECOutput
    ) -> dict[str, float]:
        with torch.no_grad():
            probs = out.prior.log_prior.exp()
            r = nll_out.responsibilities
            # expert_signal, not mu: with a shared frozen base, mu_k = f0 + r_k
            # and the homogenization watch would read the base's correlation
            # (-> 1) whatever the experts did. See NECOutput.expert_signal.
            signal = out.expert_signal.detach()
            corr = expert_output_correlation(signal)
            off = corr - torch.eye(corr.shape[0])
            dist = pairwise_expert_distance(signal)
            off_dist = dist[~torch.eye(dist.shape[0], dtype=torch.bool)]
            m = {
                "step": float(self.step_count),
                "loss": float(loss),
                "nll": float(nll_out.nll),
                "gate_entropy": float(gate_entropy(probs)),
                "sharpness": float(sharpness(r)),
                "min_batch_utilization": float(utilization(r).min()),
                "min_running_utilization": float(self.lb_buffer.fractions().min()),
                "max_offdiag_expert_corr": float(off.abs().max()),
                # scale-carrying companion to the correlation: two experts can
                # be perfectly correlated and numerically far apart, or
                # uncorrelated and identical to four decimals. The minimum is
                # the alarm — the closest pair is the one about to collapse.
                "min_pairwise_expert_distance": float(off_dist.min()),
                "mean_pairwise_expert_distance": float(off_dist.mean()),
            }
            # Persistence of the fitted chain, for any prior exposing a
            # transition matrix (duck-typed, so a future prior gets it free).
            # Logged every step rather than every log_every: a ragged history
            # is a trap for anything that reads a column out of it, and the
            # K x K solve is microseconds against a forward/backward pass.
            # Raw (not canonical) state order here: within one fit the indices
            # are stable, while a sigma-sort could swap mid-run and make the
            # series jump between regimes. Cross-fit reporting canonicalizes.
            transition = getattr(self.model.prior, "transition_matrix", None)
            if transition is not None:
                m |= persistence_metrics(transition.detach())
        self.history.append(m)
        return m

    # ------------------------------------------------- memoryless (one-shot)
    def train_step(self, batch: Batch, weight: Tensor | None = None) -> dict[str, float]:
        """One optimizer step on one cross-sectional batch (memoryless prior).

        ``weight`` (``(B,)``, optional) are the rows' decay weights (brief 09
        C.3): the data term becomes ``sum(w * nll_row) / sum(w)``; without it
        the step is exactly the unweighted one. The auxiliary terms (load
        balancing, decorrelation, the correction penalty) are unweighted
        regularisers and stay so."""
        if self.model.prior.stateful:
            raise ValueError(
                "the model's prior is stateful (HMM): shuffled one-shot batches "
                "would silently skip the forward recursion — use "
                "train_step_sequence/fit_sequence with chronological batches"
            )
        t = self.cfg.train
        self._train_mode()
        self._apply_sigma_schedule()
        # step is threaded even on the memoryless path: annealed priors
        # (Gumbel temperature) schedule off it
        ctx = PriorContext(step=self.step_count)
        out, nll_out = self._forward_nll(batch, ctx)
        loss = nll_out.nll if weight is None else weighted_mean(nll_out.per_sample_nll, weight)

        # buffer always updated: it feeds the running-utilization dashboard
        # even when the aux loss is off
        self.lb_buffer.update(nll_out.responsibilities)
        if t.aux_load_balance:
            loss = loss + t.aux_load_balance_weight * load_balance_aux(
                out.prior.log_prior.exp(), self.lb_buffer.fractions()
            )
        if t.aux_expert_decorrelation:
            # expert_signal, not mu: in correction mode mu_k = f0 + r_k shares
            # one base across all K columns, and decorrelating THAT can only be
            # achieved by inflating the corrections until they dominate the
            # base — the opposite of what the shrinkage penalty is asking for.
            loss = loss + t.aux_decorrelation_weight * expert_decorrelation_aux(
                out.expert_signal
            )
        loss = loss + self._correction_penalty(out)
        if out.prior.aux_loss is not None:
            loss = loss + out.prior.aux_loss

        self._optimize(loss)
        return self._metrics(loss, nll_out, out)

    def _correction_penalty(self, out: NECOutput) -> Tensor:
        """``alpha * mean_b (sum_k pi_k r_k)^2``, or exactly zero when off."""
        t = self.cfg.train
        if not t.aux_correction_penalty or out.corrections is None:
            return out.mu.new_zeros(())
        return t.correction_penalty_weight * correction_penalty_aux(
            out.prior.log_prior.exp(), out.corrections
        )

    def _next_fit_index(self, n_rows: int) -> Tensor:
        """The row indices of the next shuffled minibatch of the seeded epoch
        stream.

        Identical stream to chaining ``data.minibatches(batch_size, g)`` epoch
        after epoch — but the generator, the current permutation, and the
        position within it live on the instance, so a checkpoint can freeze
        the stream mid-epoch and resume it exactly. The stream depends only on
        ``TrainConfig.seed``, the batch size and the number of rows, never on
        the arm (brief 09 E.3).
        """
        t = self.cfg.train
        if self._fit_gen is None:
            self._fit_gen = torch.Generator().manual_seed(t.seed)
        if self._fit_perm is None or self._fit_pos >= len(self._fit_perm):
            self._fit_perm = torch.randperm(n_rows, generator=self._fit_gen)
            self._fit_pos = 0
        idx = self._fit_perm[self._fit_pos : self._fit_pos + t.batch_size]
        self._fit_pos += t.batch_size
        return idx

    def _next_fit_batch(self, data: Panel) -> Batch:
        """The next shuffled minibatch (see :meth:`_next_fit_index`)."""
        return data._batch(self._next_fit_index(len(data)))

    def _maybe_checkpoint(self, path: Path | None) -> None:
        if (
            path is not None
            and self.cfg.train.checkpoint_every > 0
            and self.step_count % self.cfg.train.checkpoint_every == 0
        ):
            self.save_checkpoint(path)

    def _after_step(self, path: Path | None) -> None:
        """After every optimizer step: periodic checkpoint, heartbeat, and a
        requested stop (brief 07 C.3), which checkpoints first and then raises
        :class:`~nec_moe.interrupt.RunInterrupted`."""
        self._maybe_checkpoint(path)
        if (
            self.on_heartbeat is not None
            and self.step_count % self.cfg.train.heartbeat_every == 0
        ):
            self.on_heartbeat(self.step_count)
        if stop_requested():
            if path is not None:
                self.save_checkpoint(path)
            raise RunInterrupted(
                f"stop requested: checkpointed at step {self.step_count}"
                + (f" to {path}" if path is not None else " (no checkpoint path)")
            )

    def fit(
        self,
        data: Panel,
        steps: int | None = None,
        checkpoint_path: str | Path | None = None,
        row_weight: Tensor | None = None,
    ) -> list[dict[str, float]]:
        """Minibatch training loop over a panel (memoryless prior).

        ``row_weight`` (``(len(data),)``, optional) are the experts' decay
        weights (brief 09 C), fixed for the fold; each step's loss is the
        weighted mean over its batch (:meth:`train_step`). The weights are not
        checkpointed: they are recomputed from the frozen gate on resume.

        ``steps`` are *additional* steps for this call — the sample stream and
        ``step_count`` continue across calls, so ``fit(60)`` then ``fit(40)``
        (possibly via a checkpoint reload in between) equals one ``fit(100)``.
        With ``checkpoint_path`` set, saves every
        ``TrainConfig.checkpoint_every`` steps and once at the end; resuming
        requires the *same* panel (the checkpoint stores the sampler, not the
        data). A Q26 panel built for another expert input mode is refused
        (:func:`nec_moe.features.check_panel_input_mode`, brief 08 D.4.6).
        """
        from .features import check_panel_input_mode  # late: features is data-side

        check_panel_input_mode(data, self.cfg.experts.input_mode)
        if row_weight is not None and row_weight.shape != (len(data),):
            raise ValueError(
                f"row_weight must be ({len(data)},), got {tuple(row_weight.shape)}"
            )
        t = self.cfg.train
        steps = t.steps if steps is None else steps
        ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
        metrics: list[dict[str, float]] = []
        for _ in range(steps):
            idx = self._next_fit_index(len(data))
            weight = None if row_weight is None else row_weight[idx]
            metrics.append(self.train_step(data._batch(idx), weight))
            self._after_step(ckpt)
        if ckpt is not None:
            self.save_checkpoint(ckpt)
        return metrics

    # ---------------------------------------------- stateful (time-threaded)
    def train_step_sequence(
        self,
        chunk: Sequence[Batch],
        init_state: Tensor | Sequence[Tensor] | Sequence[FilterState] | None = None,
    ) -> tuple[dict[str, float], FilterHistory]:
        """One optimizer step on a chronological chunk of per-date batches.

        Threads the **attached** filtered posteriors across timesteps within
        the chunk (backprop through the forward recursion); the prior at each
        date reads the posterior ``h`` dates back (:meth:`_lagged_context`).
        Returns the last ``h`` posteriors, detached, for the caller to carry
        into the next chunk (truncated BPTT over the regime posterior).
        """
        self._train_mode()
        self._apply_sigma_schedule()
        history = _as_history(init_state)
        per_sample: list[Tensor] = []
        penalties: list[Tensor] = []
        last: tuple[NECOutput, MixtureNLLOutput] | None = None
        for batch in chunk:
            out, nll_out = self._forward_nll(
                batch, self._lagged_context(history, batch.entity)
            )
            per_sample.append(nll_out.per_sample_nll)
            penalties.append(self._correction_penalty(out))
            # attached: the recursion's state, keyed by entity
            history.append(FilterState(batch.entity, nll_out.log_filtered))
            last = (out, nll_out)
        assert last is not None
        # mean over the chunk's dates, matching the per-sample NLL's scale so
        # alpha means the same thing on both execution paths
        loss = torch.cat(per_sample).mean() + torch.stack(penalties).mean()
        self._optimize(loss)
        self.lb_buffer.update(last[1].responsibilities)
        h = self.cfg.data.horizon_periods
        carried = [st.detach() for st in history[-h:]]
        return self._metrics(loss, last[1], last[0]), carried

    def fit_sequence(
        self,
        sequence: Sequence[Batch],
        steps: int | None = None,
        chunk_len: int = 50,
        checkpoint_path: str | Path | None = None,
    ) -> list[dict[str, float]]:
        """Chunked training over one chronological sequence (stateful prior).

        The filter state is detached at chunk boundaries and reset at each
        pass over the sequence start. The chunk cursor and carried state live
        on the instance, so — like :meth:`fit` — ``steps`` are *additional*
        steps and a checkpointed run resumes mid-pass exactly. Resuming
        requires the same ``sequence`` and ``chunk_len`` (validated against
        the chunking recorded at first call).
        """
        t = self.cfg.train
        steps = t.steps if steps is None else steps
        chunks = [
            sequence[i : i + chunk_len] for i in range(0, len(sequence), chunk_len)
        ]
        meta = {"chunk_len": chunk_len, "n_chunks": len(chunks)}
        if self._seq_meta is not None and self._seq_meta != meta:
            raise ValueError(
                f"fit_sequence chunking mismatch: trained/checkpointed with "
                f"{self._seq_meta}, this call gives {meta} — resume with the "
                "same sequence and chunk_len"
            )
        self._seq_meta = meta
        ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
        metrics: list[dict[str, float]] = []
        for _ in range(steps):
            if self._seq_ci == 0:
                self._seq_state = None  # sequence start: back to pi_0
            m, state = self.train_step_sequence(
                chunks[self._seq_ci], init_state=self._seq_state
            )
            metrics.append(m)
            self._seq_state = state
            self._seq_ci = (self._seq_ci + 1) % len(chunks)
            self._after_step(ckpt)
        if ckpt is not None:
            self.save_checkpoint(ckpt)
        return metrics

    # ------------------------------------------------------- checkpointing
    def save(self, path: str | Path) -> None:
        """Atomically persist the complete training state (see module docs).

        The checkpoint is self-contained on the *model* side (it embeds the
        config, so :meth:`load` rebuilds everything) but deliberately does not
        store the data — resuming must supply the same panel/sequence.
        """
        payload = {
            "format_version": CHECKPOINT_FORMAT,
            "config": self.cfg.to_dict(),
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "step_count": self.step_count,
            "history": self.history,
            "lb_state": self.lb_buffer.get_state(),
            "fit_state": None
            if self._fit_gen is None
            else {
                "generator": self._fit_gen.get_state(),
                "perm": self._fit_perm,
                "pos": self._fit_pos,
            },
            "seq_state": None
            if self._seq_meta is None
            else {
                "ci": self._seq_ci,
                "carried": self._seq_state,
                "meta": self._seq_meta,
            },
            # dropout draws from the global RNG — without this, a resumed
            # trajectory silently diverges from the uninterrupted one
            "torch_rng": torch.get_rng_state(),
            # so a fold resumed with zero remaining steps still reports
            # live_param_count (the audit only reruns on a backward pass)
            "grad_audit": self.grad_audit,
        }
        atomic_torch_save(payload, path)

    @staticmethod
    def checkpoint_paths(path: str | Path, keep: int | None = None) -> list[Path]:
        """``path`` and its rotated older copies ``<path>.1``, ``<path>.2``, ...,
        newest first (all that exist when ``keep`` is None)."""
        path = Path(path)
        older = sorted(
            (p for p in path.parent.glob(path.name + ".*") if p.suffix[1:].isdigit()),
            key=lambda p: int(p.suffix[1:]),
        )
        found = ([path] if path.exists() else []) + older
        return found if keep is None else found[:keep]

    @classmethod
    def checkpoint_exists(cls, path: str | Path) -> bool:
        return bool(cls.checkpoint_paths(path))

    @classmethod
    def remove_checkpoints(cls, path: str | Path) -> None:
        for p in cls.checkpoint_paths(path):
            p.unlink(missing_ok=True)

    def save_checkpoint(self, path: str | Path) -> None:
        """:meth:`save` with rotation: the newest checkpoint is at ``path``, the
        ``checkpoint_keep - 1`` before it at ``<path>.1``, ... (brief 07 C.4).

        The new state is written to a temporary file first, the older copies
        are shifted, and only then is it renamed into place, so a crash at any
        point leaves at least one complete checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keep = self.cfg.train.checkpoint_keep
        tmp = path.with_name(path.name + ".new")
        self.save(tmp)
        for i in range(keep - 1, 0, -1):
            src = path if i == 1 else path.with_name(f"{path.name}.{i - 1}")
            if src.exists():
                os.replace(src, path.with_name(f"{path.name}.{i}"))
        for stale in self.checkpoint_paths(path)[keep:]:
            stale.unlink(missing_ok=True)
        os.replace(tmp, path)

    @classmethod
    def load_latest(cls, path: str | Path) -> Trainer:
        """Load the newest checkpoint of ``path`` that loads cleanly.

        Reports when an older one had to be used (the newest was truncated or
        corrupt, e.g. by a crash during the write); raises if none loads.
        """
        candidates = cls.checkpoint_paths(path)
        if not candidates:
            raise FileNotFoundError(f"no checkpoint at {path} or its rotated copies")
        errors = []
        for candidate in candidates:
            try:
                trainer = cls.load(candidate)
            except Exception as exc:  # a truncated file: try the previous one
                errors.append(f"{candidate.name}: {type(exc).__name__}")
                continue
            if errors:
                print(f"[resume] newest checkpoint(s) unreadable ({'; '.join(errors)}); "
                      f"resumed from the older {candidate.name} at step "
                      f"{trainer.step_count}")
            return trainer
        raise RuntimeError(f"no checkpoint of {path} loads: {'; '.join(errors)}")

    @classmethod
    def load(cls, path: str | Path) -> Trainer:
        """Rebuild a :meth:`save`d trainer; resume-exact given the same data.

        Restores the global torch RNG state as saved — call sites that need a
        different RNG stream afterwards must reseed themselves.
        """
        payload = torch.load(path, weights_only=False)
        version = payload.get("format_version")
        if version != CHECKPOINT_FORMAT:
            raise ValueError(
                f"checkpoint {path} has format {version!r}, expected "
                f"{CHECKPOINT_FORMAT}"
            )
        cfg = NECConfig.from_dict(payload["config"])
        model = NECModel(cfg)
        model.load_state_dict(payload["model"])
        trainer = cls(model)
        trainer.opt.load_state_dict(payload["optimizer"])
        trainer.step_count = payload["step_count"]
        trainer.history = payload["history"]
        trainer.lb_buffer.set_state(payload["lb_state"])
        # .get: checkpoints written before the audit existed still load, and
        # simply re-audit on their next backward pass
        trainer.grad_audit = payload.get("grad_audit")
        if payload["fit_state"] is not None:
            gen = torch.Generator()
            gen.set_state(payload["fit_state"]["generator"])
            trainer._fit_gen = gen
            trainer._fit_perm = payload["fit_state"]["perm"]
            trainer._fit_pos = payload["fit_state"]["pos"]
        if payload["seq_state"] is not None:
            trainer._seq_ci = payload["seq_state"]["ci"]
            carried = payload["seq_state"]["carried"]
            # checkpoints written before audit M-1 carried one bare tensor
            trainer._seq_state = None if carried is None else _as_history(carried)
            trainer._seq_meta = payload["seq_state"]["meta"]
        trainer._apply_sigma_schedule()
        # last, so the model rebuild's own init draws don't leak into the
        # resumed stream
        torch.set_rng_state(payload["torch_rng"])
        return trainer

    # -------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self, batch: Batch) -> float:
        """Held-out mean NLL under the prior's *predictive* weights, memoryless."""
        self.model.eval()
        _, nll_out = self._forward_nll(batch, train_objective=False)
        return float(nll_out.nll)

    @torch.no_grad()
    def evaluate_sequence(
        self,
        sequence: Sequence[Batch],
        init_state: Tensor | Sequence[Tensor] | Sequence[FilterState] | None = None,
    ) -> SequenceEval:
        """No-grad filtering pass over a chronological sequence.

        Strictly causal: the prior (and ``y_hat``) at date ``t`` are computed
        from the posterior ``h`` dates back (:meth:`_lagged_context`), whose
        targets are all realised by ``t``; the date's own target enters only
        its *update*. ``init_state`` is the history to continue from, e.g. the
        training block's ``log_filtered`` stack.
        """
        self.model.eval()
        history = _as_history(init_state)
        nlls, filt, priors, preds = [], [], [], []
        for batch in sequence:
            out, nll_out = self._forward_nll(
                batch, self._lagged_context(history, batch.entity), train_objective=False
            )
            nlls.append(nll_out.per_sample_nll)
            filt.append(nll_out.log_filtered)
            priors.append(out.prior.log_prior)
            preds.append(out.y_hat)
            history.append(FilterState(batch.entity, nll_out.log_filtered))
        balanced = len({f.shape[0] for f in filt}) == 1

        def stack(xs: list[Tensor]) -> Tensor | list[Tensor]:
            return torch.stack(xs) if balanced else xs

        h = self.cfg.data.horizon_periods
        return SequenceEval(
            nll=float(torch.cat(nlls).mean()),
            log_filtered=stack(filt),
            log_prior=stack(priors),
            y_hat=stack(preds),
            final_state=history[-h:],
        )
