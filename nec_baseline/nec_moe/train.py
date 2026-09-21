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

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from .config import NECConfig
from .data import Batch, Panel
from .diagnostics import (
    expert_output_correlation,
    gate_entropy,
    persistence_metrics,
    sharpness,
    utilization,
)
from .likelihood import expert_log_likelihood
from .losses import (
    LoadBalanceBuffer,
    MixtureNLLOutput,
    correction_penalty_aux,
    expert_decorrelation_aux,
    load_balance_aux,
    mixture_nll,
)
from .model import NECModel, NECOutput
from .priors import PriorContext
from .utils import atomic_torch_save

__all__ = [
    "Trainer",
    "SequenceEval",
    "GradientAudit",
    "DeadParameterWarning",
]

#: Bumped when the checkpoint payload layout changes incompatibly.
CHECKPOINT_FORMAT = 1


class DeadParameterWarning(UserWarning):
    """Some parameters receive no gradient — see :class:`GradientAudit`."""


@dataclass
class SequenceEval:
    """Result of a no-grad filtering pass over a chronological sequence."""

    nll: float  # mean per-sample NLL over all timesteps
    log_filtered: Tensor  # (L, B, K) log-domain filtered posteriors
    log_prior: Tensor  # (L, B, K) log-domain priors (the predict step)
    y_hat: Tensor  # (L, B) causal point predictions


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
        # fit_sequence()'s cursor: which chunk is next + the carried
        # (detached) filter state, plus the chunking it was built under
        self._seq_ci: int = 0
        self._seq_state: Tensor | None = None
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
        out = self.model(batch.x_seq, batch.x_snap, ctx)
        log_lik = expert_log_likelihood(out.mu, out.log_sigma, batch.y)
        log_w = out.prior.log_prior
        if train_objective and out.prior.log_train_weights is not None:
            log_w = out.prior.log_train_weights
        return out, mixture_nll(log_w, log_lik)

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
            corr = expert_output_correlation(out.mu.detach())
            off = corr - torch.eye(corr.shape[0])
            m = {
                "step": float(self.step_count),
                "loss": float(loss),
                "nll": float(nll_out.nll),
                "gate_entropy": float(gate_entropy(probs)),
                "sharpness": float(sharpness(r)),
                "min_batch_utilization": float(utilization(r).min()),
                "min_running_utilization": float(self.lb_buffer.fractions().min()),
                "max_offdiag_expert_corr": float(off.abs().max()),
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
    def train_step(self, batch: Batch) -> dict[str, float]:
        """One optimizer step on one cross-sectional batch (memoryless prior)."""
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
        loss = nll_out.nll

        # buffer always updated: it feeds the running-utilization dashboard
        # even when the aux loss is off
        self.lb_buffer.update(nll_out.responsibilities)
        if t.aux_load_balance:
            loss = loss + t.aux_load_balance_weight * load_balance_aux(
                out.prior.log_prior.exp(), self.lb_buffer.fractions()
            )
        if t.aux_expert_decorrelation:
            loss = loss + t.aux_decorrelation_weight * expert_decorrelation_aux(out.mu)
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

    def _next_fit_batch(self, data: Panel) -> Batch:
        """The next shuffled minibatch of the seeded epoch stream.

        Identical stream to chaining ``data.minibatches(batch_size, g)`` epoch
        after epoch — but the generator, the current permutation, and the
        position within it live on the instance, so a checkpoint can freeze
        the stream mid-epoch and resume it exactly.
        """
        t = self.cfg.train
        if self._fit_gen is None:
            self._fit_gen = torch.Generator().manual_seed(t.seed)
        if self._fit_perm is None or self._fit_pos >= len(self._fit_perm):
            self._fit_perm = torch.randperm(len(data), generator=self._fit_gen)
            self._fit_pos = 0
        idx = self._fit_perm[self._fit_pos : self._fit_pos + t.batch_size]
        self._fit_pos += t.batch_size
        return data._batch(idx)

    def _maybe_checkpoint(self, path: Path | None) -> None:
        if (
            path is not None
            and self.cfg.train.checkpoint_every > 0
            and self.step_count % self.cfg.train.checkpoint_every == 0
        ):
            self.save(path)

    def fit(
        self,
        data: Panel,
        steps: int | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> list[dict[str, float]]:
        """Minibatch training loop over a panel (memoryless prior).

        ``steps`` are *additional* steps for this call — the sample stream and
        ``step_count`` continue across calls, so ``fit(60)`` then ``fit(40)``
        (possibly via a checkpoint reload in between) equals one ``fit(100)``.
        With ``checkpoint_path`` set, saves every
        ``TrainConfig.checkpoint_every`` steps and once at the end; resuming
        requires the *same* panel (the checkpoint stores the sampler, not the
        data).
        """
        t = self.cfg.train
        steps = t.steps if steps is None else steps
        ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
        metrics: list[dict[str, float]] = []
        for _ in range(steps):
            metrics.append(self.train_step(self._next_fit_batch(data)))
            self._maybe_checkpoint(ckpt)
        if ckpt is not None:
            self.save(ckpt)
        return metrics

    # ---------------------------------------------- stateful (time-threaded)
    def train_step_sequence(
        self, chunk: Sequence[Batch], init_state: Tensor | None = None
    ) -> tuple[dict[str, float], Tensor]:
        """One optimizer step on a chronological chunk of per-date batches.

        Threads the **attached** filtered posterior across timesteps within the
        chunk (backprop through the forward recursion) and returns the final
        state for the caller to detach and carry into the next chunk
        (truncated BPTT over the regime posterior).
        """
        self._train_mode()
        self._apply_sigma_schedule()
        state = init_state
        per_sample: list[Tensor] = []
        penalties: list[Tensor] = []
        last: tuple[NECOutput, MixtureNLLOutput] | None = None
        for batch in chunk:
            ctx = PriorContext(prev_filtered=state, step=self.step_count)
            out, nll_out = self._forward_nll(batch, ctx)
            per_sample.append(nll_out.per_sample_nll)
            penalties.append(self._correction_penalty(out))
            state = nll_out.log_filtered  # attached: the recursion's state
            last = (out, nll_out)
        assert last is not None and state is not None
        # mean over the chunk's dates, matching the per-sample NLL's scale so
        # alpha means the same thing on both execution paths
        loss = torch.cat(per_sample).mean() + torch.stack(penalties).mean()
        self._optimize(loss)
        self.lb_buffer.update(last[1].responsibilities)
        return self._metrics(loss, last[1], last[0]), state.detach()

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
            self._maybe_checkpoint(ckpt)
        if ckpt is not None:
            self.save(ckpt)
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
            trainer._seq_state = payload["seq_state"]["carried"]
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
        self, sequence: Sequence[Batch], init_state: Tensor | None = None
    ) -> SequenceEval:
        """No-grad filtering pass over a chronological sequence.

        Strictly causal: the prior (and ``y_hat``) at step t are computed
        before the loss sees ``y_t``; only the *update* uses ``y_t``.
        """
        self.model.eval()
        state = init_state
        nlls, filt, priors, preds = [], [], [], []
        for batch in sequence:
            ctx = PriorContext(prev_filtered=state)
            out, nll_out = self._forward_nll(batch, ctx, train_objective=False)
            nlls.append(nll_out.per_sample_nll)
            filt.append(nll_out.log_filtered)
            priors.append(out.prior.log_prior)
            preds.append(out.y_hat)
            state = nll_out.log_filtered
        return SequenceEval(
            nll=float(torch.cat(nlls).mean()),
            log_filtered=torch.stack(filt),
            log_prior=torch.stack(priors),
            y_hat=torch.stack(preds),
        )
