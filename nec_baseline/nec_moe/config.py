"""Configuration tree for the corrected NEC baseline.

One frozen dataclass tree, no scattered constructor defaults (design doc §5).
``NECConfig.validate()`` cross-checks every derived dimension at construction
time so that a defect-8-style mismatch (the prototype's encoder n=256 feeding a
head that expected 64) is a loud ``ValueError`` at build time, not silent
garbage at runtime.

Conventions
-----------
- Modules take their sub-config, never loose ints: there is exactly one place a
  dimension can be written.
- The gate's input dim is *derived* from the encoder — it is deliberately not a
  settable field, which makes the prototype's mismatch unrepresentable.
- ``prior.kind == "hmm"`` requires ``train.sequence_ordered`` (the forward
  filter needs chronological, entity-contiguous batches).
- ``prior.tvtp`` (covariate-dependent transitions) is a guarded-off config
  surface: selecting it raises ``NotImplementedError`` with a pointer (design
  doc §9bis; arXiv:2605.14976 identifiability caution).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "DataConfig",
    "EncoderConfig",
    "GateConfig",
    "ExpertConfig",
    "PriorConfig",
    "BaseConfig",
    "TrainConfig",
    "NECConfig",
    "pyramid_dims",
]

ExpertInputMode = Literal["snapshot", "snapshot_plus_hidden"]
EmissionKind = Literal["mlp", "classical"]


def pyramid_dims(first_width: int, depth: int) -> tuple[int, ...]:
    """Geometric-pyramid hidden widths: ``(w, w/2, w/4, ...)``, ``depth`` long.

    The rule Gu, Kelly and Xiu (2020) use for their benchmark networks on a
    characteristic panel (NN1--NN5 are 32; 32,16; 32,16,8; ... after Masters
    1993), and the convenience form of Q8's settled architecture choice.
    Widths floor at 1, so a deep pyramid from a narrow start degrades
    gracefully rather than producing zero-width layers.

    This is a *helper for writing* ``hidden_dims``; the explicit tuple stays
    the authoritative config field, so any non-pyramid shape remains
    expressible (a depth sweep is a sweep over tuple lengths).
    """
    if first_width < 1 or depth < 1:
        raise ValueError(
            f"pyramid_dims needs first_width >= 1 and depth >= 1, got "
            f"{first_width}/{depth}"
        )
    return tuple(max(first_width // (2**i), 1) for i in range(depth))


@dataclass(frozen=True)
class DataConfig:
    """Tensor dimensions and (optional) feature names of the data contract (§6).

    ``sequence_features`` / ``snapshot_features`` are optional documentation of
    channel order; when provided their lengths must match ``d_seq`` / ``d_snap``.
    """

    d_seq: int
    d_snap: int
    seq_len: int
    sequence_features: tuple[str, ...] = ()
    snapshot_features: tuple[str, ...] = ()
    target: str = "fwd_return"


@dataclass(frozen=True)
class EncoderConfig:
    """GRU encoder (ledger defect 5 fix: small, shallow)."""

    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.0  # inter-layer only; requires num_layers >= 2


@dataclass(frozen=True)
class GateConfig:
    """Gate head. Input dim is derived from the encoder — not settable here."""

    zero_init: bool = True  # start exactly uniform: entropy log K at step 0


@dataclass(frozen=True)
class ExpertConfig:
    """Emission model. ``input_mode`` is Decision A's flag (default: snapshot-only).

    ``kind`` selects the emission along the design's second plug axis (§7):
    ``"mlp"`` — the baseline's neural expert bank; ``"classical"`` — per-regime
    constant Gaussians (learned ``mu_k, sigma_k`` scalars, features ignored).
    ``classical`` × ``prior.kind="hmm"`` is the classical Hamilton /
    Gaussian-HMM baseline from the same skeleton. ``hidden_dims``, ``dropout``,
    ``activation``, ``input_mode`` and the correction fields are ignored by
    ``classical``.

    ``hidden_dims`` carries depth *and* width in one field (default
    ``(64, 32)`` — the two-hidden-layer block of Ye & Borde, arXiv:2608.12251,
    and the geometric pyramid of Gu--Kelly--Xiu); :func:`pyramid_dims` builds
    one from a width and a depth. It replaces the former scalar ``hidden_dim``
    outright rather than living alongside it, so there is exactly one place a
    layer shape can be written.

    Correction (residual) fields, brief 02:

    - ``correction_mode``: experts emit **corrections to a frozen base**
      rather than full forecasts, giving
      ``y_hat = f0(x) + sum_k pi_k r_k(x)``. Requires ``base.enabled``.
    - ``zero_init_head``: zero the experts' final layer so the correction is
      identically zero at step 0 and the model starts *exactly* at the base
      (Ye & Borde's zero-initialization; their ablation shows random init
      costs IC and destabilizes training). Only meaningful with
      ``correction_mode``.
    """

    n_experts: int = 2
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.05
    activation: str = "relu"  # registry key, never a hardcoded nn module
    input_mode: ExpertInputMode = "snapshot"
    kind: EmissionKind = "mlp"
    correction_mode: bool = False
    zero_init_head: bool = True


@dataclass(frozen=True)
class PriorConfig:
    """RegimePrior selection (§7). ``kind`` is a registry key.

    Registered kinds: ``soft`` (baseline), ``uniform`` (frozen-uniform-gate
    ablation), ``hard`` (top-1, hard-EM objective), ``topk``, ``gumbel``
    (memoryless family — Variation 2's comparison grid), and ``hmm``
    (recursive — Variation 3). Kind-specific fields are ignored by the others.

    HMM-only fields:

    - ``transition_diag_bias``: added to the diagonal of the transition logit
      matrix at init, so training *starts* sticky (persistence-biased — the
      transition-matrix analogue of the LSTM forget-gate bias trick).
    - ``learn_pi0``: learn the initial regime distribution's logits (else fixed
      uniform).
    - ``tvtp``: covariate-dependent transitions — guarded off (§9bis).

    Top-k-only: ``top_k`` (number of experts kept, **>= 2**: with a single
    kept entry the renormalized weight is constant 1 and the gate is
    gradient-dead — exactly ledger defect 1; use ``kind="hard"`` for trainable
    top-1 routing. ``== n_experts`` reduces to soft). Gumbel-only:
    ``tau_init``/``tau_min``/``tau_anneal_steps`` — exponential temperature
    anneal from ``tau_init`` to ``tau_min`` over ``tau_anneal_steps`` training
    steps (0 = constant ``tau_init``).
    """

    kind: str = "soft"
    transition_diag_bias: float = 2.0
    learn_pi0: bool = True
    tvtp: bool = False
    top_k: int = 2
    tau_init: float = 1.0
    tau_min: float = 0.1
    tau_anneal_steps: int = 0


@dataclass(frozen=True)
class BaseConfig:
    """The frozen base predictor ``f0`` of the residual design (brief 02).

    ``f0`` is an MLP on the characteristic snapshot, fitted on **each fold's
    training block** (with a validation tail cut from that block only) and then
    frozen — ``requires_grad_(False)`` and ``eval()``, so dropout and any
    normalization statistics stop moving. The experts then learn a
    regime-conditional *correction* on top of it.

    Why frozen rather than jointly optimized (Q7, Q19): the measured quantity
    is what a regime-conditional correction adds to a **fixed** baseline. A
    jointly trained base would co-adapt with the experts and any improvement
    could no longer be attributed to regime structure rather than to the base
    quietly reorganizing around the mixture. The deliberate contrast is
    DeepSeekMoE (arXiv:2401.06066), whose always-on *shared expert* is
    optimized jointly with the routed experts: that configuration is the one
    most likely to produce the best fitted objective, and it is out of scope
    here on purpose. Attribution is being bought with performance.

    Every value below is a config field because none of them is chosen yet;
    they are set by experiment and swept from ``run_experiment.py``.

    ``hidden_dims`` carries depth and width together (see
    :func:`pyramid_dims`). ``val_fraction`` is the *tail* of the training
    block held out for early stopping — never out-of-sample data. The base
    seed is the run seed plus ``seed_offset``, so a base can be re-seeded
    independently of the experts.
    """

    enabled: bool = False  # residual mode off: existing behaviour preserved
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.0
    activation: str = "relu"  # registry key, never a hardcoded nn module
    lr: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 1000
    batch_size: int = 128
    early_stopping_patience: int | None = None  # None = train the full budget
    val_fraction: float = 0.2  # tail of the TRAINING block only
    seed_offset: int = 0


@dataclass(frozen=True)
class TrainConfig:
    """Optimization, regularization, aux-loss gating (§4.3).

    Both aux losses are implemented-with-tests but **default off** (baseline
    objective purity; each is a pre-registered thesis ablation).
    """

    lr: float = 1e-3
    weight_decay: float = 1e-4
    gate_weight_decay: float = 1e-3  # structural: separation-proofing (M4)
    batch_size: int = 128
    steps: int = 500
    grad_clip: float | None = 5.0
    sigma_init: float = 1.0  # set near std(y) when known
    sigma_freeze_steps: int = 100  # sigma warm-start freeze
    aux_load_balance: bool = False
    aux_load_balance_weight: float = 1e-2
    load_balance_buffer_batches: int = 32  # MUST span many dates (arXiv:2501.11873)
    aux_expert_decorrelation: bool = False
    aux_decorrelation_weight: float = 1e-2
    # Correction-magnitude penalty (brief 02 §3): alpha * mean_b (sum_k pi_k
    # r_k)^2, shrinking the mixture toward the base. Ye & Borde use this form
    # but do **not** report their alpha, which is why the weight is a config
    # field selected on the training block and logged as a trial, and why a
    # sensitivity curve against it is part of the reporting contract.
    aux_correction_penalty: bool = False
    correction_penalty_weight: float = 0.0  # value unknown; set by experiment
    # Freeze the regime path (encoder, gate head, prior parameters) so only
    # the experts and their noise scales train (Q19). The gate is then an
    # input the experts cannot influence, which is what makes "what did the
    # regime structure add?" attributable.
    freeze_gate: bool = False
    seed: int = 0
    sequence_ordered: bool = False  # True required for the HMM prior
    log_every: int = 50
    # Checkpoint cadence for Trainer.fit/fit_sequence when a checkpoint_path is
    # given: save every N optimizer steps (plus always once at the end of the
    # call). 0 = periodic saves off (final save still happens).
    checkpoint_every: int = 0


@dataclass(frozen=True)
class NECConfig:
    """Root config. Construct, then ``validate()`` (NECModel does it for you)."""

    data: DataConfig
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    experts: ExpertConfig = field(default_factory=ExpertConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    base: BaseConfig = field(default_factory=BaseConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ------------------------------------------------------------------ dims
    @property
    def expert_input_dim(self) -> int:
        """Derived expert input width (Decision A)."""
        if self.experts.input_mode == "snapshot":
            return self.data.d_snap
        return self.data.d_snap + self.encoder.hidden_dim

    # ------------------------------------------------------------ validation
    def validate(self) -> NECConfig:
        d, e, x, p, t = self.data, self.encoder, self.experts, self.prior, self.train
        b = self.base

        def bad(msg: str) -> ValueError:
            return ValueError(f"NECConfig invalid: {msg}")

        # data
        if d.d_seq < 1 or d.d_snap < 1 or d.seq_len < 1:
            raise bad(
                f"dims must be >= 1, got d_seq={d.d_seq}, d_snap={d.d_snap}, "
                f"seq_len={d.seq_len}"
            )
        if d.sequence_features and len(d.sequence_features) != d.d_seq:
            raise bad(
                f"len(sequence_features)={len(d.sequence_features)} != d_seq={d.d_seq}"
            )
        if d.snapshot_features and len(d.snapshot_features) != d.d_snap:
            raise bad(
                f"len(snapshot_features)={len(d.snapshot_features)} != d_snap={d.d_snap}"
            )

        # encoder
        if e.hidden_dim < 1 or e.num_layers < 1:
            raise bad(
                f"encoder hidden_dim/num_layers must be >= 1, got "
                f"{e.hidden_dim}/{e.num_layers}"
            )
        if not 0.0 <= e.dropout < 1.0:
            raise bad(f"encoder dropout must be in [0, 1), got {e.dropout}")
        if e.dropout > 0.0 and e.num_layers < 2:
            raise bad(
                f"encoder dropout={e.dropout} is inter-layer only and requires "
                f"num_layers >= 2 (got {e.num_layers})"
            )

        # experts
        if x.n_experts < 2:
            raise bad(f"n_experts must be >= 2 (a mixture), got {x.n_experts}")
        if not x.hidden_dims or any(h < 1 for h in x.hidden_dims):
            raise bad(
                f"experts hidden_dims must be a non-empty tuple of widths >= 1, "
                f"got {x.hidden_dims}"
            )
        if not 0.0 <= x.dropout < 1.0:
            raise bad(f"experts dropout must be in [0, 1), got {x.dropout}")
        from .networks import ACTIVATION_REGISTRY  # late import: avoid cycle

        if x.activation not in ACTIVATION_REGISTRY:
            raise bad(
                f"unknown experts.activation {x.activation!r}; registered: "
                f"{sorted(ACTIVATION_REGISTRY)}"
            )
        if x.input_mode not in ("snapshot", "snapshot_plus_hidden"):
            raise bad(f"unknown experts.input_mode {x.input_mode!r}")
        from .experts import EMISSION_REGISTRY  # late import: avoid cycle

        if x.kind not in EMISSION_REGISTRY:
            raise bad(
                f"unknown experts.kind {x.kind!r}; registered: "
                f"{sorted(EMISSION_REGISTRY)}"
            )

        # prior
        from .priors import PRIOR_REGISTRY  # late import: avoid cycle

        if p.kind not in PRIOR_REGISTRY:
            raise bad(
                f"unknown prior.kind {p.kind!r}; registered: "
                f"{sorted(PRIOR_REGISTRY)}"
            )
        if p.kind == "hmm" and not t.sequence_ordered:
            raise bad(
                "prior.kind='hmm' requires train.sequence_ordered=True: the "
                "forward filter threads the previous filtered posterior, so "
                "batches must be chronological and entity-contiguous"
            )
        if p.tvtp:
            raise NotImplementedError(
                "PriorConfig.tvtp (covariate-dependent transitions) is a "
                "guarded-off extension — see design doc §9bis and the "
                "arXiv:2605.14976 identifiability caution. Start static."
            )
        if p.kind == "topk" and not 2 <= p.top_k <= x.n_experts:
            raise bad(
                f"prior.top_k must be in [2, n_experts={x.n_experts}], got "
                f"{p.top_k}: a single renormalized entry is constant 1 and the "
                "gate is gradient-dead (ledger defect 1) — use kind='hard' for "
                "trainable top-1 routing; top_k == n_experts reduces to soft"
            )
        if p.kind == "gumbel":
            if p.tau_init <= 0 or p.tau_min <= 0 or p.tau_min > p.tau_init:
                raise bad(
                    f"gumbel temperatures need 0 < tau_min <= tau_init, got "
                    f"tau_init={p.tau_init}, tau_min={p.tau_min}"
                )
            if p.tau_anneal_steps < 0:
                raise bad(f"tau_anneal_steps must be >= 0, got {p.tau_anneal_steps}")

        # train
        if t.lr <= 0 or t.batch_size < 1 or t.steps < 0:
            raise bad(
                f"train lr/batch_size/steps invalid: {t.lr}/{t.batch_size}/{t.steps}"
            )
        if t.sigma_init <= 0:
            raise bad(f"sigma_init must be > 0, got {t.sigma_init}")
        if t.load_balance_buffer_batches < 1:
            raise bad(
                "load_balance_buffer_batches must be >= 1 (and should span many "
                f"dates), got {t.load_balance_buffer_batches}"
            )
        if t.checkpoint_every < 0:
            raise bad(f"checkpoint_every must be >= 0, got {t.checkpoint_every}")
        if t.correction_penalty_weight < 0:
            raise bad(
                f"correction_penalty_weight must be >= 0, got "
                f"{t.correction_penalty_weight}"
            )
        if t.aux_load_balance and t.freeze_gate:
            # Q19 consequence 3: with a frozen gate the mean gate probability
            # has no trainable parameter behind it, so the Shazeer term's
            # gradient is exactly zero. Silently optimizing an inert loss is
            # the kind of no-op this codebase refuses to ship.
            raise bad(
                "aux_load_balance is inert when train.freeze_gate=True: the "
                "load-balancing gradient flows only through the gate, which "
                "has no trainable parameters here, so the term is exactly "
                "zero. Disable one of the two (the contrast with Ye & Borde "
                "and Shazeer et al., where the gate trains and the term does "
                "work, is a methodological note, not a config)"
            )

        # base (brief 02)
        if not b.hidden_dims or any(h < 1 for h in b.hidden_dims):
            raise bad(
                f"base hidden_dims must be a non-empty tuple of widths >= 1, "
                f"got {b.hidden_dims}"
            )
        if not 0.0 <= b.dropout < 1.0:
            raise bad(f"base dropout must be in [0, 1), got {b.dropout}")
        if b.activation not in ACTIVATION_REGISTRY:
            raise bad(
                f"unknown base.activation {b.activation!r}; registered: "
                f"{sorted(ACTIVATION_REGISTRY)}"
            )
        if b.lr <= 0 or b.batch_size < 1 or b.steps < 0:
            raise bad(
                f"base lr/batch_size/steps invalid: {b.lr}/{b.batch_size}/{b.steps}"
            )
        if b.weight_decay < 0:
            raise bad(f"base weight_decay must be >= 0, got {b.weight_decay}")
        if not 0.0 < b.val_fraction < 1.0:
            raise bad(
                f"base val_fraction must be in (0, 1) — it is a tail of the "
                f"TRAINING block — got {b.val_fraction}"
            )
        if b.early_stopping_patience is not None and b.early_stopping_patience < 1:
            raise bad(
                f"base early_stopping_patience must be >= 1 or None, got "
                f"{b.early_stopping_patience}"
            )
        if x.correction_mode and not b.enabled:
            raise bad(
                "experts.correction_mode=True requires base.enabled=True: a "
                "correction is a correction *to* something, and with no base "
                "the experts' zero-initialized heads would predict a constant 0"
            )
        if x.correction_mode and x.kind != "mlp":
            raise bad(
                f"experts.correction_mode is defined for the neural expert "
                f"bank; experts.kind={x.kind!r} has no head to zero-initialize"
            )
        return self

    # --------------------------------------------------------- serialization
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NECConfig:
        def _tupled(sub: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
            return {
                k: tuple(v) if k in keys and v is not None else v
                for k, v in sub.items()
            }

        return cls(
            data=DataConfig(
                **_tupled(d["data"], ("sequence_features", "snapshot_features"))
            ),
            encoder=EncoderConfig(**d.get("encoder", {})),
            gate=GateConfig(**d.get("gate", {})),
            # hidden_dims round-trips through JSON as a list
            experts=ExpertConfig(**_tupled(d.get("experts", {}), ("hidden_dims",))),
            prior=PriorConfig(**d.get("prior", {})),
            base=BaseConfig(**_tupled(d.get("base", {}), ("hidden_dims",))),
            train=TrainConfig(**d.get("train", {})),
        )
