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
    "TrainConfig",
    "NECConfig",
]

ExpertInputMode = Literal["snapshot", "snapshot_plus_hidden"]
EmissionKind = Literal["mlp", "classical"]


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
    Gaussian-HMM baseline from the same skeleton. ``hidden_dim``, ``dropout``
    and ``input_mode`` are ignored by ``classical``.
    """

    n_experts: int = 2
    hidden_dim: int = 64
    dropout: float = 0.05
    input_mode: ExpertInputMode = "snapshot"
    kind: EmissionKind = "mlp"


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
        if x.hidden_dim < 2:
            raise bad(f"experts hidden_dim must be >= 2, got {x.hidden_dim}")
        if not 0.0 <= x.dropout < 1.0:
            raise bad(f"experts dropout must be in [0, 1), got {x.dropout}")
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
            experts=ExpertConfig(**d.get("experts", {})),
            prior=PriorConfig(**d.get("prior", {})),
            train=TrainConfig(**d.get("train", {})),
        )
