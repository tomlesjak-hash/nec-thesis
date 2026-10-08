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
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "ComputeEnvelope",
    "COMPUTE_ENVELOPE",
    "FOLD_SCHEMES",
    "FoldConfig",
    "DataConfig",
    "EncoderConfig",
    "GateConfig",
    "ExpertConfig",
    "PriorConfig",
    "MarkovGateConfig",
    "BaseConfig",
    "TrainConfig",
    "NECConfig",
    "GATE_WEIGHTS",
    "pyramid_dims",
    "target_horizon",
]

ExpertInputMode = Literal["snapshot", "snapshot_plus_hidden"]
EmissionKind = Literal["mlp", "classical"]
HiddenInit = Literal["diversified", "identical"]


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
class ComputeEnvelope:
    """The design's compute envelope (Tom's decision 2026-10-05; Q8 update,
    Progress_Tracker section 1c; brief 09 H.1): **documented defaults, not
    hard limits**.

    - ``depths``: the depth scan, each depth's widths from
      :func:`pyramid_dims` of one fixed first width (default 1, 2, 3);
    - ``max_first_width``: the first hidden width at most 128;
    - ``max_experts``: K at most 4.

    Measured on the M4 Pro, the whole design inside this envelope takes about
    2-6 days of machine time with 10 parallel processes; depths 4-5 or a first
    width of 256 take 1-1.5 months. It says where the design is affordable,
    not which K or first width to use: those are Q18 and Q8/Q17, open, and the
    code keeps its current defaults (K = 2, widths (64, 32)).
    :meth:`notes` lists what lies outside, for a campaign to print; nothing
    refuses.
    """

    depths: tuple[int, ...] = (1, 2, 3)
    max_first_width: int = 128
    max_experts: int = 4

    def notes(self, *, first_width: int, depths: tuple[int, ...] | list[int],
              n_experts: int) -> list[str]:
        out = []
        if first_width > self.max_first_width:
            out.append(f"first width {first_width} > {self.max_first_width}")
        extra = sorted(set(depths) - set(self.depths))
        if extra:
            out.append(f"depth(s) {extra} outside the scan {self.depths}")
        if n_experts > self.max_experts:
            out.append(f"K = {n_experts} > {self.max_experts}")
        return out


#: The envelope's documented defaults (``ComputeEnvelope``).
COMPUTE_ENVELOPE = ComputeEnvelope()


_HORIZON_IN_TARGET = re.compile(r"_(\d+)d$")

#: The gate weights a fitted gate with a transition matrix can serve (Q27).
GATE_WEIGHTS: tuple[str, ...] = ("filtered", "predicted", "window")


def target_horizon(target: str) -> int | None:
    """The forward horizon a target's name declares, in periods, or ``None``.

    The real panels name their targets ``fwd_mn_ret_{h}d`` (the default,
    market-neutral), ``fwd_ret_{h}d`` and ``fwd_resid_ret_{h}d``
    (:class:`nec_moe.features.StageBSpec`), so the horizon travels with the
    target. A target over ``(t, t+h]`` is realised
    only at ``t+h``, which decides what may condition on it (audit M-1), and
    consecutive targets overlap by ``h-1`` periods, which decides how its
    daily statistics must be tested (audit E-2).
    """
    m = _HORIZON_IN_TARGET.search(target)
    return int(m.group(1)) if m else None


#: How the walk-forward evaluation splits its dates (brief 09 B).
FOLD_SCHEMES: tuple[str, ...] = ("calendar_year", "count")


@dataclass(frozen=True)
class FoldConfig:
    """The evaluation protocol's folds (Q16, decided 2026-10-05; brief 09 B).

    ``fold_scheme`` (default ``"calendar_year"``):

    - ``"calendar_year"``: one **main** fold per test year ``Y`` from
      ``first_test_year`` to ``last_test_year`` (2010-2024, 15 folds). Each
      trains on every date from the sample start to the last trading day of
      ``Y - 1``, minus the last ``purge`` dates (the target horizon, read from
      the panel), and tests on every trading day of ``Y``: an annual refit on
      an expanding window (Gu, Kelly & Xiu 2020). The **pilot** folds use
      the same rule with each of ``pilot_validation_years`` (2007, 2008,
      2009) as the validation year; the pilot never touches a date at or
      after ``first_test_year`` (:func:`nec_moe.evaluation.pilot_slice`).
    - ``"count"``: the earlier count-based split
      (:func:`nec_moe.evaluation.walk_forward_folds`): the last ``n_folds *
      test_dates_per_fold`` dates as consecutive test blocks. Kept for
      development and for panels without a calendar.
    """

    fold_scheme: Literal["calendar_year", "count"] = "calendar_year"
    first_test_year: int = 2010
    last_test_year: int = 2024
    pilot_validation_years: tuple[int, ...] = (2007, 2008, 2009)

    def validate(self) -> FoldConfig:
        if self.fold_scheme not in FOLD_SCHEMES:
            raise ValueError(
                f"unknown fold_scheme {self.fold_scheme!r}; use one of {FOLD_SCHEMES}"
            )
        if self.last_test_year < self.first_test_year:
            raise ValueError(
                f"last_test_year={self.last_test_year} is before "
                f"first_test_year={self.first_test_year}"
            )
        years = self.pilot_validation_years
        if not years or len(set(years)) != len(years) or list(years) != sorted(years):
            raise ValueError(
                f"pilot_validation_years must be distinct and increasing, got {years}"
            )
        if max(years) >= self.first_test_year:
            raise ValueError(
                f"pilot validation year {max(years)} is not before first_test_year="
                f"{self.first_test_year}: the pilot may only see pre-test data"
            )
        return self

    @property
    def test_years(self) -> tuple[int, ...]:
        return tuple(range(self.first_test_year, self.last_test_year + 1))


@dataclass(frozen=True)
class DataConfig:
    """Tensor dimensions and (optional) feature names of the data contract (§6).

    ``sequence_features`` / ``snapshot_features`` are optional documentation of
    channel order; when provided their lengths must match ``d_seq`` / ``d_snap``.

    ``horizon`` is the target's forward horizon in periods. ``None`` (the
    default) reads it from the target's name (``fwd_ret_5d`` gives 5) and
    falls back to 1 when the name declares none, as for the synthetic
    panels. See :attr:`horizon_periods`.
    """

    d_seq: int
    d_snap: int
    seq_len: int
    sequence_features: tuple[str, ...] = ()
    snapshot_features: tuple[str, ...] = ()
    target: str = "fwd_return"
    horizon: int | None = None

    @property
    def horizon_periods(self) -> int:
        """The effective forward horizon: explicit, else from the name, else 1."""
        if self.horizon is not None:
            return self.horizon
        parsed = target_horizon(self.target)
        return parsed if parsed is not None else 1


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

    ``hidden_init`` (brief 06 D, audit X-1) sets how the experts start:

    - ``"diversified"``: expert k's layers are re-drawn from its own seed,
      ``train.seed + 7919 (k + 1)``, so the experts start different (today's
      behaviour, kept so existing results reproduce);
    - ``"identical"``: every expert gets expert 0's draw, so all experts
      share their hidden-layer weights, and they share the initial
      ``log_sigma`` (``sigma_init``). Only the gate can then pull them apart.

    Heads stay zero in both under ``zero_init_head``. **The default is not a
    decision**: which start the thesis uses is open, and every trial records
    the setting. Under ``"identical"`` with ``dropout > 0`` the experts still
    diverge under a uniform gate, because each expert's dropout layer draws
    its own mask; the clean "no regime information" control needs
    ``dropout = 0``. A warm start (``Trainer.warmstart_experts``) breaks the
    symmetry on purpose and is a separate choice.
    """

    n_experts: int = 2
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.05
    activation: str = "relu"  # registry key, never a hardcoded nn module
    input_mode: ExpertInputMode = "snapshot"
    kind: EmissionKind = "mlp"
    correction_mode: bool = False
    zero_init_head: bool = True
    hidden_init: HiddenInit = "diversified"


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
class MarkovGateConfig:
    """The Hamilton (Markov switching) gate, fitted by ML and frozen (brief 03).

    Hamilton's model is **univariate**: it needs one series, and the gate is a
    date-level variable, so the series is market-level and never the
    cross-section. ``series`` is a registry key rather than a hardcoded
    choice (see :data:`nec_moe.markov_gate.SERIES_REGISTRY`); every field
    below is a config field because none of them is chosen yet.

    - ``series`` / ``series_feature`` / ``series_channel``: which date-level
      series to fit on. Must be knowable at date ``t`` — the panel's feature
      contract guarantees that for sequence features, and the target ``y`` is
      excluded by construction because it is a *forward* return. Default
      ``"crsp_market_log_return"`` (Q23, decided 2026-10-08; brief 10 B): the
      CRSP S&P 500 index's daily total return as a log return, in decimals;
      French Mkt-RF (``"market_excess_return"``, also decimals) stays
      registered for the robustness check. ``crsp_dir``: where the CRSP
      extracts live, read only for a panel built before brief 10 that does
      not carry the series itself (``None``: ``CRSPSpec``'s default,
      ``Quant Model/Data``).
    - ``k_regimes``: states in the chain. Ties to ``experts.n_experts``
      (validated); the value itself is open (Q18).
    - ``trend`` / ``switching_trend`` / ``switching_variance``: the emission.
      ``switching_variance=True`` is the usual finance setting — empirical
      regimes are distinguished more by volatility than by mean (Ang &
      Timmermann 2012) — but it is a *default*, not a decision, and both
      settings are sweepable.
    - ``order``: 0 selects ``MarkovRegression``, > 0 ``MarkovAutoregression``.
    - ``search_reps`` / ``maxiter``: the multi-start search (§5). Each start is
      a logged trial, because best-of-N is a selection event.
    - ``order_by``: the declared canonical ordering of the fitted regimes
      (§4) — a rule, not an assumption.
    - ``gate_weight`` (decision Q27, 2026-10-05; default ``"window"``): which
      row of regime probabilities the gate serves at date t, built from the
      filtered probability ``xi_t`` and the canonically ordered transition
      matrix ``A``: ``"filtered"`` is ``xi_t``; ``"predicted"`` is
      ``xi_t A``; ``"window"`` is ``(1/h) sum_{j=1..h} xi_t A^j``, the
      expected share of the target window ``(t, t+h]`` spent in each regime.
      ``h`` is the panel target's horizon (``DataConfig.horizon_periods``),
      never a field here, so it cannot disagree with the target. The same
      rule applies on training and on test dates.

    The gate's memory and estimator (brief 09 D; Q16 (d)(e)):

    - ``fit_backend``: ``"statsmodels"`` (``MarkovRegression``, BFGS after
      a few EM steps; the default) or ``"native"`` (:mod:`nec_moe.baum_welch`,
      a scaled Baum-Welch for the order-0 Gaussian model, which can weight
      its M-steps). ``"native"`` needs ``order = 0``, ``trend = "c"`` and
      switching mean and variance (what ``MarkovRegression`` fits today).
    - ``memory``: ``"full"`` (every training date weighs the same; the
      default until the pilot chooses) or ``"regime_clock"``: a second,
      weighted Baum-Welch pass whose M-steps weight regime ``k``'s terms by
      ``rho_g ** n_k(t, T)``, ``n_k`` the later regime-``k`` experience by
      the first pass's filtered probabilities, ``rho_g = 2 ** (-1 /
      gate_half_life)`` (theory notes section 7). Needs ``fit_backend =
      "native"``; with ``order > 0`` it raises ``NotImplementedError`` (the
      autoregressive gate keeps full memory). It changes only how the
      parameters are estimated: the forward filter, ``apply_causal`` and
      ``gate_weight`` are unchanged.
    - ``gate_half_life``: in regime-days; infinity reproduces the full-memory
      fit exactly (the second pass is skipped).
    - ``initial_distribution`` (native only): ``"stationary"`` (the default:
      ``rho`` is the stationary distribution of ``A``, as statsmodels'
      steady-state initialisation, so both backends maximise the same
      likelihood) or ``"estimated"`` (the textbook free ``rho = gamma_1``).
    - ``em_tol`` / ``em_maxiter``: the native unweighted fit's stopping rule
      (largest parameter change; a start that does not settle counts as not
      converged, like a statsmodels start); ``weighted_tol`` /
      ``weighted_maxiter``: the same for the weighted pass, which is a
      weighted generalised-EM heuristic and is reported, never silently
      stopped, when it does not settle; ``variance_floor_rel``: the native
      M-step's variance floor, relative to the series' variance.
    """

    series: str = "crsp_market_log_return"  # registry key (Q23, brief 10 B)
    # where the cached Kenneth French daily factors live, for the
    # market_excess_return key (read cache-first; never downloaded in a run)
    context_dir: str = "data_cache"
    # crsp_market_log_return on a panel built before brief 10: the CRSP data
    # folder whose derived/ extract holds market.parquet (None: CRSPSpec's)
    crsp_dir: str | None = None
    series_feature: str = "mkt_ret_1d"  # sequence-feature name, "sequence_feature" key
    series_channel: int = 0  # channel index for the raw-channel key
    k_regimes: int = 2
    trend: str = "c"
    switching_variance: bool = True
    switching_trend: bool = True
    order: int = 0  # 0 -> MarkovRegression; >0 -> MarkovAutoregression
    maxiter: int = 500
    start_seed: int = 0  # base seed for every stochastic start scheme
    # How starting values are generated (brief 04 B): a registry key into
    # nec_moe.markov_gate.START_SCHEME_REGISTRY -- "informed_jitter" (default),
    # "informed_grid", or "default_jitter" (the pre-brief-04 behaviour, kept
    # exactly so the B.1 comparison stays reproducible). See that registry's
    # docstring for the measurements behind the default.
    start_scheme: str = "informed_jitter"
    # default_jitter only: number of starts, and the sd of the ABSOLUTE noise
    # it adds in the CONSTRAINED space. That is the diagnosed defect -- on
    # daily returns it lands on probabilities, means and variances whose
    # scales differ by orders of magnitude, so most starts are invalid or
    # absurd. Retained for comparison, not recommended.
    search_reps: int = 20
    start_jitter: float = 0.5
    # informed_grid / informed_jitter: one data-driven centre per combination
    # of the three tuples below. q = share of (causally assigned) dates in the
    # calmest regime; w = trailing realised-vol window in dates, never centred;
    # p = diagonal persistence of the starting transition matrix, with
    # (1-p)/(K-1) off the diagonal.
    start_vol_quantiles: tuple[float, ...] = (0.5, 0.75)
    start_vol_windows: tuple[int, ...] = (20, 60)
    start_persistences: tuple[float, ...] = (0.95, 0.99)
    # dates of defined trailing vol before the expanding-quantile split (the
    # specification accepted 2026-09-26, audit G-5) may assign a regime
    # (earlier dates are left unassigned), and the smallest
    # group a centre may be built from -- both guard against a regime's mean
    # and variance being estimated from a handful of dates.
    start_min_history: int = 20
    start_min_group_size: int = 10
    # informed_jitter only: perturbed draws per centre, and the noise, applied
    # in statsmodels' UNCONSTRAINED space so every start is valid by
    # construction. sd_i = rel * max(|u_i|, floor_i), where floor_i is the
    # series std for location/scale coordinates (means; sigma, which is what
    # a variance untransforms to) and start_jitter_unit for dimensionless ones
    # (transition logits, AR terms). The floor is not optional: several
    # unconstrained coordinates are exactly 0 (e.g. the off-diagonal logits of
    # a symmetric K=3 start), and purely relative noise never moves them.
    start_draws_per_centre: int = 3
    start_jitter_rel: float = 0.1
    start_jitter_unit: float = 1.0
    # converged log-likelihoods within this many nats of their neighbour are
    # one optimum when counting distinct optima: clustering by gap, the
    # specification accepted 2026-09-26 (brief 04 B.3 as amended, audit G-5)
    distinct_optima_tol: float = 1.0
    prob_floor: float = 1e-12  # clamp before log: keeps rows normalizable
    order_by: str = "variance"  # "variance" | "mean" — ascending (§4)
    # Q27: "filtered" | "predicted" | "window" (decided); see the docstring
    gate_weight: Literal["filtered", "predicted", "window"] = "window"
    registry_tag: str = "markov_gate_starts"  # TrialRegistry tag for §5
    # brief 09 D: estimator and memory (see the docstring)
    fit_backend: Literal["statsmodels", "native"] = "statsmodels"
    memory: Literal["full", "regime_clock"] = "full"
    gate_half_life: float = math.inf  # regime-days; inf = full memory
    initial_distribution: Literal["stationary", "estimated"] = "stationary"
    em_tol: float = 1e-8
    em_maxiter: int = 5000
    weighted_tol: float = 1e-8
    weighted_maxiter: int = 5000
    variance_floor_rel: float = 1e-6


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
    block held out for early stopping — never out-of-sample data. Its
    default is **0** (brief 09 B.3, resolving audit B-2): in the main study
    the base trains on its full training block for its fixed step budget,
    with early stopping off, so the most recent and most heavily weighted
    data is never withheld; in the pilot, validation is the separate
    validation year. A positive ``val_fraction`` (with
    ``early_stopping_patience``) remains for the count-based development
    folds. The base seed is the run seed plus ``seed_offset``, so a base can
    be re-seeded independently of the experts.

    ``decay_half_life_days`` (brief 09 C.1, Q16 (d)): calendar exponential
    decay of the training rows, ``w(s) = 2 ** (-age(s) / H)`` with ``age`` in
    trading days to the fold's last training date (:mod:`nec_moe.decay`).
    ``None`` (the default, until the pilot chooses) or infinity means equal
    weights, and then the loss is exactly the unweighted one.
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
    # tail of the TRAINING block held out for early stopping; 0 = none (the
    # main study and the pilot, brief 09 B.3)
    val_fraction: float = 0.0
    seed_offset: int = 0
    # calendar decay half-life in trading days; None = equal weights (C.1)
    decay_half_life_days: float | None = None


@dataclass(frozen=True)
class TrainConfig:
    """Optimization, regularization, aux-loss gating (§4.3).

    Both aux losses are implemented-with-tests but **default off** (baseline
    objective purity; each is a pre-registered thesis ablation).
    """

    # Registry key into nec_moe.losses.OBJECTIVE_REGISTRY. Only the existing
    # mixture NLL is registered; the choice is open (Q20).
    objective: str = "mixture_nll"
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
    # call, and on a requested stop). 0 = periodic saves off (final save still
    # happens).
    checkpoint_every: int = 0
    # How many checkpoints of one fit are kept (the newest at the path itself,
    # older ones at <path>.1, <path>.2, ...). Resume loads the newest one that
    # loads cleanly, so 2 survives a crash in the middle of a checkpoint write
    # (brief 07 C.4).
    checkpoint_keep: int = 2
    # Steps between heartbeats: the trainer calls Trainer.on_heartbeat (the run
    # store stamps status.json with it, brief 07 C.5).
    heartbeat_every: int = 50
    # Interior quantile edges of each expert's gate probability for the Q24
    # expert-weight diagnostic (nec_moe.diagnostics.expert_weight_diagnostics):
    # the mean responsibility is reported within each bin. Quintiles by default.
    expert_weight_quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    # Row weights of the experts' training loss (brief 09 C.2, Q16 (d)(e)):
    # "none" | "regime_clock" (the mixture form: a row's age is the later
    # experience of its own regime, counted with the frozen gate's filtered
    # probabilities; nec_moe.decay). The half-life is in regime-days;
    # infinity means equal weights. Default "none" until the pilot chooses.
    expert_decay: Literal["none", "regime_clock"] = "none"
    expert_decay_half_life: float = math.inf


def check_gate_memory(mg: MarkovGateConfig) -> None:
    """The brief 09 D.3 rules for the gate's estimator and memory."""
    if mg.fit_backend not in ("statsmodels", "native"):
        raise ValueError(f"unknown markov_gate.fit_backend {mg.fit_backend!r}")
    if mg.memory not in ("full", "regime_clock"):
        raise ValueError(f"unknown markov_gate.memory {mg.memory!r}")
    if mg.memory == "regime_clock" and mg.order > 0:
        raise NotImplementedError(
            "markov_gate.memory='regime_clock' is implemented for the order-0 gate "
            "only; the autoregressive gate (order > 0) keeps full memory for now "
            "(brief 09 D.3)"
        )
    if mg.memory == "regime_clock" and mg.fit_backend != "native":
        raise ValueError(
            "markov_gate.memory='regime_clock' needs fit_backend='native': statsmodels "
            "takes no observation weights"
        )
    if mg.fit_backend == "native":
        if mg.order > 0:
            raise NotImplementedError(
                "fit_backend='native' fits the order-0 Gaussian model only; use "
                "'statsmodels' for the autoregressive gate"
            )
        if mg.trend != "c" or not (mg.switching_trend and mg.switching_variance):
            raise NotImplementedError(
                "fit_backend='native' fits a switching constant and a switching "
                "variance (trend='c', switching_trend and switching_variance True)"
            )
    if mg.initial_distribution not in ("stationary", "estimated"):
        raise ValueError(f"unknown initial_distribution {mg.initial_distribution!r}")
    if not mg.gate_half_life > 0:
        raise ValueError(f"gate_half_life must be > 0, got {mg.gate_half_life}")
    if not (mg.em_tol > 0 and mg.weighted_tol > 0 and mg.em_maxiter >= 1
            and mg.weighted_maxiter >= 1):
        raise ValueError("em_tol, weighted_tol must be > 0 and the maxiters >= 1")
    if not mg.variance_floor_rel >= 0:
        raise ValueError(f"variance_floor_rel must be >= 0, got {mg.variance_floor_rel}")


@dataclass(frozen=True)
class NECConfig:
    """Root config. Construct, then ``validate()`` (NECModel does it for you)."""

    data: DataConfig
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    experts: ExpertConfig = field(default_factory=ExpertConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    markov_gate: MarkovGateConfig = field(default_factory=MarkovGateConfig)
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
        b, mg = self.base, self.markov_gate

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
        if d.horizon is not None:
            if d.horizon < 1:
                raise bad(f"data.horizon must be >= 1, got {d.horizon}")
            named = target_horizon(d.target)
            if named is not None and named != d.horizon:
                raise bad(
                    f"data.horizon={d.horizon} contradicts the target "
                    f"{d.target!r}, whose name declares {named}"
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
        if x.hidden_init not in ("diversified", "identical"):
            raise bad(
                f"unknown experts.hidden_init {x.hidden_init!r}; "
                "use 'diversified' or 'identical'"
            )
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
        from .losses import OBJECTIVE_REGISTRY

        if t.objective not in OBJECTIVE_REGISTRY:
            raise bad(
                f"unknown train.objective {t.objective!r}; registered: "
                f"{sorted(OBJECTIVE_REGISTRY)}"
            )
        if t.checkpoint_every < 0:
            raise bad(f"checkpoint_every must be >= 0, got {t.checkpoint_every}")
        if t.checkpoint_keep < 1:
            raise bad(f"checkpoint_keep must be >= 1, got {t.checkpoint_keep}")
        if t.heartbeat_every < 1:
            raise bad(f"heartbeat_every must be >= 1, got {t.heartbeat_every}")
        q = t.expert_weight_quantiles
        if not q or any(not 0.0 < v < 1.0 for v in q) or any(
            b <= a for a, b in zip(q, q[1:], strict=False)
        ):
            raise bad(
                "expert_weight_quantiles must be strictly increasing values in "
                f"(0, 1), got {q}"
            )
        if t.correction_penalty_weight < 0:
            raise bad(
                f"correction_penalty_weight must be >= 0, got "
                f"{t.correction_penalty_weight}"
            )
        from .decay import EXPERT_DECAYS

        if t.expert_decay not in EXPERT_DECAYS:
            raise bad(f"unknown train.expert_decay {t.expert_decay!r}; use one of "
                      f"{EXPERT_DECAYS}")
        if not t.expert_decay_half_life > 0:
            raise bad(f"expert_decay_half_life must be > 0, got {t.expert_decay_half_life}")
        if t.expert_decay == "regime_clock" and p.kind != "markov":
            raise bad(
                "train.expert_decay='regime_clock' counts each row's age with the "
                "frozen gate's filtered regime probabilities, which only the fitted "
                f"Hamilton gate (prior.kind='markov') provides; got prior.kind={p.kind!r}"
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
        if not 0.0 <= b.val_fraction < 1.0:
            raise bad(
                f"base val_fraction must be in [0, 1) — it is a tail of the "
                f"TRAINING block, 0 for none — got {b.val_fraction}"
            )
        if b.early_stopping_patience is not None and b.early_stopping_patience < 1:
            raise bad(
                f"base early_stopping_patience must be >= 1 or None, got "
                f"{b.early_stopping_patience}"
            )
        if b.decay_half_life_days is not None and not b.decay_half_life_days > 0:
            raise bad(
                f"base decay_half_life_days must be > 0 or None, got "
                f"{b.decay_half_life_days}"
            )
        if b.early_stopping_patience is not None and b.val_fraction == 0.0:
            raise bad(
                "base early_stopping_patience needs a validation tail "
                "(val_fraction > 0); with val_fraction = 0 the base trains its "
                "full step budget"
            )
        if x.correction_mode and not b.enabled:
            raise bad(
                "experts.correction_mode=True requires base.enabled=True: a "
                "correction is a correction *to* something, and with no base "
                "the experts' zero-initialized heads would predict a constant 0"
            )
        # markov gate (brief 03). The cross-block tie is only meaningful when
        # the gate is actually selected — an unused config block must not
        # block an otherwise valid config.
        if p.kind == "markov" and mg.k_regimes != x.n_experts:
            raise bad(
                f"markov_gate.k_regimes={mg.k_regimes} must equal "
                f"experts.n_experts={x.n_experts}: the gate's regimes are the "
                "mixture's components"
            )
        if mg.search_reps < 1 or mg.maxiter < 1:
            raise bad(
                f"markov_gate search_reps/maxiter must be >= 1, got "
                f"{mg.search_reps}/{mg.maxiter}"
            )
        if mg.order < 0:
            raise bad(f"markov_gate.order must be >= 0, got {mg.order}")
        if not (mg.start_vol_quantiles and mg.start_vol_windows and mg.start_persistences):
            raise bad("markov_gate start grids must each hold at least one value")
        if any(not 0.0 < q < 1.0 for q in mg.start_vol_quantiles):
            raise bad(
                f"start_vol_quantiles must lie in (0, 1), got {mg.start_vol_quantiles}"
            )
        if any(w < 2 for w in mg.start_vol_windows):
            raise bad(f"start_vol_windows must be >= 2, got {mg.start_vol_windows}")
        if any(not 0.0 < pp < 1.0 for pp in mg.start_persistences):
            raise bad(
                f"start_persistences must lie in (0, 1) -- a persistence of "
                f"exactly 0 or 1 is not a valid transition row -- got "
                f"{mg.start_persistences}"
            )
        if mg.start_min_history < 1 or mg.start_min_group_size < 2:
            raise bad(
                f"start_min_history must be >= 1 and start_min_group_size >= 2 "
                f"(a variance needs two points), got {mg.start_min_history}/"
                f"{mg.start_min_group_size}"
            )
        if mg.start_draws_per_centre < 1:
            raise bad(
                f"start_draws_per_centre must be >= 1, got {mg.start_draws_per_centre}"
            )
        if mg.start_jitter_rel < 0 or mg.start_jitter_unit <= 0 or mg.start_jitter < 0:
            raise bad(
                "start_jitter_rel and start_jitter must be >= 0 and "
                "start_jitter_unit > 0"
            )
        if mg.distinct_optima_tol <= 0:
            raise bad(
                f"distinct_optima_tol must be > 0, got {mg.distinct_optima_tol}"
            )
        if mg.gate_weight not in GATE_WEIGHTS:
            raise bad(
                f"unknown markov_gate.gate_weight {mg.gate_weight!r}; use one of "
                f"{GATE_WEIGHTS}"
            )
        check_gate_memory(mg)
        from .markov_gate import (
            ORDERING_REGISTRY,
            SERIES_REGISTRY,
            START_SCHEME_REGISTRY,
        )

        if mg.start_scheme not in START_SCHEME_REGISTRY:
            raise bad(
                f"unknown markov_gate.start_scheme {mg.start_scheme!r}; "
                f"registered: {sorted(START_SCHEME_REGISTRY)}"
            )
        if mg.series not in SERIES_REGISTRY:
            raise bad(
                f"unknown markov_gate.series {mg.series!r}; registered: "
                f"{sorted(SERIES_REGISTRY)}"
            )
        if mg.order_by not in ORDERING_REGISTRY:
            raise bad(
                f"unknown markov_gate.order_by {mg.order_by!r}; registered: "
                f"{sorted(ORDERING_REGISTRY)}"
            )
        # A canonical ordering needs a statistic that actually varies across
        # regimes; sorting on a shared parameter returns an arbitrary order
        # that looks canonical, which is worse than refusing.
        if mg.order_by == "variance" and not mg.switching_variance:
            raise bad(
                "markov_gate.order_by='variance' requires "
                "switching_variance=True: with a shared variance the sort key "
                "is identical across regimes and the 'canonical' order would "
                "be whatever the optimizer happened to produce"
            )
        if mg.order_by == "mean" and not mg.switching_trend:
            raise bad(
                "markov_gate.order_by='mean' requires switching_trend=True "
                "(a shared mean gives every regime the same sort key)"
            )
        if p.kind == "markov" and t.sequence_ordered:
            raise bad(
                "prior.kind='markov' is a precomputed (date-keyed) gate, not a "
                "recursive one: its filter has already been run at fit time, "
                "so train.sequence_ordered must be False"
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
            markov_gate=MarkovGateConfig(
                **_tupled(
                    d.get("markov_gate", {}),
                    ("start_vol_quantiles", "start_vol_windows", "start_persistences"),
                )
            ),
            base=BaseConfig(**_tupled(d.get("base", {}), ("hidden_dims",))),
            train=TrainConfig(**_tupled(d.get("train", {}), ("expert_weight_quantiles",))),
        )
