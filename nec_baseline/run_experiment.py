"""THE CONTROL PANEL — one file, all the knobs, everything else automatic.

Edit the ``EXPERIMENT = Experiment(...)`` block below and run:

    python3.14 run_experiment.py

Two modes:

- ``mode="quick"``    — ONE model, one chronological train/test split, full
  diagnostics: monitoring dashboard, gate-utilization figure, IC + long-short
  curves, calibration check, (optional) VIX alignment. Use this while
  iterating — it answers "what is this model doing?" in a minute or two.
- ``mode="evaluate"`` — the honest protocol: multi-seed purged walk-forward
  through the shared harness, mean ± seed-std per arm, optional ridge/MLP
  baseline rows, BH-corrected claims. Use this when a number might be shown
  to anyone.

Everything is a thin driver over the tested package (`nec_moe`) — no logic is
duplicated here. Every run lands in the run store (``nec_moe.runstore``,
brief 07): ``results/<campaign>/<run_id>/`` holds ``run.json``,
``settings.json``, ``status.json``, the trial registry (``trials.jsonl``),
``metrics/``, ``figures/``, ``logs/run.log`` and ``checkpoints/``, with one
row per run in ``results/INDEX.csv``; per-security outputs (predictions) go to
``Data/derived/runs/<run_id>/``. RUNBOOK.md is the how-to; the handbook (Part
II) documents every underlying knob.

Long runs are interruptible: set ``checkpoint_every`` (steps between saves),
interrupt, and continue with

    python3.14 run_experiment.py --resume            # the EXPERIMENT block's tag
    python3.14 run_experiment.py --resume <run_id>   # a specific run

Resume is exact — completed work (sweep runs, folds) is skipped, an
interrupted fit continues from its last checkpoint on the same trajectory —
provided the settings and data are unchanged.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nec_moe import (  # noqa: E402
    RESUMABLE_STATES,
    BaseCache,
    BaseConfig,
    CRSPSpec,
    DataConfig,
    EncoderConfig,
    ExpertConfig,
    FeatureSpec,
    FoldConfig,
    MarkovGateConfig,
    MLPBaseline,
    NECConfig,
    NECModel,
    Panel,
    PriorConfig,
    PriorContext,
    ResumeRefused,
    RidgeBaseline,
    Run,
    RunStore,
    StageBSpec,
    SyntheticRegimePanel,
    SyntheticSpec,
    TrainConfig,
    Trainer,
    TrialRegistry,
    WalkForwardFold,
    base_and_correction,
    base_single_gaussian_nll,
    baseline_arm,
    build_context,
    build_crsp_panel,
    calendar_year_folds,
    corrected_claims,
    data_config_from_panel,
    data_fingerprint,
    expert_stage_seed,
    expert_weight_report,
    fit_temperature,
    fold_decay_weights,
    gate_regime_alignment,
    gate_reliability,
    graceful_interrupts,
    ic_summary,
    load_extract,
    load_french_factors,
    load_vix,
    nec_arm,
    plot_gate_utilization,
    plot_ic_series,
    plot_long_short_curve,
    plot_reliability,
    plot_sweep_report,
    plot_training_dashboard,
    plot_transition_matrix,
    rank_ic_by_date,
    run_sweep,
    settings_hash,
    target_horizon,
    trial_provenance,
    walk_forward_folds,
)
from nec_moe.evaluation import (  # noqa: E402
    _fit_gate,
    _gate_training_block,
    _heartbeat,
    _restore_gate,
    _save_predictions,
)
from nec_moe.pilot import (  # noqa: E402
    encode_settings,
    load_selection,
    selection_deviations,
)
from nec_moe.utils import atomic_torch_save  # noqa: E402

# =========================================================================== #
#  SETTINGS — this is the only part you edit
# =========================================================================== #


@dataclass
class Experiment:
    # ---------------- run identity ----------------
    tag: str = "my_experiment"      # names the run (run_id = <time>_<tag>_<hash>)
    campaign: str = "dev"           # results/<campaign>/<run_id>/ (e.g. "gates_k2")
    purpose: str = ""               # free-text label ("smoke", "debug", ...); no rule
    mode: str = "quick"             # "quick" (one model + diagnostics) | "evaluate"
    out_dir: str = "results"        # the run store's root (aggregate outputs only)
    # per-security outputs (predictions): licensed on real data, so never
    # under results/. None = Quant Model/Data/derived/runs/
    per_security_dir: str | None = None

    # ---------------- data ----------------
    data: str = "synthetic"         # "synthetic" | "crsp" | "panel_file"
    # CRSP data (data="crsp"): built from the extract in Data/derived/, which
    # scripts/extract_crsp_v2.py writes (licensed: it never leaves Data/).
    # The sample is 2000-2024 (Q16 (a), decided 2026-10-05; brief 09 A): the
    # extract reaches further back for the feature windows only.
    start: str = "2000-01-01"
    end: str = "2024-12-31"
    post_delisting_return: str = "cash"   # "cash" | "market"; not a decision (CRSPSpec)
    # panel file (data="panel_file") — e.g. the prebuilt CRSP PIT panel
    # (scripts/build_pit_panel.py; relative to this file):
    panel_file: str = (
        "../Data/derived/pit_panel_crsp_2000-01-01_2024-12-31_q26_market_neutral.pt"
    )
    # feature/target spec (CRSP builds; a panel file carries its own):
    seq_len: int = 20               # encoder window T
    # forward-return days. None = the panel's own target (a CRSP build then
    # uses StageBSpec's 5). The purge is always the TARGET's horizon; a value
    # set here that contradicts the panel's target raises (audit O-3).
    horizon: int | None = None
    # the target (Q25): "market_neutral" (decided; the forward return minus the
    # date's cross-sectional mean) | "raw" | "residual" (trailing-beta residual)
    target_kind: str = "market_neutral"
    # the snapshot inputs (Q26): "q26" (decided, 57 inputs; a CRSP build then
    # also needs the Compustat extract, scripts/extract_compustat.py) |
    # "legacy14". Every window is a FeatureSpec field (nec_moe.features)
    feature_set: str = "q26"
    # synthetic data (data="synthetic"):
    synth_dates: int = 300
    synth_entities: int = 8
    synth_regime_process: str = "iid"      # "iid" | "markov" (sticky regimes)
    synth_vol_levels: tuple[float, float] = (0.5, 2.5)  # regime separability
    # business-day date labels from this date (None: no calendar), so the
    # calendar-year folds can run on synthetic data
    synth_calendar_start: str | None = None

    # ---------------- model ----------------
    prior: str = "soft"             # soft|uniform|hard|topk|gumbel|hmm
    emission: str = "mlp"           # "mlp" (neural experts) | "classical" (Hamilton-style)
    n_experts: int = 2              # Ye & Borde ablate K in {2, 4, 6}
    encoder_hidden: int = 32
    encoder_layers: int = 1
    expert_hidden_dims: tuple[int, ...] = (64, 32)  # depth AND width; see pyramid_dims
    expert_dropout: float = 0.05
    expert_activation: str = "relu"  # relu|gelu|tanh|silu|elu (registry key)
    expert_hidden_init: str = "diversified"  # | "identical" (brief 06 D); not decided
    input_mode: str = "snapshot"    # "snapshot" | "snapshot_plus_hidden" (Decision A)
    top_k: int = 2                  # topk prior only
    tau_init: float = 1.0           # gumbel prior only …
    tau_anneal_steps: int = 0       # … 0 = constant temperature
    transition_diag_bias: float = 2.0  # hmm prior only: persistence-biased init

    # ---------------- training ----------------
    objective: str = "mixture_nll"  # OBJECTIVE_REGISTRY key; only one registered (Q20 open)
    steps: int = 600                # optimizer steps (per fold in evaluate mode)
    lr: float = 1e-3
    weight_decay: float = 1e-4      # the experts' AdamW weight decay
    batch_size: int = 256
    sigma_init: float | None = None # None = auto (std of the training target)
    sigma_freeze_steps: int = 100
    seeds: tuple[int, ...] = (0,)   # >1 seed => mean ± std in evaluate mode
    # expert warm-start (keep on; handbook I.11): pre-training on vol-sorted
    # slices of the SAME fold's training block. Not a cross-fold warm start:
    # every fold starts from fresh weights (brief 09 E.2)
    warmstart: bool = True
    warmstart_channel: int | None = None  # seq channel of the vol proxy; None = auto
    chunk_len: int = 50             # HMM truncated-BPTT chunk (dates per step)
    # decay weights (Q16 (d)(e), brief 09 C; chosen by the pilot, so the
    # defaults are equal weights): the experts' "none" | "regime_clock"
    # (age = later same-regime experience, counted with the frozen gate's
    # filtered probabilities; prior="markov" only), half-life in regime-days
    expert_decay: str = "none"
    expert_decay_half_life: float = math.inf
    # Q24 diagnostic: bins of each expert's gate probability (interior edges)
    expert_weight_quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)

    # ---------------- residual mode: frozen base + corrections (brief 02) ----
    # y_hat = f0(x) + sum_k pi_k r_k(x). All defaults off, so existing
    # behaviour is unchanged until you turn it on. None of these values is
    # chosen yet — they are what the experiment programme sweeps.
    base_enabled: bool = False        # fit f0 per fold and freeze it
    correction_mode: bool = False     # experts emit corrections, not forecasts
    zero_init_head: bool = True       # start exactly at the base (Ye & Borde)
    freeze_gate: bool = False         # freeze encoder+gate+prior (Q19)
    base_hidden_dims: tuple[int, ...] = (64, 32)
    base_dropout: float = 0.0
    base_activation: str = "relu"
    base_lr: float = 1e-3
    base_weight_decay: float = 1e-4
    base_steps: int = 1000
    base_batch_size: int = 128
    base_early_stopping_patience: int | None = None
    # tail of the TRAINING block held out for early stopping. 0 (no tail) is
    # required by the calendar-year main and pilot folds (brief 09 B.3)
    base_val_fraction: float = 0.0
    base_seed_offset: int = 0
    # the base's calendar decay half-life in trading days; None = equal weights
    base_decay_half_life_days: float | None = None
    # alpha of the correction-magnitude penalty. Ye & Borde do not report
    # theirs: select on the training block, log every candidate as a trial,
    # and publish the sensitivity curve (brief 02 §3).
    aux_correction_penalty: bool = False
    correction_penalty_weight: float = 0.0

    # ---------------- the Hamilton gate (prior="markov", brief 03) ----------
    # Fitted by maximum likelihood on each fold's training block, canonically
    # reordered, and applied to the test block with FROZEN parameters through
    # the filter. Nothing here is chosen yet.
    gate_series: str = "market_excess_return"  # registry key (see SERIES_REGISTRY)
    gate_context_dir: str = "data_cache"       # cached French factors (market_excess_return)
    gate_series_feature: str = "mkt_ret_1d"    # name, for series="sequence_feature"
    gate_series_channel: int = 0               # channel index for "sequence_channel"
    gate_trend: str = "c"
    gate_switching_variance: bool = True       # the usual finance setting
    gate_switching_trend: bool = True
    gate_order: int = 0                        # 0 -> MarkovRegression; >0 -> MarkovAR
    gate_maxiter: int = 500
    gate_start_seed: int = 0
    # starting values (brief 04 B): "informed_jitter" | "informed_grid" | "default_jitter"
    gate_start_scheme: str = "informed_jitter"
    gate_start_vol_quantiles: tuple[float, ...] = (0.5, 0.75)  # calm-regime share
    gate_start_vol_windows: tuple[int, ...] = (20, 60)         # TRAILING vol windows
    gate_start_persistences: tuple[float, ...] = (0.95, 0.99)  # start diagonal
    gate_start_min_history: int = 20
    gate_start_min_group_size: int = 10
    gate_start_draws_per_centre: int = 3       # informed_jitter draws per centre
    gate_start_jitter_rel: float = 0.1         # relative noise, unconstrained space
    gate_start_jitter_unit: float = 1.0        # noise floor, dimensionless coords
    gate_distinct_optima_tol: float = 1.0      # nats: "same optimum" tolerance
    gate_search_reps: int = 20                 # default_jitter only
    gate_start_jitter: float = 0.5             # default_jitter only (the diagnosed scheme)
    gate_prob_floor: float = 1e-12             # clamp before log of filtered probs
    gate_order_by: str = "variance"            # canonical regime order, ascending
    # Q27 (decided): "window" = the average over the target window of the
    # 1..h-step-ahead regime probabilities | "predicted" | "filtered"
    gate_weight: str = "window"
    gate_registry_tag: str = "markov_gate_starts"
    # the gate's estimator and memory (Q16 (d)(e), brief 09 D; the pilot
    # chooses the memory, so the defaults are full memory on statsmodels):
    # "statsmodels" | "native" (scaled Baum-Welch, order 0); "full" |
    # "regime_clock" (weighted second pass; native only); half-life in
    # regime-days (inf = full memory)
    gate_fit_backend: str = "statsmodels"
    gate_memory: str = "full"
    gate_half_life: float = math.inf
    gate_initial_distribution: str = "stationary"  # native: "stationary" | "estimated"
    gate_em_tol: float = 1e-8                  # native: parameter-change tolerance
    gate_em_maxiter: int = 5000
    gate_weighted_tol: float = 1e-8            # the regime-clock pass
    gate_weighted_maxiter: int = 5000
    gate_variance_floor_rel: float = 1e-6      # native variance floor / series variance

    # ---------------- checkpointing / resume ----------------
    checkpoint_every: int = 0       # save training state every N steps (0 = only at the
                                    #   end of each fit and on Ctrl+C); set for long runs
    checkpoint_keep: int = 2        # checkpoints kept per fit (survives a torn write)
    heartbeat_every: int = 50       # steps between status.json heartbeats
    stale_after: float = 1800.0     # s without a heartbeat before "running" = "crashed"
    resume: bool = False            # continue a run (CLI: --resume [run_id])
    resume_run_id: str | None = None  # which run; None = the latest interrupted/crashed
                                      #   run of this tag
    force: bool = False             # resume despite changed settings/data (CLI: --force;
                                    #   recorded in run.json)

    # ---------------- evaluation (mode="evaluate") ----------------
    # folds (Q16, brief 09 B): "calendar_year" = one fold per test year
    # first_test_year..last_test_year, each trained on everything before it
    # (expanding, purged; 2010-2024 = 15 folds) | "count" = the last
    # n_folds * test_dates_per_fold dates as test blocks (development)
    fold_scheme: str = "calendar_year"
    first_test_year: int = 2010
    last_test_year: int = 2024
    # the pilot's validation years (scripts/run_pilot.py); never >= first_test_year
    pilot_validation_years: tuple[int, ...] = (2007, 2008, 2009)
    n_folds: int = 3                # count scheme only
    test_dates_per_fold: int = 40   # count scheme only
    include_ridge: bool = True      # baseline rows in the comparison table
    include_mlp: bool = True
    backtest_quantiles: int | None = 5   # None disables the long-short backtest
    ic_hac_lags: int | None = None  # HAC lags for IC t-stats; None = horizon - 1
    ic_hac_kernel: str = "uniform"  # "uniform" (Hansen-Hodrick) | "bartlett" (Newey-West)
    cost_rate: float = 0.001        # cost per unit traded notional (10 bps)
    # long-short book for an h-day target (audit E-3): "nonoverlapping"
    # (rebalance every h dates) | "staggered" (Jegadeesh-Titman cohorts).
    # The default is NOT a decision; every trial records it.
    portfolio_scheme: str = "nonoverlapping"
    # the pre-registration lock (Q15, brief 09 F.3): the pilot's selection
    # file (scripts/run_pilot.py writes results/pilot/pilot_selection.json).
    # A run that scores any date from first_test_year on refuses to start
    # without it; its hash goes into every trial row; settings that differ
    # from its chosen ones are refused unless force_deviation (recorded).
    pilot_selection_file: str | None = None
    force_deviation: bool = False

    # ---------------- extras ----------------
    figures: bool = True
    calibration: bool = True        # reliability + temperature (soft/gumbel gates)
    alignment: bool = False         # gate-vs-VIX/factors report (real panels)
    quick_train_frac: float = 0.8   # quick mode's chronological split (purged)


EXPERIMENT = Experiment(
    # ↓↓↓ tweak here — anything not listed keeps the default above ↓↓↓
    tag="my_experiment",
    mode="quick",
    data="synthetic",
    prior="soft",
    steps=600,
    seeds=(0,),
)


# =========================================================================== #
#  Driver — you should not need to edit below this line
# =========================================================================== #

CACHE = Path(__file__).resolve().parent / "data_cache"


def _build_panel(exp: Experiment) -> tuple[Panel, int]:
    """-> (panel, purge_dates). The purge is the panel's target horizon
    (:func:`resolve_purge`)."""
    if exp.data == "synthetic":
        spec = SyntheticSpec(
            regime_process=exp.synth_regime_process,  # type: ignore[arg-type]
            vol_levels=exp.synth_vol_levels,
            beta_scale=2.0,
            noise_std=0.4,
            seed=0,
            calendar_start=exp.synth_calendar_start,
        )
        panel: Panel = SyntheticRegimePanel(spec).generate(exp.synth_dates, exp.synth_entities)
    elif exp.data == "panel_file":
        panel = torch.load(_repo_path(exp.panel_file), weights_only=False)
    elif exp.data == "crsp":
        bspec = StageBSpec(
            seq_len=exp.seq_len,
            horizon=exp.horizon if exp.horizon is not None else StageBSpec().horizon,
            target_kind=exp.target_kind,  # type: ignore[arg-type]
            features=FeatureSpec(feature_set=exp.feature_set),  # type: ignore[arg-type]
            input_mode=exp.input_mode,  # type: ignore[arg-type]
        )
        crsp_spec = dataclasses.replace(
            CRSPSpec(), start=exp.start, end=exp.end,
            post_delisting_return=exp.post_delisting_return,  # type: ignore[arg-type]
        )
        # the default window's extract covers every sub-window of it
        build = build_crsp_panel(crsp_spec, bspec, extract=load_extract(CRSPSpec()))
        print(build.coverage)
        panel = build.panel
    else:
        raise ValueError(f"unknown data mode {exp.data!r}")
    return panel, resolve_purge(exp, panel)


def resolve_purge(exp: Experiment, panel: Panel) -> int:
    """The purge: the horizon of the panel's target (audit O-3).

    Read from the target name (``fwd_ret_5d`` gives 5, the same parse
    ``DataConfig.horizon_periods`` uses); a target that declares no horizon
    (the synthetic ``y_synth``) gives 1. ``Experiment.horizon``, when set,
    must agree with a declared horizon: a 5-day panel run with ``horizon=1``
    would purge 1 day and let 4 days of labels overlap the test block.
    """
    declared = target_horizon(panel.schema.target)
    if exp.horizon is not None and declared is not None and exp.horizon != declared:
        raise ValueError(
            f"Experiment.horizon={exp.horizon} disagrees with the panel's target "
            f"{panel.schema.target!r} (horizon {declared}); the purge must be the "
            "target's horizon — set horizon=None or to match the target"
        )
    return panel.horizon


def _quick_split(exp: Experiment, panel: Panel, purge: int) -> tuple[Panel, Panel, Any]:
    """Quick mode's chronological split, **purged** like a walk-forward fold.

    The last ``1 - quick_train_frac`` share of dates is the test block; the
    ``purge`` dates before it are dropped, so no training label's forward
    window reaches the test block (audit O-1). -> (train, test, fold)."""
    dates = torch.unique(panel.date, sorted=True)
    n_test = len(dates) - int(math.floor(exp.quick_train_frac * len(dates)))
    fold = walk_forward_folds(
        panel.date, n_folds=1, test_dates_per_fold=n_test, purge_dates=purge
    )[0]
    return panel.subset_dates(fold.train_dates), panel.subset_dates(fold.test_dates), fold


def _nec_config(exp: Experiment, panel: Panel, sigma_init: float) -> NECConfig:
    if exp.data == "synthetic":
        data_cfg = DataConfig(d_seq=panel.x_seq.shape[2], d_snap=panel.x_snap.shape[1],
                              seq_len=panel.x_seq.shape[1])
    else:
        data_cfg = data_config_from_panel(panel)
    return NECConfig(
        data=data_cfg,
        encoder=EncoderConfig(hidden_dim=exp.encoder_hidden, num_layers=exp.encoder_layers),
        experts=ExpertConfig(n_experts=exp.n_experts,
                             hidden_dims=tuple(exp.expert_hidden_dims),
                             dropout=exp.expert_dropout,
                             activation=exp.expert_activation,
                             input_mode=exp.input_mode,  # type: ignore[arg-type]
                             kind=exp.emission,  # type: ignore[arg-type]
                             correction_mode=exp.correction_mode,
                             zero_init_head=exp.zero_init_head,
                             hidden_init=exp.expert_hidden_init),  # type: ignore[arg-type]
        prior=PriorConfig(kind=exp.prior, top_k=exp.top_k, tau_init=exp.tau_init,
                          tau_anneal_steps=exp.tau_anneal_steps,
                          transition_diag_bias=exp.transition_diag_bias),
        markov_gate=MarkovGateConfig(series=exp.gate_series,
                                     context_dir=_repo_path(exp.gate_context_dir),
                                     series_feature=exp.gate_series_feature,
                                     series_channel=exp.gate_series_channel,
                                     k_regimes=exp.n_experts,
                                     trend=exp.gate_trend,
                                     switching_variance=exp.gate_switching_variance,
                                     switching_trend=exp.gate_switching_trend,
                                     order=exp.gate_order,
                                     search_reps=exp.gate_search_reps,
                                     maxiter=exp.gate_maxiter,
                                     start_seed=exp.gate_start_seed,
                                     start_jitter=exp.gate_start_jitter,
                                     start_scheme=exp.gate_start_scheme,
                                     start_vol_quantiles=tuple(exp.gate_start_vol_quantiles),
                                     start_vol_windows=tuple(exp.gate_start_vol_windows),
                                     start_persistences=tuple(exp.gate_start_persistences),
                                     start_min_history=exp.gate_start_min_history,
                                     start_min_group_size=exp.gate_start_min_group_size,
                                     start_draws_per_centre=exp.gate_start_draws_per_centre,
                                     start_jitter_rel=exp.gate_start_jitter_rel,
                                     start_jitter_unit=exp.gate_start_jitter_unit,
                                     distinct_optima_tol=exp.gate_distinct_optima_tol,
                                     prob_floor=exp.gate_prob_floor,
                                     order_by=exp.gate_order_by,
                                     gate_weight=exp.gate_weight,  # type: ignore[arg-type]
                                     registry_tag=exp.gate_registry_tag,
                                     fit_backend=exp.gate_fit_backend,  # type: ignore[arg-type]
                                     memory=exp.gate_memory,  # type: ignore[arg-type]
                                     gate_half_life=exp.gate_half_life,
                                     initial_distribution=exp.gate_initial_distribution,  # type: ignore[arg-type]
                                     em_tol=exp.gate_em_tol,
                                     em_maxiter=exp.gate_em_maxiter,
                                     weighted_tol=exp.gate_weighted_tol,
                                     weighted_maxiter=exp.gate_weighted_maxiter,
                                     variance_floor_rel=exp.gate_variance_floor_rel),
        base=BaseConfig(enabled=exp.base_enabled,
                        hidden_dims=tuple(exp.base_hidden_dims),
                        dropout=exp.base_dropout,
                        activation=exp.base_activation,
                        lr=exp.base_lr,
                        weight_decay=exp.base_weight_decay,
                        steps=exp.base_steps,
                        batch_size=exp.base_batch_size,
                        early_stopping_patience=exp.base_early_stopping_patience,
                        val_fraction=exp.base_val_fraction,
                        seed_offset=exp.base_seed_offset,
                        decay_half_life_days=exp.base_decay_half_life_days),
        train=TrainConfig(objective=exp.objective,
                          lr=exp.lr, weight_decay=exp.weight_decay,
                          batch_size=exp.batch_size, steps=exp.steps,
                          sigma_init=sigma_init,
                          sigma_freeze_steps=exp.sigma_freeze_steps,
                          sequence_ordered=(exp.prior == "hmm"),
                          checkpoint_every=exp.checkpoint_every,
                          checkpoint_keep=exp.checkpoint_keep,
                          heartbeat_every=exp.heartbeat_every,
                          freeze_gate=exp.freeze_gate,
                          aux_correction_penalty=exp.aux_correction_penalty,
                          correction_penalty_weight=exp.correction_penalty_weight,
                          expert_weight_quantiles=tuple(exp.expert_weight_quantiles),
                          expert_decay=exp.expert_decay,  # type: ignore[arg-type]
                          expert_decay_half_life=exp.expert_decay_half_life),
    )


def _auto_sigma(exp: Experiment, cfg_panel: Panel, train: Panel,
                cache: BaseCache | None = None,
                window: object = "sigma") -> float:
    """``sigma_init``: the scale of what the experts actually model.

    Without a base the experts model ``y``, so its std is the right scale.
    In correction mode they model the **residual** ``y - f0(x)``, which is
    smaller — often much smaller, since the base is expected to capture most
    of a signal that is only ~0.4% monthly R-squared to begin with. Starting
    sigma at std(y) there would tell the model the noise is far larger than
    it is and flatten the responsibilities at step 0. So the default is the
    residual's std, measured on the training block by fitting the base once
    (through ``cache``, so the fold that needs the same base reuses it rather
    than refitting). Documented default, not a hidden adjustment; sigma is
    learnable from there.
    """
    if exp.sigma_init is not None:
        return exp.sigma_init
    if not exp.base_enabled:
        return float(train.y.std())
    probe = _nec_config(exp, cfg_panel, sigma_init=1.0)  # sigma irrelevant here
    cache = cache if cache is not None else BaseCache()
    fit = cache.get_or_fit(train, probe.base, window=window, seed=exp.seeds[0])
    with torch.no_grad():
        resid = train.y - fit.model(train.x_snap)
    return max(float(resid.std()), 1e-6)


def _repo_path(path: str) -> str:
    """Relative paths resolve against this file's directory, like ``CACHE``,
    so a run behaves the same whatever the working directory."""
    p = Path(path)
    return str(p if p.is_absolute() else Path(__file__).resolve().parent / p)


def _proxy_channel(exp: Experiment) -> int:
    if exp.warmstart_channel is not None:
        return exp.warmstart_channel
    return 0 if exp.data == "synthetic" else 2  # synth regime channel | vol_20d


def _warm_key(exp: Experiment):
    ch = _proxy_channel(exp)
    return (lambda p: p.x_seq[:, :, ch].std(dim=1)) if exp.warmstart else None


def _quick(exp: Experiment, panel: Panel, purge: int, run: Run) -> dict:
    train, test, fold = _quick_split(exp, panel, purge)
    cache = BaseCache()
    window = ("quick", int(train.date.min()), int(train.date.max()))
    sigma = _auto_sigma(exp, panel, train, cache, window)
    cfg = _nec_config(exp, panel, sigma)
    registry = TrialRegistry(run.trials)
    out = run.figures_dir
    ckpt = run.checkpoints_dir / "trainer.pt"
    gate_file = run.checkpoints_dir / "gate.pt"
    resumed = exp.resume and Trainer.checkpoint_exists(ckpt)
    if resumed:
        trainer = Trainer.load_latest(ckpt)
        remaining = max(exp.steps - trainer.step_count, 0)
        print(f"[resume] {ckpt} at step {trainer.step_count:,} — "
              f"{remaining:,} steps remaining")
    else:
        torch.manual_seed(exp.seeds[0])
        trainer = Trainer(NECModel(cfg))
        remaining = exp.steps
    # a precomputed (fitted-and-frozen) gate: fit on the purged training
    # split, then extend causally to the test split with the frozen
    # parameters, exactly as the walk-forward harness does
    if resumed:  # the fitted gate comes back from its own file (G-3)
        _restore_gate(trainer, panel, train, fold, gate_file)
    elif getattr(trainer.model.prior, "precomputed", False):
        gate: Any = trainer.model.prior
        _fit_gate(trainer, panel, train, fold)
        atomic_torch_save(gate.gate_state(), gate_file)
        gf = getattr(gate, "fit_result", None)
        if gf is not None:
            print(f"[gate] markov: llf {gf.llf:.2f}, durations "
                  f"{gf.expected_durations.round(1).tolist()}, perm {gf.permutation}, "
                  f"converged {gf.n_converged}/{gf.n_starts}")
    # gate frozen at construction -> base fitted on the training block and
    # frozen -> only then the experts train (brief 02 §4 ordering). On resume
    # the base weights come back with the checkpoint, and the warm-start must
    # NOT rerun: it already happened before step 0.
    if exp.base_enabled and not resumed:
        base_fit = cache.get_or_fit(train, cfg.base, window=window, seed=exp.seeds[0])
        trainer.model.attach_base(base_fit.model)
        print(f"[base] frozen f0: val MSE {base_fit.val_loss:.5f} after "
              f"{base_fit.steps_run} steps"
              f"{' (early stop)' if base_fit.stopped_early else ''}; "
              f"sigma_init={sigma:.4f}")
    if not resumed:  # the expert stage starts from a known RNG state (B-1)
        torch.manual_seed(expert_stage_seed(exp.seeds[0], fold.fold))
    # the dates the gate covers (all of them, except an AR gate's first
    # `order` training dates, audit G-2)
    expert_train, excluded = _gate_training_block(trainer, train)
    if excluded:
        print(f"[gate] {excluded} training date(s) have no filtered probability "
              "(autoregressive gate): left out of expert training")
    key = _warm_key(exp)
    if key is not None and not resumed and not (
        exp.correction_mode and exp.zero_init_head
    ):
        trainer.warmstart_experts(expert_train.full_batch(), sort_key=key(expert_train))

    trainer.on_heartbeat = _heartbeat(run, fold.fold)
    # the decay weights (brief 09 C), as the walk-forward harness uses them
    row_weight, weight_stats = fold_decay_weights(trainer, train, expert_train)
    if exp.prior == "hmm":
        trainer.fit_sequence(expert_train.time_sequence(), steps=remaining,
                             chunk_len=exp.chunk_len, checkpoint_path=ckpt)
        warm = trainer.evaluate_sequence(train.time_sequence())
        ev = trainer.evaluate_sequence(test.time_sequence(),
                                       init_state=warm.final_state)
        nll, pred = ev.nll, torch.cat(list(ev.y_hat))
    else:
        trainer.fit(expert_train, steps=remaining, checkpoint_path=ckpt,
                    row_weight=row_weight)
        nll = trainer.evaluate(test.full_batch())
        trainer.model.eval()
        with torch.no_grad():
            pred = trainer.model(
                test.x_seq, test.x_snap, PriorContext(date=test.date)
            ).y_hat
    history = trainer.history  # full trajectory, spanning any resumes
    print(f"[quick] held-out NLL = {nll:.4f}   "
          f"({len(train):,} train rows → {len(test):,} test rows)")

    results: dict = {"nll": nll}
    if exp.figures:
        plot_training_dashboard(history, n_experts=exp.n_experts,
                                path=out / "dashboard.png")
        # the dates the experts trained on and the test block; the gate also
        # covers the purge gap (Q22), but nothing is trained or scored there
        scored = panel.subset_dates(torch.cat([torch.unique(expert_train.date),
                                               fold.test_dates]))
        plot_gate_utilization(trainer, scored, path=out / "utilization.png")
        try:
            plot_ic_series(pred, test.y, test.date, path=out / "ic.png")
            if exp.backtest_quantiles:
                plot_long_short_curve(pred, test.y, test.date, test.entity,
                                      n_quantiles=exp.backtest_quantiles,
                                      cost_rate=exp.cost_rate, horizon=test.horizon,
                                      scheme=exp.portfolio_scheme, y_daily=test.y_daily,
                                      path=out / "ls.png")
        except ValueError as e:
            print(f"[figures] IC/LS skipped: {e}")
        if exp.prior == "hmm":
            plot_transition_matrix(trainer, path=out / "transition.png")

    if exp.calibration and exp.prior in ("soft", "gumbel"):
        from nec_moe import validation_tail
        tail = validation_tail(train.date, val_dates=max(10, len(set(train.date.tolist())) // 8),
                               purge_dates=purge)
        fit = fit_temperature(trainer, train.subset_dates(tail.test_dates))
        print(f"[calibration] {fit.verdict}")
        plot_reliability(gate_reliability(trainer, test), path=out / "reliability.png")
        plot_reliability(gate_reliability(trainer, test, temperature=fit.temperature),
                         path=out / "reliability_scaled.png")
        results["temperature"] = fit.temperature

    if exp.alignment and panel.date_labels is not None:
        ctx = build_context(load_vix(CACHE), load_french_factors(CACHE))
        report = gate_regime_alignment(trainer, test, ctx)
        print("[alignment] held-out gate-vs-context:\n", report.to_frame().round(3))
        run.write_metrics("alignment", report.to_frame())

    # every trial carries the parameter-count confound and, where the prior
    # has a transition matrix, how sticky the fitted chain actually is
    trial: dict[str, float] = {"nll": nll, "gate_excluded_dates": float(excluded),
                               **weight_stats}
    # §5: the base's own out-of-sample score beside every result, the
    # improvement over it, and the correction magnitude actually applied
    if trainer.model.base is not None:
        b_pred, b_nll, corr = base_and_correction(trainer, test, train=train)
        if b_pred is not None and b_nll is not None:
            # brief 06 C: NLL_single, NLL_base, NLL_full and the three gains
            b_single = base_single_gaussian_nll(trainer, test)
            trial["base_nll"] = b_nll
            trial["nll_improvement"] = b_nll - nll
            if b_single is not None:
                trial["base_single_nll"] = b_single
                trial["nll_variance_gain"] = b_single - b_nll
                trial["nll_total_gain"] = b_single - nll
            try:
                _, b_ics = rank_ic_by_date(b_pred, test.y, test.date)
                b_ic = ic_summary(b_ics).mean_ic
                trial["base_mean_ic"] = b_ic
                _, m_ics = rank_ic_by_date(pred, test.y, test.date)
                trial["ic_improvement"] = ic_summary(m_ics).mean_ic - b_ic
            except ValueError as e:  # constant cross-section: no ranking
                print(f"[base] IC comparison skipped: {e}")
        trial |= dict(corr)
        print(f"[base] held-out NLL {trial.get('base_nll', float('nan')):.4f} "
              f"vs mixture {nll:.4f}; correction |.| mean "
              f"{trial.get('correction_mean_abs', float('nan')):.4f}")
    if trainer.live_param_count is not None:
        trial["live_param_count"] = float(trainer.live_param_count)
    # Q24: each expert's training weight against the gate and the residual,
    # on the rows the experts trained on (aggregate only)
    from nec_moe.diagnostics import canonical_expert_order
    trial |= expert_weight_report(
        trainer, expert_train, order=canonical_expert_order(trainer.model.experts.log_sigma)
    )
    transition = getattr(trainer.model.prior, "transition_matrix", None)
    if transition is not None:
        from nec_moe.diagnostics import canonical_expert_order, persistence_metrics
        trial |= persistence_metrics(
            transition.detach(),
            canonical_expert_order(trainer.model.experts.log_sigma),
        )
    results |= {k: v for k, v in trial.items() if k != "nll"}
    run.write_metrics("quick", trial)
    # per-security: the test block's predictions go to Data/derived/runs/ only
    b_pred_all = (base_and_correction(trainer, test, train=train)[0]
                  if trainer.model.base is not None else None)
    _save_predictions(run.per_security_dir, 0, test, pred, b_pred_all)
    labels = panel.date_labels
    quick_years = (sorted({int(str(labels[int(d)])[:4]) for d in fold.test_dates})
                   if labels else None)
    registry.log(exp.tag, trial, config={"mode": "quick",
                 **dataclasses.asdict(exp),
                 **trial_provenance(panel, trainer.cfg, exp.portfolio_scheme),
                 # brief 09 I.1: quick mode's one chronological split
                 "fold_scheme": "quick", "protocol": "main", "test_years": quick_years,
                 **run_provenance(exp, run)},
                 seed=exp.seeds[0])
    return results


def fold_config(exp: Experiment) -> FoldConfig:
    return FoldConfig(
        fold_scheme=exp.fold_scheme,  # type: ignore[arg-type]
        first_test_year=exp.first_test_year,
        last_test_year=exp.last_test_year,
        pilot_validation_years=tuple(exp.pilot_validation_years),
    ).validate()


def main_folds(exp: Experiment, panel: Panel, purge: int) -> list[WalkForwardFold]:
    """Evaluate mode's folds: the calendar-year main folds (brief 09 B.1), or
    the count-based development split."""
    fc = fold_config(exp)
    if fc.fold_scheme == "calendar_year":
        return calendar_year_folds(
            panel, first_test_year=fc.first_test_year, last_test_year=fc.last_test_year,
            purge_dates=purge,
        )
    return walk_forward_folds(
        panel.date, n_folds=exp.n_folds, test_dates_per_fold=exp.test_dates_per_fold,
        purge_dates=purge,
    )


def _evaluate(exp: Experiment, panel: Panel, purge: int, run: Run) -> dict:
    # One cache for the whole sweep, seeded with the sigma probe: the base of
    # the earliest training window is fitted once here and reused by fold 0
    # of every arm rather than refitted.
    cache = BaseCache()
    folds = main_folds(exp, panel, purge)
    first = folds[0]
    first_train = panel.subset_dates(first.train_dates)
    window = (int(first.train_dates.min()), int(first.train_dates.max()))
    sigma = _auto_sigma(exp, panel, first_train, cache, window)
    cfg = _nec_config(exp, panel, sigma)
    warm = _warm_key(exp)
    if exp.correction_mode and exp.zero_init_head:
        warm = None  # would make the heads nonzero; see Trainer.warmstart_experts
    arms = [nec_arm(exp.prior, cfg, warmstart_key=warm)]
    d_snap = panel.x_snap.shape[1]
    if exp.include_ridge:
        arms.append(baseline_arm("ridge", lambda seed: RidgeBaseline(l2=1.0)))
    if exp.include_mlp:
        arms.append(baseline_arm("mlp", lambda seed: MLPBaseline(
            input_dim=d_snap, hidden_dims=tuple(exp.expert_hidden_dims),
            steps=exp.steps, seed=seed)))

    registry = TrialRegistry(run.trials)
    out = run.figures_dir
    resume_dir = run.checkpoints_dir  # always: a Ctrl+C leaves resumable state
    report = run_sweep(panel, arms, seeds=exp.seeds, registry=registry, tag=exp.tag,
                       steps=exp.steps, folds=folds, purge_dates=purge,
                       backtest_quantiles=exp.backtest_quantiles,
                       cost_rate=exp.cost_rate, resume_dir=resume_dir,
                       base_cache=cache, hac_lags=exp.ic_hac_lags,
                       hac_kernel=exp.ic_hac_kernel,
                       portfolio_scheme=exp.portfolio_scheme, run=run,
                       provenance_extra=run_provenance(exp, run))
    frame = report.to_frame().round(4)
    print("\n[evaluate] mean ± seed-std per arm:\n", frame.to_string())
    run.write_metrics("report", frame)
    if len(arms) > 1:
        claims = corrected_claims(report, alpha=0.10)
        print("[evaluate] BH-corrected claims (arm: reject, q):", claims)
    if exp.figures:
        plot_sweep_report(report, metric="mean_ic", path=out / "sweep_ic.png")
        if exp.backtest_quantiles:
            plot_sweep_report(report, metric="net_ir", path=out / "sweep_net_ir.png")
    return {"report": report}


class PreRegistrationRequired(ValueError):
    """A run would score the test period without the pilot's selection file
    (brief 09 F.3, Q15)."""


def scored_dates(exp: Experiment, panel: Panel, purge: int) -> torch.Tensor:
    """The date codes this run scores predictions on: quick mode's test
    split, or every fold's test block."""
    if exp.mode == "quick":
        return _quick_split(exp, panel, purge)[2].test_dates
    return torch.cat([f.test_dates for f in main_folds(exp, panel, purge)])


def touches_test_period(exp: Experiment, panel: Panel, purge: int) -> bool:
    """Does the run score any date from ``first_test_year`` on? A panel
    without a calendar (the plain synthetic generator) has no test period."""
    if panel.date_labels is None:
        return False
    labels = panel.date_labels
    return any(int(str(labels[int(d)])[:4]) >= exp.first_test_year
               for d in scored_dates(exp, panel, purge))


def selection_lock(exp: Experiment, panel: Panel, purge: int) -> dict[str, Any]:
    """The pre-registration lock (brief 09 F.3). Refuses a run that scores
    the test period without ``pilot_selection_file``, and a run whose
    settings differ from the selection's chosen ones unless
    ``force_deviation``. Returns what the run records: the file, its
    SHA-256 (recomputed from the file) and any forced deviations."""
    needed = touches_test_period(exp, panel, purge)
    if exp.pilot_selection_file is None:
        if needed:
            raise PreRegistrationRequired(
                f"this run scores dates from {exp.first_test_year} on (the test period) "
                "but no pilot_selection_file is set. Run scripts/run_pilot.py first, "
                "then pass its results/pilot/pilot_selection.json: the settings are "
                "chosen once on 2007-2009 and frozen (Q15)"
            )
        return {"pilot_selection_file": None, "pilot_selection_hash": None,
                "pilot_deviations": {}}
    selection, digest = load_selection(_repo_path(exp.pilot_selection_file))
    deviations = selection_deviations(selection, dataclasses.asdict(exp))
    if deviations and not exp.force_deviation:
        detail = "; ".join(f"{k}: chosen {c!r}, given {g!r}" for k, (c, g) in deviations.items())
        raise ValueError(
            f"settings differ from the pilot selection ({detail}). Use the chosen "
            "settings, or set force_deviation=True (recorded in run.json and every "
            "trial row)"
        )
    if deviations:
        print(f"[lock] WARNING - forced deviation from the pilot selection: {deviations}")
    return {"pilot_selection_file": str(exp.pilot_selection_file),
            "pilot_selection_hash": digest,
            "pilot_deviations": encode_settings({k: list(v) for k, v in deviations.items()})}


def run_provenance(exp: Experiment, run: Run) -> dict[str, Any]:
    """The run-level provenance every trial row carries: the pilot-selection
    hash (the pre-registration lock, F.3) and whether a deviation was forced."""
    info = run.info()
    return {"pilot_selection_hash": info.get("pilot_selection_hash"),
            "force_deviation": bool(info.get("pilot_deviations"))}


def _snapshot(exp: Experiment) -> dict[str, Any]:
    return json.loads(json.dumps(dataclasses.asdict(exp)))  # tuples -> lists


def _touches_test(panel: Panel) -> bool:
    """Both modes score a test block: true unless the panel is synthetic (Q15)."""
    return panel.data_source != "synthetic"


def _open_run(exp: Experiment, store: RunStore, panel: Panel) -> Run:
    """A new run, or with ``resume`` the run to continue (brief 07 C.6, C.7)."""
    if not exp.resume:
        return store.create(
            campaign=exp.campaign, tag=exp.tag, mode=exp.mode,
            settings=_snapshot(exp), purpose=exp.purpose,
            data_source=panel.data_source, fingerprint=data_fingerprint(panel),
            touches_test=_touches_test(panel), default_settings=_snapshot(Experiment()),
        )
    run_id = exp.resume_run_id or store.latest(
        tag=exp.tag, campaign=exp.campaign, status=RESUMABLE_STATES
    )
    if run_id is None:
        raise ValueError(
            f"nothing to resume: no interrupted or crashed run of tag {exp.tag!r} "
            f"(campaign {exp.campaign!r}) in {store.index_path}"
        )
    run = store.open(run_id)
    problems = _resume_problems(run, exp, panel)
    if problems:
        detail = "; ".join(problems)
        if not exp.force:
            raise ResumeRefused(
                f"refusing to resume {run_id}: {detail}. Resuming would mix two "
                "different runs; start a new run, or pass --force to override"
            )
        print(f"[resume] WARNING — FORCED RESUME of {run_id} despite: {detail}. "
              "The run's results now mix settings/data; recorded in run.json.")
        forced = run.info().get("forced_resumes", [])
        run.record(forced_resumes=[*forced, {"at": pd.Timestamp.now().isoformat(),
                                              "changes": problems}])
    run.status(state="running", exit_reason="")
    store.update_index(run_id, status="running", finished="", exit_reason="")
    return run


def _resume_problems(run: Run, exp: Experiment, panel: Panel) -> list[str]:
    """What differs between a stored run and this launch: settings (by hash,
    then key by key) and data (by fingerprint)."""
    problems = []
    stored, current = run.settings(), _snapshot(exp)
    if settings_hash(stored) != settings_hash(current):
        changed = sorted(
            k for k in set(stored) | set(current)
            if k not in ("resume", "resume_run_id", "force") and stored.get(k) != current.get(k)
        )
        problems.append("settings changed: " + ", ".join(
            f"{k}: {stored.get(k)!r} -> {current.get(k)!r}" for k in changed
        ))
    before, now = run.info().get("data_fingerprint"), data_fingerprint(panel)
    if before != now:
        problems.append(f"data changed (fingerprint {before} -> {now})")
    return problems


def resume_run(run_id: str, *, root: str | None = None, force: bool = False) -> dict:
    """Resume run ``run_id`` with its own stored settings (``runs.py resume``)."""
    store = RunStore(root if root is not None else _repo_path(Experiment().out_dir))
    settings = store.open(run_id).settings()
    exp = Experiment(**settings)
    exp = dataclasses.replace(exp, resume=True, resume_run_id=run_id, force=force,
                              out_dir=str(store.root))
    return main(exp)


def main(exp: Experiment) -> dict:
    if exp.mode not in ("quick", "evaluate"):
        raise ValueError(f"unknown mode {exp.mode!r} (use 'quick' or 'evaluate')")
    store = RunStore(
        _repo_path(exp.out_dir),
        _repo_path(exp.per_security_dir) if exp.per_security_dir is not None else None,
    )
    panel, purge = _build_panel(exp)
    lock = selection_lock(exp, panel, purge)  # before anything is created
    run = _open_run(exp, store, panel)
    run.record(**lock)
    with run.logging(), graceful_interrupts():
        n_dates = len(torch.unique(panel.date))
        print(f"[run] {run.run_id} (campaign {run.campaign!r}) -> {run.dir}")
        print(f"[data] {exp.data}: {len(panel):,} rows, {n_dates} dates, purge={purge}")
        try:
            if exp.mode == "quick":
                results = _quick(exp, panel, purge, run)
                metrics: Any = {k: v for k, v in results.items() if isinstance(v, float)}
            else:
                results = _evaluate(exp, panel, purge, run)
                metrics = results["report"].to_frame().round(4)
        except KeyboardInterrupt as exc:  # RunInterrupted: checkpointed first
            run.finish("interrupted", f"{type(exc).__name__}: {exc}")
            raise
        except Exception as exc:
            run.finish("failed", f"{type(exc).__name__}: {exc}")
            raise
        run.finish("completed", metrics=metrics)
    return results | {"run_id": run.run_id, "run_dir": run.dir}


def _parse_cli(exp: Experiment, argv: list[str]) -> Experiment:
    """``--resume [run_id]``: continue the latest interrupted or crashed run of
    the EXPERIMENT block's tag, or the named run. ``--force`` resumes despite
    changed settings or data (recorded)."""
    if "--force" in argv:
        exp = dataclasses.replace(exp, force=True)
    if "--resume" not in argv:
        return exp
    exp = dataclasses.replace(exp, resume=True)
    i = argv.index("--resume")
    if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
        exp = dataclasses.replace(exp, resume_run_id=argv[i + 1])
    return exp


if __name__ == "__main__":
    try:
        main(_parse_cli(EXPERIMENT, sys.argv[1:]))
    except KeyboardInterrupt:
        print("[run] interrupted: resume with  python3.14 run_experiment.py --resume",
              file=sys.stderr)
        sys.exit(130)
