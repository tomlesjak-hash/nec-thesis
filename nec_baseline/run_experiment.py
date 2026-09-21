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
duplicated here. Outputs land in ``results/<tag>/``: a ``settings.json``
snapshot (full provenance), the trial registry (``trials.jsonl``), and the
figures. The handbook (HANDBOOK.md Part II) documents every underlying knob.

Long runs are interruptible: set ``checkpoint_every`` (steps between saves),
kill the process whenever, and continue with

    python3.14 run_experiment.py --resume            # the EXPERIMENT block's tag
    python3.14 run_experiment.py --resume results/my_experiment   # a specific run

Resume is exact — completed work (sweep runs, folds) is skipped, an
interrupted fit continues from its last checkpoint on the same trajectory —
provided the settings and data are unchanged (a changed ``settings.json``
prints a loud warning).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nec_moe import (  # noqa: E402
    BaseCache,
    BaseConfig,
    DataConfig,
    EncoderConfig,
    ExpertConfig,
    MLPBaseline,
    NECConfig,
    NECModel,
    Panel,
    PriorConfig,
    RidgeBaseline,
    StageBSpec,
    SyntheticRegimePanel,
    SyntheticSpec,
    TrainConfig,
    Trainer,
    TrialRegistry,
    base_and_correction,
    baseline_arm,
    build_context,
    build_stage_b_panel,
    corrected_claims,
    data_config_from_panel,
    filter_point_in_time,
    fit_temperature,
    gate_regime_alignment,
    gate_reliability,
    ic_summary,
    load_french_factors,
    load_sp500_universe,
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
    universe_coverage_report,
    walk_forward_folds,
)

# =========================================================================== #
#  SETTINGS — this is the only part you edit
# =========================================================================== #


@dataclass
class Experiment:
    # ---------------- run identity ----------------
    tag: str = "my_experiment"      # names the results/<tag>/ folder + registry
    mode: str = "quick"             # "quick" (one model + diagnostics) | "evaluate"
    out_dir: str = "results"        # where outputs land (results/<tag>/)

    # ---------------- data ----------------
    data: str = "synthetic"         # "synthetic" | "real" | "panel_file"
    # real data (data="real"):
    start: str = "2018-01-01"
    end: str = "2024-12-31"
    source: str = "yfinance"        # "yfinance" | "stooq" (see handbook II.4)
    point_in_time: bool = False     # PIT universe filter + coverage table (slow 1st run)
    # panel file (data="panel_file") — e.g. the prebuilt PIT panel:
    panel_file: str = "data_cache/pit_panel_2015_2024.pt"
    # feature/target spec (real + panel builds):
    seq_len: int = 20               # encoder window T
    horizon: int = 5                # forward-return days; ALSO the purge length
    target_kind: str = "raw"        # "raw" | "residual" (market-neutral target)
    # synthetic data (data="synthetic"):
    synth_dates: int = 300
    synth_entities: int = 8
    synth_regime_process: str = "iid"      # "iid" | "markov" (sticky regimes)
    synth_vol_levels: tuple[float, float] = (0.5, 2.5)  # regime separability

    # ---------------- model ----------------
    prior: str = "soft"             # soft|uniform|hard|topk|gumbel|hmm
    emission: str = "mlp"           # "mlp" (neural experts) | "classical" (Hamilton-style)
    n_experts: int = 2              # Ye & Borde ablate K in {2, 4, 6}
    encoder_hidden: int = 32
    encoder_layers: int = 1
    expert_hidden_dims: tuple[int, ...] = (64, 32)  # depth AND width; see pyramid_dims
    expert_dropout: float = 0.05
    expert_activation: str = "relu"  # relu|gelu|tanh|silu|elu (registry key)
    input_mode: str = "snapshot"    # "snapshot" | "snapshot_plus_hidden" (Decision A)
    top_k: int = 2                  # topk prior only
    tau_init: float = 1.0           # gumbel prior only …
    tau_anneal_steps: int = 0       # … 0 = constant temperature
    transition_diag_bias: float = 2.0  # hmm prior only: persistence-biased init

    # ---------------- training ----------------
    steps: int = 600                # optimizer steps (per fold in evaluate mode)
    lr: float = 1e-3
    batch_size: int = 256
    sigma_init: float | None = None # None = auto (std of the training target)
    sigma_freeze_steps: int = 100
    seeds: tuple[int, ...] = (0,)   # >1 seed => mean ± std in evaluate mode
    warmstart: bool = True          # expert warm-start (keep on; handbook I.11)
    warmstart_channel: int | None = None  # seq channel of the vol proxy; None = auto
    chunk_len: int = 50             # HMM truncated-BPTT chunk (dates per step)

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
    base_val_fraction: float = 0.2    # tail of the TRAINING block only
    base_seed_offset: int = 0
    # alpha of the correction-magnitude penalty. Ye & Borde do not report
    # theirs: select on the training block, log every candidate as a trial,
    # and publish the sensitivity curve (brief 02 §3).
    aux_correction_penalty: bool = False
    correction_penalty_weight: float = 0.0

    # ---------------- checkpointing / resume ----------------
    checkpoint_every: int = 0       # save training state every N steps (0 = off);
                                    #   set for any run you might interrupt
    resume: bool = False            # continue results/<tag>/checkpoints/ (CLI: --resume)

    # ---------------- evaluation (mode="evaluate") ----------------
    n_folds: int = 3
    test_dates_per_fold: int = 40
    include_ridge: bool = True      # baseline rows in the comparison table
    include_mlp: bool = True
    backtest_quantiles: int | None = 5   # None disables the long-short backtest
    cost_rate: float = 0.001        # cost per unit traded notional (10 bps)

    # ---------------- extras ----------------
    figures: bool = True
    calibration: bool = True        # reliability + temperature (soft/gumbel gates)
    alignment: bool = False         # gate-vs-VIX/factors report (real panels)
    quick_train_frac: float = 0.8   # quick mode's chronological split


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
    """-> (panel, purge_dates). Purge = label horizon for real data; the
    synthetic target has no forward overlap, so purge 0 is the honest value."""
    if exp.data == "synthetic":
        spec = SyntheticSpec(
            regime_process=exp.synth_regime_process,  # type: ignore[arg-type]
            vol_levels=exp.synth_vol_levels,
            beta_scale=2.0,
            noise_std=0.4,
            seed=0,
        )
        return SyntheticRegimePanel(spec).generate(exp.synth_dates, exp.synth_entities), 0
    if exp.data == "panel_file":
        panel = torch.load(exp.panel_file, weights_only=False)
        return panel, exp.horizon
    if exp.data == "real":
        bspec = StageBSpec(
            seq_len=exp.seq_len,
            horizon=exp.horizon,
            target_kind=exp.target_kind,  # type: ignore[arg-type]
        )
        if exp.point_in_time:
            u = load_sp500_universe(CACHE)
            tickers = tuple(sorted(u.members_union(exp.start, exp.end)))
            panel = build_stage_b_panel(exp.start, exp.end, str(CACHE),
                                        tickers=tickers, source=exp.source, spec=bspec)
            panel = filter_point_in_time(panel, u)
            years = range(int(exp.start[:4]) + 1, int(exp.end[:4]) + 1)
            print(universe_coverage_report(
                u, [f"{y}-01-05" for y in years],
                available=set(panel.entity_labels or ())))
        else:
            print("[note] default universe is survivorship-biased — "
                  "pipeline-grade only (set point_in_time=True for claims)")
            panel = build_stage_b_panel(exp.start, exp.end, str(CACHE),
                                        source=exp.source, spec=bspec)
        return panel, exp.horizon
    raise ValueError(f"unknown data mode {exp.data!r}")


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
                             zero_init_head=exp.zero_init_head),
        prior=PriorConfig(kind=exp.prior, top_k=exp.top_k, tau_init=exp.tau_init,
                          tau_anneal_steps=exp.tau_anneal_steps,
                          transition_diag_bias=exp.transition_diag_bias),
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
                        seed_offset=exp.base_seed_offset),
        train=TrainConfig(lr=exp.lr, batch_size=exp.batch_size, steps=exp.steps,
                          sigma_init=sigma_init,
                          sigma_freeze_steps=exp.sigma_freeze_steps,
                          sequence_ordered=(exp.prior == "hmm"),
                          checkpoint_every=exp.checkpoint_every,
                          freeze_gate=exp.freeze_gate,
                          aux_correction_penalty=exp.aux_correction_penalty,
                          correction_penalty_weight=exp.correction_penalty_weight),
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


def _proxy_channel(exp: Experiment) -> int:
    if exp.warmstart_channel is not None:
        return exp.warmstart_channel
    return 0 if exp.data == "synthetic" else 2  # synth regime channel | vol_20d


def _warm_key(exp: Experiment):
    ch = _proxy_channel(exp)
    return (lambda p: p.x_seq[:, :, ch].std(dim=1)) if exp.warmstart else None


def _quick(exp: Experiment, panel: Panel, purge: int, out: Path,
           registry: TrialRegistry) -> dict:
    train, test = panel.split_by_date(exp.quick_train_frac)
    cache = BaseCache()
    window = ("quick", int(train.date.min()), int(train.date.max()))
    sigma = _auto_sigma(exp, panel, train, cache, window)
    cfg = _nec_config(exp, panel, sigma)
    ckpt = (out / "checkpoints" / "trainer.pt"
            if (exp.checkpoint_every or exp.resume) else None)
    resumed = exp.resume and ckpt is not None and ckpt.exists()
    if resumed:
        trainer = Trainer.load(ckpt)  # type: ignore[arg-type]
        remaining = max(exp.steps - trainer.step_count, 0)
        print(f"[resume] {ckpt} at step {trainer.step_count:,} — "
              f"{remaining:,} steps remaining")
    else:
        torch.manual_seed(exp.seeds[0])
        trainer = Trainer(NECModel(cfg))
        remaining = exp.steps
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
    key = _warm_key(exp)
    if key is not None and not resumed and not (
        exp.correction_mode and exp.zero_init_head
    ):
        trainer.warmstart_experts(train.full_batch(), sort_key=key(train))

    if exp.prior == "hmm":
        trainer.fit_sequence(train.time_sequence(), steps=remaining,
                             chunk_len=exp.chunk_len, checkpoint_path=ckpt)
        warm = trainer.evaluate_sequence(train.time_sequence())
        ev = trainer.evaluate_sequence(test.time_sequence(),
                                       init_state=warm.log_filtered[-1])
        nll, pred = ev.nll, torch.cat(list(ev.y_hat))
    else:
        trainer.fit(train, steps=remaining, checkpoint_path=ckpt)
        nll = trainer.evaluate(test.full_batch())
        trainer.model.eval()
        with torch.no_grad():
            pred = trainer.model(test.x_seq, test.x_snap).y_hat
    history = trainer.history  # full trajectory, spanning any resumes
    print(f"[quick] held-out NLL = {nll:.4f}   "
          f"({len(train):,} train rows → {len(test):,} test rows)")

    results: dict = {"nll": nll}
    if exp.figures:
        plot_training_dashboard(history, n_experts=exp.n_experts,
                                path=out / "dashboard.png")
        plot_gate_utilization(trainer, panel, path=out / "utilization.png")
        try:
            plot_ic_series(pred, test.y, test.date, path=out / "ic.png")
            if exp.backtest_quantiles:
                plot_long_short_curve(pred, test.y, test.date, test.entity,
                                      n_quantiles=exp.backtest_quantiles,
                                      cost_rate=exp.cost_rate, path=out / "ls.png")
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
        report.to_frame().to_csv(out / "alignment.csv")

    # every trial carries the parameter-count confound and, where the prior
    # has a transition matrix, how sticky the fitted chain actually is
    trial: dict[str, float] = {"nll": nll}
    # §5: the base's own out-of-sample score beside every result, the
    # improvement over it, and the correction magnitude actually applied
    if trainer.model.base is not None:
        b_pred, b_nll, corr = base_and_correction(trainer, test)
        if b_pred is not None and b_nll is not None:
            trial["base_nll"] = b_nll
            trial["nll_improvement"] = b_nll - nll
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
    transition = getattr(trainer.model.prior, "transition_matrix", None)
    if transition is not None:
        from nec_moe.diagnostics import canonical_expert_order, persistence_metrics
        trial |= persistence_metrics(
            transition.detach(),
            canonical_expert_order(trainer.model.experts.log_sigma),
        )
    results |= {k: v for k, v in trial.items() if k != "nll"}
    registry.log(exp.tag, trial, config={"mode": "quick",
                 **dataclasses.asdict(exp)}, seed=exp.seeds[0])
    return results


def _evaluate(exp: Experiment, panel: Panel, purge: int, out: Path,
              registry: TrialRegistry) -> dict:
    # One cache for the whole sweep, seeded with the sigma probe: the base of
    # the earliest training window is fitted once here and reused by fold 0
    # of every arm rather than refitted.
    cache = BaseCache()
    first = walk_forward_folds(
        panel.date, n_folds=exp.n_folds,
        test_dates_per_fold=exp.test_dates_per_fold, purge_dates=purge,
    )[0]
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

    resume_dir = (out / "checkpoints"
                  if (exp.checkpoint_every or exp.resume) else None)
    report = run_sweep(panel, arms, seeds=exp.seeds, registry=registry, tag=exp.tag,
                       steps=exp.steps, n_folds=exp.n_folds,
                       test_dates_per_fold=exp.test_dates_per_fold, purge_dates=purge,
                       backtest_quantiles=exp.backtest_quantiles,
                       cost_rate=exp.cost_rate, resume_dir=resume_dir,
                       base_cache=cache)
    frame = report.to_frame().round(4)
    print("\n[evaluate] mean ± seed-std per arm:\n", frame.to_string())
    frame.to_csv(out / "report.csv")
    if len(arms) > 1:
        claims = corrected_claims(report, alpha=0.10)
        print("[evaluate] BH-corrected claims (arm: reject, q):", claims)
    if exp.figures:
        plot_sweep_report(report, metric="mean_ic", path=out / "sweep_ic.png")
        if exp.backtest_quantiles:
            plot_sweep_report(report, metric="net_ir", path=out / "sweep_net_ir.png")
    return {"report": report}


def main(exp: Experiment) -> dict:
    base = Path(exp.out_dir)
    if not base.is_absolute():
        base = Path(__file__).resolve().parent / base
    out = base / exp.tag
    out.mkdir(parents=True, exist_ok=True)
    settings_path = out / "settings.json"
    snapshot = json.loads(json.dumps(dataclasses.asdict(exp)))  # tuples -> lists
    if exp.resume and settings_path.exists():
        previous = json.loads(settings_path.read_text())
        changed = sorted(k for k in snapshot
                         if k != "resume" and previous.get(k) != snapshot[k])
        if changed:
            print(f"[resume] WARNING: settings changed since launch: {changed} — "
                  "resume assumes identical settings and data")
    settings_path.write_text(json.dumps(snapshot, indent=2))
    registry = TrialRegistry(out / "trials.jsonl")

    panel, purge = _build_panel(exp)
    n_dates = len(torch.unique(panel.date))
    print(f"[data] {exp.data}: {len(panel):,} rows, {n_dates} dates, "
          f"purge={purge}; outputs → {out}")

    if exp.mode == "quick":
        return _quick(exp, panel, purge, out, registry)
    if exp.mode == "evaluate":
        return _evaluate(exp, panel, purge, out, registry)
    raise ValueError(f"unknown mode {exp.mode!r} (use 'quick' or 'evaluate')")


def _parse_cli(exp: Experiment, argv: list[str]) -> Experiment:
    """``--resume [path]``: resume the EXPERIMENT block's run, or — with a
    path to ``results/<tag>`` (or its ``checkpoints/``) — that specific run."""
    if "--resume" not in argv:
        return exp
    exp = dataclasses.replace(exp, resume=True)
    i = argv.index("--resume")
    if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
        run_dir = Path(argv[i + 1]).resolve()
        if run_dir.name == "checkpoints":
            run_dir = run_dir.parent
        exp = dataclasses.replace(exp, tag=run_dir.name, out_dir=str(run_dir.parent))
    return exp


if __name__ == "__main__":
    main(_parse_cli(EXPERIMENT, sys.argv[1:]))
