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
    baseline_arm,
    build_context,
    build_stage_b_panel,
    corrected_claims,
    data_config_from_panel,
    filter_point_in_time,
    fit_temperature,
    gate_regime_alignment,
    gate_reliability,
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
    run_sweep,
    universe_coverage_report,
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
    n_experts: int = 2
    encoder_hidden: int = 32
    encoder_layers: int = 1
    expert_hidden: int = 64
    expert_dropout: float = 0.05
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
        experts=ExpertConfig(n_experts=exp.n_experts, hidden_dim=exp.expert_hidden,
                             dropout=exp.expert_dropout,
                             input_mode=exp.input_mode,  # type: ignore[arg-type]
                             kind=exp.emission),  # type: ignore[arg-type]
        prior=PriorConfig(kind=exp.prior, top_k=exp.top_k, tau_init=exp.tau_init,
                          tau_anneal_steps=exp.tau_anneal_steps,
                          transition_diag_bias=exp.transition_diag_bias),
        train=TrainConfig(lr=exp.lr, batch_size=exp.batch_size, steps=exp.steps,
                          sigma_init=sigma_init,
                          sigma_freeze_steps=exp.sigma_freeze_steps,
                          sequence_ordered=(exp.prior == "hmm"),
                          checkpoint_every=exp.checkpoint_every),
    )


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
    sigma = exp.sigma_init or float(train.y.std())
    cfg = _nec_config(exp, panel, sigma)
    ckpt = (out / "checkpoints" / "trainer.pt"
            if (exp.checkpoint_every or exp.resume) else None)
    if exp.resume and ckpt is not None and ckpt.exists():
        trainer = Trainer.load(ckpt)
        remaining = max(exp.steps - trainer.step_count, 0)
        print(f"[resume] {ckpt} at step {trainer.step_count:,} — "
              f"{remaining:,} steps remaining")
    else:
        torch.manual_seed(exp.seeds[0])
        trainer = Trainer(NECModel(cfg))
        key = _warm_key(exp)
        if key is not None:
            trainer.warmstart_experts(train.full_batch(), sort_key=key(train))
        remaining = exp.steps

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
    sigma = exp.sigma_init or float(panel.y.std())
    cfg = _nec_config(exp, panel, sigma)
    arms = [nec_arm(exp.prior, cfg, warmstart_key=_warm_key(exp))]
    d_snap = panel.x_snap.shape[1]
    if exp.include_ridge:
        arms.append(baseline_arm("ridge", lambda seed: RidgeBaseline(l2=1.0)))
    if exp.include_mlp:
        arms.append(baseline_arm("mlp", lambda seed: MLPBaseline(
            input_dim=d_snap, hidden_dim=exp.expert_hidden,
            steps=exp.steps, seed=seed)))

    resume_dir = (out / "checkpoints"
                  if (exp.checkpoint_every or exp.resume) else None)
    report = run_sweep(panel, arms, seeds=exp.seeds, registry=registry, tag=exp.tag,
                       steps=exp.steps, n_folds=exp.n_folds,
                       test_dates_per_fold=exp.test_dates_per_fold, purge_dates=purge,
                       backtest_quantiles=exp.backtest_quantiles,
                       cost_rate=exp.cost_rate, resume_dir=resume_dir)
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
