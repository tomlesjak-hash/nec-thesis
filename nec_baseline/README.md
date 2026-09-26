# nec_baseline — corrected NEC (regime-conditional MoE)

Research baseline for the thesis. **Design spec:** `../NEC_sources/NEC_baseline_design.md`
— every architectural choice traces to that document (defect ledger fixes, mixture-NLL
objective, shared-likelihood / pluggable-prior gate abstraction serving both thesis
variations).

The original prototype this corrects lives untouched at `../OP model/nec/nec_hybrid_explained.py`.

## The control panel (start here)

All the knobs in one file — edit the `Experiment(...)` block at the top of
[`run_experiment.py`](run_experiment.py) (data, model, prior, training length, seeds,
evaluation) and run:

```bash
python3.14 run_experiment.py
```

`mode="quick"` = one model + full diagnostics (figures, calibration); `mode="evaluate"`
= multi-seed purged walk-forward with baselines and corrected claims. Outputs +
provenance land in `results/<tag>/`. Everything below is the machinery it drives.

Long runs are interruptible: set `checkpoint_every=<N>` in the settings block, kill the
process whenever, continue with `python3.14 run_experiment.py --resume` — completed
sweep runs and folds are skipped, an interrupted fit resumes from its last (atomically
written) checkpoint on the bit-exact same trajectory (`Trainer.save/load`; handbook
II.6).

## Quickstart (smoke: train on synthetic regime data)

```bash
cd nec_baseline
python3.14 -m pytest tests/ -x -q          # full suite (< ~90 s CPU)
python3.14 -m pytest tests/test_smoke.py -q  # just the end-to-end smoke test
```

```python
from nec_moe import (DataConfig, NECConfig, NECModel, SyntheticRegimePanel,
                     SyntheticSpec, Trainer)

spec = SyntheticSpec(seed=0)
panel = SyntheticRegimePanel(spec).generate(n_dates=300, n_entities=8)
train, test = panel.split_by_date(0.8)

cfg = NECConfig(data=DataConfig(d_seq=spec.d_seq, d_snap=spec.d_snap, seq_len=spec.seq_len))
model = NECModel(cfg)
trainer = Trainer(model)

# Break the symmetric-mixture local optimum (design doc §4.3): warm-start each
# expert on a slice of the data sorted by an observable regime proxy (window
# vol here; VIX / realized vol on real data). The gate stays unsupervised.
proxy = train.x_seq[:, :, 0].std(dim=1)
trainer.warmstart_experts(train.full_batch(), sort_key=proxy)

trainer.fit(train, steps=500)
print("held-out NLL:", trainer.evaluate(test.full_batch()))
```

> **Why the warm-start?** With a genuinely symmetric two-regime task (sign-flipped
> per-regime coefficients), both experts otherwise collapse to the pooled regression and
> the gate has no signal to learn — the classic symmetric-mixture local optimum (Module 8).
> Seed diversity alone doesn't escape it; the vol/VIX-sorted-slice warm-start is the design
> doc's prescribed device. It is a training step, not a model change, and does not supervise
> the gate.

## Routing mechanisms are a config sweep

`PriorConfig.kind` selects the gate's prior: `soft` (baseline), `uniform`
(frozen-uniform-gate ablation — no learned routing), `hard` (top-1 with the hard-EM
objective; the *trainable* version of the prototype's argmax), `topk` (k ≥ 2 enforced —
top-1-renormalized is gradient-dead, use `hard`), `gumbel` (annealed temperature),
`hmm`, and `markov` (Hamilton, fitted separately and frozen).

> **`hmm` is not Hamilton.** `HMMRegimePrior` is a jointly fitted latent Markov mixture
> with homogeneous, covariate-independent transitions, estimated by gradient descent
> through the forward recursion rather than by maximum likelihood; the encoder and gate
> head are dormant under it (it uses the gate logits for shape inference only). Under
> Q19 it is a **baseline arm**, not one of the compared regime mechanisms. The Hamilton
> arm is `prior.kind="markov"` (`MarkovSwitchingRegimePrior`), fitted by maximum
> likelihood on each fold's training block, canonically reordered, and applied to the
> test block with frozen parameters through the **filter** — never the smoother.
> Comparing the two estimators' transition matrices and expected durations is itself a
> reportable result. `ExpertConfig.kind` selects the emission along the second plug
axis: `mlp` (neural experts) or `classical` (per-regime constant Gaussians —
`classical` × `hmm` **is the classical Hamilton baseline**, parameter-recovery-tested;
compare it on held-out NLL and regime recovery, not rank-IC: constant per-date
predictions have no cross-sectional ranking). The HMM-gated variant:

```python
from nec_moe import PriorConfig, TrainConfig
cfg = NECConfig(
    data=DataConfig(d_seq=2, d_snap=3, seq_len=8),
    prior=PriorConfig(kind="hmm"),
    train=TrainConfig(sequence_ordered=True),
)
# then Trainer.fit_sequence(panel.time_sequence(), ...) — filtering only, never smoothing.
```

## Walk-forward evaluation (purged, fit-once-per-window)

```python
from nec_moe import walk_forward_evaluate, Trainer, NECModel

result = walk_forward_evaluate(
    panel,
    make_trainer=lambda: Trainer(NECModel(cfg)),   # fresh model per window
    n_folds=4, test_dates_per_fold=20,
    purge_dates=5,          # = the label horizon: drops train dates whose
                            #   forward-return label overlaps the test block
    steps=400,
    warmstart_key=lambda p: p.x_seq[:, :, 0].std(dim=1),
    backtest_quantiles=4,   # optional: quantile long-short backtest
    cost_rate=0.001,        #   cost per unit traded notional (10 bps)
)
print(result.pooled_ic)         # mean rank-IC, ICIR, t-stat over all test dates
print(result.pooled_portfolio)  # long-short gross/net, IR, turnover, cost drag
for f in result.folds:          # per-fold NLL, IC, portfolio, expert order (σ-sort)
    print(f)
```

Works for both prior families: HMM folds are scored by a strictly-causal filtering pass
warmed through the (past) training window. Still missing from the full Module 5 protocol:
deflated Sharpe (needs the trial registry) and multiple-testing corrections.

## Stage B: CRSP daily data (licensed)

```python
from nec_moe import CRSPSpec, StageBSpec, build_crsp_panel, data_config_from_panel

spec = CRSPSpec()   # release ciz202512, S&P 500 (INDNO 1000500), 2015-01-01..2024-12-31
build = build_crsp_panel(spec, StageBSpec(seq_len=20, horizon=5))  # from Data/derived/
panel = build.panel      # entity labels are PERMNOs; panel.data_source == "crsp_ciz202512"
print(build.coverage)    # per year: members with and without usable rows
cfg = NECConfig(data=data_config_from_panel(panel))
# panel plugs into Trainer / walk_forward_evaluate unchanged
```

Two scripts, run from `nec_baseline/` (they need the `crsp` extra, i.e. pyarrow):

1. `python3.14 scripts/extract_crsp.py` streams the CRSP CIZ files once (from the
   extracted `Data/crspdata/ciz202512_ascii/` copy, else from `Data/ciz202512_ascii.zip`,
   never unzipped whole) and keeps the rows of every PERMNO that was an S&P 500 member in
   the window, with lookback and lead. Resumable: the byte offset of every block is on
   record.
2. `python3.14 scripts/build_pit_panel.py` builds the point-in-time panel from that
   extract and writes the panel, its coverage table and an aggregate build report.

`run_experiment.py` reads it with `data="crsp"` (build from the extract) or
`data="panel_file"` (the prebuilt panel).

- **Licence.** CRSP is licensed to the university; redistribution is prohibited.
  `Quant Model/Data/` is gitignored, and everything derived from CRSP (extracts, parquet
  files, built panels, coverage tables, registries of CRSP runs) lives in `Data/derived/`
  and nowhere else: never in `data_cache/`, `results/`, `figs/` or `Master Thesis/`.
  Tests use CIZ-format fixtures with the real headers and invented numbers; tests that
  read the real files are marked `crsp_data` and skip without `Data/`. Committed reports
  quote aggregate statistics only (counts, ranges, means, rates), never rows or
  per-security values.
- **Construction** (`nec_moe/crsp.py`, each rule checked against the release metadata and
  tested): daily log return `log(1 + DlyRet)`, a missing return stays missing; `DlyRet`
  already includes the delisting return, and a forward window that runs past a stock's
  final return is completed with `CRSPSpec.post_delisting_return` (`"cash"` = 0 or
  `"market"`; the default is not a decision) so no row is dropped because of a future
  delisting; dollar volume `|DlyPrc| x DlyVol`; drawdown compounded inside its own window;
  share volume made split-invariant with `DlyCumFacShr` ratios inside the window; the
  market series is `log(1 + DlyTotRet)` of `market_indno`.
- **Anti-leakage:** every feature at date t is computed from data ≤ t (verified by
  `test_no_lookahead` and its CRSP-fixture twin); the forward return target is the only
  forward-looking column; snapshot features are cross-sectionally rank-normalized per
  date. Walk-forward purge should equal the target horizon (`purge_dates=spec.horizon`).
- **Provenance:** every `TrialRegistry` row carries `data_source`,
  `post_delisting_return` and the experts' `hidden_init` (`trial_provenance`); the
  registry refuses a row without them.

## Stage C: context data + gate–regime alignment (the interpretability question)

```python
from nec_moe import load_vix, load_french_factors, build_context, gate_regime_alignment

ctx = build_context(load_vix("data_cache"), load_french_factors("data_cache"))
report = gate_regime_alignment(trainer, panel, ctx)   # works for soft AND hmm priors
print(report.to_frame())  # per expert: corr with VIX / factors / realized vol,
                          # plus high- vs low-VIX tercile utilization
```

VIX (CBOE official CSV) is **diagnostics only**: it never enters training (the gate stays
unsupervised, which is what keeps "do learned experts correspond to regimes?" a
*testable* question). The Kenneth French factors are diagnostics too, with one exception:
the Hamilton gate's registered series `market_excess_return` (brief 03 §2) is French
daily Mkt-RF, so that series is the gate's input (never a target or an expert feature). First real run (30-name panel,
2015–2024, unsupervised soft gate): the high-σ expert's utilization correlated **+0.81
with VIX** / +0.85 with realized market vol and ≈0 with directional factors — a
volatility-regime split discovered from the mixture likelihood alone. Single seed,
survivorship-biased universe: tooling demonstration, not a thesis claim.

## Single-model baselines (the comparison's other side)

```python
from nec_moe import RidgeBaseline, MLPBaseline, walk_forward_evaluate_baseline

ridge = walk_forward_evaluate_baseline(
    panel, lambda: RidgeBaseline(l2=1.0),
    n_folds=4, test_dates_per_fold=20, purge_dates=5,
    backtest_quantiles=4, cost_rate=0.001,
)
```

`RidgeBaseline` (closed form, unpenalized intercept — Module 4 conventions) and
`MLPBaseline` (the *same* network class as one NEC expert + a learnable noise σ,
trained on the same Gaussian NLL — so held-out NLLs are directly comparable and the
comparison isolates routing structure at matched capacity). Same folds, same purging,
same scoring code path as the NEC harness. Planted-truth tests pin both directions:
on sign-flip regime data the pooled relationship is ~zero and single models are
structurally blind (NEC IC ≈ 0.9 vs ~0); on regime-free data ridge scores IC > 0.8 —
the baselines lose above for structural reasons, not because they're broken.
LightGBM (syllabus candidate #2) is deliberately not wired in: heavy optional
dependency, and the repo carries a separate LGBM pipeline.

## Point-in-time universe (S&P 500 membership from CRSP)

```python
from nec_moe import CRSPSpec, read_membership, membership_universe, filter_point_in_time

u = membership_universe(read_membership(CRSPSpec()))  # StkIndMembership spells, INDNO 1000500
u.members_asof("2018-06-01")                          # PERMNOs whose spell covers the date
panel = filter_point_in_time(panel, u)                # drop (PERMNO, date) rows outside it
```

`build_crsp_panel` does this for you: it builds features over every PERMNO that was a
member at some point in the window (history before joining may feed features), keeps a
row only on dates its PERMNO was a member (`MbrStartDt`/`MbrEndDt`, both inclusive), and
re-ranks the snapshot features among the members (audit D-1). CRSP carries the departed
names and their delisting returns, so both survivorship components are covered by the
data; `build.coverage` still reports, per year, members without usable rows. The
membership INDNO is `1000500` ("CRSP Index of the S&P 500 Universe"): `1000502` has no
constituents in `StkIndMembership` (evidence in the `CRSPSpec` docstring). Companies with
two share classes in the index are kept as two PERMNOs.

## Selection-aware inference (trial registry + deflated Sharpe + FDR)

Every sweep configuration is a logged trial; picking a winner is a recorded event:

```python
from nec_moe import (TrialRegistry, ic_pvalue, benjamini_hochberg,
                     deflated_sharpe_ratio, trial_provenance)
import statistics

reg = TrialRegistry("trials.jsonl")            # append-only JSONL, reopenable
for cfg_name, result in sweep_results.items(): # one row per configuration/seed
    reg.log("routing_sweep", {"mean_ic": result.pooled_ic.mean_ic,
                              "p": ic_pvalue(result.pooled_ic),
                              "sharpe": sharpe},
            config={"prior": cfg_name, **trial_provenance(panel, cfg)})  # provenance required

# the claim family, corrected (FDR — the factor-zoo discipline):
pvals = [r.metrics["p"] for r in reg.trials("routing_sweep")]
reject, qvals = benjamini_hochberg(pvals, alpha=0.10)

# 'best of N' is a selection event — best() records it with the candidate count:
winner = reg.best("routing_sweep", "sharpe")
d = deflated_sharpe_ratio(winner_returns,
                          n_trials=reg.n_trials("routing_sweep"),
                          sr_variance=statistics.pvariance(
                              reg.metric_values("routing_sweep", "sharpe")))
print(d.psr_zero, d.dsr)  # naive confidence vs. selection-honest confidence
```

The factor-zoo test pins the point: the best of 60 pure-noise strategies gets naive
PSR > 0.95 and DSR < 0.75, while genuine skill of the same apparent size survives
deflation. Bonferroni is included for FWER when the family is small and the claim
is singular.

See the design doc §6 and §13 for the contract and remaining scope.
