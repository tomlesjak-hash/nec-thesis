# nec_baseline — corrected NEC (regime-conditional MoE)

Research baseline for the thesis. **Design spec:** `../NEC_sources/NEC_baseline_design.md`
— every architectural choice traces to that document (defect ledger fixes, mixture-NLL
objective, shared-likelihood / pluggable-prior gate abstraction serving both thesis
variations).

The original prototype this corrects lives untouched at `../OP model/nec/nec_hybrid_explained.py`.

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
top-1-renormalized is gradient-dead, use `hard`), `gumbel` (annealed temperature), and
`hmm` (Variation 3). `ExpertConfig.kind` selects the emission along the second plug
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

## Stage B: real daily data (Stooq / yfinance)

```python
from nec_moe import build_stage_b_panel, StageBSpec, data_config_from_panel

panel = build_stage_b_panel(
    "2015-01-01", "2024-12-31", "data_cache",
    source="yfinance",                    # or "stooq" — see caveat below
    spec=StageBSpec(seq_len=20, horizon=5),
)
cfg = NECConfig(data=data_config_from_panel(panel))
# panel plugs into Trainer / walk_forward_evaluate unchanged
```

- The **cache CSV is the interface** (`Date,Open,High,Low,Close,Volume`, one file per
  symbol under `data_cache/`): downloads are cache-first, features/panels never touch
  the network, and all Stage-B tests run offline against fixture CSVs.
- **Stooq caveat:** its CSV endpoint sits behind a JavaScript anti-bot challenge (as of
  mid-2026) which this code deliberately does *not* circumvent — either download the CSVs
  in a browser into the cache (the error message gives URL + filename), or use
  `source="yfinance"` (the syllabus's prototyping fallback; `pip install yfinance`).
- **Anti-leakage:** every feature at date t is computed from data ≤ t (verified by
  `test_no_lookahead`); the forward return target is the only forward-looking column;
  snapshot features are cross-sectionally rank-normalized per date. Walk-forward purge
  should equal the target horizon (`purge_dates=spec.horizon`).
- **SURVIVORSHIP BIAS:** `DEFAULT_UNIVERSE` is a static list of *today's* large caps —
  pipeline-verification grade, not thesis-claim grade. Point-in-time universe +
  delistings (Module 12) remain TODO before any performance claim.

## Stage C: context data + gate–regime alignment (the interpretability question)

```python
from nec_moe import load_vix, load_french_factors, build_context, gate_regime_alignment

ctx = build_context(load_vix("data_cache"), load_french_factors("data_cache"))
report = gate_regime_alignment(trainer, panel, ctx)   # works for soft AND hmm priors
print(report.to_frame())  # per expert: corr with VIX / factors / realized vol,
                          # plus high- vs low-VIX tercile utilization
```

VIX (CBOE official CSV) and Kenneth French daily factors are **diagnostics only** —
they never enter training (the gate stays unsupervised, which is what keeps "do learned
experts correspond to regimes?" a *testable* question). First real run (30-name panel,
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

## Selection-aware inference (trial registry + deflated Sharpe + FDR)

Every sweep configuration is a logged trial; picking a winner is a recorded event:

```python
from nec_moe import (TrialRegistry, ic_pvalue, benjamini_hochberg,
                     deflated_sharpe_ratio)
import statistics

reg = TrialRegistry("trials.jsonl")            # append-only JSONL, reopenable
for cfg_name, result in sweep_results.items(): # one row per configuration/seed
    reg.log("routing_sweep", {"mean_ic": result.pooled_ic.mean_ic,
                              "p": ic_pvalue(result.pooled_ic),
                              "sharpe": sharpe}, config={"prior": cfg_name})

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
