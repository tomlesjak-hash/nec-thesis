# Pre-registration: <experiment name>

> Copy this file to `results/prereg_<tag>.md`, fill every field, **commit it
> before running a single trial**, and put the commit hash in the registry
> notes. The registry then makes deviations visible instead of silent. Fields
> marked (†) are the ones reviewers check first.

## 1. Identity

- **Experiment tag** (the `TrialRegistry` tag): `<tag>`
- **Date frozen / commit hash**: `<YYYY-MM-DD> / <git hash>`
- **Thesis variation**: `<Variation 2: routing comparison | Variation 3: HMM-gated>`
- **Research question (one sentence)** (†): e.g. *Does soft routing improve
  out-of-sample cross-sectional rank-IC over single-model baselines and hard/
  top-k/Gumbel alternatives on a point-in-time U.S. large-cap panel?*

## 2. Data (frozen before any model sees it)

- **Panel**: `<e.g. pit_panel_2015_2024 via scripts/build_pit_panel.py>`
- **Universe + coverage table** (†): `<results/pit_coverage_*.csv>` — publish
  alongside every result; state the residual-bias posture
  ("survivorship-mitigated with documented residual coverage").
- **StageBSpec**: `seq_len=<>`, `horizon=<>`, `target_kind=<raw|residual>`,
  `beta_window=<>` (if residual), `cs_rank=<>`, `min_names_per_date=<>`
- **Known data limitations**: `<delisting returns absent; yfinance-grade prices; …>`

## 3. Candidate grid (†)

The complete set of arms. Anything evaluated later that is not on this list is
exploratory and must be labeled as such.

| arm | prior | emission | K | other config |
|---|---|---|---|---|
| `<soft>` | soft | mlp | 2 | `<…>` |
| `<…>` | | | | |

- **Seeds**: `<e.g. 0–4>` (≥ 5 for any headline number)
- **Baselines included**: `<ridge(l2=…), mlp(hidden=…), hamilton, uniform-gate>`

## 4. Tuning plan (before the grid runs)

- **Tuned on**: validation tail of fold 0's training window,
  `val_dates=<>`, `purge_dates=<horizon>` (via `tune`, tag `<tag>_tune`)
- **Tuning candidates**: `<the exact list>`
- **Tuning metric / mode**: `<mean_ic max | nll min>`
- **Frozen after tuning**: yes — per-fold re-tuning is not permitted.

## 5. Evaluation protocol (†)

- **Split**: `n_folds=<>`, `test_dates_per_fold=<>`,
  `purge_dates=<horizon — must equal §2's horizon>`, fit-once-per-window.
- **Primary metric**: `<pooled rank-IC (mean, ICIR, t)>`
- **Secondary metrics**: `<held-out NLL; net long-short IR at cost_rate=<>,
  n_quantiles=<>; regime-alignment corr (held-out folds); gate ECE>`
- **Models scored on NLL/regime metrics only** (no cross-section):
  `<classical Hamilton>`

## 6. Claim family and corrections (†)

- **The family**: the `<N>` arms of §3 (seeds are replicates, not hypotheses).
- **Replicate combination**: median p across seeds (fixed here, per handbook II.9).
- **Correction**: Benjamini–Hochberg at `alpha = <0.10>` via `corrected_claims`.
- **Winner selection**: `registry.best(<tag>, <metric>)` — the selection event
  is logged; the winner's Sharpe is reported **deflated**
  (`deflated_sharpe_ratio` with `n_trials = registry.n_trials(<tag>)`).

## 7. Outcomes, committed in advance (†)

- **Success looks like**: `<e.g. soft's q-value < 0.10 AND pooled IC t > 2 AND
  positive net IR after costs, stable in sign across ≥ 4/5 seeds>`
- **A negative result still teaches**: `<e.g. no arm survives BH ⇒ evidence
  that routing structure does not add cross-sectional signal at this horizon
  on this universe — reported with the same prominence>`
- **Stopping rule**: `<the grid runs exactly once as specified; no extension
  without a new pre-registration>`

## 8. Deviations log (append-only, dated)

| date | what changed | why | effect on interpretation |
|---|---|---|---|
| | | | |
