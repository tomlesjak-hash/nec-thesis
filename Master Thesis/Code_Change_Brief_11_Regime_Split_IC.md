# Code change brief 11: the IC curve split by regime, and the sector-regime check (2000-2006, descriptive)

Date: 2026-10-09. Written for **Claude Code running locally on Tom's Mac**. Five parts (0, A-D), each ending in its
own commit. Do **not** push. Tom pushes himself.

Repository `Quant Model/nec_baseline`, package `nec_moe`; the git root is `Quant Model/`. Same rules as briefs 02-10:
every number is a config field with a documented default; every claim gets a test; nothing open is chosen by the
code.

---

## Why this brief exists

The IC decay curve (brief 10 C, run 2026-10-09, recorded in Q21) confirmed h = 5, but it **pools regimes**: 2000-2006
mixes the 2000-02 bear market with the calm of 2003-06. A signal whose sign or size depends on the market state can
average to nothing. The thesis rests on the claim that the cross-section of returns depends on the regime, and the
only evidence so far is a crude check (2015-2024, days split at median trailing volatility; Q21 point 2). This brief
replaces it with the study's own gate, on pre-pilot years only, and asks two questions:

1. **Does each signal's IC differ between calm and stress, and how does the difference change with the horizon?**
   (Parts A-B.)
2. **Do industries have regimes of their own beyond the market's?** A descriptive input to the open Q28 (how the
   industry-level gate is built). (Part C.)

Both are **descriptive diagnostics**. Their results do not choose K (Q18), the industry-gate form (Q28), the gate's
memory, or the features (Q26). They go into the thesis as pre-registered motivating evidence.

---

## Licence and research-integrity rules (unchanged from brief 10)

- CRSP, Compustat and anything derived per security stay in `Quant Model/Data/`; `results/` holds aggregate outputs
  only; never commit anything under `Data/`; stage by explicit path; `Claude outputs/` stays untracked.
- **Hard date guard: 2000-01-01 to 2006-12-31.** No market return, stock return, gate fit or target from 2007 or later
  is read. Forward windows end by 2006-12-31, so the last scored date moves back by h (as in brief 10 C). The guard
  refuses any later end date with no override flag.
- **Write and test on synthetic data; do not run on real data.** Tom runs parts A-C after reviewing the code.

---

## 0. Read first

- `Master Thesis/Advisor_Questions.md`: the Q21 resolution and its **IC decay result** (2026-10-09); **Q27** (window
  gate weight); **Q23** and **Q22** (gate series, filter); **Q1** and the open **Q28** (industry-level gate); **Q18**
  (K, open).
- `Master Thesis/Progress_Tracker.md` sections 1 and 1b.
- Code to reuse, not duplicate:
  - `scripts/ic_decay.py`: `check_end`, `restrict`, `forward_log_returns`, `market_neutral`, `spearman_ic_by_date`,
    `load_inputs`, `cut_at`, the `Settings` pattern, the output/provenance writer;
  - `nec_moe/evaluation.py`: `hac_variance`, `ic_summary`;
  - `nec_moe/markov_gate.py`: `MarkovSwitchingRegimePrior` (fit, filtered probabilities, canonical ordering by
    variance, multi-start), `gate_weight_probs(filtered, transition, kind, horizon)`, the `crsp_market_log_return`
    series;
  - `nec_moe/characteristics.py`: `_sector_mean` and the sector conventions (`min_sector_names`).

**Commit 0, documents only.** Check `git status`; stage by explicit path only these, if modified or new:

- `Master Thesis/Advisor_Questions.md` (Q21 IC-decay result)
- `Master Thesis/Progress_Tracker.md`
- `Master Thesis/Code_Change_Brief_11_Regime_Split_IC.md` (this file)
- `Master Thesis/Methodology_2026-10-08.pdf` and `Master Thesis/Methodology_tex/`

List anything else that is modified to Tom and leave it out. Message: "Docs 2026-10-09: IC decay result (Q21),
methodology document, brief 11".

---

## A. Regime probabilities on 2000-2006

New module `nec_moe/regime_split.py` (pure functions) and script `scripts/ic_regime_split.py`.

1. **Series:** the CRSP S&P 500 index daily log return (`crsp_market_log_return`, decimals), 2000-01-03 to the last
   trading day of 2006.
2. **Fit:** the Hamilton gate on that whole span, `memory="full"` (equal weights), switching mean and variance,
   canonical ordering by variance, the existing multi-start. **K = 2 primary, K = 3 secondary** (config field
   `k_list`, default `(2, 3)`). This does not decide Q18: both are reported.
   - Equal weights because the pilot has not chosen the gate's memory; equal weights are the reference arm.
3. **Probabilities:** only **filtered** probabilities, never smoothed. For each horizon h, the gate weight is the
   window average of Q27, `gate_weight_probs(filtered, A, "window", h)`. The filtered probability itself is kept as a
   robustness variant (`weight_kind` in `{"window", "filtered"}`, default `"window"`).
4. **Stated caveat** (write it into the JSON and the figure caption): the gate's *parameters* are estimated on all of
   2000-2006, so the probabilities are causal in the data they filter but not in the parameters. This is acceptable for
   a descriptive in-sample diagnostic and is not a forecast.
5. **Parameter-free robustness state:** `stress_vol(t) = 1` if the trailing 20-day standard deviation of the market's
   log return (data up to t) is above its 2000-2006 median, else 0. This links the result to the earlier crude check.
6. **Aggregate outputs:** regime variances, means, transition matrix, expected durations, share of days per regime,
   the multi-start trace; a figure of the market's cumulative log return with the stress probability shaded.

Tests (synthetic): a planted two-regime series is recovered (ordering by variance); probabilities come from
`filtered_marginal_probabilities` or the native forward pass, never a smoother (assert by construction); the window
weight equals the filtered weight moved by A and averaged; the guard refuses 2007.

Commit: "Brief 11 A: regime probabilities on 2000-2006 for the split IC (descriptive)".

---

## B. The IC curve split by regime

**Signals** (config `signals`): the 16 of brief 10 C, plus the two industry signals `ret_20d_ind_rel` (within-industry
reversal) and `ind_mom_12_1` (industry momentum): 18. **Horizons** `(1, 2, 3, 5, 10)`. The daily IC series
`IC_t(h)` is computed exactly as in brief 10 C (Spearman across the universe, market-neutral forward log return).

For each signal, horizon and K, with regime weights `w_k(t) = wbar_k(t; h)` (they sum to 1 on each date):

1. **Regime-weighted mean IC (descriptive):**
   `IC_k = sum_t w_k(t) IC_t / sum_t w_k(t)`, with the Kish effective number of dates per regime,
   `(sum_t w_k)^2 / sum_t w_k^2`.
2. **Pure-regime IC with inference (primary):** the regression without intercept
   `IC_t = sum_k c_k w_k(t) + e_t`.
   - `c_k` is the IC in a date that is fully in regime k; with K = 2, `c_stress - c_calm` is the regime contrast.
   - Standard errors from the sandwich `(X'X)^-1 S (X'X)^-1`, with `S` the HAC variance of `x_t e_t`.
   - **Two HAC choices, both reported:** Hansen-Hodrick, uniform kernel, lag h - 1 (the overlap of the target, as in
     brief 10 C), and Newey-West, Bartlett kernel, lag `max(h - 1, nw_min_lag)`, `nw_min_lag` default 20, because the
     regressor (a persistent regime probability) makes the residuals' autocorrelation outlast the target overlap.
     Reuse `hac_variance` for the kernels; do not write a second HAC implementation.
   - Report `c_k`, their t-statistics, the contrast `c_stress - c_calm` with its t-statistic under both HAC choices.
3. **Hard-split robustness:** the same with `w_stress(t) = 1[wbar_stress(t; h) > 0.5]`, and with the parameter-free
   `stress_vol` state of A.5.
4. **Multiple testing.** Two families, declared before the run (write them in the JSON):
   - **Primary hypotheses (pre-specified from the literature, K = 2, h = 5, Hansen-Hodrick):**
     - H1: short-term reversal is stronger (more negative IC) in stress, for `ret_5d`, `ret_20d`, `ret_20d_ind_rel`
       (Nagel 2012; Hameed & Mian 2015);
     - H2: momentum is weaker in stress, for `mom_12_1` and `ind_mom_12_1` (Cooper, Gutierrez & Hameed 2004;
       Wang & Xu 2015; Daniel & Moskowitz 2016).
     Five one-sided tests; report raw p-values and Holm-adjusted p-values.
   - **Everything else is exploratory**: all signals x horizons x both K; report Benjamini-Hochberg q-values across
     the family and say so in the output.
5. **Outputs** (aggregate only): `results/diagnostics/ic_regime_2000_2006.csv` and `.json` (signal, h, K, weight kind,
   regime, weighted IC, effective dates, c_k, t-statistics under both HAC choices, contrast, p, q, family), and figures:
   - small multiples, one panel per signal: IC against h, one line per regime (calm and stress, K = 2), with
     plus-or-minus two standard error bands;
   - one summary panel: the contrast `c_stress - c_calm` against h for the primary signals.
   **Use a palette with enough distinct colours, or line styles, for every line** (brief 10 C's figure reused colours).

Tests (synthetic): a planted signal with IC +a in calm and -a in stress is recovered by `c_k`, while its pooled IC is
near zero; with regime-independent IC the contrast is near zero and the test holds its size over many simulations
(report the rejection rate at 5%); the weighted mean equals the plain mean when all weights are equal; the decomposition
is exact on noise-free data; Holm and Benjamini-Hochberg match a hand-worked example.

Commit: "Brief 11 B: IC curve split by regime, with pre-specified primary tests (descriptive)".

---

## C. Do sectors have their own regimes? (descriptive input to Q28)

Script `scripts/sector_regimes.py`, functions in `nec_moe/regime_split.py`.

1. **Sector returns:** for each GICS sector and date, the value-weighted (previous day's market equity) average of
   members' daily simple returns over the universe rows with that sector code, as a log return. Skip a sector-date with
   fewer than `min_sector_names` (default 5, the existing convention) and report which sectors are too thin in
   2000-2006 (for example, Real Estate was part of Financials until 2016). Equal-weighted as a robustness variant.
2. **Two series per sector:** the **sector-relative** log return (sector minus the CRSP index) and the **raw** sector
   log return.
3. **Fit** a 2-state Hamilton gate per sector and series, 2000-2006, equal weights, filtered probabilities, ordering
   by variance; the market gate is the K = 2 fit of part A.
4. **Measures** (all aggregate, sector level):
   - share of days with sector stress probability above 0.5;
   - **share of days a sector is stressed while the market is calm** (sector above 0.5, market below 0.5), and the
     conditional probability of sector stress given market calm;
   - correlation of each sector's stress probability with the market's;
   - expected durations and the number of stress episodes lasting at least `min_episode_days` (default 5);
   - the 11 x 11 co-stress matrix (share of days two sectors are both stressed).
   The expectation to check: raw sector regimes largely copy the market's; sector-relative regimes should not.
5. **Outputs:** `results/diagnostics/sector_regimes_2000_2006.{csv,json}` and one figure: a heat strip of each
   sector's stress probability through 2000-2006 with the market's on top.

This part informs Q28 sub-questions 1 (sector-relative or raw), 2 (granularity) and 6 (the descriptive check). It does
**not** choose the industry-gate form.

Tests (synthetic): a planted sector-only stress episode is detected in the relative series and not in the market;
value weights use the previous day's market equity; thin sectors are skipped and reported.

Commit: "Brief 11 C: sector regimes on 2000-2006, descriptive input to Q28".

---

## D. Docs and finish

- `RUNBOOK.md`: how Tom runs parts A-C (one command each, real data, 2000-2006 only).
- `HANDBOOK.md`: one section on the regime-split diagnostic and its caveats (in-sample parameters; descriptive).
- Run the full test suite; report the counts, the commits, and the exact commands for Tom.
- Report anything that touches an open question (Q18, Q28, Q29) without acting on it.

Commit: "Brief 11 D: docs".

---

## How the results will be read (written now, before the run)

- **Supports the thesis:** reversal clearly stronger in stress and momentum weaker or reversed in stress at h = 5, with
  contrasts that change with the horizon. Then the regime gate has a documented target.
- **Neutral:** contrasts of the expected sign but not significant after Holm. Report as consistent but underpowered:
  2000-2006 holds one long stress episode.
- **Against:** no regime contrast for any primary signal. Report it plainly; it weakens the motivation but does not
  change the design, which is fixed and pre-registered. Regime-dependence can still appear in 2007-2024.
- **Sectors:** if sector-relative stress while the market is calm is rare, an industry gate adds little beyond the
  market gate; if common, Q28 has its evidence. Either way the result is reported, not tuned on.

## What this brief does not decide (do not pick answers)

- K (Q18). The industry-gate form (Q28). The gate's memory (pilot). The features (Q26, unchanged). The error function
  (Q20).

---

## Literature (for Claude Code: what each supplies)

| Reference | Use in this brief |
|---|---|
| Nagel, S. (2012), "Evaporating liquidity", *Review of Financial Studies* 25(7), 2005-2039 | Short-term reversal returns are pay for liquidity provision and rise with market volatility (VIX). H1. |
| Hameed, A. & Mian, G. M. (2015), "Industries and stock return reversals", *JFQA* 50(1-2), 89-117 | Reversal is stronger within industries; motivates `ret_20d_ind_rel` in H1. |
| Cooper, M., Gutierrez, R. & Hameed, A. (2004), "Market states and momentum", *Journal of Finance* 59(3), 1345-1365 | Momentum profits depend on the market state (positive after up markets, absent after down markets). H2. |
| Daniel, K. & Moskowitz, T. (2016), "Momentum crashes", *Journal of Financial Economics* 122(2), 221-247 | Momentum crashes in high-volatility rebounds after bear markets; why momentum's pooled IC can be near zero. H2. |
| Wang, K. Q. & Xu, J. (2015), "Market volatility and momentum", *Journal of Empirical Finance* 30, 79-91 | Market volatility predicts momentum payoffs (negatively). H2. |
| Avramov, D., Cheng, S. & Hameed, A. (2016), "Time-varying liquidity and momentum profits", *JFQA* 51(6), 1897-1923 | Momentum is weaker when market illiquidity is high; context for the liquidity signals (exploratory). |
| Stambaugh, R., Yu, J. & Yuan, Y. (2012), "The short of it: Investor sentiment and anomalies", *Journal of Financial Economics* 104(2), 288-302 | Anomalies (including idiosyncratic volatility) vary with market conditions; context for the exploratory family. |
| Hansen, L. P. & Hodrick, R. (1980), *Journal of Political Economy* 88(5); Newey, W. & West, K. (1987), *Econometrica* 55(3) | The two HAC estimators of part B. |
| Holm, S. (1979), *Scandinavian Journal of Statistics* 6(2); Benjamini, Y. & Hochberg, Y. (1995), *JRSS B* 57(1) | Multiple-testing corrections for the primary and exploratory families. |
| Hamilton, J. (1989), *Econometrica* 57(2); Kim, C.-J. & Nelson, C. (1999), *State-Space Models with Regime Switching*, ch. 4 | Filtered versus smoothed probabilities; why only filtered ones are used. |
| Ghahramani, Z. & Jordan, M. (1997), "Factorial hidden Markov models", *Machine Learning* 29 | Background for Q28's factorial option, which part C informs. |
