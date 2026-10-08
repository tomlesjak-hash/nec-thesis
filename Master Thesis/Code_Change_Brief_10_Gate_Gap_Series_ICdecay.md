# Code change brief 10: the gate filter through the purge gap, the CRSP gate series, the IC decay script

Date: 2026-10-08. Written for **Claude Code running locally on Tom's Mac**. Four parts, done in order
(0 to C). Each part ends in its own commit. Do **not** push. Tom pushes himself.

Repository `Quant Model/nec_baseline`, package `nec_moe`; the git root is `Quant Model/`. Same rules as
briefs 02 to 09:

- every numeric quantity is a config field with a documented default;
- every claim gets a test that pins it;
- this brief contains only decisions Tom has made. Where something is open it says so, and the code
  must not pick an answer.

---

## Licence and research-integrity rules (unchanged from brief 09)

- CRSP, Compustat and anything derived from them stay in `Quant Model/Data/`. Per-security outputs go
  only to `Data/derived/...`. `results/` holds aggregate outputs only. Never commit anything under
  `Data/`. Stage files by explicit path. `Claude outputs/` stays untracked.
- **No out-of-sample performance number on the real panel is printed, written or read.** That covers
  every date from 2010 on, and the pilot's validation years 2007-2009.
- You **do not run** the pilot, the campaign or the IC decay script (part C) on real data. Tom runs them.
- Real-data work allowed: fitting the gate on the new market series for diagnostics only (convergence,
  regime variances, expected durations; no scoring), to compare with the French series.

---

## 0. Read first, then commit today's documents

**Decisions** (all 2026-10-08; read the full text in `Master Thesis/Advisor_Questions.md`, RESOLVED section):

1. **Q22** (b): the gate's filter runs over every trading day, including the purge gap. Parameters are
   still fitted on the training block only.
2. **Q23** (b): the gate's series is the CRSP value-weighted S&P 500 universe index (INDNO 1000500),
   daily total return, as a log return. French Mkt-RF stays registered as a robustness check.
3. **Q21 follow-up**: an IC decay curve on 2000-2006 (part C).
4. **Q1** (industry-level gate), **Q2**, **Q4**: decided, but **nothing to code**. How the industry gate
   is implemented is **open (Q28)**, and so is a factor-return gate (**Q29**). Do not start either.

Also read `Master Thesis/Progress_Tracker.md` sections 1 and 1b.

**Commit 0, documents only.** Tom's documents were edited today and are not committed. Stage by explicit
path, after checking `git status` and `git diff --stat` show only these:

- `Master Thesis/Advisor_Questions.md`
- `Master Thesis/Progress_Tracker.md`
- `Master Thesis/Improvements_and_Extensions.md`
- `Master Thesis/Model_Derivation_BaseCase.md`
- `Master Thesis/MoE_Formulation_tex/s0_howto.tex`
- `Master Thesis/MoE_Formulation_tex/s9_checklist.tex`
- `Master Thesis/MoE_HMM_Gate_Formulation.pdf`
- `Master Thesis/Code_Change_Brief_10_Gate_Gap_Series_ICdecay.md` (this file)

If anything else is modified, list it to Tom and leave it out. Commit message: "Docs 2026-10-08: Q1, Q2,
Q4, Q22, Q23 decided; Q28, Q29 opened; Q21 follow-up; brief 10".

---

## A. The gate's filter runs through the purge gap (Q22)

**Now.** The gate is fitted on `fold.train_dates`, then `apply_causal` runs it over
`train_dates + test_dates`, so the purge-gap days are missing from the market series and one step of A
bridges h + 1 trading days at each fold boundary. Places that build that date set:

- `nec_moe/evaluation.py` (around line 1083, the gate fit-and-apply helper);
- `nec_moe/pilot.py` (around line 294, the gate-memory selection);
- `tests/test_pilot.py` (line 194);
- `scripts/smoke_run_2026_09.py` (line 286; a historical script: leave it, or note it in its docstring).

Search for any other place that applies the gate to train plus test dates.

**Change.**

1. Add one helper (for example `gate_span_dates(panel, fold)`) that returns **every panel date from the
   first training date to the last test date**, gap included, and use it everywhere the gate is applied
   causally.
2. `fit` is unchanged: it still sees only `fold.train_dates`. The existing assertion that the fit panel
   ends before the test block stays.
3. Fitted training dates are never overwritten (existing rule in `_remember_filtered`); gap dates get
   filtered probabilities from the causal application like test dates.
4. Pilot: `predictive_log_density` is now evaluated over the contiguous span; the validation score still
   sums **only** the test dates' contributions.
5. Labels: nothing changes. The purge still removes the gap days' rows from **training**; only the
   gate's market series includes them.

**Tests.**

- On a synthetic panel with a purge: the test-date filtered probabilities equal those from one filter
  pass, with the frozen parameters, over the full contiguous series (to float tolerance), for both
  backends (statsmodels and native).
- The gap dates are not in the training rows of the experts or the base.
- Changing a gap-day market return changes the first test date's gate weight; changing a gap-day
  **target** changes nothing in training.
- The fit still raises if handed a date in or after the test block.

Commit: "Brief 10 A: the gate filter runs through the purge gap (Q22)".

---

## B. The gate's series: CRSP S&P 500 index log return (Q23)

**Now.** `MarkovGateConfig.series = "market_excess_return"` and `Experiment.gate_series` use French
Mkt-RF (`_french_market_excess` in `markov_gate.py`). The CRSP market series is in the extract as
`market.parquet` (`DlyTotRet` of `CRSPSpec.market_indno`, 1000500), and the panel features already use
`log1p` of it (`crsp.py`, `mkt_ret`). It is **not** one of the panel's sequence features, so the existing
`sequence_feature` key does not cover it.

**Change.**

1. Add a `SERIES_REGISTRY` entry, for example `"crsp_market_log_return"`: `log(1 + DlyTotRet)` of the
   panel's market index, aligned to the panel's dates through `date_labels`.
   - Preferred source: store the series with the panel at build time (for example in
     `panel.metadata`, keyed by calendar date), so the gate never reads files during a run. For panels
     built before this change, fall back to the extract's `market.parquet`, located from what the panel's
     metadata records; raise with a clear message if neither is available.
   - A panel date with no market return raises (as the French builder does); never forward-fill.
2. Make it the default in `MarkovGateConfig.series` and `Experiment.gate_series`. Keep
   `market_excess_return` registered for the robustness check.
3. **Scale.** Check the units the French series is passed to the fit in (decimal or percent) and pass the
   CRSP series in the same units, so starting values, the variance floor and convergence behave the same.
   State the units in the docstring and in a test.
4. Make sure the run settings and the trial registry record the gate series; if they already do, nothing
   to add.
5. Check whether the backprop-HMM baseline reads the gate series registry. If it does, it follows the new
   default; if it has its own input, leave it and report what it uses.
6. **Diagnostic only (allowed on real data):** fit the Hamilton gate (K = 2 and 3) on the 2000-2008
   training block with both series and write convergence, regime variances, expected durations and the
   correlation of the two series' filtered stress probabilities to
   `results/benchmark/gate_series_comparison.json`. No scoring, no dates after 2008.

**Tests.**

- The new series equals `log1p(DlyTotRet)` on a fixture, aligned by date.
- A missing date raises.
- The default is the new key; the French key still works.
- The units match the French path.

Commit: "Brief 10 B: CRSP S&P 500 index log return as the gate series (Q23)".

---

## C. The IC decay script (Q21 follow-up)

`scripts/ic_decay.py`. **Write and test it; do not run it on real data.** Tom runs it.

**What it computes.** For each signal and each horizon h in `horizons` (default 1, 2, 3, 5, 10): on each
date, the Spearman rank correlation across the universe between the signal and the **market-neutral**
forward return over (t, t+h] (sum of the next h daily log returns, the same delisting fill as the target,
minus the date's equal-weighted cross-sectional mean over stocks with a valid value). Then the mean IC
over dates and its Hansen-Hodrick t-statistic with lag h - 1 (reuse the existing implementation).

**Signals** (configurable; default): the two Short-Term Reversal and two Momentum characteristics of the
JKP block and the 12 characteristics of the short-horizon market block, by their names in the `q26`
schema. Use the characteristics as ranked in the panel.

**Hard guard.** Dates restricted to 2000-01-01 to 2006-12-31 by default, and the script **refuses** any
end date on or after 2007-01-01 (the pilot's validation years and the test period), with no override
flag.

**Output** (aggregate only): `results/diagnostics/ic_decay_2000_2006.csv` and `.json` (signal, h, mean
IC, t-statistic, number of dates) and one figure, IC against h per signal. Forward returns for h other
than 5 are computed inside the script from the extract; nothing per-security is written outside
`Data/derived/`.

**Tests** (synthetic): a planted signal that predicts only the next day's return gives an IC that falls
with h as expected; the market-neutral demeaning holds; the guard refuses 2007 and later; the
Hansen-Hodrick lag is h - 1.

Commit: "Brief 10 C: IC decay script, 2000-2006 only (Q21 follow-up)".

---

## Finish

- Update `nec_baseline/HANDBOOK.md` / `RUNBOOK.md` where they describe the gate's input or how it is
  applied across folds, and add the IC decay script to the runbook.
- Run the full test suite; report the counts.
- Report to Tom: the commits, what part B's diagnostic found (convergence and durations under both
  series), anything you found that touches an open question (Q28, Q29) without acting on it, and the
  exact command for him to run part C.

## What this brief does not decide (do not pick answers)

- How the industry-level gate is implemented (Q28).
- A factor-return gate (Q29).
- K (Q18), the error function (Q20), widths (Q8), baselines (Q12).
