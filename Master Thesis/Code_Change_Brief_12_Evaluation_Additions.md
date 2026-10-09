# Code change brief 12: evaluation additions (leg split, decile monotonicity, IC by year and episode, execution lag)

Date: 2026-10-09. Written for **Claude Code running locally on Tom's Mac**. Five parts (0, A-D), each ending in its
own commit. Do **not** push. Tom pushes himself.

Repository `Quant Model/nec_baseline`, package `nec_moe`; git root `Quant Model/`. Same rules as briefs 02-11: every
number is a config field with a documented default; every claim gets a test; nothing open is chosen by the code.

---

## Why this brief exists

Tom reviewed the LightGBM pipeline in `OP model/lgbm` and adopted four reporting additions (Advisor_Questions **Q9,
update 2026-10-09**; Progress_Tracker decision register, 2026-10-09). They are **reporting only**:

- they change no training, no model, no gate, no feature, no setting chosen by the pilot;
- every existing metric must come out bit-for-bit unchanged;
- they are fixed now, before any test-period (2010-2024) number is computed.

Still **open, do not implement**: the primary metric family (Q9 (a)), transaction costs in headline results
(Q9 (c)), training-target outlier clipping (Q20), residual reversal as an input (Improvements E3).

---

## Licence and research-integrity rules (unchanged)

- CRSP/Compustat and anything derived per security stay in `Data/`; `results/` aggregate only; never commit `Data/`;
  stage by explicit path; `Claude outputs/` untracked.
- **No out-of-sample number on the real panel** (2007 onward, which covers the pilot's validation years and the test
  period) is printed, written or read by you. Build and test on synthetic panels only. Tom runs everything real.
- Nothing in this brief may change the pilot's selection lock or any setting it froze. If any change would alter a
  frozen setting, a run hash or resume compatibility of existing runs, stop and ask.

---

## 0. Read first, then commit today's documents

Read: `Advisor_Questions.md` **Q9** (with the 2026-10-09 update, which is the specification), **Q21** (horizon and
Hansen-Hodrick), **Q25** (market-neutral target); `Progress_Tracker.md` sections 1 and 1b; `nec_moe/evaluation.py`
(`rank_ic_by_date`, `ic_summary`, `hac_variance`, `_quantile_legs`, `long_short_by_date`, `long_short_book`,
`portfolio_summary`, `walk_forward_evaluate`, `walk_forward_evaluate_baseline`); `nec_moe/regime_split.py` (brief 11,
regime-weighted means); `nec_moe/registry.py`; `nec_moe/features.py` (how the target is built, `forward_daily_returns`).

**Commit 0, documents only**, staged by explicit path if modified or new:
`Master Thesis/Advisor_Questions.md`, `Master Thesis/Progress_Tracker.md`,
`Master Thesis/Improvements_and_Extensions.md`, `Master Thesis/Code_Change_Brief_12_Evaluation_Additions.md`.
List anything else that is modified to Tom and leave it out. Message: "Docs 2026-10-09: Q9 reporting additions,
extension E3, brief 12".

---

## A. Long leg against short leg

For each test date, using the same quantile legs as the existing book (`_quantile_legs`, same `n_quantiles`):

1. **Leg returns.** Long leg = mean market-neutral target of the top quantile; short leg = minus the mean of the bottom
   quantile. The existing gross spread is long minus short of the means, so **long leg + short leg = existing gross
   spread** exactly (assert in a test). Because the target is demeaned with equal weights over the universe, each leg
   is already measured against the universe average.
2. **Half-sample ICs.** The rank IC computed separately among names in the upper half and among names in the lower half
   of that date's forecast distribution (ranks recomputed inside each half).
3. Summaries for both: mean, standard error with Hansen-Hodrick lags h - 1 (`ic_summary` / `hac_variance`), number of
   dates. Both under the existing `portfolio_scheme` handling (non-overlapping or staggered) for the leg returns.

Tests (synthetic): legs sum to the spread; a planted signal that only ranks the losers produces a short-leg return and
lower-half IC but a long leg and upper-half IC near zero; existing metrics unchanged.

Commit: "Brief 12 A: long and short leg returns, half-sample ICs (Q9 update)".

---

## B. Decile monotonicity

1. Each test date, sort names into `n_sort_groups` (default 10) by forecast; record each group's mean market-neutral
   h-day return. Report the time-series mean per decile with Hansen-Hodrick standard errors.
2. **Monotonic relation test** (Patton & Timmermann 2010): null hypothesis that mean returns are not increasing,
   alternative that every adjacent difference (decile j+1 minus decile j) is positive. Statistic: the minimum adjacent
   difference of the means. p-value by the **stationary bootstrap** (Politis & Romano 1994) on the date-by-decile
   series, recentred under the null as in Patton & Timmermann. Config: `mr_bootstrap_reps` (default 1000),
   `mr_mean_block` (default `max(10, 2h)` trading days, so blocks span the target overlap), `mr_seed`.
3. Also report the Spearman correlation between decile number and mean return (a descriptive shape summary).

Tests (synthetic): strictly increasing true means reject at a high rate; flat means hold size near 5% over repeated
simulations (report the rejection rate); a single extreme decile with flat others does not reject; bootstrap is
reproducible under a fixed seed.

Commit: "Brief 12 B: decile returns and the Patton-Timmermann monotonicity test (Q9 update)".

---

## C. IC and leg returns by calendar year and by named episode

1. **Calendar years** 2010-2024: mean rank IC, ICIR, long-short spread, long leg and short leg, each with standard
   errors, per year.
2. **Named episodes**, config field `episodes` with these defaults (fixed in Q9; do not change them):

   | name | start | end |
   |---|---|---|
   | euro_debt_flash_crash_2010 | 2010-04-26 | 2010-07-02 |
   | us_downgrade_2011 | 2011-07-25 | 2011-10-31 |
   | china_oil_2015_16 | 2015-08-17 | 2016-02-29 |
   | q4_selloff_2018 | 2018-10-01 | 2018-12-31 |
   | covid_crash_2020 | 2020-02-20 | 2020-04-30 |
   | covid_rebound_2020 | 2020-05-01 | 2020-12-31 |
   | bear_market_2022 | 2022-01-03 | 2022-10-31 |
   | regional_banks_2023 | 2023-03-08 | 2023-05-05 |

   A date belongs to an episode if its **formation date** t lies in the window. Report the same statistics as for
   years, plus the number of dates; flag episodes with fewer than `min_episode_dates` (default 20) formation dates.
3. **By regime** for gated arms: the regime-weighted means of brief 11 (`regime_split`), using each arm's own
   window-average gate weights, for IC, spread and both legs.
4. Output: one aggregate table per run (and a campaign-level summary across seeds and folds), written with the run's
   other metrics. Aggregate only.

Tests (synthetic): episode membership by formation date; a planted signal that works only inside one window is found
there and not elsewhere; year tables add up to the full-sample totals in the right proportions.

Commit: "Brief 12 C: IC and leg returns by year, named episode and regime (Q9 update)".

---

## D. The execution-lag knob

1. New config field `execution_lag` (integer trading days, default **0**, meaning the current behaviour). With L > 0,
   the forecast made at the close of t is scored against the **market-neutral return over (t + L, t + L + h]**:
   the sum of the stock's daily log returns over those days (post-delisting fill as for the target), minus the
   equal-weighted mean over **date t's** universe names with a valid value. This is an **evaluation-only label**.
2. It must never reach training: assert that the training loss and every fitted object are identical whatever the
   value of `execution_lag`. Build it either in the panel as a separate evaluation-only column or at evaluation time
   from the extract; choose the route that leaves the training panel, the run hashes and resume of existing runs
   untouched, and say which you chose.
3. At the end of the sample (last date 2024-12-31) the last L formation dates have no lagged label and are dropped from
   the L > 0 scores only. Reading returns from the following test year to score a fold is allowed: it is evaluation of
   that fold, not training of any fold.
4. Every metric of parts A-C, the headline IC and the existing book are reported at `execution_lags` (default
   `(0, 1)`), side by side. Hansen-Hodrick lags stay h - 1.
5. Registry: record `execution_lag` on every reported row.

Tests (synthetic): L = 0 reproduces existing numbers exactly; a planted signal that predicts only day t+1's return has
positive IC at L = 0 and none at L = 1; training artefacts are identical across L; the lagged label uses only returns
after t + L.

Commit: "Brief 12 D: execution_lag knob, headline metrics at L = 0 and 1 (Q9 update)".

---

## Finish

- `RUNBOOK.md` / `HANDBOOK.md`: the new tables and how to read them (legs, monotonicity, years and episodes, lag).
- Full test suite; report counts, commits, and anything touching open questions (Q9 (a), Q9 (c), Q20) not acted on.
- Confirm explicitly that no existing metric changed and the pilot lock is untouched.

## Literature

| Reference | Use |
|---|---|
| Patton, A. & Timmermann, A. (2010), "Monotonicity in asset returns: New tests with applications to the term structure, the CAPM, and portfolio sorts", *Journal of Financial Economics* 98(3), 605-625 | The monotonic relation test of part B |
| Politis, D. & Romano, J. (1994), "The stationary bootstrap", *Journal of the American Statistical Association* 89(428), 1303-1313 | Bootstrap for dependent, overlapping series |
| Hansen, L. P. & Hodrick, R. (1980), *Journal of Political Economy* 88(5) | Standard errors for overlapping h-day targets |
| Daniel, K. & Moskowitz, T. (2016), "Momentum crashes", *Journal of Financial Economics* 122(2) | Why the 2020 rebound is a named episode |
| Nagel, S. (2012), "Evaporating liquidity", *Review of Financial Studies* 25(7) | Why stress episodes are expected to change short-horizon signals |
