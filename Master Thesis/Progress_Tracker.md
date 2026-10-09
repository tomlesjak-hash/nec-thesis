# Progress tracker

Everything added to the model or the thesis, with the date and where the detail lives. Section 1 is the
current state and is edited in place. Section 1b is the decision register: every big decision, oldest
first, only appended to. Section 1c is the compute budget. Section 2 is a log, newest first, and is only appended to. The
detail stays in the source documents; this file points to them.

- **Decisions and open questions:** `Advisor_Questions.md`
- **Code briefs:** `Code_Change_Brief_*.md`, `Code_Audit_Brief_05.md`
- **What the code does:** `Code_Audit_2026-09-25.md` (most recent full reading), `Code_State_2026-09-21.md` (older)
- **What is left to code:** `WORK_QUEUE.md`
- **The maths:** `Model_Derivation_BaseCase.md`

---

## 1. Current state (2026-10-05)

| Component | What it is | Status | Where |
|---|---|---|---|
| Data | CRSP daily stock file (CIZ `ciz202512`), re-extracted from the full daily file with open/high/low/close/bid/ask (brief 08 C); CRSP/Compustat Merged (`cfz202607`) for fundamentals, GICS and the link. **Sample decided: 2000-2024** (was 2015-2024; panel still built for 2015-2024 until re-run). Free data retired | CRSP and Compustat pipelines implemented (briefs 06, 08 C); 2000-2024 extract not yet built | Q3, Q16 (a), Q26; `crsp.py`; `Data/` (gitignored) |
| Universe | S&P 500 point-in-time from CRSP membership spells, INDNO 1000500 (not 1000502, which has no constituents), bounds inclusive, 502-508 members per day; dual-class companies kept as two PERMNOs. Delisting returns already in `DlyRet` (rule a); forward windows past a delisting completed by `post_delisting_return` (`"cash"` default, not a decision) | Implemented (brief 06 A) | `crsp.py`, `universe.py` |
| Frequency and horizon | Daily Hamilton gate, daily cross-section, 5-day forward target (h = 5, overlapping by 4 days; Hansen-Hodrick t-stats, purge = h). Already what the code does | **Decided 2026-10-05** (option B); no code change | Q21 |
| Target | **Market-neutral return** (`fwd_mn_ret_5d`): the forward return minus the date's equal-weighted cross-sectional mean over the stocks with a valid target. Reporting market-neutral throughout | **Decided 2026-10-01**; implemented, now the default (brief 08 A, 7bb3a4d) | Q25 |
| Features | Three families: price/volume (CRSP), fundamentals (Compustat via CCM, point-in-time), industry (GICS, Compustat; ICB dropped because CRSP's ICB stops in Oct 2023). Rule: exactly two representatives per JKP theme (13 themes, 26 characteristics), plus a short-horizon market block (6 themes × 2), 11 sector dummies, industry momentum, within-industry reversal and 4 within-sector fundamentals: 40 characteristics (22 market, 18 fundamental), 57 inputs. Rank to [-1, 1]. Missing: minimum-count windows, fundamentals carried forward (12-month cap), then 0 plus a flag per family; rows no longer dropped. The legacy 14-feature set is kept only as `feature_set="legacy14"` | **Decided 2026-10-05**; implemented (brief 08 C, D: 488baad, 0633379; real-data fixes be77fb7, 5524da1); first coverage report run | Q26 |
| Base | MLP, trained on the training block, then frozen | Decided, implemented | Q7; brief 02 |
| Gate | **Partly stock-specific at the industry level (Q1, decided 2026-10-08): market regime combined with an industry-level regime; implementation open (Q28), not implemented.** Market regime model fitted separately, then frozen, on the CRSP value-weighted S&P 500 index's daily log return (Q23); the filter runs through the purge gap (Q22); gate weight is the **average over the 5-day target window of the h-step-ahead regime probabilities** (built from the filtered probability and A; never the smoothed one); implemented as `gate_weight="window"` (brief 08 B, 3f1fac5). Memory: regime-clock forgetting, implemented (brief 09 D) | Decided. Hamilton implemented, the autoregressive variant too (brief 07, G-2); jump, Wasserstein and TVTP to come. The backprop-HMM baseline now runs on the point-in-time panel (state keyed by stock, M-5) | Q19, Q27; briefs 03, 04 B, 07 |
| Experts | MLP corrections to the base, $\hat y = f_0 + \sum_k \pi_k r_k$, zero-initialised heads | Implemented. Hidden-layer initialisation is a switch, `hidden_init` (brief 06 D); which one is used is **open** (X-1) | brief 02; audit X-1 |
| Number of regimes K | | **Open** | Q18 |
| Error function | Only the mixture NLL is registered, behind a seam | **Open** | Q20 |
| Depth grid | Mechanism by depth, depths set by the compute budget | Decided | Q11 |
| Evaluation | Walk-forward with the purge read from the target's horizon (quick mode purged too); rank IC with Hansen-Hodrick t-stats for overlapping targets; NLL reported three ways (`NLL_single`, `NLL_base`, `NLL_full`); a horizon-consistent long-short book, `portfolio_scheme` "nonoverlapping" or "staggered" (default **not decided**). Every trial row carries `data_source`, `post_delisting_return`, `hidden_init` and `portfolio_scheme` | Implemented and audited (briefs 06 C, 07 A) | audit E-1, E-2, E-3, O-1, O-3; `evaluation.py`, `registry.py` |
| Runs and resume | Every run in `results/<campaign>/<run_id>/` with settings, status, registry, metrics, figures, log and `SUMMARY.md`, one row in `results/INDEX.csv`; per-stock outputs only in `Data/derived/runs/`. Ctrl+C checkpoints and stops; resume is exact and refuses changed settings or data unless forced | Implemented (brief 07 B, C); how-to in `nec_baseline/RUNBOOK.md` | `runstore.py`, `scripts/runs.py` |
| Likelihood for the experts | Per row: the composite (independence) likelihood of the per-date model; one log-sum-exp per row. Already what the code does (`losses.mixture_nll`) | **Decided 2026-10-05**; no code change | Q24; PDF Section 7.3 and Appendix D |
| Gate through the purge gap | Whether the filter sees the gap's market returns | **Open** | Q22 |
| Training window and weighting | Expanding window. Experts: regime-clock decay; base: calendar decay, long half-life; gate: regime-clock forgetting (weighted Baum-Welch), memory chosen by predictive log-likelihood. Half-lives from a preset grid, chosen once in the pilot | **Decided 2026-10-05**; implemented (brief 09, commits f76f6eb-0e515eb); pilot not yet run. Test 2010-2024 with annual refits (15 folds); pilot on 2007-2009 with a regime-balanced loss; main study on full blocks with frozen settings. Every day, fresh starts, paired seeds, process-parallel. Training scheme fully decided except leave-one-episode-out (Q16 c) | Q16; `Training_Window_and_Weighting_Theory.md` |

## 1b. Decision register

Every big decision, oldest first, with what it replaced and where the reasoning lives. Rows are only
added. A decision that is later reversed keeps its row, and the reversal gets its own row.

| Date | Decision | Instead of / why | Where |
|---|---|---|---|
| 2026-09-18 | Data is CRSP (US equities); Wind only as a possible China extension | free data | Q3; advisor meeting |
| 2026-09-18 | Report the full mechanism-by-depth grid, nothing tuned away; depths set by the compute budget | fixing or tuning depth | Q11; advisor meeting |
| 2026-09-18 | No width sweep: widths follow the pyramid rule from a fixed first layer | sweeping widths | advisor meeting |
| 2026-09-18 | No sweep over K: K fixed by argument and held across all gates | sweeping K | advisor meeting; K itself is Q18 (open) |
| 2026-09-18 | Order of work: data, then compute budget, then parameters; start with the base and a generic loss | designing the loss first | advisor meeting |
| 2026-09-21 | Gate fitted separately and frozen, for every gate | joint training | Q19 |
| 2026-09-21 | Base pre-trained and frozen; experts are corrections to it (residual design) | joint training | Q7; brief 02 |
| 2026-09-26 | CRSP replaces free price and universe data everywhere | yfinance/Stooq/Wikipedia | brief 06 |
| 2026-09-26 | S&P 500 universe from CRSP membership INDNO 1000500, bounds inclusive | 1000502 (no constituents) | brief 06 A.3 erratum |
| 2026-09-26 | Delisting rule (a): CIZ `DlyRet` already contains the delisting return | compounding `DelRet` in | brief 06 |
| 2026-09-26 | Overlapping-target t-statistics: Hansen-Hodrick | Newey-West (worse size in simulation) | audit E-2 |
| 2026-09-26 | Report the volatility-only NLL gain next to the correction gain | one combined number | audit E-1 follow-up |
| 2026-10-01 | Target: market-neutral return (forward return minus the date's equal-weighted cross-sectional mean) | raw, CAPM residual, standardised, vol-scaled | Q25 |
| 2026-10-05 | All reporting on the market-neutral basis, no raw-return reporting | reporting raw returns too | Q25 clarification |
| 2026-10-05 | Daily gate, daily cross-section, 5-day target (option B) | h = 1, monthly | Q21 |
| 2026-10-05 | Features: two representatives per JKP theme plus a 6-theme short-horizon market block and an industry block; 40 characteristics, 57 inputs | the 14 legacy price features | Q26 |
| 2026-10-05 | Ranks mapped to [-1, 1] | [-0.5, 0.5] | Q26 part 2 |
| 2026-10-05 | Missing values: minimum-count windows, fundamentals carried forward for at most 12 months, then 0 plus a flag per family; rows no longer dropped | dropping rows; interpolation (look-ahead) | Q26 part 3 |
| 2026-10-05 | Industry: sector dummies, within-sector ranks of 4 fundamentals, industry momentum and within-industry reversal | none, or a subset | Q26 part 4 |
| 2026-10-05 | Sector source: GICS from Compustat | ICB (CRSP stops filling it in Oct 2023) | Q26 part 4 |
| 2026-10-05 | Point-in-time fundamentals: usable the trading day after the later of report and filing date; surprises dated by the report date | quarter end plus a lag | Q26 part 5; brief 08 C |
| 2026-10-05 | IVAOQ, IVSTQ, MIBQ, PSTKQ count as 0 when blank in total accruals and net operating assets | strict "missing" (left those features on about 5% of rows) | commit 5524da1 (after the first coverage report) |
| 2026-10-05 | Experts trained on the per-row likelihood, read as the composite (independence) likelihood of the per-date model | per-date likelihood | Q24; PDF Appendix D |
| 2026-10-05 | Gate weight for the 5-day target: average of the 1- to 5-step-ahead regime probabilities | filtered or one-step predicted | Q27; PDF Section 8.9 |
| 2026-10-05 | Backprop-HMM baseline keeps its own h-step predict | moving it to window weights | Q27 correction |
| 2026-10-05 | Sample 2000-2024 | 2015-2024 (too few stress episodes) | Q16 (a) |
| 2026-10-05 | Expanding training window | rolling | Q16 (b) |
| 2026-10-05 | Experts: regime-clock decay (age = later same-regime experience; one half-life in regime-days) | equal weights; calendar decay (erases crises) | Q16 (d)(e); `Training_Window_and_Weighting_Theory.md` |
| 2026-10-05 | Base: calendar exponential decay, long half-life | equal weights | Q16 (d) |
| 2026-10-05 | Gate: regime-clock forgetting via a two-pass weighted Baum-Welch; memory chosen by one-step predictive log-likelihood on validation | full history; calendar forgetting (can lose the stress state) | Q16 (d)(e); theory notes section 7 |
| 2026-10-05 | Every half-life chosen once from a preset grid (equal weights always included) in the pilot on early validation blocks, then frozen and pre-registered | tuning per fold or on test data | Q16; Q15 |
| 2026-10-05 | Compute planning on the laptop (Apple M4 Pro, 48 GB) until the school node is confirmed | — | this log |
| 2026-10-05 | Test period 2010-2024, refitted annually (15 folds) | testing from 2008 (pilot would see no crisis) | Q16 |
| 2026-10-05 | Pilot on validation years 2007-2009 chooses every setting once (half-lives, gate memory, learning rate, weight decay, step budget), with a regime-balanced validation loss; then frozen and pre-registered | tuning in every fold; a calm-only validation | Q16; Q15 |
| 2026-10-05 | Main study trains on the full block: no held-out tail, no per-fold early stopping (resolves B-2); fallback to regime-balanced early stopping if the pilot's best step budget drifts | the last-20% validation tail | Q16; audit B-2 |
| 2026-10-05 | Train on every day under a fixed step budget | every 5th day (saves nothing under a step budget) | Q16 |
| 2026-10-05 | Fresh initialisation every fold; warm starts only as a fallback (shrink and perturb) | warm starts (worse generalisation; break zero-init residual design) | Q16; Ash & Adams 2020 |
| 2026-10-05 | Seeds paired across arms; 10 per grid cell, 30 for the primary cell, final count from the pilot's measured seed variance | independent seeds per arm | Q16; Q11 |
| 2026-10-05 | Engineering: one process per performance core, large batches fixed in the pilot, features built once, base cached | default single-process training | Q16 |
| 2026-10-05 | Compute envelope for the design: depth scan 1-3 (pyramid widths from a fixed first width), first width at most 128, K at most 4; runs on the laptop, 10 parallel processes | depths 1-5 and first width 256 (1-1.5 months of laptop time; Gu, Kelly & Xiu find depths 4-5 add nothing) | section 1c; Q8, Q11 |
| 2026-10-05 | Brief 09 written: implements the sample extension, calendar-year folds, the pilot, decay weights, the weighted Baum-Welch gate, the parallel runner | — | `Code_Change_Brief_09_Training_Scheme.md` |
| 2026-10-08 | The gate's filter runs over every trading day, including the purge gap; parameters still fitted on training only | skipping the gap (one step of A bridged 6 days) | Q22 |
| 2026-10-08 | Gate input: CRSP value-weighted S&P 500 universe index (1000500), daily log total return; French Mkt-RF kept as a robustness check | French Mkt-RF (external file, whole market) | Q23 |
| 2026-10-08 | Gate is partly stock-specific at the industry level: market regime combined with an industry-level regime, both frozen; how it is implemented is open (Q28) | one market-wide weight vector per date | Q1, Q28 |
| 2026-10-08 | Framing: comparison-led; no constructive element (the differentiable jump-model gate is not pursued) | adding a design-led contribution | Q2, Q4 |
| 2026-10-08 | Horizon stays h = 5; add an IC decay curve on 2000-2006 and a horizon profile (h = 1, 3, 5, 10) for the final model as robustness | switching to h = 3 or h = 1 | Q21 follow-up |
| 2026-10-09 | IC decay curve (2000-2006) confirms h = 5: one-week reversal and abnormal volume peak at 5, slow signals still build, nothing favours 3; horizon question closed; Q26 inputs unchanged despite weak pooled ICs | switching horizon; pruning weak signals on outcomes | Q21 follow-up result |
| 2026-10-09 | Reporting additions for every arm: long-leg vs short-leg returns and half-sample ICs; decile monotonicity (Patton-Timmermann test); IC by year and by 8 pre-named episodes; `execution_lag` knob, headline tables at L = 0 and 1 | headline IC and long-short only | Q9 update; brief 12 |
| 2026-10-09 | Residual short-term reversal (`resid_mom20`) kept as an extension, not added to the inputs (pilot already run) | changing Q26 after the pilot | Improvements E3 |


---

## 1c. Compute budget (laptop, measured 2026-10-05; real pipeline timed 2026-10-06)

**Machine.** Apple M4 Pro: 10 performance and 4 efficiency cores, 48 GB, GPU (MPS) available.

**Benchmark.** Script `nec_baseline/scripts/benchmark_compute.py`, run on synthetic data only. Results
in `nec_baseline/results/benchmark/compute_benchmark_20261005_233042.{csv,json}`.

- **One thread per run.** A single core reaches 30-230 GFLOP/s; for K = 3, widths 64-32, batch
  4,096, that is 238 steps/s. Extra threads in one run give nothing (244 steps/s at 4 threads, 212
  at 8).
- **Parallel runs.** 10 runs on the performance cores give 6.9x (1,622 steps/s in total); adding the
  efficiency cores gives 7.6x. The largest model scales 5.5x at 10 runs (shared memory bandwidth).
  Plan: 10 parallel runs.
- **GPU (MPS).** It has a fixed cost of about 1.6-3 ms per step, whatever the model size. It loses to
  10 CPU runs on small models and roughly matches them on the largest, so it is at most one extra
  worker for large configurations.
- **Experts computed as one batched product:** no gain on CPU, so this option is dropped.
- **Cost of a step:** steps × batch × (6P - 2·d·w1) FLOPs, which does not depend on the number of
  training rows (fixed step budget).
- **The real pipeline** (timed 2026-10-06; `nec_baseline/scripts/time_real_pipeline.py`, timing only,
  results in `nec_baseline/results/benchmark/real_pipeline_timing.json`). One job on the real
  2000-2024 panel: Hamilton gate, K = 3, widths 64-32, batch 4,096, 2,000 steps, the 2009 pilot fold
  (training 2000-2008, 1.13M rows), one thread. **5.57 ms per step against the benchmark's 4.19 ms:
  an overhead factor of 1.33.** The first timing gave 24.4 ms (×5.8); most of that was an unused GRU
  encoder and a per-row date lookup in the gate, both removed without changing any result
  (`HANDBOOK.md` II.0c). Per job, outside the step cost: gate fit 29 s (statsmodels, K = 3, 24
  starts), base fit 0.2 s; peak memory 3.9 GB.

**What counts as one run.** One run trains one model, for one fold, from one seed. The design
multiplies:

| Block | Count | Runs |
|---|---|---|
| Gate × depth grid | 6 arms (5 gates + no-regime control) × depths × 10 seeds × 15 folds | 900 per depth |
| Primary cell, extra seeds | 20 seeds × 15 folds | 300 |
| Stage-2 construction arms | 5 arms × 10 seeds × 15 folds | 750 |
| Base depth sweep | depths × 10 seeds × 15 folds (about half the cost of a K = 2 model) | 150 per depth |
| Pilot | about 50 settings × 3 folds × 3 seeds | 450 |
| Robustness: no-decay arm for every gate (primary depth) | 6 × 10 × 15 | 900 |
| Robustness: leave-one-episode-out (primary cell) | | about 30 |

- **Depths 1-3:** 5,580 clean runs.
- **Depths 1-5:** 7,680 clean runs.
- The hours below assume **everything is run twice**: bugs, fixes and changed settings in practice
  double the clean count.

**Laptop hours.** Assumptions: everything run twice, 10 parallel runs, batch 4,096, and ×1.33 for the
real pipeline's overhead (measured 2026-10-06 for K = 3, widths 64-32; applied to every
configuration, as the earlier ×2.5 assumption was). Widths follow the pyramid rule from the first
width. Steps per run are set in the pilot, so three values are shown. The hours are linear in the
overhead factor, so these are the 2026-10-05 table's values times 1.33/2.5 (0.53); the earlier
table is in git history (commit ffc8bd3).

| Depth scan | K | First width | 3k steps | 10k steps | 30k steps |
|---|---|---|---|---|---|
| 1-3 | 3 | 64 | 7 h | 23 h (1.0 d) | 71 h (2.9 d) |
| 1-3 | 3 | 128 | 14 h | 46 h (1.9 d) | 137 h (5.7 d) |
| 1-3 | 4 | 64 | 9 h | 30 h (1.3 d) | 91 h (3.8 d) |
| 1-3 | 4 | 256 | 38 h | 126 h (5.2 d) | 377 h (15.7 d) |
| 1-5 | 3 | 64 | 11 h | 37 h (1.5 d) | 110 h (4.6 d) |
| 1-5 | 3 | 128 | 21 h | 70 h (2.9 d) | 211 h (8.8 d) |
| 1-5 | 4 | 64 | 14 h | 47 h (1.9 d) | 140 h (5.8 d) |
| 1-5 | 4 | 256 | 58 h | 194 h (8.1 d) | 581 h (24.2 d) |

Depths 4 and 5 are extrapolated from the measured cost of going from depth 2 to depth 3.

**Not in the table: the gate fits.** The campaign runner fits the gate in every job. At the measured
29 s per fit (Hamilton gate, K = 3), the 11,160 jobs of the depth 1-3 design run twice add at most
about 13 h at 10 workers (fewer: the no-regime control and the baselines fit no gate; the other
gates' fit times are not measured yet). The gate depends only on the fold and its own settings, so
fitting it once per (fold, gate) and sharing it across depths and seeds would remove most of this.

**Envelope decided 2026-10-05:** depths 1-3, first width at most 128, K at most 4.

**Reading.** With moderate widths (first width 64-128) and K ≤ 4, the whole study, run twice, takes
about 1-2 days of machine time at 10,000 steps per run and about 3-6 days at 30,000, plus up to half
a day of gate fits. That fits a laptop. The corner of wide experts (first width 256) with long
training (30k steps) and a depth scan to 5 now takes about 24 days (was 1-1.5 months at ×2.5): still
impractical, and still the only corner compute rules out. Within the rest, the binding limit is
statistical (Q17).

**To confirm:**

- ~~the real-pipeline overhead, from one timed real run~~: done 2026-10-06, ×1.33;
- the steps per run, from the pilot;
- the gate fit times of the other gates (jump, Wasserstein, TVTP) once they exist;
- the slowdown from heat over multi-day runs (budget 10-20%).


---

## 2. Log (newest first)

### 2026-10-09 (brief 11 results: regime-split IC and sector regimes, 2000-2006)
- **Market regimes (K = 2):** one long stress block, January 2000 to August 2003, calm after; daily volatility 0.69%
  calm against 1.53% stress; expected durations 229 and 177 days; all 24 starts at one optimum. K = 3: 0.60%, 0.98%,
  1.75%, durations 110, 46, 79 days.
- **Regime-split IC, reading "neutral" by the rule written before the run:** four of the five primary contrasts have
  the expected sign (reversal stronger in stress, momentum weaker), none survives Holm (smallest adjusted p 0.076,
  one-month reversal). The K = 2 contrast is largely 2000-03 against 2003-06, so slow drift is inside it; the
  trailing-volatility split keeps the one-month reversal contrast (t = -2.16) but not the one-week one. Exploratory:
  11 of 175 contrasts at q < 0.10, none at q < 0.05 (mostly abnormal volume stronger in stress, and one-month
  reversal at K = 3): hypotheses for 2007-2024, not findings.
- **Sector regimes (input to Q28):** between the two pre-written outcomes. Sector-relative regimes still follow the
  market's (median correlation 0.64, against 0.76 for raw sector returns); a sector is stressed while the market is
  calm on 6-9% of days (P = 10-16%); clearest own episode Energy 2005-06. Real Estate empty (inside Financials until
  2016), Communication Services thin (median 12 names). Caveat: "sector minus index" partly cancels big sectors.
- All recorded in the methodology PDF (new section on pre-pilot diagnostics). No design choice changed.

### 2026-10-09 (evaluation additions from the LightGBM pipeline; brief 12)
- Reviewed `OP model/lgbm` (LightGBM on China A-shares). Adopted for evaluation (Q9 update): long vs short leg,
  decile monotonicity, IC by year and named episode, and an `execution_lag` knob (L = 0 and 1 reported).
  Residual short-term reversal recorded as extension E3. Training-target outlier clipping: held for later (Q20).
  Turnover and cost reporting: explained; stays with Q9 (c).
- **Code_Change_Brief_12** written for these evaluation additions.

### 2026-10-09 (IC decay result; brief 11)
- **IC decay curve run** (brief 10 C, by Tom; commit 00b9604): rank IC of 16 signals at h = 1-10 on 2000-2006.
  One-week reversal peaks at h = 5 (IC -2.46%, t = -4.21), abnormal volume too (+1.07%, t = 4.05); fast signals
  (idiosyncratic volatility, max return, momentum) fade by day 2-3; slow ones (MA50 gap, spread, earnings dummy,
  one-month reversal) keep building. **Decided: h = 5 confirmed; Q26 inputs unchanged.** Recorded in Q21.
- **Code_Change_Brief_11** written: the IC curve split by the gate's regime (2000-2006, descriptive) and the Q28
  descriptive sector check (sector-relative regimes against the market regime).

### 2026-10-08 (methodology document)
- **`Methodology_2026-10-08.pdf`** (40 pages after adding a "why we chose this" box, with what was given up, to every decision; sources in `Methodology_tex/`): the whole methodology to date, section
  by section (data, target and horizon, features, model, gate, training, evaluation, compute, status), every open
  question grouped with a suggested order, the decision register, a glossary.
- Section 1 corrected: the training scheme and the gate's memory are implemented (brief 09), not pending.

### 2026-10-08 (Q2 and Q4 decided; horizon follow-up)
- **Decided: Q2 and Q4.** The comparison framing stands; no constructive element. The proposal is
  comparison-led.
- **Q21 follow-up:** h = 5 stays. IC decay curve on 2000-2006 and a horizon profile for the final model,
  as robustness.
- **Code_Change_Brief_10** written: Q22 (filter through the purge gap), Q23 (CRSP index as the gate
  series), the IC decay script. Q28 and Q29 stay open, nothing coded for them.

### 2026-10-08 (Q29 raised: factor-return gate)
- **New open question Q29:** a gate variant whose regimes are defined by daily factor returns (market, size,
  value, momentum) instead of the market return alone, motivated by momentum crashes in falling-volatility
  rebounds. Records how the Hamilton gate infers the regime and why stock characteristics stay out of the
  gate. Nothing decided.

### 2026-10-08 (Q1 decided: industry-level gate)
- **Decided (Tom, final): the gate is partly stock-specific, at the industry level.** Each stock's gate weight
  combines the market regime with an industry-level regime; both frozen. **How it is implemented stays open: new Q28** (factorial, coupled, sector-only or pooled chains;
  what the industry chain is fitted on; granularity; number of states). PDF and Model_Derivation need a
  section once Q28 is settled.

### 2026-10-08 (Q22 and Q23 decided)
- **Decided: Q22 (b).** The gate's filter runs over every trading day, including the purge gap. The purge
  removes labels, never information; the parameters are still fitted on the training block only.
- **Decided: Q23 (b).** The gate is fitted on the CRSP value-weighted S&P 500 universe index (1000500),
  daily log total return. French Mkt-RF kept as a robustness check. Clarified in Q23: the gate sees only
  this one series, not the stock inputs, and it is not market-neutralised (only the target is).
- Both need small code changes; they go into the next code brief.

### 2026-10-06 (real pipeline timed; compute budget updated)
- **Brief 09 implemented** (parts A-I, f76f6eb..0e515eb). The real pipeline timed once (timing only): 24.4 ms per
  step, ×5.8 the benchmark; after removing the unused GRU encoder and vectorising the gate's date lookup (1f7e67a,
  results unchanged) 5.57 ms, **×1.33**. Section 1c updated: the laptop-hours table rescaled from the assumed ×2.5
  (the depth 1-3 envelope now takes about 1-2 days at 10k steps, 3-6 days at 30k), plus the per-job gate fits
  (29 s each, at most about 13 h in total).

### 2026-10-05 (compute envelope; brief 09)
- **Decided: the compute envelope.** Depth scan 1-3 with pyramid widths from a fixed first width; first width at
  most 128; K at most 4; 10 parallel processes on the M4 Pro. All 5 gates plus a control, the depth scan, seeds,
  robustness runs and a full re-run fit in about 2-6 days of machine time (section 1c). K itself (Q18) and the
  widths (Q8/Q17) are still to be chosen inside this envelope.
- **Brief 09 written** (`Code_Change_Brief_09_Training_Scheme.md`) for local Claude Code: implements every training-scheme
  decision of today, with a reading list.

### 2026-10-05 (compute benchmark on the M4 Pro)
- **Benchmark** (`nec_baseline/scripts/benchmark_compute.py`, synthetic data only; results in
  `nec_baseline/results/benchmark/compute_benchmark_20261005_233042.*`). M4 Pro, 10 performance + 4 efficiency cores,
  48 GB, MPS available.
  - One thread per run: 30-230 GFLOP/s per core; e.g. K=3, widths 64-32, batch 4,096: 238 steps/s. Extra threads
    give nothing (244 steps/s at 4 threads, 212 at 8).
  - Parallel runs: 10 processes give 6.9x (1,622 steps/s), 14 give 7.6x; the largest model scales 5.5x at 10.
  - GPU (MPS): about 1.6-3 ms fixed cost per step, so it loses to 10 CPU processes for small models and roughly
    matches them for the largest; usable as one extra worker for large configurations.
  - Experts as one batched product: no gain on CPU (dropped).
- **Budget for the whole design** (4,650 runs: 6 arms x 3 depths x 10 seeds x 15 folds, the 30-seed primary cell,
  Stage-2 arms, pilot, base sweep; 10 parallel processes, batch 4,096, x2.5 for real-pipeline overhead), at
  10,000 steps per run: K=3 with first width 64 about 18 hours; K=4 with first width 256 about 96 hours. At 30,000
  steps, at most about 12 days. **Compute on the laptop is not the binding constraint** for K <= 4, depth <= 3,
  first width <= 256; the data (Q17) is. To confirm: real-pipeline overhead (one timed real run) and steps per run
  (pilot).

### 2026-10-05 (compute levers decided)
- **Decided:** every-day training under a fixed step budget; fresh start every fold; seeds paired across arms (10 per
  grid cell, 30 primary, final count from the pilot); process-parallel runs, large batches, features built once.
  Next: benchmark on the M4 Pro, then the compute budget.

### 2026-10-05 (test period, refits, pilot)
- **Decided:** test 2010-2024, refitted annually (15 folds); a pilot on validation years 2007-2009 chooses every
  setting once with a regime-balanced validation loss; the main study trains on full blocks with frozen settings and
  no held-out tail (resolves audit B-2). Recorded in Q16 and the decision register.

### 2026-10-05 (gate memory decided; brief 08 implemented; decision register)
- **Decided: the gate gets regime-clock forgetting** (two-pass weighted Baum-Welch), with its memory chosen by the
  one-step predictive log-likelihood of the market series on validation (Nystrup, Madsen & Lindström 2017: HMM
  parameters drift, but plain forgetting loses on the worst days). Q16 marked partly resolved: (a), (b), (d), (e)
  decided; (c) open.
- **Brief 08 implemented** by the cloud session: A target (7bb3a4d), B gate weight and Q24 notes (3f1fac5), C data
  layer (488baad), D features (0633379), E registry/docs/scripts (bd7f730); fixes from the first real-data run
  (be77fb7); Tom's zero-fill decision for four blank balance-sheet items (5524da1).
- **Decision register added** (section 1b): every big decision since the 2026-09-18 meeting in one table.

### 2026-10-05 (sample and data weighting)
- **Decided: sample 2000-2024** (was 2015-2024): about twice the stress episodes; 2000 is the earliest start with
  Compustat GICS history. Expanding window, never rolling. Laptop for compute: Apple M4 Pro, 48 GB.
- **Decided: experts use regime-clock decay** (a row's age is the amount of later same-regime experience, from the
  gate's filtered probabilities; one half-life in regime-days). **Base uses calendar exponential decay with a long
  half-life.** Half-lives chosen once in the pilot on early validation blocks, then frozen.
- **Open: the gate's memory** (full history, calendar forgetting, or regime-clock forgetting via weighted
  Baum-Welch; chosen by the gate's one-step predictive log-likelihood on validation).
- Theory written up in `Training_Window_and_Weighting_Theory.md` (new). Not yet in Advisor_Questions (Q16).

### 2026-10-05 (sector source changed; brief 08)
- **Sector source changed to GICS (Compustat) from ICB** (Tom's decision): CRSP's ICB field is NOAVAIL for every
  S&P 500 member from October 2023. Recorded in Q26 part 4.
- **Code_Change_Brief_08** written: implements Q25 (target), Q26 (features, data pipeline), Q27 (gate weight),
  documents Q24, adds the expert-weight diagnostic.

### 2026-10-05 (gate weight decided)
- **Decided: Q27, the window-average gate weight.** The gate weight for the 5-day target is the average of the
  1- to 5-day-ahead regime probabilities (filtered belief moved j steps with A). It is exact for the expected
  5-day return, consistent with the model, and equal to the one-step predicted weight at h = 1. Descriptive
  check on the CRSP index 2015-2024: it differs from the filtered weight by 0.055 on average (max 0.10).
  **Not implemented:** needs a `gate_weight` config switch in the Hamilton gate (the backprop-HMM baseline is
  unchanged: its latent is per target window, so its h-step predict is already consistent). PDF, checklist and Model_Derivation_BaseCase.md updated.

### 2026-10-05 (likelihood decided)
- **Decided: Q24, per row.** Reframed as the composite (independence) likelihood of the per-date model: each
  per-row term is the exact one-stock marginal, so the estimator is consistent, robust to the failure of A3
  (correlated stocks), and keeps the frozen gate in charge of which data each expert learns from. Costs:
  efficiency, no Hessian standard errors (not used), softer specialisation. No code change.
- **PDF revised** (`MoE_HMM_Gate_Formulation.pdf`, 51 pp): Section 7 rewritten around the per-row objective;
  new Appendix D, the full theory of per date versus per row; checklist updated.

### 2026-10-05 (features decided)
- **Q26 resolved: final list.** 26 JKP characteristics plus a 12-feature short-horizon market block (volatility
  dynamics, tails, volume and liquidity, price path, earnings timing, return dynamics), the industry block and 2
  flags: 57 inputs. Cross-checked against the Trexquant work (return-dynamics theme added from it; intraday,
  calendar and macro variables not transferable). CRSP daily file has open/high/low/bid/ask for 100% of
  S&P 500 stock-days; trade counts only 28%.
- **Decided (Q26, partly resolved, full theory recorded):** three feature families; exactly two representatives per JKP
  theme; rank to [-1, 1]; causal missing-value treatment (minimum-count windows, fundamentals carried forward with a
  12-month cap, then 0 plus a per-family flag; rows no longer dropped); industry used three ways (sector dummies,
  within-sector ranks for a subset of fundamentals, industry momentum and within-industry reversal).
- **Pending:** the list of representatives, with the fundamentals under review. Redundancy check (rank correlations,
  clustering, PCA) and the group ablation to run later.
- **Data check (aggregate only):** CRSP/Compustat Merged in `Data/` has the link history, quarterly statements, report
  dates (99.9% coverage for S&P 500 member firm-quarters, fiscal 2013-2024) and filing dates.
- **Found in code:** `rel_ret_20d` and `rel_vol_20d` are exact duplicates of `ret_20d` and `vol_20d` after ranking.

### 2026-10-05 (frequency and horizon decided; target clarified)
- **Decided: option B for Q21.** Daily gate, daily cross-section, h = 5 (resolved in
  `Advisor_Questions.md` with the full motivation: statistical power on 2015-2024, regime contrast
  clearest at h = 5, less noise than next-day returns, regime timescale, gate identification, already
  implemented and audited). Confirms the code as it is; nothing to change.
- **Descriptive check behind it** (CRSP S&P 500 members 2015-2024, aggregate only, full-sample signal
  diagnostics, not model results): reversal and momentum rank ICs on the market-neutral target at
  h = 1, 5, 21, split by trailing market volatility. Regime contrast large at h = 5 and 21; h = 21 has
  only ~118 non-overlapping periods, too few for power.
- **Q25 clarified:** market-neutral returns throughout, reporting included (no raw-return reporting);
  measured noise reduction (32% of daily variance, 28% at 5 days, 50% in 2020); implicit unit-beta caveat.

### 2026-10-01 (target decided; formulation notes)
- **Decided: the target is the market-neutral return** (Q25, resolved, with the full reasoning and the
  alternatives considered: raw/excess, cross-sectionally standardised or ranked, volatility-scaled,
  factor-residual). Defined as the forward return minus the date's equal-weighted cross-sectional
  mean. The code's `target_kind="residual"` is a different object (trailing-beta residual) and is
  not the decision. **Not implemented yet**; needs a brief (new target kind, base and experts retrained).
- **Formulation notes.** `MoE_HMM_Gate_Formulation.pdf` (sources in `MoE_Formulation_tex/`, plan in
  `MoE_Formulation_Course_Plan.md`): the base-case HMM-gated MLP MoE derived from Bishop, with warm-up
  exercises and solutions. It flags one new open item, not yet an advisor question: whether the gate
  should be the predicted probability $\sum_j A_{jk}\xi_{t\mid t}(j)$ (consistent with the
  generative model) or the filtered $\xi_{t\mid t}$ currently used.

### 2026-10-01 (brief 07 implemented)
- **Commits** (not pushed): A, the open audit bugs (ccf8378); B, the run store (1b29f84); C, resume
  (8715328); D, documentation (RUNBOOK, HANDBOOK, this tracker, audit section 12, the brief).
  351 tests pass, 1 skipped, **0 xfailed**: the last three pinned audit bugs (B-1, G-2, G-3) are fixed.
- **Fixed.** O-1 (quick mode purged), O-3 (purge = the target's horizon; a contradicting `horizon`
  raises), E-3 (horizon-consistent long-short book), B-1 (the base no longer disturbs the random
  stream), G-2 (autoregressive Hamilton gate), M-5 (HMM baseline keyed by stock), S-3 (the current
  design end to end through the control panel), G-3 (resume under a fitted gate).
- **Built.** The run store and `scripts/runs.py`; exact resume after Ctrl+C, a crash between folds,
  mid-sweep, or a torn checkpoint, with refusal on changed settings or data. `RUNBOOK.md` explains
  starting, watching, stopping and resuming runs (including `caffeinate -i`).
- **Not decided** (scope fence): `portfolio_scheme` default ("nonoverlapping"), Q16, Q18, Q20, Q21,
  Q22, Q23, X-1, B-2, the post-delisting fill. For G-2 the test-block filter keeps the harness's span
  (purge gap skipped), so Q22 stays open.
- **Found.** A resume was close but not exact until the reloaded base was built outside the restored
  random stream; the exact-equality test caught it. Some synthetic test runs had written prediction
  files into `Data/derived/runs/` through a test-isolation gap; fixed and removed (no real data).

### 2026-09-26 (brief 06 implemented)
- **Question added.** Q23, which market series the Hamilton gate is fitted on (French Mkt-RF or a
  CRSP index). Open; the gate keeps French Mkt-RF for now.
- **Diagnostic (2026-09-27).** Rows dropped only because a missing return sits in the forward target
  window: **15** of 1,264,598 (0.0012%), from 3 missing prices whose next CRSP return spans the gap
  (P1); nothing changed (`Smoke_Run_2026-09-26_CRSP.md` section 6).
- **Commits** (not pushed): step 0, the audit fixes (9c10945) and these documents (87216a2); A, the
  CRSP data layer (2552a32); B, free data retired (494c21e); C, the NLL split (f44f573); D, the
  `hidden_init` switch (3cef208); E, G-5 accepted (6dd3276); F, the CRSP integration run (d5c78d6);
  G, documentation and audit section 11. 309 tests pass, 1 skipped, 3 xfailed.
- **Decided (Tom).** The S&P 500 membership INDNO is **1000500**, not 1000502: 1000502 has no
  constituents in `StkIndMembership`; 1000500 has 2,084 spells over 1,956 PERMNOs and 502-508
  members per day in 2015-2024, bounds inclusive (70 count changes versus 276 for exclusive
  bounds). Erratum added to brief 06 A.3.
- **Decided (Tom).** Delisting rule (a): CIZ `DlyRet` already includes the delisting return
  (`MetaSIZtoCIZ`; on the real data `DlyRet` equals `DelRet` on every delisting row, 5,376 of 5,376
  market-wide in 2015-2024). Nothing is compounded in.
- **Decided (Tom).** A forward window that runs past a stock's final CRSP return is completed with a
  post-delisting return, `post_delisting_return` = `"cash"` (0) or `"market"`; default `"cash"`,
  not a decision, recorded on every trial. It touched 407 panel rows. The 2 member delistings
  without a delisting return (GDR/FING) keep missing targets; nothing is imputed.
- **Found.** Same-seed expert starts are bit-identical across gate arms (private generator; B-1
  does not reach them). Dropout masks are drawn independently per expert, so the `"identical"`
  control needs dropout 0. The Hamilton gate's input series is French daily Mkt-RF (brief 03 §2),
  so the French factors are not "diagnostics only" as brief 06 B says; the docstrings now say so.
- **Integration run on CRSP** (`Smoke_Run_2026-09-26_CRSP.md`): every fold's gate converged (24 of 24
  starts), stationary probabilities 0.32-0.68, expected durations 32-72 days; no stop condition.
  No out-of-sample number was looked at or reported.

### 2026-09-26
- **Data.** CRSP/Compustat Merged downloaded: `cfz202607_ascii.zip` (95 files, 13.2 GB unzipped) and
  `clz202607_ascii.zip` (11 files). Sizes match, no CRC errors. Fundamentals cover fiscal 1950 to
  2026 for 46,957 firms; the link table has 125,736 links; `filingdates` gives the actual 10-K and
  10-Q filing dates, needed for point-in-time fundamentals.
- **Code (Claude Code).** The five critical audit findings were fixed and pinned by tests: D-1
  (point-in-time ranks), D-2 (adjusted prices in dollar volume), E-1 (the base is scored under the
  same mixture density with the corrections at zero), E-2 (overlap-robust t-stats; Hansen-Hodrick
  chosen over Newey-West after a size simulation), M-1 (HMM prior lag). 272 tests pass. Not yet
  committed; brief 06 commits them.
- **Decided.** CRSP replaces the free price and universe data in the whole pipeline.
- **Decided.** Report the volatility-only NLL gain as its own number next to the correction gain
  (follow-up to E-1).
- **Decided.** Expert hidden-layer initialisation becomes a config switch, identical or diversified
  (X-1). Which one is used is not decided.
- **Accepted.** Both start-value deviations from brief 04 (G-5): the expanding volatility quantile and
  the 1-nat clustering gap.
- **Questions added.** Q21 (frequency and horizon), Q22 (gate filter through the purge gap), Q16 ask
  (e) (crash-preserving weights, regime-clock decay, and testing which periods matter, with Data
  Shapley, influence functions and Mulliner et al. 2025).
- **Brief 06 written:** `Code_Change_Brief_06_CRSP_And_Decisions.md`.

### 2026-09-25
- **Data.** CRSP stock file `ciz202512_ascii.zip` downloaded and verified (34 files, 66.4 GB
  unzipped; monthly 1925-12 to 2025-12, 40,518 PERMNOs). `Data/` added to `.gitignore`.
- **Code.** Brief 04 committed: A, the objective registry seam (866914c); B, informed starting values
  for the Hamilton fit, 20 of 20 starts converging (1b7561e); C, the first end-to-end smoke run on free
  data (fd1a8ed). The gate tracks VIX (Spearman 0.81); seed variance of the base is as large as the
  differences between arms. Diagnostic only, not quotable.
- **Audit.** Brief 05 run: `Code_Audit_2026-09-25.md` (9a29368). 5 critical, 12 major, 19 minor,
  13 notes, with a mutation table.
- **Question extended.** Q16 (d), time decay, with reading.

### 2026-09-22
- **Maths.** `Model_Derivation_BaseCase.md`, the full derivation of the base case.

### 2026-09-21
- **Decided.** Q19, gate fitted separately and frozen, for all four gates. Q7, base pre-trained and
  frozen. **Opened** Q20, the error function.
- **Code.** Brief 01 survey and persistence diagnostics (7c3e873). Brief 02, residual mixture with a
  frozen base (811a0e1) and its corrections (15403df). Brief 03, the gate interface and the Hamilton
  gate (1bfe7ca).

### 2026-09-18 to 2026-09-19
- **Advisor meeting.** Data is CRSP (Q3 resolved). The mechanism-by-depth grid survives (Q11
  resolved). No width sweep, no sweep over K. Sequence: data, then compute budget, then parameters.

### 2026-07
- **Pre-thesis baseline** (`nec_baseline`): synthetic regime panels, free-data Stage B panel,
  walk-forward harness, multi-seed sweeps, tuning discipline, calibration, checkpointing, figures.

---

## 3. Open, in one place

- **Advisor questions open:** Q28 (industry-gate implementation), Q29 (factor-return gate), Q5, Q6 (design half), Q8, Q9, Q10, Q12 to Q15, Q16 (c) and the rest of the training scheme, Q17, Q18, Q20.
- **Audit findings still open:** B-2 (decided 2026-10-05 to be resolved by training on full blocks;
  not yet coded); the majors S-2 (the superseded 2026-09-25 smoke report's NLL columns)
  and X-1 (the `hidden_init` choice); and the minors and notes listed in audit section 12.
- **Next in `WORK_QUEUE.md`:** documentation drift and `DECISIONS.md`, diagnostics (ICC, gate
  permutation test), the remaining gates, pre-registration.

## 4. Parked ideas

- **Future study, beyond the thesis:** `Improvements_and_Extensions.md` (created 2026-10-05). E1: how to improve the model for one-day forecasting. E2 (2026-10-08): the regime-gated MoE as a stacking meta-model over machine-learning alphas. E3 (2026-10-09): residual short-term reversal as an input. E4 (2026-10-09): recurrent or transformer experts on raw price sequences.
- Switching the gate's input from French Mkt-RF to a CRSP index series (Q23).

- Testing which historical periods matter (Q16 e, extension; diagnostic only).
- Leave-one-episode-out: train without 2008, test on it (Q16 point 3).
