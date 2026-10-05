# Progress tracker

Everything added to the model or the thesis, with the date and where the detail lives. Section 1 is the
current state and is edited in place. Section 2 is a log, newest first, and is only appended to. The
detail stays in the source documents; this file points to them.

- **Decisions and open questions:** `Advisor_Questions.md`
- **Code briefs:** `Code_Change_Brief_*.md`, `Code_Audit_Brief_05.md`
- **What the code does:** `Code_Audit_2026-09-25.md` (most recent full reading), `Code_State_2026-09-21.md` (older)
- **What is left to code:** `WORK_QUEUE.md`
- **The maths:** `Model_Derivation_BaseCase.md`

---

## 1. Current state (2026-10-01)

| Component | What it is | Status | Where |
|---|---|---|---|
| Data | CRSP daily stock file (CIZ `ciz202512`, to Dec 2025) feeds the whole pipeline: a resumable extract and the point-in-time panel (1,264,598 rows, 2,516 dates, 2015-2024) in `Data/derived/`. Free data (yfinance, Stooq, Wikipedia) retired. CRSP/Compustat Merged downloaded, not used | Implemented (brief 06 A, B); integration run on CRSP passed (F) | Q3; `crsp.py`; `Data/` (gitignored, licensed) |
| Universe | S&P 500 point-in-time from CRSP membership spells, INDNO 1000500 (not 1000502, which has no constituents), bounds inclusive, 502-508 members per day; dual-class companies kept as two PERMNOs. Delisting returns already in `DlyRet` (rule a); forward windows past a delisting completed by `post_delisting_return` (`"cash"` default, not a decision) | Implemented (brief 06 A) | `crsp.py`, `universe.py` |
| Frequency and horizon | Daily Hamilton gate, daily cross-section, 5-day forward target (h = 5, overlapping by 4 days; Hansen-Hodrick t-stats, purge = h). Already what the code does | **Decided 2026-10-05** (option B); no code change | Q21 |
| Target | **Market-neutral return**: the forward return minus the date's equal-weighted cross-sectional mean over the stocks with a valid target. Not the code's existing `target_kind="residual"` (a trailing-beta CAPM-style residual). Code still defaults to `"raw"` | **Decided 2026-10-01**; implementation pending (no brief yet) | Q25 |
| Features | Three families: price/volume (CRSP), fundamentals (Compustat via CCM, point-in-time), industry (GICS, Compustat; ICB dropped because CRSP's ICB stops in Oct 2023). Rule: exactly two representatives per JKP theme (13 themes, 26 characteristics), plus a short-horizon market block (6 themes × 2), 11 sector dummies, industry momentum, within-industry reversal and 4 within-sector fundamentals: 40 characteristics (22 market, 18 fundamental), 57 inputs. Rank to [-1, 1]. Missing: minimum-count windows, fundamentals carried forward (12-month cap), then 0 plus a flag per family; rows no longer dropped. Code still has the 14 price/volume features ranked to [-0.5, 0.5] (two exact duplicates after ranking) | **Decided 2026-10-05**; not implemented (needs a CRSP re-extract with OHLC and bid/ask, and the Compustat pipeline) | Q26 |
| Base | MLP, trained on the training block, then frozen | Decided, implemented | Q7; brief 02 |
| Gate | Regime model fitted separately, then frozen; gate weight is the **average over the 5-day target window of the h-step-ahead regime probabilities** (built from the filtered probability and A; never the smoothed one). Code still uses the filtered probability | Decided. Hamilton implemented, the autoregressive variant too (brief 07, G-2); jump, Wasserstein and TVTP to come. The backprop-HMM baseline now runs on the point-in-time panel (state keyed by stock, M-5) | Q19, Q27; briefs 03, 04 B, 07 |
| Experts | MLP corrections to the base, $\hat y = f_0 + \sum_k \pi_k r_k$, zero-initialised heads | Implemented. Hidden-layer initialisation is a switch, `hidden_init` (brief 06 D); which one is used is **open** (X-1) | brief 02; audit X-1 |
| Number of regimes K | | **Open** | Q18 |
| Error function | Only the mixture NLL is registered, behind a seam | **Open** | Q20 |
| Depth grid | Mechanism by depth, depths set by the compute budget | Decided | Q11 |
| Evaluation | Walk-forward with the purge read from the target's horizon (quick mode purged too); rank IC with Hansen-Hodrick t-stats for overlapping targets; NLL reported three ways (`NLL_single`, `NLL_base`, `NLL_full`); a horizon-consistent long-short book, `portfolio_scheme` "nonoverlapping" or "staggered" (default **not decided**). Every trial row carries `data_source`, `post_delisting_return`, `hidden_init` and `portfolio_scheme` | Implemented and audited (briefs 06 C, 07 A) | audit E-1, E-2, E-3, O-1, O-3; `evaluation.py`, `registry.py` |
| Runs and resume | Every run in `results/<campaign>/<run_id>/` with settings, status, registry, metrics, figures, log and `SUMMARY.md`, one row in `results/INDEX.csv`; per-stock outputs only in `Data/derived/runs/`. Ctrl+C checkpoints and stops; resume is exact and refuses changed settings or data unless forced | Implemented (brief 07 B, C); how-to in `nec_baseline/RUNBOOK.md` | `runstore.py`, `scripts/runs.py` |
| Likelihood for the experts | Per row: the composite (independence) likelihood of the per-date model; one log-sum-exp per row. Already what the code does (`losses.mixture_nll`) | **Decided 2026-10-05**; no code change | Q24; PDF Section 7.3 and Appendix D |
| Gate through the purge gap | Whether the filter sees the gap's market returns | **Open** | Q22 |
| Sample weighting | Time decay; crash-preserving weights | **Parked**, not in code | Q16 (d), (e) |

---

## 2. Log (newest first)

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

- **Advisor questions open:** Q1, Q2, Q4, Q5, Q6 (design half), Q8, Q9, Q10, Q12 to Q18, Q20, Q22, Q23.
- **Not yet a question:** whether to extend the sample before 2015 (loader `start` is a config field).
- **Audit findings still open:** B-2 (the base withholds its validation tail with early stopping
  off; interacts with Q16); the majors S-2 (the superseded 2026-09-25 smoke report's NLL columns)
  and X-1 (the `hidden_init` choice); and the minors and notes listed in audit section 12.
- **Next in `WORK_QUEUE.md`:** documentation drift and `DECISIONS.md`, diagnostics (ICC, gate
  permutation test), the remaining gates, pre-registration.

## 4. Parked ideas

- **Future study, beyond the thesis:** `Improvements_and_Extensions.md` (created 2026-10-05). E1: how to improve the model for one-day forecasting.
- Switching the gate's input from French Mkt-RF to a CRSP index series (Q23).

- Time decay of old data (Q16 d).
- Keeping crash periods at full weight, regime-clock decay, and testing which periods matter (Q16 e).
- Leave-one-episode-out: train without 2008, test on it (Q16 point 3).
