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

## 1. Current state (2026-09-26)

| Component | What it is | Status | Where |
|---|---|---|---|
| Data | CRSP daily and monthly stock file (CIZ format, to Dec 2025); CRSP/Compustat Merged (July 2026 release) | Downloaded and verified. The pipeline switches to CRSP in brief 06 | Q3; `Data/` (gitignored, licensed) |
| Universe | S&P 500 point-in-time | Currently Wikipedia reconstruction on free prices; brief 06 moves it to CRSP membership | `universe.py` |
| Frequency and horizon | Code runs daily with a 5-day forward target | **Open** | Q21 |
| Base | MLP, trained on the training block, then frozen | Decided, implemented | Q7; brief 02 |
| Gate | Regime model fitted separately, then frozen; output is the filtered probability, never the smoothed one | Decided. Hamilton implemented; jump, Wasserstein and TVTP to come | Q19; briefs 03, 04 B |
| Experts | MLP corrections to the base, $\hat y = f_0 + \sum_k \pi_k r_k$, zero-initialised heads | Implemented. Initialisation of the hidden layers becomes a switch in brief 06 (X-1) | brief 02; audit X-1 |
| Number of regimes K | | **Open** | Q18 |
| Error function | Only the mixture NLL is registered, behind a seam | **Open** | Q20 |
| Depth grid | Mechanism by depth, depths set by the compute budget | Decided | Q11 |
| Evaluation | Walk-forward with purge; rank IC with Hansen-Hodrick t-stats for overlapping targets; NLL improvement over the base | Implemented and audited. Volatility-only NLL gain added in brief 06 | audit E-1, E-2 |
| Gate through the purge gap | Whether the filter sees the gap's market returns | **Open** | Q22 |
| Sample weighting | Time decay; crash-preserving weights | **Parked**, not in code | Q16 (d), (e) |

---

## 2. Log (newest first)

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

- **Advisor questions open:** Q1, Q2, Q4, Q5, Q6 (design half), Q8, Q9, Q10, Q12 to Q18, Q20, Q21, Q22.
- **Audit findings still open:** B-2 (the base withholds its validation tail with early stopping
  off; interacts with Q16), and the majors B-1, E-3, O-1, O-3, S-2, S-3, M-5, G-2, G-3.
- **Next in `WORK_QUEUE.md`:** documentation drift and `DECISIONS.md`, diagnostics (ICC, gate
  permutation test), the remaining gates, pre-registration.

## 4. Parked ideas

- Time decay of old data (Q16 d).
- Keeping crash periods at full weight, regime-clock decay, and testing which periods matter (Q16 e).
- Leave-one-episode-out: train without 2008, test on it (Q16 point 3).
- Compustat fundamentals as features. The data is in hand; whether to use it is not decided.
