# Code change brief 09: the training scheme (sample, folds, pilot, decay weights, weighted gate, parallel runs)

Date: 2026-10-05. Written for **Claude Code running locally on Tom's Mac** (Apple M4 Pro, 10 performance +
4 efficiency cores, 48 GB). Unlike brief 08, you can see `Quant Model/Data/`, so the real-data steps in
part A and the timing in part H can be run. They must stay aggregate-only.

Nine parts, done in order (A to I). Each part ends in its own commit. Do **not** push. Tom pushes himself.

Repository `Quant Model/nec_baseline`, package `nec_moe`; the git root is `Quant Model/`. Same rules as
briefs 02 to 08:

- every numeric quantity is a config field with a documented default;
- every claim gets a test that pins it;
- this brief contains only decisions Tom has made. Where something is open it says so, and the code
  must not pick an answer.

---

## Licence and research-integrity rules

- CRSP, Compustat and anything derived from them stay in `Quant Model/Data/`. Per-security outputs go
  only to `Data/derived/...`. `results/` holds aggregate outputs only. Never commit anything under
  `Data/`. Stage files by explicit path. `Claude outputs/` stays untracked.
- **No out-of-sample performance number on the real panel is printed, written or read.** That covers
  every date from 2010 on: IC, NLL, portfolio returns, anything that scores predictions.
- You build the pilot (part F) and test it on synthetic panels, but **you do not run the pilot on real
  data**. Tom runs it after reviewing the code.
- Real-data work allowed in this brief:
  - building the extracts and the panel for 2000-2024;
  - the existing aggregate feature-coverage report;
  - fitting the gate on the market series for diagnostics (convergence, state durations; no scoring);
  - one timed training run with **timing output only** (part H).

---

## 0. Read first

**Decisions** (all resolved 2026-10-05; read the full text, not only the summaries):

1. `Master Thesis/Advisor_Questions.md`, **Q16** (the resolution block covers (a), (b), (d), (e), the
   test period, refits and pilot, and the compute levers). Also **Q24** (per-row composite
   likelihood), **Q27** (window-average gate weight), **Q19** (gate fitted separately and frozen) and
   **Q8** (the compute-envelope update).
2. `Master Thesis/Training_Window_and_Weighting_Theory.md`: the theory and formulas for the decay
   weights and the gate's memory. Sections 4-8 are the specification for parts C, D and F.
3. `Master Thesis/Progress_Tracker.md`, sections 1b (decision register) and 1c (compute budget).
4. `Master Thesis/MoE_HMM_Gate_Formulation.pdf`:
   - Section 5: the HMM, forward pass, Baum-Welch with scaling. This is the notation for part D.
   - Section 7 and Appendix D: per-row composite likelihood; decay weights multiply each row's term.
   - Section 8.9: the window-average gate weight.
5. `Code_Change_Brief_08_Target_Features_Gate.md`: what was just built.

**Code:**

- `run_experiment.py` (`Experiment`);
- `nec_moe/evaluation.py` (`walk_forward_folds`, `walk_forward_evaluate`);
- `nec_moe/markov_gate.py` (`MarkovSwitchingRegimePrior`, the multi-start fit, `apply_causal`,
  `gate_weight`);
- `nec_moe/train.py`;
- `nec_moe/base.py` (`fit_base`, `BaseCache`, `val_fraction`, early stopping);
- `nec_moe/losses.py`;
- `nec_moe/sweep.py`;
- `nec_moe/runstore.py`;
- `scripts/extract_crsp_v2.py`, `scripts/extract_compustat.py`, `scripts/feature_coverage_report.py`,
  `scripts/benchmark_compute.py`.

**Literature: read before implementing the named part.** About 3 hours in total. The page and section
pointers are what matters.

| For part | Read | What to take from it |
|---|---|---|
| D | Rabiner, L. R. (1989), "A tutorial on hidden Markov models...", *Proc. IEEE* 77(2), sections III-C, V-A (scaling) | Baum-Welch re-estimation formulas, and scaled forward-backward so nothing underflows on 6,000 dates |
| D | Bishop (2006), *PRML*, sections 13.2.1-13.2.4 | Same in the notation of the formulation PDF; the M-step for Gaussian emissions and for A |
| D | Nystrup, Madsen & Lindström (2017), "Long memory of financial time series and hidden Markov models with time-varying parameters", *J. Forecasting* 36(8), section 3 | Exponentially weighted HMM estimation, effective memory, and the **one-step predictive log-likelihood** as the criterion for choosing the memory (part F) |
| D | Hamilton (1994), *Time Series Analysis*, chapter 22.4 | The filter, and h-step regime forecasts with the transition matrix (already used by Q27) |
| C | Pesaran, Pick & Pranovich (2013), "Optimal forecasts in the presence of structural breaks", *J. Econometrics* 177(2), sections 2-3 | Why down-weighting beats dropping data; exponential weights as the robust choice |
| C | Varin, Reid & Firth (2011), "An overview of composite likelihood methods", *Statistica Sinica* 21, section 2 | Weighted composite likelihood: row weights keep the estimating equation unbiased |
| B, G | López de Prado (2018), *Advances in Financial Machine Learning*, chapter 7 | Purging between training and test with overlapping labels (already used; extend to calendar-year folds) |
| B | Gu, Kelly & Xiu (2020), *RFS* 33(5), section 2.3 (sample splitting) | Annual refits with an expanding window: the protocol being mirrored |
| E | Ash & Adams (2020), "On warm-starting neural network training", *NeurIPS* | Why every fold starts from fresh weights |
| G | PyTorch docs, "Multiprocessing best practices" and `torch.set_num_threads` | Spawn start method on macOS; one thread per worker; no CUDA/MPS tensors shared between processes |

---

## A. Sample 2000-2024 (Q16 (a))

**A.1** `Experiment.start` defaults to `"2000-01-01"` and `end` to `"2024-12-31"`.

The extracts must reach far enough back for the feature windows:

- about 5 years of CRSP before 2000 (the 1,260-day beta and the years 2-5 seasonality);
- Compustat for 12-quarter growth (`debt_gr3`) and 8-quarter volatility (`niq_su`).

The brief 08 helper derives the lookback from the longest feature window. Use it, do not hardcode
dates.

**A.2** GICS history starts in mid-1999 (the `INDFROM` of the earliest rows). Check the sector-missing
share in 2000 in the coverage report; if it is materially above later years, report it rather than
patch it.

**A.3** Build the extracts and the panel for 2000-2024 with the existing scripts, into new folders
(brief 08 C.1: never overwrite an extract). Run `feature_coverage_report` for 2000-2024 and add a
**by-year table for 2000-2009**, in particular:

- the RDQ coverage for universe firm-quarters (it was 99.9% for fiscal 2013-2024; earlier years may be
  lower);
- the flag rates.

Aggregate only.

Commit: "Brief 09 A: sample 2000-2024 (Q16 a); extracts and coverage".

---

## B. Calendar-year walk-forward and the pilot folds (Q16)

**B.1** Add calendar-year folds alongside the existing count-based `walk_forward_folds`. Config fields:

- `fold_scheme: "calendar_year" | "count"`, default `"calendar_year"`;
- `first_test_year = 2010`;
- `last_test_year = 2024`;
- `pilot_validation_years = (2007, 2008, 2009)`.

**Main folds.** For each test year Y from 2010 to 2024:

- training dates: every date from the sample start up to the last trading day of Y-1, minus the last
  `purge` dates;
- test dates: every trading day of Y;
- `purge` is the target horizon (5), read from the panel as today.

That gives 15 folds.

**Pilot folds.** For each validation year V in (2007, 2008, 2009):

- training dates: sample start to the end of V-1, minus the purge;
- validation dates: year V.

**B.2 Hard guard.** Pilot mode must **refuse** to touch any date at or after `first_test_year`. Assert
on the panel slice it receives, not only on the fold dates. Main mode keeps the existing purge
checks. Tests:

- fold dates on a synthetic multi-year calendar;
- no test date appears in any training block;
- the pilot raises if handed a 2010 date.

**B.3 No held-out tail in main mode (resolves audit B-2).**

- In main mode the base trains on its full training block with early stopping off, and
  `BaseConfig.val_fraction = 0`.
- The experts train for the fixed step budget.
- In pilot mode, validation is the separate validation year, never a tail of the training block.

Commit: "Brief 09 B: calendar-year folds 2010-2024, pilot folds 2007-2009, B-2".

---

## C. Decay weights for the base and the experts (Q16 (d)(e))

Specification: `Training_Window_and_Weighting_Theory.md`, sections 4-6. Let T be the last training
date of the fold.

**C.1 Base: calendar exponential decay.** A training row on date s gets

```
w_base(s) = 2 ** ( -age(s) / H_base ),    age(s) = number of trading days from s to T
```

- Config `BaseConfig.decay_half_life_days: float | None`. `None` means equal weights.
- Default: `None`, until the pilot chooses (part F).

**C.2 Experts: regime-clock decay (mixture form).** With xi_u(k), the frozen gate's **filtered**
probabilities on the training block:

```
n_k(s, T) = sum over u = s+1 .. T of xi_u(k)          # later regime-k experience, in regime-days
w_exp(s)  = sum over k of xi_s(k) * rho ** n_k(s, T),  rho = 2 ** (-1 / H_expert)
```

- Config `TrainConfig.expert_decay: "none" | "regime_clock"` and `expert_decay_half_life: float`
  (in regime-days). Default `"none"` until the pilot chooses.
- Compute n_k with a reverse cumulative sum. It is O(T·K): do not loop over pairs of dates.
- Use the filtered probabilities, not the window-average gate weights and not the smoothed
  probabilities.
- The per-expert form (expert k's gradient weighted by rho to the power n_k only) is an **open minor
  point**: do not implement it. Leave a one-line note in the docstring.

**C.3 Using the weights.** Both losses become weighted means:

```
loss = sum(w * per_row_loss) / sum(w)
```

- This applies to the per-row mixture NLL for the experts and to the base's loss.
- Normalise by the sum, so the step size does not depend on the half-life.
- The weights are fixed per fold, computed once before training, and stored with the fold's
  aggregate diagnostics as summary statistics only:
  - the Kish effective sample size `(sum w)^2 / sum w^2`, overall and per regime;
  - the weight mass by calendar year.

**C.4 Tests:**

- With H = infinity (or `None`), training is bit-identical to unweighted.
- With a gate that is certainly in regime k throughout, the regime-clock weights reduce to calendar
  decay at the same half-life.
- A rare-regime row's weight decays slower than a common-regime row of the same calendar age
  (pin a number).
- The weights depend only on data up to T.

Commit: "Brief 09 C: calendar decay for the base, regime-clock decay for the experts".

---

## D. The gate: regime-clock forgetting with a weighted Baum-Welch (Q16 (d)(e), theory notes section 7)

The Hamilton gate currently fits with statsmodels, which takes no weights. Add a native implementation
for the order-0 Gaussian model (switching mean and variance). This is what `MarkovRegression` fits
today.

**D.1 Unweighted native Baum-Welch.** Implement it first:

- scaled forward-backward, as in Rabiner section V-A or the formulation PDF Section 5;
- M-steps for the initial probabilities, A, the means and the variances;
- a variance floor as a config field;
- multi-start using the existing start-value machinery.

Test it against statsmodels on synthetic and on the real market series: log-likelihood, parameters
and filtered probabilities agree to tolerance. That is the correctness anchor before any weighting.

**D.2 Weighted fit, two passes.**

1. **Pass 1.** Unweighted fit on the training block, giving filtered probabilities xi_t(k).
2. **Regime-clock weights.** For each regime k and date t:
   ```
   omega_t(k) = rho_g ** n_k(t, T),   rho_g = 2 ** (-1 / H_gate)
   ```
   with n_k as in C.2. Each regime ages on its own clock.
3. **Pass 2.** Baum-Welch whose E-step is unchanged and whose M-steps weight each term by that
   regime's clock:
   - means and variances of regime k: the sums of gamma_t(k)·m_t etc. become sums of
     omega_t(k)·gamma_t(k)·m_t etc.;
   - transitions out of regime j: the sums of xi_t(j, l) and gamma_t(j) become
     omega_t(j)·xi_t(j, l) and omega_t(j)·gamma_t(j).

   Iterate to a tolerance on parameter change (config). Log the weighted objective each iteration.
   This is a weighted generalised-EM heuristic: report if it fails to settle, never silently stop.
4. Keep the canonical state ordering (by variance) after pass 2.

**D.3 Config.**

- `MarkovGateConfig.memory: "full" | "regime_clock"`. Default `"full"` until the pilot chooses.
- `gate_half_life: float` (regime-days).
- `fit_backend: "statsmodels" | "native"`. `"native"` is required when `memory = "regime_clock"`.
- With `order > 0` (the autoregressive gate) and `memory = "regime_clock"`: raise
  NotImplementedError with a clear message. The AR gate keeps full memory for now.

**D.4 Downstream unchanged.** After the weighted fit, everything is as before:

- the forward filter over training and test dates with the frozen parameters (`apply_causal`);
- `gate_weight = "window"` (Q27).

The regime-clock weights affect only how the parameters are estimated, never the filter.

**D.5 Tests:**

- H = infinity reproduces the unweighted fit exactly.
- On a synthetic series whose calm-regime variance drifts over time while stress keeps its
  parameters, the weighted fit tracks the recent calm variance and keeps the stress variance.
- A long calm stretch at the end does **not** remove the stress state (the failure mode in theory
  notes section 7.3); compare against calendar forgetting at the same half-life.

Commit: "Brief 09 D: native Baum-Welch and regime-clock weighted gate fit".

---

## E. Training protocol details (Q16, compute levers)

**E.1 Fixed step budget per fold.** `TrainConfig.steps` and `BaseConfig.steps` are set by the pilot.
There is no per-fold early stopping in main mode. Train on every trading day's rows; do not thin to
every fifth day.

**E.2 Fresh initialisation every fold: no cross-fold warm start.**

- Add a test that the parameters at the start of fold f do not depend on fold f-1's trained
  parameters: same seed and same fold give the same starting weights, whatever ran before.
- The existing within-run expert "warmstart" (`Experiment.warmstart`, slice pre-training) is a
  different thing. Leave it as it is and say so in the docstring.

**E.3 Seeds paired across arms.**

- Every arm in a campaign (gate × depth × construction) runs on the same seed list.
- A seed fixes the base, the expert initialisation and the data order, independently of the arm.
- The registry records the seed, so comparisons can be made seed by seed.
- Test: two arms with the same seed and fold see identical mini-batch index sequences and identical
  base parameters.

**E.4 Batch size.** It remains a config field. The pilot grid (part F) includes 1,024 and 4,096.

Commit: "Brief 09 E: fixed step budget, fresh starts, paired seeds".

---

## F. The pilot (Q16): selection on 2007-2009 only, then freeze

**F.1 Script.** Add `scripts/run_pilot.py` and a pilot config file listing preset grids.

| Setting | Default grid |
|---|---|
| H_base (trading days) | infinity, about 10 years, about 5 years |
| H_expert (regime-days) | infinity, plus two regime-day values equivalent to roughly 10 and 5 calendar years for the stress regime |
| H_gate (regime-days) | the same scale as H_expert |
| learning rate | small grid |
| weight decay | small grid |
| batch size | 1,024 and 4,096 |
| step budget | small grid |

**F.2 Choosing the settings.**

- **Gate memory** is chosen **first and separately**, by the gate's **one-step predictive
  log-likelihood of the market series** on each validation year, summed over the 3 pilot folds. The
  filter runs forward through the validation year with frozen parameters, as in Nystrup et al.
  - Report separately the contribution of the 20 worst days.
  - Ties go to the longer memory.
- **Other settings** use the mixture's **regime-balanced validation loss**. Each validation row is
  weighted by
  ```
  v(s) = sum over k of wbar_k(s) / M_k
  ```
  where wbar is the window-average gate weight and M_k is the mean of wbar_k over the validation
  rows. Every regime then contributes equal total weight. The loss is averaged over 3 pilot folds and
  the pilot seeds.
- **Training length check.** Record the best step budget per pilot fold. If it differs by more than
  a factor of 2 across the three folds, the selection file says so; the fallback (regime-balanced
  early stopping) is then Tom's call, so do not implement it pre-emptively.

**F.3 Freezing.**

- The script writes `results/pilot/pilot_selection.json` (aggregate: the chosen settings, the grid,
  per-setting validation summaries) and its hash.
- Main-mode runs **refuse to start** unless a selection file is given. Its hash is recorded in every
  trial row: this is the pre-registration lock (Q15).
- Main mode also refuses settings that differ from the selection file unless `force_deviation=True`,
  which is recorded.

**F.4 Testing.** Test end to end on a synthetic multi-year panel. **Do not run it on the real
panel.**

Commit: "Brief 09 F: pilot runner, regime-balanced validation loss, selection lock".

---

## G. Parallel runner (compute levers; benchmark in Progress_Tracker 1c)

**G.1 Job list.** A campaign is a list of jobs (arm, depth, seed, fold). Add `scripts/run_campaign.py`:

- a process pool (spawn start method) with `n_workers` (config, default 10 = the performance cores);
- each worker calls `torch.set_num_threads(1)` and runs one job at a time.

**G.2 Build data once.** Load the panel and features once per campaign and give them to workers
**read-only**: save the panel tensors to a `.npy`/`.pt` file in `Data/derived/` and memory-map it in
each worker. Never rebuild features per job.

Base predictions are cached per (fold, seed, base depth) with the existing `BaseCache`. Workers must
not race on the cache: one writer, or a lock file.

**G.3 Resume.** Each job is resume-safe through the run store (brief 07).

- A crashed or interrupted campaign restarts only the unfinished jobs.
- Ctrl+C stops all workers cleanly.
- `caffeinate -i` is recommended in `RUNBOOK.md`.

**G.4 Tests (synthetic):**

- a 2-worker campaign of 4 tiny jobs gives results identical to running them sequentially;
- killing one job and resuming completes only that job.

Commit: "Brief 09 G: process-parallel campaign runner".

---

## H. Compute envelope and one real timing run (Progress_Tracker 1c)

**H.1 Envelope as documented defaults** (not hard limits):

- depth scan `(1, 2, 3)` with `pyramid_dims` from a fixed first width;
- first width at most 128;
- K at most 4.

Put this in `HANDBOOK.md`. **Do not choose K or the first width.** Those are Q17/Q18 and stay open;
keep the current defaults.

**H.2 One timed real run.** On the real 2000-2024 panel, run **one** job (any gate, K = 3, widths
64-32, batch 4,096, 2,000 steps, the 2009 pilot fold, i.e. training 2000-2008) and record **timing
only**:

- seconds per step in the real pipeline;
- data loading, gate fit, base fit, expert training;
- peak memory.

Write `results/benchmark/real_pipeline_timing.json`. Compute the **overhead factor**: real
seconds per step divided by the benchmark's for the same configuration
(`results/benchmark/compute_benchmark_20261005_233042.json`). Disable every metric that scores
predictions for this run. Do not print or save any IC, NLL or return.

Commit: "Brief 09 H: envelope defaults and real-pipeline timing".

---

## I. Docs, registry, checks

**I.1 Registry.** Every trial row also records:

- `fold_scheme`, the test year, and the pilot or main mode;
- `start` / `end`;
- `H_base`, `expert_decay`, `H_expert`, gate `memory`, `H_gate`, `fit_backend`;
- the pilot-selection hash;
- the seed.

**I.2 Docs.** Update `HANDBOOK.md` and `RUNBOOK.md`: the pilot-then-main workflow, the campaign runner,
the selection lock, the envelope.

**I.3 Checks.** pytest (all), ruff and mypy from `nec_baseline/`. Report:

- the counts;
- anything skipped and why;
- the overhead factor from H.2;
- the 2000-2009 coverage findings from A.3;
- every place where you made a conservative reading of this brief.

Commit: "Brief 09 I: registry fields and docs". **Do not push.**

---

## What this brief does not decide (do not pick answers)

- K (Q18), the widths and depths inside the envelope (Q8/Q17), the error function (Q20), the gate
  series (Q23), the purge-gap filter (Q22), leave-one-episode-out (Q16 (c)).
- The per-expert form of the regime-clock decay (open minor point).
- The fallback to early stopping (only if the pilot shows unstable step budgets; Tom decides).
- Running the pilot or any main-study run on real data.
