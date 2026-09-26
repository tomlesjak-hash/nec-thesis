# Code change brief 04: objective seam, Hamilton starting values, first real-data run

Date: 2026-09-25. Three items, done in order: A, then B, then C. C depends on B.

Repository `Quant Model/nec_baseline`, package `nec_moe`. Same rules as briefs 02 and 03: **every
numeric quantity is a config field with a documented default and no hardcoded constant**, and every
claim gets a test that pins it.

---

## 0. Read first

**For A**
1. `Code_Change_Brief_02_Residual_Frozen_Base.md` section 3.0, which specified this seam. It was not
   implemented: there is no `OBJECTIVE_REGISTRY` and no `TrainConfig.objective` in the package.
2. Q20 in `Advisor_Questions.md`, which is why the seam exists and why nothing but the existing
   objective may be registered.
3. `Model_Derivation_BaseCase.md` sections 3.3, 3.4 and Stage 4: the two candidate objectives and
   why they weight the experts differently.

**For B**
4. The statsmodels documentation for `MarkovRegression` and `MarkovSwitching.fit`, specifically
   `start_params`, the `transformed` argument of `fit`, and `transform_params` /
   `untransform_params`. The whole diagnosis below rests on which parameter space a start lives in.
5. Hamilton, J. D. (1990), "Analysis of time series subject to changes in regime," *Journal of
   Econometrics* 45(1 to 2), 39 to 70. The EM estimator and the role of starting values.
6. Hamilton, J. D. (1994), *Time Series Analysis*, Chapter 22, and Kim, C.-J. and Nelson, C. R.
   (1999), *State-Space Models with Regime Switching*, Chapter 4, for the likelihood being
   multimodal and why multi-start is standard practice.

**For C**
7. `README.md`, the Stage B section (free-data panel, cache contract, survivorship warning).
8. `run_experiment.py`, the settings block, which drives everything.
9. Gu, S., Kelly, B. and Xiu, D. (2020), *Review of Financial Studies* 33(5), for the order of
   magnitude a real cross-sectional signal has. Any real-data result that looks dramatically better
   than theirs is a bug until proven otherwise.

---

## A. The objective registry seam

Implement brief 02 section 3.0 exactly:

- `OBJECTIVE_REGISTRY` in `losses.py`, and `TrainConfig.objective: str = "mixture_nll"`.
- Register **only** the existing mixture negative log likelihood, under `"mixture_nll"`.
- Route the call in `Trainer` (currently `return out, mixture_nll(log_w, log_lik)` in `train.py`)
  through the registry.
- An unknown key raises in `NECConfig.validate` with the registered keys listed.
- Write `objective` into every `TrialRegistry` entry, beside `correction_penalty_weight`, because
  the penalty weight is not comparable across objectives.

**Do not** add squared error, Huber, rank or IC objectives, and do not write any docstring arguing
for one. Q20 is open.

Test: the registry path and the direct call produce **bit-identical** losses and gradients on the
same batch, for both the memoryless and the stateful training paths.

---

## B. Starting values for the Hamilton fit

### B.1 What is wrong, measured 2026-09-25

The WORK_QUEUE note on `start_jitter` recorded a trade-off: small jitter sends every start to the
same basin, large jitter reaches distinct optima but most starts fail. The cause turns out to be
simpler than a trade-off, and the "distinct optima" were not what they looked like.

**Cause 1: the perturbation is in the wrong space and at the wrong scale.** `_jitter` adds
`N(0, start_jitter)` noise to `model.start_params`. In statsmodels that vector is in the
**constrained** (natural) parameterisation, and `fit` defaults to `transformed=True`, so the noise
lands directly on transition probabilities, regime means and regime variances. Those live on wildly
different scales for daily returns: statsmodels' own default start on a realistic series was

    p[0->0]=0.5, p[1->0]=0.5, const=(0, 3.8e-5), sigma2=(1.2e-5, 1.2e-4)

An absolute perturbation of 0.05 is a thousand times a daily mean and produces negative variances
and probabilities outside [0, 1]. Those starts raise and are counted as failures.

**Cause 2: every start is centred on an uninformed point.** The default start has transition
probabilities of 0.5, meaning no persistence at all, which is the opposite of what a regime model
of daily returns expects.

**Measured, 20 starts each, simulated daily-scale two-state series of 3,000 days:**

| Series | Scheme | Converged | Distinct optima | Best log-lik |
|---|---|---|---|---|
| well separated, sticky | current, jitter 0.05 | 9 / 20 | 1 | 9248.95 |
| well separated, sticky | current, jitter 0.20 | 6 / 20 | 1 | 9248.95 |
| well separated, sticky | current, jitter 0.50 | 4 / 20 | 2 | 9248.95 |
| well separated, sticky | informed + relative jitter | 20 / 20 | 1 | 9248.95 |
| weak separation | current, jitter 0.05 | 7 / 20 | 1 | 9508.75 |
| weak separation | informed + relative jitter | 20 / 20 | 4 | 9508.75 |
| mean switch only | current, jitter 0.05 | 7 / 20 | 1 | 9409.16 |
| mean switch only | informed + relative jitter | 20 / 20 | 5 | **9412.44** |
| mean switch only | informed grid, 27 centres | 27 / 27 | 2 | 9409.16 |

Three things follow.

1. **More than half the starts fail under the current scheme even at the smallest jitter**, on
   realistic scales.
2. **"Every start in the same basin" is not itself a defect.** On the well-separated series the
   likelihood has one dominant optimum and every valid start should find it. The second optimum
   the current scheme reached at jitter 0.5 was 577 nats *worse*, not a better one it had
   discovered.
3. **Where the likelihood is genuinely multimodal, the current scheme cannot see it**, because it
   only ever explores from one centre and loses most starts. In the mean-switch case the informed
   scheme found an optimum 3.3 nats better than anything the current scheme reached.

### B.2 The fix

Make the start-generation scheme a registry, `markov_gate.start_scheme`, with three entries:

- **`"default_jitter"`** — the current behaviour, kept exactly, for comparison and so the table
  above is reproducible. Not the default.
- **`"informed_grid"`** — deterministic, data-driven centres. For each combination of a
  volatility quantile `q`, a trailing window `w` and a persistence `p` from config-supplied tuples:
  compute trailing realised volatility over `w` days, split the training-block dates into `K`
  groups by volatility quantile, set each regime's mean and variance to that group's sample mean
  and variance, and set the transition matrix to `p` on the diagonal with `(1-p)/(K-1)` off it.
  **Use a trailing window, never a centred one**, so no future observation enters even at
  initialisation.
- **`"informed_jitter"`** — the informed centres above, each perturbed **in the unconstrained
  space**: `untransform_params`, add noise **scaled relative to each parameter's magnitude**,
  `transform_params` back. A start built this way is valid by construction.

Set the default to `"informed_jitter"`. The measurements in B.1 are the justification; record them
in the docstring. All grid values (`q`, `w`, `p`, relative scale, number of jittered draws per
centre) are config fields.

Build the start vector by `param_names`, never by position; the existing `_regime_moments` already
explains why positional indexing breaks when a switching flag changes.

### B.3 What to record per fit

Every start remains a logged trial. Add to `MarkovFit.metrics()`:

- `gate_n_failed` (raised) separately from `gate_n_nonconverged` (ran but did not converge);
- `gate_n_distinct_optima`, the number of distinct converged log-likelihoods after rounding to one
  nat;
- `gate_best_minus_second`, the gap between the best and second-best distinct optimum.

The last two are what actually answer "is the likelihood multimodal on this data", which the
current metrics cannot.

### B.4 Tests

1. Every start produced by `informed_grid` and `informed_jitter` is valid **before** fitting:
   variances positive, transition probabilities in [0, 1], rows summing to one.
2. On a simulated daily-scale series, `informed_jitter` converges at least 95 per cent of starts.
3. On the same series, the best log-likelihood from `informed_jitter` is at least that of
   `default_jitter`.
4. The informed centre uses only data at or before each date: perturbing the series after date `t`
   leaves the volatility split at `t` unchanged.
5. Seeded runs are reproducible.
6. `default_jitter` reproduces the pre-change behaviour exactly, so the comparison stays honest.

---

## C. First end-to-end run on real data

### C.1 Purpose, and the rule that governs it

This is an **integration run**, not an experiment. Every number so far has come from synthetic
data. The goal is to find out whether the whole pipeline behaves sensibly when the Hamilton gate
meets real returns, and how long it takes.

**Nothing from this run may be quoted.** The universe is survivorship biased and the source is not
CRSP. Label every table and figure in the output: *DIAGNOSTIC ONLY — free data, survivorship
biased, not CRSP.* And **do not change any setting in response to what the results show.** Record
observations; do not tune. The first plausible-looking number is exactly the moment a result
quietly becomes a claim.

### C.2 Setup

- Data: the existing offline cache, 2015 to 2024, with the Kenneth French factors and VIX already
  cached. Nothing downloaded.
- Market series for the gate: the market excess return, as registered.
- `K = 2`. **A placeholder for plumbing, not a decision**; Q18 is open. Say so in the output.
- Base on, `correction_mode=True`, `zero_init_head=True`, gate frozen, objective `"mixture_nll"`
  (the only one registered; again not a decision).
- Three walk-forward folds, one or two seeds, a modest step count. Enough to exercise every path,
  no more.

### C.3 Arms

1. **Base only.** The floor.
2. **Residual mixture, Hamilton gate** (`informed_jitter` starts).
3. **Residual mixture, frozen uniform gate.** The control: identical capacity, no regime
   information. Without it, any gain in arm 2 could be the extra parameters rather than the gate.

Share the fitted base across all three arms through the existing `BaseCache`.

### C.4 Report

Write `Smoke_Run_2026-09-25.md` in the Master Thesis folder, with figures in a subfolder beside it.
Tag every trial `smoke_2026_09` with `source` recorded.

**Per fold, the gate:**
- fitted regime means, variances and transition matrix, canonically ordered;
- expected durations;
- `n_failed`, `n_nonconverged`, `n_distinct_optima`, `best_minus_second`;
- a plot of the filtered probability of the high-variance regime over the fold's dates, with VIX
  overlaid. Whether the calm/turbulent split lines up with VIX is the single most informative
  sanity check available.

**Per arm, pooled and per fold:**
- mean rank IC and ICIR, the base's own values beside them, and the improvement over the base;
- out-of-sample negative log likelihood and its improvement over the base;
- the out-of-sample correction magnitude, mean and distribution of `sum_k pi_k r_k`;
- pairwise distance between the fitted corrections;
- live parameter count.

**Timing:** wall clock per fold, split into gate fit, base fit and expert training. This feeds the
compute budget in Q11.

**A short observations section** listing anything that looks wrong or surprising. No
interpretation of performance.

### C.5 Stop conditions

Stop and report, rather than work around, if any of these happen:

- the gate fails to converge on any fold;
- the fitted regimes are degenerate (a stationary probability below 0.02, or expected duration
  below two days);
- any arm's improvement over the base is larger than 0.05 in mean rank IC. At this signal level
  that is almost certainly leakage, and finding where is more important than finishing the run.

---

## Tests, summarised

A: bit-identical registry path. B: six tests in B.4. C: no new unit tests required; the run itself
is the test, and its report is the deliverable.

## Scope fence

Implement A, B and C only. Do not start the permutation test, the jump model, Wasserstein, TVTP,
the CRSP seam or the pre-registration. Do not register any objective other than `"mixture_nll"`.
Do not tune any hyperparameter in response to C's results.
