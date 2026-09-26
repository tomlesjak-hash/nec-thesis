# Code audit 2026-09-25 (brief 05)

Audit of `Quant Model/nec_baseline` (package `nec_moe`, `run_experiment.py`, `scripts/`) at commit
`fd1a8ed`, carried out as `Code_Audit_Brief_05.md` specifies. **No production code was changed.**
Suspected bugs are pinned by tests marked `xfail(strict=True)` in
`nec_baseline/tests/test_audit_2026_09_25.py`. Mutations were made only in a scratch git worktree,
which was deleted afterwards.

Read first, as the brief requires: `Model_Derivation_BaseCase.md`, briefs 01 to 04,
`Code_State_2026-09-21.md`, Q7, Q19 and Q20, and `Smoke_Run_2026-09-25.md`.

---

## 0. Summary

### 0.1 Baseline (brief section 2), recorded before any code was read

| Item | Result |
|---|---|
| Commit | `fd1a8ed8029c46f0a83491b3f7dbc44f062dbc15` (code files clean; the only tracked change was `Advisor_Questions.md`, not code) |
| Test suite | **243 passed, 0 failed, 3 skipped, 0 xfailed**, 246 collected, 87.1 s (88 s wall) |
| Skipped | the three network tests, gated on `NEC_NETWORK_TESTS=1` (`test_stage_b.py:263`, `test_stage_c.py:214`, `test_universe.py:175`) |
| ruff 0.15.21 | 0 violations over `nec_moe/ tests/ scripts/ run_experiment.py` |
| mypy 2.3.0 | 0 errors in 30 source files (`nec_moe/`, `run_experiment.py`; CI does not type-check `scripts/` or `tests/`) |
| Environment | Python 3.14.4, torch 2.12.1, statsmodels 0.14.6, pytest 9.1.1, macOS arm64 |

Ten slowest tests: `test_informed_jitter_best_is_at_least_default_jitters` 7.73 s;
`test_hmm_transition_reflects_true_persistence` 6.51 s; `test_canonical_ordering_is_stable_across_seeds`
4.93 s; `test_hamilton_parameter_recovery` 4.85 s; `test_hmm_recovers_persistent_regime` 3.68 s;
`test_transition_matrix_orientation_is_row_stochastic` 3.48 s; `test_run_sweep_end_to_end` 3.37 s;
`test_informed_jitter_converges_at_least_95_percent` 3.35 s; `test_alignment_hmm_filtered_path` 3.01 s;
`test_markov_gate_runs_through_the_harness_and_is_date_level` 2.67 s.

The baseline passed, so the audit proceeded. After the audit: 243 passed, 3 skipped, **6 xfailed** (the
audit's pinned bugs), 0 failed.

### 0.2 Critical properties and their mutations (brief section 5)

| # | Property | Mutation made | Caught by | Tests failing |
|---|---|---|---|---|
| P1 | Features use only data at or before t | shift one feature (ret_5d) by -1 day (uses t+1) | `test_stage_b::test_no_lookahead` | 1 |
| P2 | Target is the forward return at the configured horizon | shift the target by one day (fwd return from t+1) | `test_residual_target::test_residual_target_is_zero_for_a_market_clone`, `test_residual_target::test_residual_target_removes_the_market_component`, `test_stage_b::test_feature_values_hand_checked` | 3 |
| P3 | Purge equals the horizon | purge set to 0 in walk_forward_folds | `test_evaluation::test_purge_removes_label_overlap`, `test_evaluation::test_walk_forward_folds_chronology_and_purge`, `test_markov_gate::test_fit_called_once_per_fold_on_training_block_only` (+3 more) | 6 |
| P4 | Gate uses filtered, never smoothed, probabilities | gate table built from smoothed, not filtered, probabilities | `test_markov_gate::test_applied_parameters_are_identical_to_the_training_fit`, `test_markov_gate::test_every_start_is_logged_as_a_trial`, `test_markov_gate::test_filtering_over_the_extended_series_leaves_training_dates_unchanged` (+7 more) | 10 |
| P5 | Gate is fitted on the training block only | gate fitted on train plus test (guard left in place) | `test_markov_gate::test_fit_called_once_per_fold_on_training_block_only` | 1 |
| P6 | Test-block gate uses the frozen parameters | test-block gate re-fitted on the extended series (in-code assert kept) | `test_markov_gate::test_applied_parameters_are_identical_to_the_training_fit`, `test_markov_gate::test_every_start_is_logged_as_a_trial`, `test_markov_gate::test_filtering_over_the_extended_series_leaves_training_dates_unchanged` (+6 more) | 9 |
| P6b | (P6 with the in-code assertion also removed) | as P6, and the in-code parameter-identity assert disabled | `test_markov_gate::test_frozen_apply_matches_a_manual_filter_run` | 1 |
| P7 | Base is fitted on the training block only | base fitted on the full panel instead of the training block | `test_run_reporting::test_pooled_base_ic_is_the_base_scored_over_the_same_dates` | 1 |
| P8 | Base and gate are frozen during expert training | base freeze() leaves requires_grad on (attach check kept) | `test_residual_base::test_base_and_gate_tensors_are_unchanged_by_training`, `test_residual_base::test_base_and_improvement_reach_the_registry`, `test_residual_base::test_base_validation_split_is_a_tail_of_training_only` (+21 more) | 24 |
| P8b | (P8 with attach_base's frozen check also removed) | as P8, and attach_base's frozen check removed (base trains) | `test_residual_base::test_base_and_gate_tensors_are_unchanged_by_training`, `test_residual_base::test_base_validation_split_is_a_tail_of_training_only`, `test_residual_base::test_dead_parameter_audit_sees_the_frozen_ends` (+2 more) | 5 |
| P9 | At initialisation y_hat equals f0 exactly | expert heads initialised randomly in correction mode | `test_residual_base::test_posterior_equals_prior_at_initialisation`, `test_residual_base::test_residual_mixture_recovers_the_regime_conditional_part`, `test_residual_base::test_zero_init_starts_exactly_at_the_base` | 3 |
| P10 | Gate weights sum to one where f0's coefficient relies on it | prior scaled by 0.9 in y_hat (after the normalisation guard) | `test_residual_base::test_every_prior_is_normalized_in_correction_mode[gumbel]`, `test_residual_base::test_every_prior_is_normalized_in_correction_mode[hard]`, `test_residual_base::test_every_prior_is_normalized_in_correction_mode[soft]` (+4 more) | 7 |
| P11 | Rank normalisation is per date | rank normalisation across the whole panel, not per date | `test_stage_b::test_cross_sectional_ranks` | 1 |
| P12 | Tuning uses only the training block's tail | tuning tail loses its purge (validation_tail purge forced to 0) | `test_tuning::test_validation_tail_is_the_purged_last_block` | 1 |
| P13 | Regimes are canonically ordered before storage | canonical regime reordering skipped (identity permutation) | **NONE: the mutation survives** | 0 |
| P14 | The base is shared across arms | base refitted per arm (cache never hits) | `test_residual_base::test_base_cache_shares_across_gate_arms_and_splits_on_config` | 1 |
| P15 | Every selection is logged with its multiplicity | selection-event row dropped from registry.best | `test_objective_seam::test_every_trial_row_records_objective_beside_penalty_weight`, `test_selection_inference::test_registry_best_records_selection_event`, `test_selection_inference::test_registry_to_corrections_workflow` (+1 more) | 4 |
| P16 | Mixture NLL is computed in the log domain | mixture NLL via exp-then-log instead of logsumexp | `test_loss_math::test_log_domain_stability` | 1 |
| X1 | Gate series is causal (Mkt-RF of date t, not t+1) | gate series uses the NEXT day's Mkt-RF (look-ahead in gate input) | `test_markov_gate::test_market_excess_return_is_french_mkt_rf_aligned_by_date` | 1 |
| X2 | Informed starts use trailing, never centred, windows | informed-start volatility window centred (future data at init) | `test_markov_starts::test_informed_split_uses_no_data_after_each_date[20]`, `test_markov_starts::test_informed_split_uses_no_data_after_each_date[60]` | 2 |
| X3 | The apply path never overwrites a fitted gate row | apply path may overwrite fitted rows of the gate table | **NONE: the mutation survives** | 0 |
| X4 | Rank IC is computed per date | rank IC pooled over all dates instead of per date | `test_evaluation::test_rank_ic_skips_degenerate_dates`, `test_residual_base::test_base_and_improvement_reach_the_registry`, `test_run_reporting::test_pooled_base_ic_is_the_base_scored_over_the_same_dates` (+2 more) | 5 |
| X5 | Frozen base stays in eval mode during training | frozen base left in train mode during expert training | `test_residual_base::test_frozen_base_stays_in_eval_mode_during_training` | 1 |
| X6 | log_sigma is frozen during its warm-up | log_sigma warm-up freeze disabled | `test_checkpoint::test_fit_resume_matches_uninterrupted`, `test_persistence_diagnostics::test_sigma_freeze_is_reported_as_frozen_not_dead` | 2 |
| X7 | Stateful test filter is warmed through the training block | stateful test-block filter not warmed through the training block | **NONE: the mutation survives** | 0 |
| X8 | freeze_gate freezes the encoder | freeze_gate leaves the encoder trainable | `test_residual_base::test_base_and_gate_tensors_are_unchanged_by_training`, `test_residual_base::test_dead_parameter_audit_sees_the_frozen_ends`, `test_residual_base::test_frozen_parameters_never_enter_the_optimizer` | 3 |
| X9 | Base-comparison NLL uses the mixture's own noise scale | base-comparison NLL uses a fixed sigma of 1 (metric definition) | **NONE: the mutation survives** | 0 |
| X10 | Training goes through the objective registry | objective registry routes to a different (scaled) objective | `test_objective_seam::test_registry_path_is_bit_identical_on_the_memoryless_path`, `test_objective_seam::test_registry_path_is_bit_identical_on_the_stateful_path` | 2 |

Each mutation was made in a scratch worktree at `fd1a8ed`. The full offline suite was run and the
failing tests recorded; then the file was restored and, at the end, the worktree was deleted.
Variants P6b and P8b also remove the in-code guard, to see whether the tests themselves pin the
property or only the guard does. X1 to X10 are further critical properties found while reading.

**What the table shows:**

- **P13 survives.** Removing the canonical regime reordering passes every test. The informed starts
  already assign regime 0 to the calmest volatility group, so the sort is a no-op in every existing
  test, including the two named for it (G-6).
- **P7 is not really caught.** Its single "catch" is a float coincidence in
  `test_pooled_base_ic_is_the_base_scored_over_the_same_dates` (E-10). No test pins that the base is
  fitted on the training block only (B-5).
- **P6b** is caught only by `test_frozen_apply_matches_a_manual_filter_run`. The test named for the
  property, `test_applied_parameters_are_identical_to_the_training_fit`, compares a quantity the apply
  path never writes.
- **P12 is only partly expressible as a mutation.** The mutation removes the tuning tail's purge.
  "Tuning sees the test block" cannot be written as a code mutation, because `tune()` never holds the
  test block: it trusts its caller (T-2).
- **P14** is caught only by an object-identity check. A per-arm refit gives a numerically identical
  base, except through B-1.
- **X3 survives, but is an equivalent mutant.** `extend_causal_table` already filters out fitted dates
  before `_write` is reached. With both guards removed (X3c), `test_causal_table_never_overwrites_a_fitted_row`
  fails, as it should.
- **X7 survives** (E-9).
- **X9 survives:** nothing pins the base-comparison NLL (E-1).
- Most properties are caught by only one to three tests. P1, P5, P11, P12, P14, P16 and X1 each hang
  on a single test.

### 0.3 Findings by severity

*Update 2026-09-26:*
- G-6 and B-5 are closed by new tests, and the break-it table was re-run (section 9).
- The five Critical findings (D-1, D-2, E-1, E-2, M-1) are then fixed in code and pinned by tests, and the table was run again on the fixed code: every fix-reversal mutation is caught (section 10).
- D-1 and D-2 still need the real panel rebuilt.
- One new Major finding, M-5, is added.
- Counts now: 5 Critical (all fixed), 12 Major (2 closed), 19 Minor, 13 Notes.

| Severity | Count | Findings |
|---|---|---|
| Critical | 5 | D-1, D-2, M-1, E-1, E-2 (all fixed 2026-09-26, section 10) |
| Major | 12 | B-1, B-5 (closed), X-1, G-2, G-3, G-6 (closed), E-3, O-1, O-3, S-2, S-3, M-5 (new) |
| Minor | 19 | D-3, D-4, B-2, B-3, B-4, X-2, G-4, G-8, M-2, M-3, T-1, T-2, E-4, E-5, E-9, E-10, O-2, O-4, O-5 |
| Note | 13 | D-5, D-6, D-7, G-1, G-5, M-4, T-3, E-6, E-7, E-8, O-6, O-7, S-1 |

**Six of the findings are pinned by xfail tests:** D-1, B-1, E-1, M-1, G-2 and G-3, in
`tests/test_audit_2026_09_25.py`.

### 0.4 Is it safe to build the next component on this code?

**Not yet. Build nothing that produces or interprets a number until the Critical findings are
fixed.**

*Update 2026-09-26:* items 1 and 2 below are fixed in code (section 10). Item 2 still needs the panel rebuilt, which requires re-downloading the price cache, and the smoke run needs re-running after that. Items 3 and 4 and the smoke test (S-3) are still open.

The structural core is sound, and mostly well tested:
- the `RegimePrior.fit` / `PrecomputedRegimePrior` interface;
- the purged walk-forward;
- the frozen base and frozen gate, with zero-initialised corrections that start exactly at the base;
- the log-domain objective;
- the causal Hamilton gate.

The next gate (jump model or Wasserstein) could be written against that interface today. But every
number the pipeline produces on real data is affected by at least one Critical finding. That covers
the inputs, the headline comparison metric and the significance tests. Anything built next could not be
evaluated honestly.

**Fix first, in this order:**

1. **E-1 and E-2**, which define the headline quantities:
   - the base-comparison NLL must use the same mixture density;
   - IC t-stats, p-values, BH claims and PSR/DSR need overlap-robust (Newey-West, lag `h-1`) standard
     errors or non-overlapping dates.
2. **D-1 and D-2**, the look-ahead in the panel's inputs: rank after the point-in-time filter, and
   compute dollar volume from unadjusted prices. Then rebuild the panel.
3. **B-1**, so that an arm's result no longer depends on the order the arms ran. Needed before any
   cross-arm comparison.
4. **O-1 and O-3**: the quick-mode purge, and asserting the purge against the target's horizon (this
   also closes G-4).
5. **The critical-property test gaps**: G-6 (canonical ordering) and B-5 (base fitted on the training
   block only), both closed 2026-09-26 (section 9), and a smoke test of the current design (S-3),
   still open.

**Then, before the paths they affect are used:**
- M-1 before the backprop-HMM baseline arm is run on the 5-day target;
- E-3 before any portfolio number is reported;
- G-2 before an autoregressive Hamilton gate;
- G-3 before relying on mid-fold resume.

**Decisions for Tom, not bugs:**
- X-1: whether the experts' hidden layers start identical or diversified. This decides what the
  uniform control measures.
- G-1: whether the gate's filter should include the purge gap.
- B-2: whether the base should train on its validation tail when there is no early stopping.
- G-5: whether to accept the two start-scheme deviations from brief 04.

---

## 1. Data

`market_data.py`, `universe.py`, `features.py`, `context_data.py`, `data.py`,
`scripts/build_pit_panel.py`.

**Verdict.** Every feature for a single ticker at date `t` uses only data up to `t`. Sequence windows
end at the row's own date. The target is exactly `log(C_{t+5}/C_t)`, and the point-in-time filter
removes the right rows. The tests pin all of these. Two look-ahead channels sit outside what those
tests can see.

**Rank normalisation (D-1).** Each date's ranks are computed over every candidate ticker, before the
point-in-time filter removes non-members. On average about 103 non-member names per date took part in
the ranking. Many of them are only in the candidate list because they joined the index later.

**Auto-adjusted prices (D-2).** yfinance auto-adjusted closes carry every later dividend and spin-off
back in time. So `dollar_vol_20d` at `t` embeds corporate actions from after `t`.

Both are properties of how the panel was built, not of the model. Both must be fixed, and the panel
rebuilt, before any number from it is interpreted. The synthetic generator is useful for plumbing
but tests at an easy signal-to-noise ratio, and has no forward horizon at all.

| ID | Severity | File and line | Description | Evidence | Proposed fix |
|---|---|---|---|---|---|
| D-1 | Critical, **fixed 2026-09-26** (section 10; panel rebuild pending) | `features.py:285-286`; `scripts/build_pit_panel.py:77-80`; `universe.py:259` | Snapshot features are rank-normalised per date over **every candidate ticker** (the 2015 to 2024 membership union). `filter_point_in_time` then only drops rows; it does not re-rank. A member's feature value on date `t` therefore depends on names that are not in the index that day, including names that join later and are in the candidate list only because they grew. Ordering among members is preserved per feature; spacing is not. | On all 48 sampled dates of `pit_panel_2015_2024.pt`, members' ranks deviate from their own-cross-section ranks by up to 0.073 (0.0728 on 2016-11-11). The build dropped 246,708 non-member rows over 2,390 dates. Pinned by `test_D1_point_in_time_ranks_use_only_that_dates_members` (xfail). | Filter to point-in-time membership **before** ranking: pass the membership mask into `build_panel`, or re-rank `x_snap` per date inside `filter_point_in_time`. Ranks of ranks among members equal member-only ranks exactly. Apply `min_names_per_date` after the filter too (D-4). Rebuild the panel. |
| D-2 | Critical, **fixed 2026-09-26** (section 10; panel rebuild pending) | `market_data.py:153-154`; `features.py:190` | `yfinance.download(..., auto_adjust=True)` back-adjusts OHLC for every dividend, split and spin-off up to the download date. Return-based features are unaffected, because a price ratio only involves adjustments inside its own window. `dollar_vol_20d = log mean(close x volume)` is a **level**, however, and carries the cumulative adjustment factor of all corporate actions after `t`. | Cached 2015-06-25 closes: XOM 52.49 (traded near 83.5), KO 28.38 (near 39.5), T 12.65 (near 35.8). These are factors of about 0.63, 0.72 and 0.35 from 2015 to 2024 events, so the log factor reaches about -1.0 on a feature whose cross-sectional spread is a few log units. The magnitude of predictive leakage is unmeasured and is probably small, but it is look-ahead. | Extend the cache contract to hold the raw close and the adjusted close (`auto_adjust=False`). Compute returns from adjusted prices and dollar volume from raw close x raw volume. For CRSP use `abs(PRC) x VOL`. Do not let a future CRSP loader write adjusted closes into the single `Close` column. |
| D-3 | Minor | `features.py:176-211` | Rolling windows and the forward target are computed over each ticker's **own rows**, not the market calendar. A ticker with missing days gets windows and an `h`-row target spanning more than `h` trading days. | 10 of 590 cached tickers miss market days inside their own range: cce 158, hot 55, col 28, har 12, teg 8, and five with 1. | Reindex each ticker to the market calendar before computing features. Missing days become NaN and are dropped by validity. |
| D-4 | Minor | `features.py:246-251` | `min_names_per_date` and the date codes are computed on the pre-filter candidate set, so after the point-in-time filter a date may hold fewer names than the threshold. Not observed on the current panel: the minimum is 402 names per date. | Code reading. | Apply the threshold after filtering (with D-1). |
| D-5 | Note | `context_data.py:146-159` | French factors are the current vintage. The library is revised with each release, so the gate series is revised Mkt-RF, not what was known in real time. | Code and source reading. | State it in the methodology. Record the factor file's "created by" line with each run. |
| D-6 | Note | `universe.py` | The Wikipedia reconstruction is best effort. Tickers are reused over time: `cce` shows a 158-day gap and may splice two companies under one entity code. | D-3 evidence. | Key entities on (ticker, date range), and on CRSP PERMNO later. |
| D-7 | Note | `data.py:310-345` | The synthetic generator makes `y` contemporaneous with `x` (no forward horizon). `x_seq` is i.i.d. per row, and noise sd 0.4 against beta scale 2 gives R-squared near 0.96. Tests built on it cannot exercise horizon-related look-ahead (M-1, O-1) and run at an SNR two orders of magnitude above the real one. | Code reading. | Add a generator mode with an `h`-day overlapping target and a realistic SNR, for leakage tests. |

| Test file | What it claims to test | Verdict | Mutations caught | Gaps |
|---|---|---|---|---|
| `test_stage_b.py` | Cache parsing, hand-checked features, per-ticker no look-ahead, window alignment, rank range, plumbing | adequate | P1, P2, P11 | P1 is caught by one test only, and only because the shifted feature is among those probed. No panel-level no-look-ahead test (ranking against membership, D-1). No calendar alignment (D-3). Cannot see adjusted prices (D-2). The harness test is plumbing only. |
| `test_universe.py` | Parsing, roll-back epochs, leave-and-rejoin, filter, coverage maths | strong | none | The interaction of the filter with ranking (D-1). |
| `test_residual_target.py` | Market-neutral target: beta recovery, trailing beta, clone gives zero, market removal | strong | P2 | None material. |
| `test_stage_c.py` (parsers) | VIX and French parsers, cache-first loaders, context join | strong | none | None material. |

---

## 2. Model

### 2a. Base: `base.py`, `networks.py`, `BaseCache` keying

**Verdict.** The base is an MLP on the snapshot, fitted on the block it is given, and returned frozen:
`requires_grad=False` and `eval()`. The cache key covers every `BaseConfig` field plus window, seed
and input width, and every one of those fields is tested.

The serious problem is a side effect. `fit_base` reseeds and consumes the **global** torch RNG, but
only on a cache miss. Expert dropout draws from that same RNG. So an arm's result depends on whether
the base had already been cached, which means on the order the arms ran (B-1). This does not bias the
comparison in expectation, but it makes an arm's number irreproducible except by rerunning the whole
sweep in the same order. The smoke run used expert dropout 0.05.

| ID | Severity | File and line | Description | Evidence | Proposed fix |
|---|---|---|---|---|---|
| B-1 | Major | `base.py:121` (called from `evaluation.py:_attach_base`) | `fit_base` calls `torch.manual_seed(seed)` and trains, consuming the global RNG, **only on a cache miss**. With expert dropout above 0, the same arm, seed and base give different results depending on whether the base was fitted in that run or reused. | `test_B1_arm_result_does_not_depend_on_base_cache_state` (xfail): fold NLL 2.53368 vs 2.53349 with an identical base. The difference vanishes at dropout 0 (checked). | Give `fit_base` a private `torch.Generator` for sampling and wrap it in `torch.random.fork_rng()` so the global state is untouched. Alternatively, reseed the global RNG from (seed, fold) immediately before expert training. |
| B-2 | Minor | `base.py:114` | The validation tail (`val_fraction=0.2`: the most recent 20% of training dates) is withheld from base training **even when `early_stopping_patience=None`**; it then only feeds a reported `val_loss`. The experts train on the full block, so on the tail they learn from out-of-sample base residuals. Derivation Stage 5 step 3 can be read either way; this is reported, not decided. | Code reading. | Decide and pre-register: without early stopping, either train on the whole block or document the withholding. |
| B-3 | Minor | `base.py:114` | The base's train/validation split has no purge. With patience set, the first `h` validation labels overlap the last training labels, which makes the early-stopping criterion optimistic. This stays within the training block; there is no test leakage. | Code reading. | Split with `validation_tail(..., purge_dates=h)`. |
| B-4 | Minor | `base.py:168-193` | The cache key has no panel identity: no target, feature set or date mapping. Two panels with equal window codes and width would share a base inside one cache. Every current caller builds one cache per panel, so this cannot happen today. | Code reading. | Add a panel fingerprint to the key. |
| B-5 | Major (test gap), **closed 2026-09-26** (section 9) | `evaluation.py:794` (`_attach_base(trainer, train, ...)`) | No test pins that the base is fitted on the **training block only** (critical property P7). Fitting it on the full panel, test block included, is caught by one test, and only by accident: a 1e-12 float comparison in `test_pooled_base_ic_is_the_base_scored_over_the_same_dates` that a different base happens to perturb (E-10). | P7 mutation; failure traced to a float32 rounding difference, not to the base's scope. | Add a spy test in the style of `test_fit_called_once_per_fold_on_training_block_only`: record the date range `fit_base` receives per fold and assert it is the fold's training dates. |

### 2b. Experts: `experts.py`, zero initialisation, correction mode, `r` materialised separately

**Verdict.** The core properties hold and are tested bitwise:
- Zero initialisation starts the model **exactly** at the base.
- `r` is the experts' raw output, captured before the base is added.
- Every between-expert statistic reads `r`.
- The warm-start is refused in the zero-init mode.

One real disagreement with the derivation remains (X-1).

| ID | Severity | File and line | Description | Evidence | Proposed fix |
|---|---|---|---|---|---|
| X-1 | Major (brief disagreement) | `experts.py:115-116` | Derivation §3.1 says all K experts start **identical**, so that "symmetry is broken by the gate rather than by initialisation". The code re-initialises each expert's hidden layers from a different seed (`seed + 7919*(k+1)`); only the zeroed heads are identical. Under the uniform-gate control the experts therefore still diverge, so the control is not "the same capacity with nothing to separate the experts". Which one is intended is Tom's decision. | Smoke run: RMS distance between `r_0` and `r_1` is 0.028 to 0.039 under the uniform gate (observation 8). | Add a config field (e.g. `experts.diversify_init`) and record the choice. Report both variants of the control if it matters. |
| X-2 | Minor | `experts.py:116`; `priors.py:193`; `model.py:83`; `evaluation.py:680`; `baselines.py:130`; `evaluation.py:145` | Numeric constants outside config: seed stride 7919; normalisation tolerances 1e-4 and 1e-5; clamps of 1e-12; the MLP baseline's weight decay 1e-4; `rank_ic_by_date(min_names=3)`. This goes against the brief-02 rule that every numeric quantity is a config field. | Code reading. | Promote them to config fields, or document each as a structural constant. |

### 2c. Gate interface: `RegimePrior.fit`, `PrecomputedRegimePrior` and its refusal rules

**Verdict.** The interface is sound:
- `fit` is called once per fold, on a training panel whose last date is asserted to precede the test
  block.
- The table refuses unfitted lookups and unseen dates, and never lets the apply path overwrite a
  fitted row.
- Rows must be normalised, and lookups carry no gradient.

What is missing is persistence. The table is a plain tensor attribute, so it is not in `state_dict`,
and a checkpoint cannot restore it (G-3).

| ID | Severity | File and line | Description | Evidence | Proposed fix |
|---|---|---|---|---|---|
| G-3 | Major (known, confirmed) | `evaluation.py:776-783`; `priors.py:180-188` | The resume branch of `walk_forward_evaluate` loads the trainer checkpoint but never re-fits the gate, and `PrecomputedRegimePrior`'s table is not in `state_dict`. A fold interrupted mid-fit therefore cannot resume under the Hamilton gate; `Trainer.load` of any precomputed-gate checkpoint yields an unfitted gate. | `test_G3_mid_fold_resume_with_a_precomputed_gate_matches_uninterrupted` (xfail): "forward before fit()". | Re-fit and re-apply the gate in the resume branch (the fit is deterministic), or persist the table through `get_extra_state`/`set_extra_state` or a buffer. |
| G-8 | Minor | `priors.py:184` | The table is a plain attribute, not a registered buffer, so `.to(device)` leaves it on the CPU. | Code reading. | Register it as a buffer or extra state (fixes G-3 too). |

### 2d. Hamilton gate: `markov_gate.py`

**Verdict.** Each of the following is enforced in code and tested:
- **Filtered, never smoothed:** a string test, backed by a numeric test against a manual filter.
- **Fitted on the training block only.**
- **Applied with frozen parameters:** element-wise identity is asserted in code, and a numeric test
  checks the served probabilities against a manual filter.
- **Row-stochastic orientation**, tested on an asymmetric chain.
- **Every start logged.**

The informed starts are valid by construction and causal (a perturbation test and its centred-window
counter-test), and the smoke run reproduces exactly at `fd1a8ed`.

Weaknesses:
- The autoregressive variant (`order > 0`), which is offered in config, cannot run at all (G-2).
- The canonical ordering by variance is enforced in code but untested: removing it passes every test, because the informed starts already put the calm regime first (G-6).
- The purge-gap skip is confirmed and small (G-1).

| ID | Severity | File and line | Description | Evidence | Proposed fix |
|---|---|---|---|---|---|
| G-1 | Note (known issue 1, confirmed) | `evaluation.py:571-573`; `markov_gate.py:689-710` | The frozen filter is applied over the training dates followed directly by the test dates, as brief 03 §3.2 specifies ("train plus test"). The five purge-gap market returns are not in the series, and one transition step bridges six trading days. This is not look-ahead: it uses less information, not more. | Real panel, same frozen fit, gap skipped vs contiguous span. Max abs dP(high) on test dates is 7.05e-3, 7.88e-3 and 4.33e-4 for folds 0, 1 and 2, always on the first test date, and above 1e-3 on at most 2 test dates per fold (smoke report §2, reproduced). | A design decision for Tom. If the contiguous span is preferred: apply over `dates[:test_max+1]`. That is causal and adds only the known purge-gap returns. |
| G-2 | Major | `markov_gate.py:577, 587-589` | With `order > 0` (MarkovAutoregression) statsmodels returns filtered probabilities for `nobs - order` dates, and `fit` writes them against all training dates, so the fit raises. The option is offered in `MarkovGateConfig.order` and `run_experiment.gate_order`, and is untested. | `test_G2_markov_autoregression_gate_fits` (xfail): "expected (300, 2), got (299, 2)". | Drop the first `order` dates from the fitted table, and from the lookup's information set. Add the AR case to the recovery tests. |
| G-4 | Minor (known) | `evaluation.py` (`_fit_gate`, `walk_forward_evaluate`) | The brief 03 §3 horizon-alignment assertion was never implemented: nothing ties the gate's filtered date to the target horizon or the purge (see O-3). | Code reading; brief 03 §3. | Parse the horizon from the target and assert `purge_dates >= horizon` in the harness. |
| G-5 | Note (brief disagreements) | `markov_gate.py:212-248, 448-459` | (a) Informed centres cut volatility with an **expanding** quantile, not a training-block quantile as brief 04 B.2 describes. This is stricter, and the docstring says why. (b) Distinct optima are clustered by **gap at most 1 nat**, not "rounding to one nat" as brief 04 B.3 says. This avoids splitting two values on either side of a rounding boundary. | Code and brief reading. | **Accepted 2026-09-26** (brief 06 E): both are the specification; the `markov_gate.py` docstrings now say so. No code change. |
| G-6 | Major (test gap), **closed 2026-09-26** (section 9) | `markov_gate.py:647-650` | The canonical regime reordering (critical property P13) is untested. The informed starts assign regime 0 to the lowest-volatility group, so statsmodels' raw labels already come out calm-first, and the permutation is the identity in every test. Removing the sort changes nothing any test observes. The code itself does reorder correctly. | P13 mutation (identity permutation): 0 of 243 tests fail. | Add a test that forces turbulent-first raw labels: fit from an explicit start vector with the two regimes' moments swapped, or apply `_canonical_order` to a result with permuted labels. Assert ascending variance, the matching permutation, and the permuted transition matrix. |

### 2e. Baseline gates: soft, hard, top-k, Gumbel, uniform, backprop HMM, `gate.py`, `encoder.py`

**Verdict.** The memoryless priors are correct and well tested:
- normalisation;
- the hard-EM gradient `pi - onehot`;
- top-k at least 2, with k = K reducing to soft;
- Gumbel frequencies and annealing;
- uniform is gradient-free.

The encoder is causal (no bidirectional option) and uses LayerNorm, not BatchNorm, with a
cross-sample-coupling test.

**The backprop-HMM baseline has look-ahead on any multi-day target (M-1).** Its prior at `t` is the
predict step from the posterior at `t-1`. That posterior was updated with `y_{t-1}`, which for a
5-day forward return is only realised at `t+4`. Brief 01 §0 and `Code_State_2026-09-21.md` §3 both
call this path "already correct". That holds only for a 1-day horizon. The primary Hamilton gate is
unaffected, because it filters the market return `r_t`, which is known at `t`.

| ID | Severity | File and line | Description | Evidence | Proposed fix |
|---|---|---|---|---|---|
| M-1 | Critical (for the HMM baseline arm), **fixed 2026-09-26** (section 10) | `priors.py:507-522`; `train.py:570-599, 735-759`; `evaluation.py:517-534`; `alignment.py:48-50`; `plots.py` (utilisation) | With an `h`-day forward target, the stateful prior at `t` conditions on `y_{t-1}`, realised at `t-1+h`: a look-ahead of `h-1` = 4 trading days on the real panel. It enters training, the test-block predictions (the filter is updated with each test date's `y`), and the HMM alignment and utilisation diagnostics. Nothing forbids `prior="hmm"` with the 5-day panel. | `test_M1_stateful_prior_ignores_targets_not_yet_realised` (xfail): the prior at `t` moves when `y_{t-1}` is perturbed, with target `fwd_ret_5d`. The HMM tests all run on synthetic data with no horizon. | Carry the horizon into the stateful path, and update the filter only with targets dated at or before `t-h` (a lagged update). Or refuse stateful priors when the horizon exceeds 1. Correct the brief 01 §0 and Code_State wording. |
| M-2 | Minor | `config.py:validate` | `freeze_gate=True` with a trainable memoryless prior (soft, hard, top-k, Gumbel) freezes a zero-initialised gate: exactly uniform for soft, a constant arg-max for hard, forever. The config is accepted silently, so a "frozen soft" arm is a uniform arm under another name. | Code reading; the brief 02 memory note. | Refuse `freeze_gate` unless the prior is precomputed or uniform. |
| M-3 | Minor | `config.py:validate`; `model.py:155-163` | `freeze_gate=True` with `input_mode="snapshot_plus_hidden"` feeds the experts a frozen, randomly initialised GRU state (random features). Accepted silently. | Code reading. | Refuse the combination, or warn. |
| M-4 | Note (known issue 4) | `model.py:112-113, 175-176`; `train.py:224-243` | The encoder and gate head are still built and run on every forward pass on the frozen path, which costs compute. They are frozen and outside the optimiser. They count in `total_param_count` but not `live_param_count`, and are listed under the audit's `frozen`. They are not counted anywhere they should not be. | Code reading; `test_frozen_parameters_never_enter_the_optimizer`. | Optionally skip the encoder when the prior is precomputed and `input_mode="snapshot"`. |
| M-5 | Major (found 2026-09-26) | `train.py` (`_lagged_context`, `evaluate_sequence`, `train_step_sequence`); `priors.py:507-530` | The backprop-HMM baseline threads each date's per-row posterior into the next date **by row position**. That assumes the same entities in the same order on every date, which a point-in-time panel does not have: membership changes. When a date's cross-section has a different size, the forward pass raises; when sizes happen to match but the names differ, it silently hands one stock's regime state to another. | A panel with one name dropped on one date: `ValueError: shape mismatch for 'prev_filtered': expected (3, 2), got (4, 2)`. | Key the recursion's state by entity (carry a per-entity table and look up each row's previous posterior; new entrants start from pi_0), or make the HMM regime date-level. Until then the HMM baseline arm cannot run on the point-in-time panel. |

### 2f. Assembly: `likelihood.py`, `model.py`

**Verdict.** The mixture assembly is correct:
- The log density is computed directly in the log domain.
- The base is added inside `forward` as `mu_k = f0 + r_k`, so the likelihood, the fused NLL and `y_hat`
  need no special case.
- `sum_k pi_k = 1` is asserted where `f0`'s coefficient depends on it, and tested for every
  memoryless prior.
- The posterior equals the prior at initialisation (tested).

No findings beyond X-2's hardcoded tolerance.

| Test file | What it claims to test | Verdict | Mutations caught | Gaps |
|---|---|---|---|---|
| `test_residual_base.py` | Zero init is exact; frozen tensors unchanged; optimiser exclusion; cache sharing and keying; penalty; recovery (brief 02 test 5); low-SNR sanity; registry fields | strong on freezing, zero init and cache; **misleading on recovery** | P3, P8, P8b, P9, P10, P14, X4, X5, X8 | "THE test" (recovery) and the low-SNR test run with `freeze_gate=False` and a **trainable soft gate**: the jointly trained configuration, not the frozen-gate design. The audit test expects sigmas to be counted but runs with `sigma_freeze_steps=0` (T-1). |
| `test_markov_gate.py` | Brief 03 tests 1 to 7: fit once on train, forward-before-fit, frozen params, no smoother, recovery, ordering, manual-filter agreement | adequate | P3, P4, P5, P6, P6b, X1 | `test_applied_parameters_are_identical_to_the_training_fit` compares `fit_result.params` with itself, since apply never writes it; the real protection is the in-code assertion plus test 7. `test_non_convergence_is_reported_not_silently_absorbed` passes on a start-validation `ValueError` and never reaches non-convergence. No AR case (G-2). Removing the canonical reordering (P13) passes every test, including `test_canonical_ordering_is_stable_across_seeds` and `test_transition_matrix_orientation_is_row_stochastic` (G-6). |
| `test_markov_starts.py` | Brief 04 B.4: start validity, at least 95% convergence, best at least legacy, causal split, reproducibility, verbatim legacy | strong | X2 | None material. |
| `test_gate_abstraction.py` | Shared likelihood core; HMM filter matches a reference to 1e-10; causality probe; row-stochastic; persistence recovery | strong for `h=1` | none | The causality probe perturbs a **future date** only. With a multi-day target, the previous date's `y` is itself future information (M-1); synthetic data has no horizon. |
| `test_hamilton.py` | Classical x HMM parameter recovery; the four-corner grid; moment warm-start | strong | none | Same horizon blind spot as above. |
| `test_routing_variants.py` | Every memoryless prior's contract | strong | P10 | None material. |
| `test_defects.py` | The prototype's ledger defects: gate gradient, all experts in loss, logits, no BatchNorm coupling, capacity, feature contract | strong | none | Runs on the jointly trained configuration, as designed. |

---

## 3. Training

`losses.py` (objective registry, mixture NLL, correction penalty, load balancing, decorrelation),
`train.py` (freezing, sigma schedule, dead-parameter audit, live parameter count, sequence training,
checkpoint and resume), `tuning.py`.

**Verdict.** The objective is right:
- The mixture NLL is one fused `logsumexp`, and the registry path is bit-identical to the direct call.
- The penalty acts on the combined correction.
- Load balancing is refused when the gate is frozen.

Freezing is right. The frozen modules are excluded from the optimiser by construction and put back in
`eval()` after every `model.train()`, and both are tested by comparing tensors.

Checkpoint and resume are bit-exact on the paths they cover, but those are only the soft
configuration (G-3 for the Hamilton gate). `live_param_count` depends on the sigma schedule (T-1).
Tuning protects the test block only by trusting its caller (T-2).

| ID | Severity | File and line | Description | Evidence | Proposed fix |
|---|---|---|---|---|---|
| T-1 | Minor (known issue 3, confirmed) | `train.py:283-297` | `live_param_count` is a snapshot at the first backward pass. With `sigma_freeze_steps > 0` (default 100), `log_sigma` is `frozen` at that instant and excluded, although it trains from step 100 onward. The derivation's trainable set is psi plus `{log s_k}`, so the count is short by K. Two tests encode opposite expectations and pass only because they use different freeze settings. | Smoke run: 6,146 = 2 x 3,073, with no sigmas. `test_dead_parameter_audit_sees_the_frozen_ends` counts sigmas (freeze 0); `test_sigma_freeze_is_reported_as_frozen_not_dead` (freeze 50). | Count parameters that are trainable at any point of the schedule (those currently frozen only by the sigma schedule count as live). Or audit after the freeze ends. |
| T-2 | Minor | `tuning.py:79-121`; `tests/test_tuning.py:83-85` | `tune()` has no guard that `train_panel` excludes the outer test block; the docstring relies on the caller. The test's "no contamination" assertion checks the fixture the test itself built, not the tuner. No production code calls `tune()` yet. | Code reading; P12 mutation (below). | Take `(panel, fold)` and slice the training block internally, asserting as `_fit_gate` does. |
| T-3 | Note | `base.py:123`; `train.py:181` | Weight decay is AdamW's decoupled decay, not the derivation's `lambda * ||phi||^2` L2 term. The two are equivalent for SGD, not for Adam. | Code and derivation reading. | State which is used in the methodology. |

| Test file | What it claims to test | Verdict | Mutations caught | Gaps |
|---|---|---|---|---|
| `test_loss_math.py` | NLL and responsibilities against a hand reference (1e-10); `pi - r` gate gradient; responsibility-weighted expert gradient; 40-sigma stability; gauge invariance; sigma stationarity; load-balance scope | strong | P16 | None material. |
| `test_objective_seam.py` | Registry path bit-identical in loss and gradients (both paths); only `mixture_nll` registered; unknown keys refused; objective recorded beside the penalty weight | strong | P15, X10 | None material. |
| `test_checkpoint.py` | Resume-exact `fit`, `fit_sequence`, walk-forward and sweep resume; CLI | strong for what it covers | X6 | Only the soft, no-base configuration. No residual-mode or precomputed-gate resume (G-3). |
| `test_tuning.py` | Purged validation tail; planted winner; every trial and one selection event logged | adequate (weak on the tuner's own scope) | P3, P12, P15, X4 | The contamination assertion is tautological (T-2). |
| `test_persistence_diagnostics.py` | Durations, stationary distribution, persistence logging; dead-parameter audit; the Wasserstein stub raises | strong | X6 | Its expectation for `log_sigma` contradicts `test_residual_base.py` (T-1). |

---

## 4. Evaluation

`evaluation.py`, `baselines.py`, `multiple_testing.py`, `registry.py`, `sweep.py`.

**Verdict.** The mechanics are right and well tested:
- The folds are chronological and purged. With purge = `h`, the last training label ends the day
  before the test block.
- Rank IC is per date and rank based.
- The long-short book and turnover maths are right.
- The registry is append-only and records selection events.
- The Benjamini–Hochberg, Bonferroni, PSR and DSR arithmetic is correct.

Two statistical errors sit on top of these mechanics, and both reach headline numbers:

- **E-1.** The "NLL improvement over the base" does not isolate the correction. The base is scored
  as one Gaussian while the mixture is a scale mixture of the experts' sigmas, so the "improvement"
  is non-zero even when every correction is exactly zero.
- **E-2.** Every p-value treats the daily ICs as independent, although consecutive 5-day targets
  overlap by 4 days. On the real panel the naive t-statistic is 1.6 to 1.8 times the
  autocorrelation-robust one. The same overlap makes the daily-formed long-short series (E-3)
  mis-scaled.

| ID | Severity | File and line | Description | Evidence | Proposed fix |
|---|---|---|---|---|---|
| E-1 | Critical, **fixed 2026-09-26** (section 10) | `evaluation.py:640-700` (sigma at 662) | `base_and_correction` scores the base as a **single Gaussian** at the prior-weighted sigma, while the mixture's NLL is a **scale mixture** of the experts' different sigmas. `nll_improvement` therefore includes the variance structure, not only the correction. Under the Hamilton gate the experts can lower NLL with regime-dependent sigmas alone. The docstring's claim that the two NLLs "differ only in the mean" is false; it also says "responsibility-weighted" where the code uses the prior. | `test_E1_nll_improvement_is_zero_when_every_correction_is_zero` (xfail): with every correction exactly 0 and sigmas (0.6, 2.4), the "improvement" is -0.038. It is 0 when the sigmas are equal. Mutation X9 (base NLL at sigma 1) survives every test: nothing pins the base NLL's definition. | Score the base with the **same** mixture density and corrections set to zero: `-logsumexp_k(log pi_k + log N(y; f0, s_k^2))`. If wanted, report the variance-only gain separately. Re-derive the smoke report's NLL columns. |
| E-2 | Critical, **fixed 2026-09-26** (section 10) | `evaluation.py:196-209`; `multiple_testing.py:77-163`; `sweep.py:177-183, 373-386` | `t_stat = ICIR * sqrt(T)`, `ic_pvalue`, `corrected_claims` (BH over arms) and PSR/DSR all assume independent periods. The target is a 5-day forward return, so consecutive daily ICs share 4 of 5 return days, and consecutive long-short returns overlap likewise. | Real panel, three single-feature predictors: lag-1 autocorrelation of the daily IC is 0.63, 0.78 and 0.78, dying out by lag 5. The iid t-stat is 1.62x, 1.79x and 1.81x the Newey-West (lag 4) t-stat. | Use HAC (Newey-West, lag `h-1`) standard errors in `ic_summary` and PSR/DSR, or evaluate on non-overlapping dates (every `h`-th date). Carry the horizon into the metric functions. |
| E-3 | Major | `evaluation.py:217-303` | The long-short book is re-formed **daily** on 5-day forward returns. Each date's gross is a 5-day return, while turnover and costs are charged per day, so the net return mixes horizons. The overlapping holding periods also autocorrelate the series, which overstates its IR and t-stat. | Code reading; E-2's measurement of the overlap. | Use Jegadeesh-Titman overlapping portfolios (1/`h` of the book re-formed per day, each held `h` days), or rebalance every `h` dates. Charge costs on the same horizon. |
| E-4 | Minor | `sweep.py:177-209` | The arm summary mixes pooled statistics (`mean_ic`, `icir`, `p` from the pooled IC) with fold means (`base_mean_ic`, `base_icir`, `ic_improvement`), so `icir` and `base_icir` are not comparable. `WalkForwardResult.pooled_base_ic` exists but is unused, and `nll` is an unweighted fold mean. | Code reading. | Use pooled numbers for both sides. |
| E-5 | Minor (known issue 2, confirmed) | `evaluation.py:576-614` (`_gate_report`) | Gate starts are logged once per (arm, seed) run, although the fit is seed-independent, so with S seeds the tag holds S times the distinct starts. No production code reads that tag yet: DSR inputs are supplied by the caller. It becomes Critical the moment the tag feeds a DSR or correction. Conceptually, gate starts are optimiser restarts of one estimator, not strategy candidates. | Smoke registry: 144 gate-start rows for 72 distinct starts. | Log starts once per (fold, gate-config fingerprint), under a family kept apart from strategy trials. Document which N feeds the DSR. |
| E-6 | Note | `sweep.py:17-26` | `n_trials(tag)` counts seeds as trials for the DSR, while `corrected_claims` treats seeds as replicates. | Code reading. | Decide and pre-register. |
| E-7 | Note | `evaluation.py:813-815` | `expert_order` (a sigma sort of the experts) is recorded per fold. Under a frozen precomputed gate, the pairing of expert k with regime k is fixed by the gate's own canonical order, so a sigma re-sort can contradict it. | Code reading. | Under precomputed gates, key per-regime statistics on `gate_permutation` only. |
| E-8 | Note | `registry.py:64-95` | `log()` re-reads the whole file to assign `trial_id`, which is O(n^2) over a sweep. Harmless at 160 rows; slow at tens of thousands. | Code reading. | Count lines once and cache. |
| E-9 | Minor (test gap) | `evaluation.py:527-529` | Nothing tests that the stateful test-block filter is warmed through the training block: starting it cold (from pi_0) passes every test (mutation X7). Losing the warm-up is not look-ahead, but the HMM baseline's test predictions would silently change. | X7 mutation: 0 tests fail. | Assert that the first test date's prior equals the predict step from the last training date's posterior. |
| E-10 | Minor | `tests/test_run_reporting.py:117` | `test_pooled_base_ic_is_the_base_scored_over_the_same_dates` compares a float32 pooled mean to a mean of fold means with `abs=1e-12`, which is tighter than float32 summation guarantees. It passes deterministically today, but a harmless reordering of a sum could break it, and it is the test that "caught" P7 by accident. | P7 diagnosis. | Use `abs=1e-6`, or compute both sides in float64. |

| Test file | What it claims to test | Verdict | Mutations caught | Gaps |
|---|---|---|---|---|
| `test_evaluation.py` | Chronology and purge; label overlap; IC maths; portfolio maths; memoryless and HMM harness | strong on maths, adequate on the harness | P3, X4 | No overlapping-target inference (E-2, E-3). Harness tests use the jointly trained configuration. |
| `test_selection_inference.py` | Registry round-trip; selection events; PSR, DSR, BH, Bonferroni maths; factor-zoo deflation | strong on arithmetic | P15 | Every series is i.i.d., so the independence assumption is never challenged (E-2). |
| `test_sweep.py` | One row per (arm, seed); planted separation; BH over arms; seeding | adequate | none | Soft and uniform arms only; no base or gate arms. |
| `test_baselines.py` | Ridge exact recovery and shrinkage; MLP deterministic; baselines blind where regimes are real, winning where absent | adequate | none | None material. |
| `test_run_reporting.py` | Timing split and excluded from equality; correction quantiles; pairwise distance; gate means and transitions; pooled base IC | adequate | P3, P4, P6, P7 (by accident, E-10), P8, P8b, X4 | Does not test what the NLL improvement means (E-1). |

---

## 5. Diagnostics

`diagnostics.py`, `alignment.py`, `calibration.py`, `plots.py`.

**Verdict.** `diagnostics.py` is correct and tested:
- durations `1/(1-p_kk)`;
- the stationary distribution solved by least squares, validated on 2- and 3-state chains;
- persistence metrics that are registry-safe, with absorbing states handled;
- pairwise distance as a metric;
- `wasserstein_template_tracking` raises instead of being a silent stub.

Calibration is scoped to softmax gates and validated by planted over-confidence.

The HMM paths in `alignment.py` and the utilisation plot use the filtered posterior, which on a
multi-day target includes future returns. That is M-1's look-ahead, reaching the diagnostics too.
`plots.py` is presentation only, and its tests only check that figures render.

| ID | Severity | File and line | Description | Evidence | Proposed fix |
|---|---|---|---|---|---|
| Q-1 | (see M-1) | `alignment.py:48-50`; `plots.py` | The HMM alignment and utilisation series use the posterior updated with `y_t`, a forward return. The gate-VIX alignment for the HMM arm is therefore look-ahead on a multi-day target. | Code reading; M-1. | Fixed with M-1. For diagnostics, use the prior (predict step) or a lagged posterior. |

| Test file | What it claims to test | Verdict | Mutations caught | Gaps |
|---|---|---|---|---|
| `test_persistence_diagnostics.py` | See section 3 | strong | see section 3 | — |
| `test_stage_c.py` (alignment) | Planted-truth alignment recovery; HMM filtered path; no overlap raises | adequate | none | No horizon. |
| `test_calibration.py` | Identical experts give ECE 0; planted over-confidence detected and repaired on the validation tail; honest gate gives T near 1 | strong | none | None material. |
| `test_plots.py` | Figures render | weak | none | Asserts axis counts and file sizes only. Acceptable for presentation code. |

---

## 6. Orchestration

`run_experiment.py`, config round-tripping through `to_dict` and `from_dict`, CI.

**Verdict.** The settings block reaches every `BaseConfig` and `MarkovGateConfig` field. The config
round-trips through JSON with its tuple fields intact; I verified this, but the existing tests
round-trip only through a dict. `NECConfig.validate` is thorough: unknown registry keys, inert load
balancing, `k_regimes != n_experts`, degenerate sort keys, and a markov gate with sequence ordering
are all refused.

Two gaps can leak labels:

- **O-1.** Quick mode, which is the default mode, splits train and test with no purge.
- **O-3.** The purge is taken from `Experiment.horizon` and never checked against the panel's own
  target horizon.

CI runs a different Python version with unpinned dependencies, and it has not yet run on the four
audit-era commits, which are local only.

| ID | Severity | File and line | Description | Evidence | Proposed fix |
|---|---|---|---|---|---|
| O-1 | Major | `run_experiment.py:404` | Quick mode (`mode="quick"` is the default in the settings block) splits with `Panel.split_by_date`, which has **no purge**. With a 5-day target, the last 5 training labels overlap the test period, and the gate and base are also fitted on that unpurged block. | Code reading: `purge` is passed to `_quick` but used only for calibration. | Split with `walk_forward_folds(n_folds=1, test_dates_per_fold=..., purge_dates=purge)`. |
| O-2 | Minor | `run_experiment.py:484-488` | Quick-mode temperature scaling is fitted on a "validation tail" of the training block that the model was trained on, so the calibration is in-sample. | Code reading. | Hold the tail out of training, as `tuning.py` does. |
| O-3 | Major | `run_experiment.py:261, 283`; `evaluation.py` (harness) | `purge = Experiment.horizon` is never checked against the panel's target (`schema.target == "fwd_ret_5d"`). Setting `horizon=1` on the 5-day panel silently purges 1 day. This is the missing brief 03 §3 assertion (G-4). | Code reading. | Parse the horizon from `schema.target`, or store it on the panel, and assert `purge_dates >= horizon` in `walk_forward_evaluate`. |
| O-4 | Minor | `run_experiment.py:560-562`; `baselines.py:128-131` | The MLP baseline arm gets the default `sigma_init=1.0` for a target with sd near 0.046, and a hardcoded weight decay 1e-4 that also decays `log_sigma`. | Code reading. | Pass the same auto sigma. Exclude `log_sigma` from decay. |
| O-5 | Minor | `.github/workflows/ci.yml` | CI uses Python 3.12 against 3.14 locally, with torch and statsmodels unpinned, although the Hamilton gate relies on statsmodels internals (`regime_transition` layout, `param_names`, `mle_retvals`). mypy skips `scripts/` and `tests/`. Local `main` is 4 commits ahead of `origin`, so CI has not run on them. | CI file; `git log`. | Pin versions (a lock file) and add 3.14 to the matrix. Type-check `scripts/`. Push. |
| O-6 | Note | `config.py:648-675` | The JSON round-trip of `NECConfig` with tuple fields is correct but untested; tests round-trip through a dict only. | Verified during the audit (equal after `json.dumps`/`loads`). | Add a JSON round-trip test. |
| O-7 | Note | `registry.py` | The brief 01 §7 registry-level `source` field is not implemented; the smoke script stores `source` in each row's config instead. | Code reading. | Part of the loader/CRSP seam item. |

| Test file | What it claims to test | Verdict | Mutations caught | Gaps |
|---|---|---|---|---|
| `test_run_experiment.py` | The control panel drives quick and evaluate modes end to end | weak for the current design | none | Runs the soft gate without a base. No markov, base, correction-mode or frozen-gate path through `_evaluate`. Synthetic data with purge 0, so O-1 and O-3 are invisible. |

---

## 7. The brief's known issues (section 6)

| # | Issue | Real? | How serious | Finding |
|---|---|---|---|---|
| 1 | The gate's filter skips the purge gap | Yes | Small: max abs dP(high) of 7.9e-3, on the first test date only, above 1e-3 on at most 2 test dates per fold. Not look-ahead. A design decision for Tom. | G-1 |
| 2 | The registry double-counts gate starts across seeds | Yes: 144 rows for 72 distinct starts | Minor today, because nothing reads the tag. It would be a wrong multiplicity the moment it fed a DSR or correction. Gate restarts should not share the strategy family's N at all. | E-5 |
| 3 | `log_sigma` excluded from the live parameter count | Yes, whenever `sigma_freeze_steps > 0` | Minor: short by K. **Not correct** against the derivation's trainable set, and the tests disagree with each other. | T-1 |
| 4 | Encoder and gate head vestigial on the frozen path | Built and run on every forward pass (compute only); frozen; outside the optimiser; in `total_param_count`, not `live_param_count` | No miscounting found. The one risky combination (`snapshot_plus_hidden` with a frozen gate) is M-3. | M-4, M-3 |
| 5 | The smoke report was generated at "1b7561e plus uncommitted changes" | Yes: the report's footer says so | **Resolved by reproduction.** Rerun at `fd1a8ed`, every number matches exactly (section 8). The footer should say `fd1a8ed`; not changed here, since the brief allows committing only this report and xfail tests. | S-1 |

---

## 8. The smoke run and the smoke tests

### 8.1 Reproduction

`scripts/smoke_run_2026_09.py` was rerun unchanged at clean commit `fd1a8ed`, with only its three
output paths redirected to scratch so that the committed report was not overwritten.

- `results.json`, excluding wall-clock timing and the environment line: **0 differences**, covering
  every gate metric, fold metric and pooled metric for both seeds and both arms.
- `trials.jsonl`: all **160 rows identical** apart from timestamps and timing.
- Report sections 1 to 3: identical apart from the output path and the gate-fit seconds (4.8/4.6/4.4
  vs 4.6/4.5/4.5).
- Provenance line: now reads `fd1a8ed`. The code at `1b7561e` plus uncommitted changes was exactly
  what `fd1a8ed` committed.

Exactness here depends on running the arms in the same order (B-1): a single arm rerun alone would
not reproduce its committed numbers.

### 8.2 Does the script do what the report says?

| Check | Result |
|---|---|
| Three arms as specified | Yes. Hamilton (`prior="markov"`, `informed_jitter`), frozen uniform, and base-only read off the shared base. |
| One base shared across arms | Yes. One `BaseCache`; the script checks that `base_ic` is identical across arms in every (seed, fold), and it held. |
| C.5 gate conditions | Implemented as written: stationary < 0.02 and duration < 2 days, checked on a pre-flight fit per fold before any expert trains, and the harness's fit is verified identical to the pre-flight fit. Non-convergence is read as "no start converged", and the report states that reading. |
| C.5 leakage condition | Implemented pooled and per fold with the 0.05 threshold. It is **one-sided** (signed improvement > 0.05), which matches the brief's wording; a large negative "improvement" would not stop the run. |
| "Diagnostic only" on every table and figure | Yes. Every generated table is prefixed by the label, every figure carries it as a footer, and so do the hand-written sections. |

### 8.3 Is each observation supported by what the script measures?

| Obs. | Supported? | Comment |
|---|---|---|
| 1 (no stop condition fired) | Yes | Thresholds and values as reported. |
| 2 (one optimum, stable fits) | Yes | |
| 3 (split lines up with VIX) | Yes | The correlations are measured; the episode list is read off the figures. |
| 4 (test blocks mostly calm) | Yes | |
| 5 (purge-gap skip) | Yes | Measured correctly; G-1. |
| 6 (corrections not small next to the base) | Yes as arithmetic | |
| 7 (common shift leaves rank IC unchanged but changes NLL) | **No** | The quantiles are pooled over each fold, not taken within each date, so "common to all names on a date" is not measured. The NLL side is confounded by E-1: part of the NLL change comes from the experts' sigmas, not the correction. |
| 8 (uniform-gate experts large and cancelling; fits Q20) | **Partly** | The cancellation arithmetic is supported. The attribution to posterior weighting is not measured, and the diversified initialisation (X-1) and the RNG order effect (B-1) are unexamined contributors. |
| 9 (base IC varies across seeds) | Yes | |
| 10 (training covers under a pass) | Yes | |
| 11 (timing) | Yes | |
| 12 (capacity of the control) | Yes, with T-1 | The count excludes `log_sigma`. "Brief 02 §5 reads it this way" overstates the brief, which does not mention the sigmas. |
| 13 (double count) | Yes | E-5. |
| 14 (data use) | Yes | |

**The NLL-improvement columns of the smoke report are not interpretable (E-1).** Its IC-improvement
t-statistics would be overstated if computed (E-2); the report computes none.

### 8.4 The smoke tests

| Test file | What it actually exercises | Verdict | Mutations caught | Gaps |
|---|---|---|---|---|
| `test_smoke.py` | The **jointly trained soft gate** with no base: warm-start, 400 steps, regime AUC above 0.8, NLL drop, and a broken-gradient guard. It is the real `Trainer` path, but for a configuration the thesis no longer uses. | misleading as a smoke test of the current design | none | No frozen Hamilton gate, frozen base or correction mode. Runs `split_by_date`, with no purge. |
| `test_run_experiment.py` | The control panel's quick and evaluate modes with the soft gate and HMM | weak for the current design | see section 6 | As above. |
| `test_run_reporting.py`; `test_markov_gate.py::test_markov_gate_runs_through_the_harness_and_is_date_level`; `test_residual_base.py` | Pieces of the current design through `walk_forward_evaluate` | adequate for those pieces | see above | None of them drives `run_experiment._evaluate` or `scripts/smoke_run_2026_09.py`, which is the path experiments take. |

| ID | Severity | File and line | Description | Evidence | Proposed fix |
|---|---|---|---|---|---|
| S-1 | Note | `Master Thesis/Smoke_Run_2026-09-25.md` (footer) | The footer names `1b7561e + uncommitted`; the reproduction shows the numbers belong to `fd1a8ed`. | Section 8.1. | Update the footer when the report is next edited. |
| S-2 | Major | the smoke report, section 3 and observations 7 and 8 | The report's NLL-improvement columns rest on E-1. Observation 7 is not measured by the script, and observation 8's mechanism is not measured. | Section 8.3. | After E-1 is fixed, regenerate the report's NLL columns (a rerun is allowed only once the fix lands) and reword observations 7 and 8. |
| S-3 | Major | `tests/test_smoke.py`; `tests/test_run_experiment.py` | No end-to-end test exercises the current design through the driver experiments use. Every "smoke" test passes on the old jointly trained configuration. | Section 8.4. | Add the smoke test described below. |

**What an adequate smoke test for the current design should run and assert.** Drive
`run_experiment.main` in evaluate mode on a small synthetic panel that has a real forward horizon
(say `h=5`, from the D-7 generator mode) and date labels, so that the gate series and the purge are
real. Use `prior="markov"`, `base_enabled`, `correction_mode`, `zero_init_head` and `freeze_gate`,
with 2 folds, 2 seeds, the uniform control arm and a few dozen steps. Assert:

1. `purge == horizon` and every fold's last training label ends before its test block.
2. The gate is fitted once per fold on the training block, the frozen parameters are unchanged in the
   test block, and the table rows equal a manual filter run.
3. At step 0 `y_hat == f0` bitwise, and the posterior equals the prior.
4. After training, the base, gate and encoder tensors are unchanged.
5. `base_ic` is identical across arms in every (seed, fold), and an arm's result does not depend on
   arm order (B-1).
6. `live_param_count` equals the expert parameters plus the `log_sigma`s.
7. The NLL improvement is exactly 0 when the corrections are forced to 0 (E-1).
8. Every registry row carries objective, penalty weight and source, and gate starts appear once per
   (fold, gate config).
9. The p-values use the horizon-aware standard error (E-2).

Also give the smoke script's `gate_preflight` and `consistency_checks` a unit test on synthetic data.


---

## 9. Re-run of the break-it table (2026-09-26): closing G-6 and B-5

**No fixes have been applied yet.** No production code has changed since this audit (`9a29368`), so
this re-run does not test any fix. It does two things:

1. It closes the two critical-property test gaps the audit found, which needed tests, not code
   changes, because the code already behaves correctly.
2. It re-runs every mutation to confirm that the new tests catch what they should and that nothing
   else changed.

Repeat this table once the Critical findings are fixed.

**New tests** (both pass on the current code):

- `tests/test_markov_gate.py::test_canonical_reordering_relabels_turbulent_first_raw_output` closes
  **G-6** (P13). A test-only start scheme (`monkeypatch` on `START_SCHEME_REGISTRY`) swaps one informed
  centre's regime moments, so statsmodels returns the **turbulent regime as raw label 0**. The test
  asserts that the raw permutation really was `(1, 0)`, so the sort had real work to do. It then checks
  that the stored moments, the transition matrix, the durations, the metrics and the
  filtered-probability table all come out calm-first. The table is compared against a manual filter
  with its columns permuted, and column 0 is checked against the true simulated calm path.
- `tests/test_residual_base.py::test_base_is_fitted_on_each_folds_training_block_only` closes **B-5**
  (P7). A spy on `nec_moe.base.fit_base` records the dates the base actually receives in each fold of
  `walk_forward_evaluate` and asserts they are exactly that fold's purged training dates: never a
  purge-gap date, never a test date, once per fold.

**Method.** A fresh scratch worktree at `9a29368` with the two new test files copied in. The same 28
mutations as section 0.2, plus three stricter variants written to make sure the new tests could not
pass by accident:
- **P13b:** the probability table is left unpermuted while the moments are still sorted.
- **P13c:** the transition matrix is left unpermuted.
- **P7b:** the base is fitted on the training block **plus the purge gap**, the subtlest version of
  the leak.

Unmutated baseline in the worktree: 245 passed, 3 skipped, 6 xfailed. The worktree was deleted
afterwards; nothing from a mutation reached the main checkout.

| # | Mutation | Tests failing, audit run | Tests failing, re-run | Tests newly catching it |
|---|---|---|---|---|
| P1 | shift one feature (ret_5d) by -1 day (uses t+1) | 1 | 1 | — |
| P2 | shift the target by one day (fwd return from t+1) | 3 | 3 | — |
| P3 | purge set to 0 in walk_forward_folds | 6 | **7** | `test_base_is_fitted_on_each_folds_training_block_only` |
| P4 | gate table built from smoothed, not filtered, probabilities | 10 | **11** | `test_canonical_reordering_relabels_turbulent_first_raw_output` |
| P5 | gate fitted on train plus test (guard left in place) | 1 | 1 | — |
| P6 | test-block gate re-fitted on the extended series (in-code assert kept) | 9 | 9 | — |
| P6b | as P6, and the in-code parameter-identity assert disabled | 1 | 1 | — |
| P7 | base fitted on the full panel instead of the training block | 1 | **2** | `test_base_is_fitted_on_each_folds_training_block_only` |
| P7b | base fitted on the training block PLUS the purge gap | new | **3** | `test_base_is_fitted_on_each_folds_training_block_only`, `test_pooled_base_ic_is_the_base_scored_over_the_same_dates`, `test_residual_mixture_recovers_the_regime_conditional_part` |
| P8 | base freeze() leaves requires_grad on (attach check kept) | 24 | **25** | `test_base_is_fitted_on_each_folds_training_block_only` |
| P8b | as P8, and attach_base's frozen check removed (base trains) | 5 | 5 | — |
| P9 | expert heads initialised randomly in correction mode | 3 | 3 | — |
| P10 | prior scaled by 0.9 in y_hat (after the normalisation guard) | 7 | 7 | — |
| P11 | rank normalisation across the whole panel, not per date | 1 | 1 | — |
| P12 | tuning tail loses its purge (validation_tail purge forced to 0) | 1 | 1 | — |
| P13 | canonical regime reordering skipped (identity permutation) | 0 | **1** | `test_canonical_reordering_relabels_turbulent_first_raw_output` |
| P13b | gate table columns left unpermuted (moments still sorted) | new | **1** | `test_canonical_reordering_relabels_turbulent_first_raw_output` |
| P13c | transition matrix left unpermuted (moments still sorted) | new | **1** | `test_canonical_reordering_relabels_turbulent_first_raw_output` |
| P14 | base refitted per arm (cache never hits) | 1 | **2** | `test_B1_arm_result_does_not_depend_on_base_cache_state` (the B-1 xfail flipping to a strict XPASS; see below) |
| P15 | selection-event row dropped from registry.best | 4 | 4 | — |
| P16 | mixture NLL via exp-then-log instead of logsumexp | 1 | 1 | — |
| X1 | gate series uses the NEXT day's Mkt-RF (look-ahead in gate input) | 1 | 1 | — |
| X2 | informed-start volatility window centred (future data at init) | 2 | 2 | — |
| X3 | apply path may overwrite fitted rows of the gate table | 0 | 0 | — |
| X4 | rank IC pooled over all dates instead of per date | 5 | 5 | — |
| X5 | frozen base left in train mode during expert training | 1 | 1 | — |
| X6 | log_sigma warm-up freeze disabled | 2 | 2 | — |
| X7 | stateful test-block filter not warmed through the training block | 0 | 0 | — |
| X8 | freeze_gate leaves the encoder trainable | 3 | 3 | — |
| X9 | base-comparison NLL uses a fixed sigma of 1 (metric definition) | 0 | 0 | — |
| X10 | objective registry routes to a different (scaled) objective | 2 | 2 | — |

**Reading the table:**

- **Both gaps are closed.**
  - P13 and its two variants are each caught by the new ordering test. In the audit run nothing
    caught P13.
  - P7 is caught by the new base-scope test as well as the accidental float check (E-10).
  - P7b, fitting on the purge gap, is caught by the new test and two others.
- **Nothing regressed.** No mutation lost a catching test, and every other row matches the audit run.
  The new tests also add catches to P3, P4 and P8.
- **P14's extra catch is not new protection.** Refitting the base in every arm reseeds the global RNG
  every time, which removes the arm-order effect that `test_B1_arm_result_does_not_depend_on_base_cache_state`
  pins. That xfail therefore passes, and strict mode reports the pass as a failure. The only direct
  protection of P14 is still the object-identity check.
- **X3, X7 and X9 still survive, as expected.**
  - X3 is an equivalent mutant (section 0.2).
  - X7 is E-9 and X9 is part of E-1, which await their fixes and tests.

**Status changes:** G-6 and B-5 are **closed** (Major, test gaps). The remaining findings,
including all five Critical ones, are open.


---

## 10. Critical fixes (2026-09-26)

All five Critical findings are fixed in code, each pinned by tests. Every test asserts the property,
not the implementation. The three audit xfails for fixed findings (D-1, E-1, M-1) now pass as plain
tests; B-1, G-2 and G-3 remain pinned as xfail.

**Suite:** 272 passed, 3 skipped (the network tests), 3 xfailed. ruff and mypy are clean.

| Finding | What changed | Tests that pin it |
|---|---|---|
| **E-1** base-comparison NLL | `NECModel.corrections_disabled()` forces every correction to exactly zero. `base_and_correction` scores the base as the same model through the same evaluation path, stateful warm-up included (`train=` is passed from the harness and quick mode). The base NLL is `-mean logsumexp_k(log pi_k + log N(y; f0, s_k^2))`, and `nll_improvement` is exactly zero when the corrections are. | `test_E1_nll_improvement_is_zero_when_every_correction_is_zero`, `test_nll_improvement_is_exactly_zero_before_any_training[uniform, hmm]`, `test_base_nll_is_the_mixture_density_at_the_base`, `test_corrections_disabled_is_a_scoped_switch` |
| **E-2** overlap-robust inference | `ic_summary` computes its t-stat from a HAC long-run variance over `h-1` lags, where `h` is read from the panel's target (`fwd_ret_5d` gives 4 lags). This flows through the harness, the baseline harness, `run_sweep`, `tune` and the control panel (`ic_hac_lags`, `ic_hac_kernel`), and sweep rows record `ic_hac_lags`. PSR and DSR take the same correction through an effective sample size. | `tests/test_overlap_inference.py` (11 tests), including a Monte Carlo size test |
| **D-1** point-in-time ranks | `FeatureSchema.rank_normalized` records that a panel's snapshot features are per-date ranks. `filter_point_in_time` then re-ranks the survivors per date among themselves, which equals ranking over the members alone, exactly. That fixes both the PIT script and the control panel's real-data path. | `test_D1_point_in_time_ranks_use_only_that_dates_members`, `test_filter_leaves_an_unranked_panel_untouched`, `test_build_panel_records_whether_it_rank_normalized` |
| **D-2** adjusted prices | yfinance is fetched with `auto_adjust=False` into a new cache file (`*.raw.csv`) holding the raw close (split-adjusted only) and `Adj Close`. Returns, drawdown and the target use the adjusted close; dollar volume uses raw close x volume, where the split factors cancel. Legacy auto-adjusted files are never read. | `test_features_at_t_ignore_dividends_paid_after_t` (which also shows the old contract failing), `test_yfinance_cache_holds_raw_and_adjusted_close` |
| **M-1** HMM look-ahead | `DataConfig.horizon` (explicit, else read from the target name, else 1). The HMM prior at `t` is the posterior from `h` dates back carried forward `h` predict steps. The trainer carries the last `h` posteriors between chunks and in checkpoints, and training, evaluation and the test-block warm-up all use the same lagged recursion. The HMM alignment and utilisation diagnostics report the prior, not the posterior (Q-1). `h = 1` is unchanged: every existing HMM reference test passes. | `test_M1_stateful_prior_ignores_targets_not_yet_realised`, `test_prior_is_the_posterior_h_dates_back_carried_h_steps`, `test_training_and_evaluation_use_the_same_lagged_recursion`, `test_hmm_alignment_series_is_the_prior_not_the_posterior`, `test_horizon_config_is_read_from_the_target_and_validated`, `test_fit_sequence_resume_is_exact_with_a_multi_day_target` |

**A refinement to the audit's own recommendation (E-2).** The audit proposed Newey-West at lag `h-1`.
Measured before choosing, on 2,000 simulated overlapping 5-period series per length, the rejection
rate of a nominal 5% test:

| T | i.i.d. | Newey-West, lag h-1 | Newey-West, lag 2(h-1) | Hansen-Hodrick, lag h-1 |
|---|---|---|---|---|
| 120 | 40.2% | 13.2% | 11.2% | 7.3% |
| 360 | 38.4% | 12.0% | 9.4% | 6.0% |
| 500 | 37.4% | 10.8% | 7.7% | 5.4% |

Bartlett weights shrink exactly the autocovariances the overlap creates. The default is therefore
**Hansen-Hodrick (uniform weights) at lag `h-1`**, which is exact for this structure. Bartlett stays
available, and is used automatically, and recorded on the summary, if the uniform estimate is ever
non-positive; that never happened in 6,000 draws. The audit's measured t-stat inflation (1.6 to
1.8x, against Newey-West) understates the true inflation somewhat.

**Break-it table on the fixed code.** A fresh scratch worktree held `9a29368` with the fixes overlaid,
and each mutated file was restored from a snapshot of its fixed version. It ran the 31 mutations of
section 9 plus ten **fix-reversal** mutations, each of which undoes one fix:

| # | Fix undone | Tests failing | Caught by |
|---|---|---|---|
| F1 | D-1 undone: no re-rank after the point-in-time filter | 1 | `test_D1_point_in_time_ranks_use_only_that_dates_members` |
| F2 | D-2 undone: dollar volume from the adjusted close | 1 | `test_features_at_t_ignore_dividends_paid_after_t` |
| F3 | D-2 undone: yfinance fetched auto-adjusted again | 1 | `test_yfinance_cache_holds_raw_and_adjusted_close` |
| F4 | E-1 undone: corrections_disabled has no effect | 2 | `test_base_nll_is_the_mixture_density_at_the_base`, `test_corrections_disabled_is_a_scoped_switch` |
| F5 | E-2 undone: harness default HAC lags forced to 0 | 2 | `test_harness_uses_horizon_minus_one_lags_by_default[fwd_ret_5d-None-4]`, `test_sweep_rows_record_the_lags_their_p_values_rest_on` |
| F6 | E-2 weakened: uniform kernel silently uses Bartlett weights | 3 | `test_a_non_positive_uniform_estimate_falls_back_to_bartlett_and_says_so`, `test_hac_t_stat_has_honest_size_on_overlapping_series`, `test_long_run_variance_is_the_kernel_formula[uniform]` |
| F7 | E-2 undone in PSR/DSR: effective sample size ignored | 1 | `test_psr_and_dsr_count_overlapping_periods_honestly` |
| F8 | M-1 undone: prior from the previous date's posterior, one step | 2 | `test_M1_stateful_prior_ignores_targets_not_yet_realised`, `test_prior_is_the_posterior_h_dates_back_carried_h_steps` |
| F9 | M-1 undone in diagnostics: HMM alignment reads the posterior | 1 | `test_hmm_alignment_series_is_the_prior_not_the_posterior` |
| F10 | M-1 weakened: multi-step predict collapsed to one step | 1 | `test_prior_is_the_posterior_h_dates_back_carried_h_steps` |

**Every fix reversal is caught.** Among the earlier mutations:
- None lost a catching test (lost: none).
- X9, the base-NLL definition, which no test caught before E-1, is now caught.
- Counts changed only upward: P2 3 → 4, P8 25 → 30, P9 3 → 6, P10 7 → 8, X9 0 → 1, X10 2 → 3.
- X3 (an equivalent mutant) and X7 (E-9, not a Critical finding) still survive, as expected.
- F8, the M-1 trainer reversal, is caught by the hand-written reference test and the look-ahead test,
  but not by the training-equals-evaluation test. That is by design: both paths share the reverted
  helper, and the reference test pins the rule itself.
- The worktree was deleted afterwards.

**What the fixes do not yet cover.**

- **The real panel must be rebuilt.** `data_cache/pit_panel_2015_2024.pt` was built with the old
  ranking and the auto-adjusted prices, so it still carries D-1 and D-2. The D-2 fix needs the raw
  closes, so the rebuild re-downloads the price cache: about 590 tickers plus SPY from yfinance. That
  has not been done without Tom's go-ahead.
- **The smoke run is stale.** `Smoke_Run_2026-09-25.md` predates all five fixes: its NLL columns
  (E-1), and its inputs (D-1, D-2). It should be re-run once the panel is rebuilt, and its footer
  updated (S-1).
- **New finding M-5 (Major).** The backprop-HMM baseline threads its state by row position, so it
  cannot run on a panel whose membership changes (section 2e).
- **Other findings.** Every other open finding is as before, including the Majors B-1, X-1, E-3,
  O-1, O-3, S-2, S-3, G-2 and G-3, and D-4, whose `min_names_per_date` is still counted before the
  filter.

---

## 11. Brief 06 (2026-09-26): CRSP replaces the free data, three small decisions

Implemented in one commit per section, not pushed: step 0 (9c10945 fixes, 87216a2 documents), A
(2552a32), B (494c21e), C (f44f573), D (3cef208), E (6dd3276), F (d5c78d6), G (this section).

**Suite:** 309 passed, 1 skipped (a network test), 3 xfailed (B-1, G-2, G-3). ruff and mypy are clean.
The three tests that read the licensed CRSP files (`crsp_data`) run here and skip without `Data/`.

**Decisions recorded** (Tom, 2026-09-26):

1. **Membership INDNO 1000500.** The brief's guess, `1000502` ("S&P 500 Composite", family
   `1100502`), has 0 rows in `StkIndMembership`: it is S&P's index level series. `1000500` ("CRSP
   Index of the S&P 500 Universe", family `1100500`) has 2,084 spells over 1,956 PERMNOs; `1000501`
   has identical spells. Members per date lie between 502 and 508 over all 2,516 trading days of
   2015-2024; the excess over 500 is companies with two share classes (7 in the window, kept as two
   PERMNOs). Bounds are inclusive: the per-date count changes 70 times under inclusive bounds and
   276 times if either bound is exclusive, and 202 of 239 spell starts abut an end on the previous
   trading day. Erratum added to brief 06 A.3; evidence in the `CRSPSpec` docstring.
2. **Delisting rule (a).** CIZ `DlyRet` already includes the delisting return: `MetaSIZtoCIZ` maps
   legacy `DLRET` to both `DelRet` and `DlyRet`, and on the real data all 5,518 delistings dated
   2015-2024 have a stock row on `DelDlyDt` (flagged `DlyDelFlg = Y`) whose `DlyRet` equals
   `DelRet` in all 5,376 cases with a value (largest difference 1e-16); the other 142 are missing in
   both under the same codes. Nothing is compounded in. The delisting row is kept as the stock's
   final return and is never a panel row.
3. **`post_delisting_return` switch.** A forward window that runs past a stock's final CRSP return
   is completed with a post-delisting return, `"cash"` (0) or `"market"` (the `market_indno`
   return), default `"cash"`, which is not a decision. Without it the row would be dropped because
   of a future event (look-ahead selection) and the delisting return would reach only one date. The
   residual target is computed on the completed window. It touched 407 rows of the 2015-2024 panel.
   The 2 member delistings without a delisting return (DelActionType/DelReasonType GDR/FING) keep
   missing targets; nothing is imputed. The setting is recorded on every registry row.

**Closed by this brief:**

| Item | What changed | Pinned by |
|---|---|---|
| **E-1 follow-up** | `NLL_single` (the base as one Gaussian, the pre-fix definition at 9a29368), `NLL_base` and `NLL_full`, reported as `nll_variance_gain`, `nll_improvement` and `nll_total_gain` in the harness, sweep rows, quick mode and the smoke script. The variance gain is attributed to the experts' noise scales under the gate; neither gain is ranked | `test_total_gain_is_variance_gain_plus_improvement`, `test_variance_gain_is_zero_when_every_sigma_is_equal`, `test_total_gain_is_the_variance_gain_when_every_correction_is_zero`, `test_nll_single_is_the_pre_fix_base_nll`, `test_sweep_rows_carry_all_three_nll_quantities` |
| **X-1** (switch only) | `ExpertConfig.hidden_init`: `"diversified"` (default, bit-identical to before) or `"identical"`. Which one the thesis uses is **not decided**. Recorded on every registry row. Checked: for one seed the experts' start is bit-identical across gate arms (uniform, soft, hmm, markov); the draws use a private generator, so B-1's global reseed does not reach them. Reported: dropout masks are drawn independently per expert, so identical experts diverge under a uniform gate unless dropout is 0 | `tests/test_expert_init.py` (9 tests) |
| **G-5** | Accepted: the expanding volatility quantile and the 1-nat gap clustering are the specification; docstrings say so. No code change | table row above |
| §10 "real panel must be rebuilt" | The CRSP panel is built from scratch with the D-1 re-rank; the free-data panel is obsolete and unused | `test_crsp.py` |
| §10 "smoke run is stale" | Superseded by the CRSP integration run (`Smoke_Run_2026-09-26_CRSP.md`): gate converged on every fold, no stop condition; no out-of-sample number reported | the run |
| WORK_QUEUE 7 provenance | Every registry row carries `data_source`, `post_delisting_return` and `hidden_init`; `TrialRegistry.log` refuses a row without them | `test_registry_refuses_a_row_without_provenance`, `test_data_source_and_fill_reach_every_registry_row` |

**Changed along the way.** The free loaders went with section B, and so did the tests of their
contract, including the two D-2 tests (`test_features_at_t_ignore_dividends_paid_after_t`,
`test_yfinance_cache_holds_raw_and_adjusted_close`); the break-it rows F2 and F3 no longer apply.
D-2's property holds on CRSP by construction (dollar volume is `|DlyPrc| x DlyVol`, no adjusted
price exists) and is pinned by `test_dollar_volume_is_raw_price_times_raw_volume`. The feature layer
now works from daily returns: the drawdown compounds returns inside its window, and share volume
uses split factors inside the window only (`test_split_leaves_the_volume_zscore_unchanged`).

**Observed, not changed:**

- The Hamilton gate's registered series `market_excess_return` (brief 03 §2) is French daily
  Mkt-RF from `context_data.py`, so the French factors are not "diagnostics only, never a training
  signal" as brief 06 B puts it: that series is the gate's input. The docstrings now say so. The
  integration run kept the series, since F allows only the data source to change.
- Under rule A.4.1 a single missing return invalidates the stock's rows for up to 120 trading days
  (`mom_120d` needs every return in its window). On the panel this costs little: 99% or more of
  each year's members have rows.

**Not closed** (scope fence): B-1 and every other open finding stay open, including the Majors
E-3, O-1, O-3, S-2, S-3, G-2, G-3 and M-5, and D-4 (`min_names_per_date` still counted before the
point-in-time filter). X-1 has its switch; the choice between the two starts is still Tom's. B-2 and Q16 were not touched; G-1 is now Q22 and the filter still skips the
purge gap; Q21 (frequency and horizon) and Q20 (the objective) are unanswered; Compustat is not used.
