# Brief 05: full audit of the existing code and tests

Date: 2026-09-25. Repository `Quant Model/nec_baseline`, package `nec_moe`.

**Purpose.** Before any new component is built, establish that everything built so far is correct,
free of look-ahead, and covered by tests that would actually fail if it were wrong. This is an
**audit, not a development task.**

---

## 0. Ground rules

1. **Do not change production code** (`nec_moe/`, `run_experiment.py`, `scripts/`). Findings go in
   the report with a proposed fix. Fixes happen afterwards, once Tom has read the report.
2. **You may add tests** that expose a suspected bug. Mark each with `pytest.mark.xfail(strict=True)`
   and a reason that cites the finding ID, so the suite stays green while the bug stays pinned.
3. **Mutation checks happen only in a scratch git worktree** that is deleted afterwards. Nothing from
   a mutation is ever committed.
4. **No new features, no CRSP, no tuning.** The only re-run allowed is reproducing the existing smoke
   run exactly (section 7).
5. Where the code's behaviour and a brief disagree, report it. Do not decide which one is right.

## 1. Read first

- `Model_Derivation_BaseCase.md` — the maths the code is supposed to implement.
- Briefs 01 to 04 and `Code_State_2026-09-21.md` — what each part was meant to do.
- Q7, Q19 and Q20 in `Advisor_Questions.md` — the design decisions the code must respect.
- `Smoke_Run_2026-09-25.md` — the first real-data run and its observations.

## 2. Baseline, before reading any code

Run and record, at a clean commit:

- the full test suite: passed, failed, skipped, xfailed, total count, wall time, the ten slowest tests;
- ruff and mypy, with the counts;
- the commit hash.

If anything fails here, stop and report before auditing further.

## 3. What to check in every section

**(a) Correctness against the maths.** Does the code compute what `Model_Derivation_BaseCase.md` and
the briefs say? Check signs, normalisations, log versus probability domain, off-by-one indexing on
dates, broadcasting shapes.

**(b) Look-ahead.** Could any quantity at date `t` depend on data after `t`, or on test-block data at
fit time? This is the single most important check in the audit.

**(c) Configuration.** Is anything that should be a config field hardcoded? Are defaults documented?
Does `NECConfig.validate` reject invalid combinations?

**(d) Test quality.** For every test file covering the section:

- Does each test assert the property its name and docstring claim, or something weaker (only shapes,
  only "runs without error", comparing a quantity with itself)?
- **Would it fail if the code were wrong?** For each critical property in section 5, make the mutation
  listed there in the scratch worktree and record whether at least one test fails. A property whose
  mutation no test catches is untested, whatever the coverage says.
- Are tolerances meaningful, or loose enough to pass anything?
- Is synthetic data at a realistic signal-to-noise ratio, or only at an easy one?
- Is randomness seeded, so the test is deterministic?
- Is anything the code promises **not** tested at all?

Give every test file a verdict: **strong**, **adequate**, **weak** (passes but proves little), or
**misleading** (gives confidence it has not earned).

## 4. The sections, in order

1. **Data.** `market_data.py`, `universe.py`, `features.py`, `context_data.py`, `data.py`,
   `scripts/build_pit_panel.py`. Special attention: every feature at `t` uses data at or before `t`;
   rank normalisation is per date; the forward target is aligned to the right horizon; point-in-time
   membership really excludes names outside the index on each date.
2. **Model.**
   - 2a base: `base.py`, `networks.py`, `BaseCache` keying.
   - 2b experts: `experts.py`, zero initialisation, correction mode, `r` materialised separately.
   - 2c gate interface: `RegimePrior.fit`, `PrecomputedRegimePrior` and its refusal rules.
   - 2d Hamilton gate: `markov_gate.py`, the informed starts, canonical ordering, frozen-parameter
     application, filtered-only probabilities, multi-start logging.
   - 2e baseline gates: soft, hard, top-k, Gumbel, uniform, backprop HMM, `gate.py`, `encoder.py`.
   - 2f assembly: `likelihood.py`, `model.py`.
3. **Training.** `losses.py` (objective registry, mixture NLL, correction penalty, load balancing,
   decorrelation), `train.py` (freezing, sigma schedule, dead-parameter audit, live parameter count,
   sequence training, checkpoint and resume), `tuning.py`.
4. **Evaluation.** `evaluation.py` (walk-forward folds, purge, rank IC, ICIR, long-short, the base
   comparison, correction magnitude), `baselines.py`, `multiple_testing.py`, `registry.py`,
   `sweep.py`.
5. **Diagnostics.** `diagnostics.py`, `alignment.py`, `calibration.py`, `plots.py`.
6. **Orchestration.** `run_experiment.py`, config round-tripping through `to_dict` and `from_dict`,
   CI configuration.
7. **The smoke run and the smoke tests.** See section 7 below.

## 5. Critical properties and their mutations

Each of these, if broken, would invalidate a thesis result. For each, make the listed mutation in the
scratch worktree, run the suite, and record which test catches it, or that none does.

| # | Property | Mutation to try |
|---|---|---|
| P1 | Features use only data at or before `t` | shift one feature by `-1` day |
| P2 | Target is the forward return at the configured horizon | shift the target by one day |
| P3 | Purge equals the horizon | set the purge to 0 |
| P4 | Gate uses filtered, never smoothed, probabilities | substitute the smoothed series |
| P5 | Gate is fitted on the training block only | fit it on train plus test |
| P6 | Test-block gate uses the frozen parameters | re-fit on the extended series |
| P7 | Base is fitted on the training block only | fit it on the full panel |
| P8 | Base and gate are actually frozen during expert training | leave the base's `requires_grad` on |
| P9 | At initialisation `y_hat` equals `f0` exactly | initialise the expert head randomly |
| P10 | Gate weights sum to one where `f0`'s coefficient relies on it | scale the prior by 0.9 |
| P11 | Rank normalisation is per date | normalise across the whole panel |
| P12 | Tuning uses only the training block's tail | let tuning see the test block |
| P13 | Regimes are canonically ordered before storage | skip the reordering |
| P14 | The base is shared across arms | refit the base per arm |
| P15 | Every selection is logged with its multiplicity | drop the selection-event row |
| P16 | Mixture NLL is computed in the log domain | exponentiate densities then take the log |

Add any further critical property you find while reading, with its own mutation.

## 6. Known issues to confirm or refute

From the smoke run's observations and earlier briefs. For each, say whether it is real, and how
serious.

1. **The gate's filter skips the purge gap** (smoke observation 5). Confirm the size of the effect.
   This is a design decision for Tom, not a bug to fix.
2. **The registry double-counts gate starts across seeds** (smoke observation 13). If the deflated
   Sharpe or any correction reads this tag, the multiplicity is wrong.
3. **`log_sigma` is excluded from the live parameter count** (smoke observation 12). Correct or not?
4. **The encoder and gate head are vestigial on the frozen-gate path.** Are they still built,
   trained, or counted anywhere they should not be?
5. **The smoke report was generated at "commit 1b7561e plus uncommitted changes".** A result that
   cannot be tied to a commit cannot be reproduced. See section 7.

## 7. The smoke run and the smoke tests

**The smoke run** (`scripts/smoke_run_2026_09.py`, `Smoke_Run_2026-09-25.md`):

- Reproduce it exactly at the current clean commit and confirm the report's numbers match. If they do
  not, explain every difference.
- Check that the script does what the report says: three arms as specified, one base shared across
  them, the stop conditions in brief 04 section C.5 implemented as written (thresholds, pooled and per
  fold), the "diagnostic only" labelling on every table.
- Say whether each observation in the report is supported by what the script actually measures.

**The smoke tests** (`tests/test_smoke.py` and any other end-to-end tests):

- What does each actually exercise? A smoke test is only useful if it runs the real pipeline path
  that experiments use, not a simplified copy of it.
- Does it cover the current design (frozen Hamilton gate, frozen base, correction mode), or only the
  older jointly trained configuration?
- Recommend what an adequate smoke test for the current design should run and assert.

## 8. The report

Write `Code_Audit_2026-09-25.md` in the Master Thesis folder.

**Per section:** a one-paragraph verdict; a findings table; a test-quality table.

Findings table columns: ID (e.g. `D-3` for data, `G-2` for gate), severity, file and line,
description, evidence, proposed fix.

Severity:
- **Critical**: could invalidate a result (look-ahead, wrong maths, a frozen component that is not
  frozen, a wrong multiplicity feeding the corrections).
- **Major**: wrong behaviour on a non-default path, or a test that gives false confidence about a
  critical property.
- **Minor**: robustness, clarity, missing validation.
- **Note**: worth knowing, no action needed.

Test-quality table columns: file, what it claims to test, verdict, mutations caught, gaps.

**At the top of the report:** the baseline results from section 2; the P1 to P16 mutation table with
the catching test for each; a count of findings by severity; and a plain statement of whether it is
safe to build the next component on this code, and if not, what must be fixed first.

Commit the report and any `xfail` tests. Nothing else.
