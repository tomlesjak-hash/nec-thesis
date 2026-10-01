# Code change brief 07: open bugs, a run store, and resume you can rely on

Date: 2026-09-30. Three parts, done in order: A (bugs), then B (run store), then C (resume). C builds
on B. Each lettered section ends in its own commit.

Repository `Quant Model/nec_baseline`, package `nec_moe`; the git root is `Quant Model/`. Same rules as
briefs 02 to 06: **every numeric quantity is a config field with a documented default and no
hardcoded constant**, every claim gets a test that pins it, and the brief contains only decisions Tom
has made. Where something is still open it says so, and the code must not pick an answer.

---

## Licence rules (unchanged from brief 06)

- CRSP and anything derived from it stays inside `Quant Model/Data/`. That includes every
  **per-security** output of a run: predictions, corrections, per-stock residuals, per-stock
  responsibilities. Those go to `Data/derived/runs/<run_id>/`, never to `results/`.
- `results/` holds **aggregate** outputs only: settings, per-fold and pooled summary statistics,
  date-level series that contain no per-security values, figures, logs.
- Test fixtures use invented numbers. Tests that need `Data/` skip without it.
- No out-of-sample performance number on the real panel is printed, reported or read in this brief.
  All testing uses synthetic panels or the invented CIZ fixture.

---

## 0. Read first

1. `Code_Audit_2026-09-25.md`: findings B-1 (section 2a), G-2 and G-3 (2d), M-5 (2e), E-3 (4), O-1 and
   O-3 (6), S-3 (8.4), and sections 10 and 11.
2. `run_experiment.py`, `nec_moe/evaluation.py` (`walk_forward_evaluate`, its `resume_dir` path),
   `nec_moe/sweep.py` (`run_sweep`, its `resume_dir`), `nec_moe/train.py` (checkpoint save and load),
   `nec_moe/base.py` (`fit_base`, `BaseCache`), `nec_moe/priors.py` (`PrecomputedRegimePrior`,
   `HMMRegimePrior`), `nec_moe/registry.py`.
3. Jegadeesh, N. and Titman, S. (1993), "Returns to buying winners and selling losers", *Journal of
   Finance* 48(1), the overlapping-portfolio construction, for E-3.

---

## A. Fix the open bugs

Remove each finding's `xfail` marker where one exists; the test must then pass as a plain test.

**A.1 O-1, quick mode has no purge.** `mode="quick"` splits with `Panel.split_by_date`, which has no
purge gap. Give the quick split the same purge as the walk-forward harness (the target horizon, see
A.2). The gate and the base are fitted on the purged training block only. Test: with a 5-day target,
no training row's forward window overlaps the test block, in quick mode.

**A.2 O-3, the purge is never checked against the target.** Read the horizon from the panel's target
(`DataConfig.horizon_periods`, already used by E-2) and use it as the purge. If `Experiment.horizon`
is set and disagrees with the panel's target, raise with both values in the message. Test: a 5-day
panel with `horizon=1` raises.

**A.3 E-3, the long-short book mixes horizons.** Each date's gross is an h-day forward return while
turnover and costs are charged per day, and the overlapping holding periods autocorrelate the series.
Make the book horizon-consistent. Add a config field in the evaluation settings (and the `Experiment` block),
`portfolio_scheme: "nonoverlapping" | "staggered"`:

- `"nonoverlapping"`: form a portfolio every h dates, hold it h dates, charge turnover once per
  rebalance. Returns are h-day and non-overlapping.
- `"staggered"`: the Jegadeesh-Titman construction, h overlapping cohorts each held h days, each
  day's return the equal-weighted average of the cohorts' **daily** returns, turnover charged daily
  on the combined book. This needs each stock's daily forward returns; if the panel doesn't carry
  them, add them as a non-feature column built under the same timing rule as the target.

The default is `"nonoverlapping"`, and the docstring says the default is not a decision. Record the
scheme on every registry row. Tests: with h = 1 both schemes equal the current daily book; with
h > 1, on a synthetic panel with known returns, each scheme reproduces a hand-computed book, and costs
are charged at the stated frequency.

**A.4 B-1, results depend on arm order.** `fit_base` seeds and consumes the **global** RNG only on a
cache miss, so expert dropout differs between a run that fitted the base and one that reused it. Give
the base its own `torch.Generator` (seeded from the run seed plus `BaseConfig.seed_offset`) and never
touch the global RNG in `fit_base`; seed the expert stage explicitly at its start. The existing
B-1 xfail test must pass.

**A.5 G-2, the autoregressive Hamilton gate fails.** With `order > 0`, statsmodels returns filtered
probabilities for `nobs - order` dates. Align them to `dates[order:]`. The first `order` training
dates have no filtered probability: exclude them from expert training and report how many were
excluded per fold. On the test block, run the filter over a contiguous span that starts `order` dates
before the first test date, so every test date has a filtered probability. The G-2 xfail test must pass.

**A.6 M-5, the backprop-HMM baseline breaks on a changing cross-section.** It threads each date's
per-row posterior to the next date by row position. Key the carried state by **entity** (PERMNO):
keep a per-entity table, look up each row's previous posterior by its entity, start new entrants from
`pi_0`, and drop exits. This preserves the baseline's current definition; making the regime
date-level instead would change the model, which is not decided. Tests: dropping one name on one date
runs, and every surviving entity's state equals what it would be without the drop; permuting the row
order within dates leaves every output unchanged.

**A.7 S-3, no end-to-end test of the current design.** Add a test that drives `run_experiment.main`
with a synthetic sticky-regime panel in the current design: `base_enabled=True`,
`correction_mode=True`, `prior="markov"`, `freeze_gate=True`, in both `quick` and `evaluate` mode,
with two seeds. Assert that it completes, that every registry row carries the provenance fields, that
`nll_improvement` is exactly 0 before expert training, and that the outputs land where section B says.
Mark it slow if needed; it must run in CI.

---

## B. The run store

Today every run writes to `results/<tag>/`, with no index, no status, and no separation between
aggregate and per-security outputs. Build one storage architecture that every entry point uses:
`run_experiment.py`, `run_sweep`, `walk_forward_evaluate`, and the scripts.

### B.1 Layout

```
nec_baseline/results/
  INDEX.csv                      one row per run, appended at launch, updated at exit
  <campaign>/                    a folder per campaign, named by the user (e.g. "dev", "gates_k2")
    <run_id>/                    run_id = YYYYMMDD-HHMMSS_<tag>_<8-char settings hash>
      run.json                   identity: run_id, campaign, tag, purpose, mode, created, git commit,
                                 dirty flag, data_source, data fingerprint, package versions, device
      settings.json              the full Experiment snapshot (as today)
      status.json                state, current arm/seed/fold/step, last heartbeat, exit reason
      trials.jsonl               the trial registry (as today)
      metrics/                   aggregate results: per-fold and pooled summaries (json and csv)
      figures/
      logs/run.log               everything printed, timestamped
      checkpoints/               resume state (section C); gitignored
      SUMMARY.md                 written at exit: settings diff against defaults, status, metrics table
Data/derived/runs/<run_id>/      per-security outputs (licensed; gitignored with Data/)
```

- **`purpose`** is a free-text label the user sets in the settings block (for example "smoke",
  "debug", "gate comparison"). It carries no rule.
- **`touches_test`** (boolean in `run.json` and `INDEX.csv`) is true for any run that computed a
  test-block metric on a real panel. It feeds the adaptive-overfitting accounting of Q15. It is
  recorded, never enforced.
- **Data fingerprint**: a hash of the panel's shape, date range, entity set and target column
  checksum, plus the CRSP release. Recorded in `run.json`; used by section C to refuse a resume on
  changed data.

### B.2 Git

Add to `.gitignore`: `nec_baseline/results/**/checkpoints/` and `nec_baseline/results/**/logs/`.
Everything else under `results/` stays trackable (aggregate only, by the licence rules). The existing
`results/smoke_2026_09/` and `results/pit_coverage_2015_2024.csv` move into
`results/legacy/` with an `INDEX.csv` row each; nothing is deleted.

### B.3 A small command-line tool

`scripts/runs.py` with:

- `list [--campaign C] [--status S]`: the index as a table.
- `show <run_id>`: `SUMMARY.md` plus status.
- `diff <run_id> <run_id>`: settings differences between two runs.
- `resume <run_id>`: section C.
- `latest [--campaign C] [--status interrupted]`: the most recent matching run id.

No command deletes anything.

### B.4 Tests

The layout is created exactly as specified; `INDEX.csv` gets one row per run and is updated at exit;
per-security outputs never appear under `results/` (scan the files a synthetic run writes); the
settings hash is stable across processes and changes when any setting changes; `touches_test` is
false for synthetic runs.

---

## C. Resume after an interruption

Resume already exists at three levels: `run_sweep` skips completed (arm, seed) runs,
`walk_forward_evaluate` skips completed folds, and `Trainer` checkpoints mid-fit every
`checkpoint_every` steps. Make it complete and robust.

**C.1 Fix G-3: resume under a fitted gate.** A resumed fold never re-fits the gate, and
`PrecomputedRegimePrior`'s table is not in `state_dict`, so the resumed forward pass raises. Persist
each fold's fitted gate (fitted parameters, the canonical ordering, and the date-keyed log-prior
table) to `checkpoints/fold_<i>_gate.pt` at fit time, and reload it on resume. The G-3 xfail test must
pass.

**C.2 Persist the base per fold.** Save each fold's frozen base to `checkpoints/fold_<i>_base.pt`
with its cache key, and reload it on resume instead of refitting. `BaseCache` stays in-memory within a
run; this is the run's own resume state.

**C.3 Clean interruption.** On SIGINT (Ctrl+C) or SIGTERM, finish the current optimizer step, write a
checkpoint, set `status.json` to `interrupted`, and exit. A second signal exits immediately.

**C.4 Safe writes.** Every checkpoint and every results file is written to a temporary file and
renamed into place, so a crash mid-write never leaves a corrupt file. Keep the last
`checkpoint_keep` checkpoints per fit (config field); on resume, load the newest one that loads
cleanly and report if an older one had to be used.

**C.5 Crash detection.** `status.json` carries a heartbeat timestamp updated every
`heartbeat_every` steps (config field). A run whose status is `running` but whose heartbeat is older
than `stale_after` seconds (config field) is reported as `crashed` by `runs.py list`, and is
resumable like an interrupted one.

**C.6 Refuse an unsafe resume.** Resume compares the stored settings hash and data fingerprint with
the current ones. If either differs, refuse and print what changed. `--force` overrides with a loud
warning and records the override in `run.json`. Today a changed `settings.json` only prints a warning.

**C.7 One command.** `python3.14 run_experiment.py --resume` resumes the most recent interrupted or
crashed run of the settings block's tag. `--resume <run_id>` resumes a specific run. Also
`scripts/runs.py resume <run_id>`.

**C.8 Tests.** The central property is **an interrupted and resumed run equals an uninterrupted one
exactly**, on CPU with a fixed seed. Pin it for:

- the current design (residual base, correction mode, Markov gate, frozen), interrupted mid-fit
  inside a fold, and between folds;
- a sweep interrupted mid-arm;
- a run killed during a checkpoint write (simulate by truncating the newest checkpoint), which
  resumes from the previous one and still matches.

Also: SIGINT leaves status `interrupted` and a loadable checkpoint; a changed setting or changed data
refuses to resume; `--force` resumes and records the override.

---

## D. Documentation and commits

- **`RUNBOOK.md`** in `nec_baseline/`, written for Tom rather than for a developer: how to start a
  run, name a campaign and purpose, watch progress, interrupt it, resume it, find its results, compare
  two runs, and where licensed outputs go. On macOS, note `caffeinate -i` for long runs so the
  machine doesn't sleep.
- `HANDBOOK.md`: a section on the run store and resume, pointing to `RUNBOOK.md`.
- `WORK_QUEUE.md`, `Progress_Tracker.md` (one dated log entry, updated rows), and a section 12 in
  `Code_Audit_2026-09-25.md` listing what this brief closed and what it did not.

**Commits.** One commit per section: A, B, C, D. Stage by explicit path; never `git add -A`,
`git add .` or `git commit -a`. Before each commit check `git diff --cached --name-only`: nothing under
`Data/`, `data_cache/`, `Claude outputs/`, or any `checkpoints/` or `logs/` folder. Do not push.

## Tests, summarised

A: one test per finding, and the three audit xfails become plain passes. B: B.4. C: C.8. The suite,
ruff and mypy pass before every commit.

## Scope fence

Implement A to D only. Do not:

- decide Q16 (window, folds), Q18 (K), Q20 (loss), Q21 (frequency, horizon), Q22 (purge gap in the
  gate filter), Q23 (gate input series), X-1 (`hidden_init`), B-2 (the base's validation tail) or the
  post-delisting fill;
- make the HMM baseline's regime date-level (A.6 keeps its current definition);
- add gates (jump, Wasserstein, TVTP), objectives, or features;
- tune any hyperparameter;
- print or report any out-of-sample number from the real panel.

If a fix turns out to need one of these decisions, stop and report instead of choosing.
