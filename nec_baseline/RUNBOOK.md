# Runbook: running, watching, stopping and resuming experiments

How to run things day to day. HANDBOOK.md explains what the code does; this file explains
how to drive it. Every command below is run from the `nec_baseline/` folder.

---

## 1. Start a run

Open `run_experiment.py`, find the `EXPERIMENT = Experiment(...)` block near the top, and set
what you want. Anything you leave out keeps its default. Three settings name the run:

| setting | what it does | example |
|---|---|---|
| `tag` | a short name for this kind of run; it becomes part of the run id | `"k2_markov"` |
| `campaign` | the folder that groups related runs | `"dev"`, `"gates_k2"` |
| `purpose` | a free-text note to yourself; it has no effect | `"debug the base"`, `"smoke"` |

`mode="quick"` trains one model on one split and draws the diagnostic figures (minutes).
`mode="evaluate"` runs the full walk-forward protocol over every seed (longer).

Then start it:

```bash
python3.14 run_experiment.py
```

The first lines it prints give the run id and the folder, for example

```
[run] 20261001-095437_k2_markov_8a242625 (campaign 'dev') -> .../results/dev/20261001-095437_k2_markov_8a242625
```

The run id is the start time, the tag, and an 8-character fingerprint of the settings, so
two runs with different settings never share an id.

### Long runs: keep the Mac awake

On macOS a long run stops when the machine goes to sleep. Start it through `caffeinate`:

```bash
caffeinate -i python3.14 run_experiment.py
```

`-i` keeps the Mac from idle-sleeping for as long as the run lasts, and lets it sleep again
afterwards. Closing a laptop's lid still puts it to sleep, so leave the lid open (or use an
external display and power). If it does sleep or lose power, nothing is lost beyond the last
checkpoint; see section 5.

For a long run also set `checkpoint_every`, for example `checkpoint_every=50`. That saves
the training state every 50 steps. With the default 0 it is saved only at the end of each
fit and when you press Ctrl+C, so a power cut would lose the fit in progress.

---

## 2. Watch it

```bash
python3.14 scripts/runs.py list                    # every run: status, purpose, data
python3.14 scripts/runs.py list --campaign dev     # one campaign
python3.14 scripts/runs.py list --status running
```

Status is one of `running`, `completed`, `failed`, `interrupted`, or `crashed`. A run shows
as `crashed` when it says `running` but its heartbeat stopped more than `stale_after`
seconds ago (30 minutes by default): the process died without saying so (power cut,
sleep, killed from outside).

To follow one run as it goes:

```bash
tail -f results/<campaign>/<run_id>/logs/run.log   # everything it prints, timestamped
cat results/<campaign>/<run_id>/status.json        # current arm, seed, fold and step
```

`status.json` is refreshed every `heartbeat_every` training steps (50 by default) and at
each new arm, seed and fold.

---

## 3. Stop it

Press **Ctrl+C once**. The run finishes the training step it is on, saves a checkpoint,
marks itself `interrupted`, and exits. During training that takes a second or two; if it is
fitting the gate or the base at that moment, it stops at the first training step after the
fit. Press **Ctrl+C again** only if you need it to stop at once; then whatever was not saved
is lost (the run is still marked `interrupted` and resumable from its last checkpoint).

Stopping it from another terminal (`kill <pid>`, which sends SIGTERM) behaves like the
first Ctrl+C.

---

## 4. Resume it

```bash
python3.14 run_experiment.py --resume              # the latest interrupted or crashed run
                                                   #   with the settings block's tag
python3.14 run_experiment.py --resume <run_id>     # a specific run
python3.14 scripts/runs.py resume <run_id>         # the same, using the run's own settings
```

Resuming continues the same run in the same folder. Finished folds and seeds are skipped,
and a fit that was cut off continues from its checkpoint. The result is **exactly** what the
run would have produced without the interruption.

**Resume refuses if anything changed.** It compares the settings with the ones the run
started with, and the data with a fingerprint taken at the start. If either differs it stops
and prints what changed, for example

```
refusing to resume <run_id>: settings changed: steps: 600 -> 1000. ...
```

Usually the right answer is to start a **new** run with the new settings. If you really mean
to continue the old run under the change (say, to give it more steps), add `--force`:

```bash
python3.14 run_experiment.py --resume <run_id> --force
```

It prints a loud warning and records the override in the run's `run.json`, so the mixed
history is on the record.

If the newest checkpoint was damaged by a crash during the save, resume uses the one before
it and says so; the result is still exact. Two checkpoints are kept per fit
(`checkpoint_keep`).

---

## 5. After a crash, sleep or power cut

The run stays marked `running` until its heartbeat goes stale, then `runs.py list` shows
it as `crashed`. Resume it exactly as in section 4. `--resume` without a run id picks it up.
If you don't want to wait for it to count as stale, give the run id.

---

## 6. Find the results

```bash
python3.14 scripts/runs.py show <run_id>                  # its SUMMARY.md and status
python3.14 scripts/runs.py latest --campaign dev          # the newest run id
python3.14 scripts/runs.py latest --status interrupted    # the one to resume
```

Everything a run produces is in `results/<campaign>/<run_id>/`:

| file or folder | what is in it |
|---|---|
| `SUMMARY.md` | written at the end: status, which settings differ from the defaults, the main numbers |
| `run.json` | who and what: purpose, git commit (and whether the code had uncommitted changes), data source and fingerprint, package versions, any forced resume |
| `settings.json` | every setting of the run |
| `status.json` | state, where it got to, last heartbeat, why it ended |
| `trials.jsonl` | the trial registry: one row per arm and seed (and per gate start) |
| `metrics/` | the numbers: `quick.json`, or `report.csv` plus `<arm>_seed<s>_folds.csv` and `_pooled.json` |
| `figures/` | the plots |
| `logs/run.log` | everything printed (not committed) |
| `checkpoints/` | what resume needs (not committed) |

`results/INDEX.csv` has one row per run, so you can also open it in a spreadsheet.

---

## 7. Compare two runs

```bash
python3.14 scripts/runs.py diff <run_id_a> <run_id_b>
```

prints every setting that differs, one per line (`steps: 600 -> 1000`). The two
`SUMMARY.md` files then show the numbers side by side.

---

## 8. Licensed data: what goes where

CRSP is licensed: nothing derived from it may leave `Quant Model/Data/`.

- **Per-stock outputs** (each stock's predictions) are written to
  `Data/derived/runs/<run_id>/`, never to `results/`. That folder is inside `Data/` and is
  never committed.
- **`results/` holds aggregate numbers only**: settings, per-fold and pooled summaries,
  figures, logs. It is safe to commit, except `logs/` and `checkpoints/`, which git already
  ignores.
- When committing, stage files by name (`git add <path>`), never `git add -A`.

`touches_test` in `run.json` and `INDEX.csv` is `True` for any run that scored a test block
on a real panel. Every such run is a look at the test data, which the pre-registration (Q15)
has to count. The code records it and does not stop you. Until the pre-registration is
written, keep real-panel runs to what you have decided to do.

---

## 9. Quick reference

| I want to... | command |
|---|---|
| start a run | `python3.14 run_experiment.py` |
| start a long run safely | `caffeinate -i python3.14 run_experiment.py` (with `checkpoint_every` set) |
| see all runs | `python3.14 scripts/runs.py list` |
| follow one run | `tail -f results/<campaign>/<run_id>/logs/run.log` |
| stop cleanly | Ctrl+C once |
| resume the last stopped run | `python3.14 run_experiment.py --resume` |
| resume a specific run | `python3.14 scripts/runs.py resume <run_id>` |
| resume despite a change | add `--force` |
| read a run's results | `python3.14 scripts/runs.py show <run_id>` |
| compare two runs | `python3.14 scripts/runs.py diff <a> <b>` |

No command here deletes anything. Old runs stay until you remove their folders yourself.
