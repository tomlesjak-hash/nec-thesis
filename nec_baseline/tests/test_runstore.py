"""The run store (brief 07 section B): layout, index, licence split, hashes.

Synthetic runs only; every test runs against a temporary store (the autouse
fixture in conftest redirects both roots), so nothing touches the real
``results/`` or the licensed ``Data/``.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

import run_experiment as rx  # noqa: E402
from nec_moe import RunStore, settings_hash  # noqa: E402
from nec_moe.runstore import VOLATILE_SETTINGS  # noqa: E402
from nec_moe.train import DeadParameterWarning  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _exp(tmp_path: Path, **kw) -> rx.Experiment:
    base = dict(
        tag="rs", campaign="unit", purpose="test", out_dir=str(tmp_path / "results"),
        per_security_dir=str(tmp_path / "Data" / "derived" / "runs"), data="synthetic",
        synth_dates=100, synth_entities=6, encoder_hidden=8, expert_hidden_dims=(8, 4),
        expert_dropout=0.0, steps=10, sigma_freeze_steps=0, seeds=(0,), n_folds=2,
        test_dates_per_fold=10, include_ridge=True, include_mlp=False,
        backtest_quantiles=3, calibration=False,
        fold_scheme="count",  # a synthetic panel without a calendar
    )
    return rx.Experiment(**(base | kw))


def _main(exp: rx.Experiment) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        return rx.main(exp)


def test_layout_is_created_exactly_as_specified(tmp_path: Path):
    result = _main(_exp(tmp_path, mode="quick", figures=True))
    run_dir: Path = result["run_dir"]
    run_id = result["run_id"]
    assert run_dir == tmp_path / "results" / "unit" / run_id
    stamp, tag, digest = run_id.split("_")
    assert len(stamp) == 15 and stamp[8] == "-" and tag == "rs" and len(digest) == 8
    assert sorted(p.name for p in run_dir.iterdir()) == sorted([
        "run.json", "settings.json", "status.json", "trials.jsonl", "metrics",
        "figures", "logs", "checkpoints", "SUMMARY.md",
    ])
    info = json.loads((run_dir / "run.json").read_text())
    for key in ("run_id", "campaign", "tag", "purpose", "mode", "created", "git_commit",
                "dirty", "data_source", "data_fingerprint", "versions", "device",
                "settings_hash", "touches_test"):
        assert key in info, key
    assert info["purpose"] == "test" and info["data_source"] == "synthetic"
    assert json.loads((run_dir / "settings.json").read_text())["tag"] == "rs"
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "completed" and status["heartbeat"]
    log = (run_dir / "logs" / "run.log").read_text()
    assert "[run] " + run_id in log and log[:4].isdigit()  # timestamped lines
    assert (run_dir / "metrics" / "quick.json").exists()
    assert list((run_dir / "figures").glob("*.png"))
    summary = (run_dir / "SUMMARY.md").read_text()
    assert "Settings that differ from the defaults" in summary
    assert "| synth_dates | 100 |" in summary


def test_index_gets_one_row_per_run_updated_at_exit(tmp_path: Path):
    store = RunStore(tmp_path / "results")
    first = _main(_exp(tmp_path, mode="quick", figures=False))
    second = _main(_exp(tmp_path, mode="evaluate", figures=False))
    with (tmp_path / "results" / "INDEX.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["run_id"] for r in rows] == [first["run_id"], second["run_id"]]
    for r in rows:
        assert r["status"] == "completed" and r["finished"] and r["touches_test"] == "False"
    # a run that fails is recorded as such at exit
    with pytest.raises(ValueError):
        _main(_exp(tmp_path, mode="quick", prior="no_such_prior"))
    last = store.index()[-1]
    assert last["status"] == "failed" and "no_such_prior" in last["exit_reason"]


def test_per_security_outputs_never_appear_under_results(tmp_path: Path):
    """Scan every file a synthetic quick and evaluate run write under
    results/: aggregate formats only, no per-row columns or keys; the
    predictions are in the per-security folder."""
    runs = [_main(_exp(tmp_path, mode=m, figures=True)) for m in ("quick", "evaluate")]
    per_row_keys = {"entity", "pred", "prediction", "predictions", "base_pred", "y"}
    for result in runs:
        for path in result["run_dir"].rglob("*"):
            if path.is_dir() or "checkpoints" in path.parts:
                continue  # resume state (gitignored), not an output
            assert path.suffix in {".json", ".jsonl", ".csv", ".md", ".log", ".png"}, path
            if path.suffix == ".csv":
                with path.open(newline="") as fh:
                    header = next(csv.reader(fh))
                assert not per_row_keys & {h.lower() for h in header}, path
            if path.suffix == ".json":
                assert not per_row_keys & set(json.loads(path.read_text())), path
        preds = list((tmp_path / "Data" / "derived" / "runs" / result["run_id"])
                     .rglob("*_predictions.pt"))
        assert preds, "per-security outputs go to Data/derived/runs/<run_id>/"


def test_settings_hash_is_stable_across_processes_and_tracks_every_setting():
    snap = rx._snapshot(rx.Experiment())
    here = settings_hash(snap)
    code = (
        "import sys, json; sys.path.insert(0, '.'); import run_experiment as rx; "
        "from nec_moe import settings_hash; "
        "print(settings_hash(rx._snapshot(rx.Experiment())))"
    )
    other = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.strip()
    assert other == here and len(here) == 8
    for key in snap:
        changed = dict(snap) | {key: "__changed__"}
        if key in VOLATILE_SETTINGS:
            assert settings_hash(changed) == here  # volatile: asking to resume
        else:
            assert settings_hash(changed) != here, key


def test_touches_test_is_false_for_synthetic_and_true_for_a_real_panel(tmp_path: Path):
    result = _main(_exp(tmp_path, mode="quick", figures=False))
    info = json.loads((result["run_dir"] / "run.json").read_text())
    assert info["touches_test"] is False
    real = dataclasses.replace(
        rx.SyntheticRegimePanel(rx.SyntheticSpec(seed=1)).generate(20, 4),
        data_source="crsp_ciz202512",
    )
    assert rx._touches_test(real) is True


def test_runs_cli_lists_shows_diffs_and_finds_the_latest(tmp_path: Path, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import runs

    a = _main(_exp(tmp_path, mode="quick", figures=False))
    b = _main(_exp(tmp_path, mode="quick", figures=False, steps=12))
    root = ["--root", str(tmp_path / "results")]
    assert runs.main([*root, "list", "--campaign", "unit"]) == 0
    listed = capsys.readouterr().out
    assert a["run_id"] in listed and b["run_id"] in listed
    assert runs.main([*root, "show", a["run_id"]]) == 0
    assert "# Run " + a["run_id"] in capsys.readouterr().out
    assert runs.main([*root, "diff", a["run_id"], b["run_id"]]) == 0
    assert capsys.readouterr().out.strip() == "steps: 10 -> 12"
    assert runs.main([*root, "latest", "--campaign", "unit"]) == 0
    assert capsys.readouterr().out.strip() == b["run_id"]


def test_the_log_tee_survives_its_closed_log(tmp_path: Path, capsys):
    """A reference to the log tee can outlive Run.logging (anything that
    cached sys.stdout, and the io finalizer, which flushes at garbage
    collection). Writing, flushing and finalizing it after the log closed
    must not raise, and a late write goes to the console only."""
    import gc

    run = RunStore(tmp_path / "results").create(
        campaign="unit", tag="tee", mode="quick", settings={"a": 1}
    )
    with run.logging():
        tee = sys.stdout
        print("inside")
    tee.write("late\n")
    tee.flush()
    del tee
    gc.collect()  # the finalizer used to raise "I/O operation on closed file"
    log = run.log_path.read_text()
    assert "inside" in log and "late" not in log
    assert "late" in capsys.readouterr().out
