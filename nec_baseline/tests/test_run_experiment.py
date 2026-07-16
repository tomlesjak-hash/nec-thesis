"""The control panel drives the whole machinery end to end from one settings block."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from run_experiment import Experiment, main


def _tiny(**overrides) -> Experiment:
    base = dict(
        data="synthetic",
        synth_dates=120,
        synth_entities=8,
        synth_vol_levels=(0.5, 2.5),
        encoder_hidden=16,
        expert_hidden=16,
        expert_dropout=0.0,
        steps=120,
        lr=3e-3,
        sigma_freeze_steps=20,
        seeds=(0,),
    )
    base.update(overrides)
    return Experiment(**base)


def test_quick_mode_end_to_end(tmp_path: Path):
    pytest.importorskip("matplotlib")
    exp = _tiny(tag="smoke_quick", mode="quick", out_dir=str(tmp_path),
                figures=True, calibration=True)
    result = main(exp)
    out = tmp_path / "smoke_quick"
    # provenance snapshot + registry row + the diagnostics figures
    settings = json.loads((out / "settings.json").read_text())
    assert settings["prior"] == "soft" and settings["mode"] == "quick"
    assert (out / "trials.jsonl").exists()
    for fig in ["dashboard.png", "utilization.png", "ic.png", "ls.png",
                "reliability.png", "reliability_scaled.png"]:
        assert (out / fig).stat().st_size > 5_000, fig
    assert result["nll"] < 2.0 and "temperature" in result


def test_quick_mode_hmm_variant(tmp_path: Path):
    pytest.importorskip("matplotlib")
    exp = _tiny(tag="smoke_hmm", mode="quick", out_dir=str(tmp_path),
                prior="hmm", synth_regime_process="markov", chunk_len=25,
                steps=60, calibration=True)  # calibration auto-skips for hmm
    result = main(exp)
    out = tmp_path / "smoke_hmm"
    assert (out / "transition.png").exists()  # the HMM-specific figure
    assert "temperature" not in result  # calibration correctly skipped


def test_evaluate_mode_end_to_end(tmp_path: Path):
    exp = _tiny(tag="smoke_eval", mode="evaluate", out_dir=str(tmp_path),
                seeds=(0, 1), n_folds=2, test_dates_per_fold=15,
                include_ridge=True, include_mlp=False, figures=False)
    result = main(exp)
    report = result["report"]
    assert [a.arm for a in report.arms] == ["soft", "ridge"]
    frame_path = tmp_path / "smoke_eval" / "report.csv"
    assert frame_path.exists()
    # the planted comparison holds even through the control panel
    scores = {a.arm: a.mean["mean_ic"] for a in report.arms}
    assert scores["soft"] > 0.3 and abs(scores["ridge"]) < 0.15
    # registry: 2 arms x 2 seeds
    lines = (tmp_path / "smoke_eval" / "trials.jsonl").read_text().splitlines()
    assert len([ln for ln in lines if ln.strip()]) == 4


def test_bad_mode_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown mode"):
        main(_tiny(tag="bad", mode="banana", out_dir=str(tmp_path)))
