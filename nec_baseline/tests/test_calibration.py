"""Gate calibration: reliability math + planted-overconfidence repair.

Two planted truths anchor the machinery: identical experts force r ≡ π (ECE
exactly 0 — the calibrated-by-construction case), and artificially sharpened
gate logits force r ≠ π in the overconfident direction — the fitted
temperature must be > 1, and it must improve both held-out NLL and ECE.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
import torch
from conftest import small_config

from nec_moe import (
    NECModel,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    fit_temperature,
    gate_reliability,
    validation_tail,
)
from nec_moe.calibration import _reliability_from


def _trained(sharpen: float = 1.0):
    """A well-trained soft-gate model; sharpen>1 multiplies the gate weights,
    manufacturing overconfidence without touching what the gate has learned."""
    torch.manual_seed(0)
    spec = SyntheticSpec(
        vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=0
    )
    panel = SyntheticRegimePanel(spec).generate(n_dates=200, n_entities=8)
    train, test = panel.split_by_date(0.8)
    trainer = Trainer(NECModel(small_config(sigma_init=1.5, lr=3e-3, batch_size=256)))
    trainer.warmstart_experts(train.full_batch(),
                              sort_key=train.x_seq[:, :, 0].std(dim=1))
    trainer.fit(train, steps=250)
    if sharpen != 1.0:
        with torch.no_grad():
            trainer.model.gate.linear.weight.mul_(sharpen)
            trainer.model.gate.linear.bias.mul_(sharpen)
    return trainer, train, test


def test_identical_experts_are_perfectly_calibrated():
    """log-lik constant across experts => r == pi exactly => ECE == 0."""
    z = torch.randn(500, 2) * 2.0
    log_lik = torch.randn(500, 1).expand(500, 2)  # same density per expert
    report = _reliability_from(z, log_lik, temperature=1.0, n_bins=10)
    assert report.ece == pytest.approx(0.0, abs=1e-7)
    assert sum(report.bin_weight) == pytest.approx(1.0)
    frame = report.to_frame()
    assert list(frame.columns) == ["bin_lo", "bin_hi", "confidence", "outcome", "weight"]


def test_planted_overconfidence_is_detected_and_repaired():
    trainer, train, test = _trained(sharpen=4.0)
    honest, _, _ = _trained(sharpen=1.0)

    # detection: the sharpened gate is measurably worse calibrated out of sample
    rel_sharp = gate_reliability(trainer, test)
    rel_honest = gate_reliability(honest, test)
    assert rel_sharp.ece > rel_honest.ece

    # repair: T fitted on a validation TAIL of the training window (protocol),
    # then judged on the untouched test block
    tail = validation_tail(train.date, val_dates=25, purge_dates=5)
    val = train.subset_dates(tail.test_dates)
    fit = fit_temperature(trainer, val)
    assert fit.temperature > 1.5, fit  # softening, roughly undoing the x4
    assert fit.nll_after < fit.nll_before  # better held-out density on val
    assert "overconfident" in fit.verdict

    rel_fixed = gate_reliability(trainer, test, temperature=fit.temperature)
    assert rel_fixed.ece < rel_sharp.ece  # and the TEST diagram improves too
    assert rel_fixed.nll < rel_sharp.nll


def test_honest_gate_fits_temperature_near_one():
    trainer, train, _ = _trained(sharpen=1.0)
    tail = validation_tail(train.date, val_dates=25, purge_dates=5)
    fit = fit_temperature(trainer, train.subset_dates(tail.test_dates))
    assert 0.5 < fit.temperature < 2.0
    # a scalar cannot buy much on an already-trained mixture
    assert fit.nll_before - fit.nll_after < 0.05


def test_calibration_rejects_non_softmax_priors():
    torch.manual_seed(0)
    panel = SyntheticRegimePanel(SyntheticSpec(seed=0)).generate(40, 6)
    hmm = Trainer(NECModel(small_config(prior_kind="hmm")))
    with pytest.raises(ValueError, match="forward filter"):
        gate_reliability(hmm, panel)
    hard = Trainer(NECModel(small_config(prior_kind="hard")))
    with pytest.raises(ValueError, match="softmax-predictive"):
        fit_temperature(hard, panel)
    with pytest.raises(ValueError, match="temperature"):
        gate_reliability(Trainer(NECModel(small_config())), panel, temperature=0.0)


def test_reliability_plot_renders(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    from nec_moe import plot_reliability

    trainer, _, test = _trained(sharpen=1.0)
    report = gate_reliability(trainer, test)
    fig = plot_reliability(report, path=tmp_path / "rel.png")
    assert (tmp_path / "rel.png").stat().st_size > 5_000
    matplotlib.pyplot.close(fig)
