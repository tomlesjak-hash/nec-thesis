"""Figure functions: every plot renders headlessly, saves, and guards its inputs.

These are artifact tests, not pixel tests: each function must produce a Figure
with the promised structure and write a non-trivial PNG. Backend is forced to
Agg before matplotlib loads so the suite runs headless anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # before any pyplot import

import pytest
import torch

matplotlib = pytest.importorskip("matplotlib")

import pandas as pd
from conftest import small_config

from nec_moe import (
    ArmSummary,
    NECModel,
    SweepReport,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    plot_gate_utilization,
    plot_ic_series,
    plot_long_short_curve,
    plot_sweep_report,
    plot_training_dashboard,
    plot_transition_matrix,
)


@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    spec = SyntheticSpec(vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=0)
    panel = SyntheticRegimePanel(spec).generate(n_dates=120, n_entities=8)
    trainer = Trainer(NECModel(small_config(sigma_init=1.5, lr=3e-3, batch_size=256)))
    trainer.warmstart_experts(panel.full_batch(),
                              sort_key=panel.x_seq[:, :, 0].std(dim=1))
    history = trainer.fit(panel, steps=120)
    return trainer, panel, history


def _saved(path: Path) -> None:
    assert path.exists() and path.stat().st_size > 5_000  # a real PNG, not a stub


def test_training_dashboard(trained, tmp_path: Path):
    trainer, _, history = trained
    fig = plot_training_dashboard(history, n_experts=2, path=tmp_path / "dash.png")
    assert len(fig.axes) == 4
    _saved(tmp_path / "dash.png")
    matplotlib.pyplot.close(fig)
    with pytest.raises(ValueError, match="empty history"):
        plot_training_dashboard([], n_experts=2)


def test_gate_utilization_with_and_without_context(trained, tmp_path: Path):
    trainer, panel, _ = trained
    fig = plot_gate_utilization(trainer, panel, path=tmp_path / "util.png")
    _saved(tmp_path / "util.png")
    matplotlib.pyplot.close(fig)

    # synthetic panels join a context frame indexed by integer date codes
    codes = sorted(int(d) for d in torch.unique(panel.date))
    ctx = pd.DataFrame({"vix": [10.0 + (c % 7) for c in codes]}, index=codes)
    fig2 = plot_gate_utilization(trainer, panel, context=ctx,
                                 path=tmp_path / "util_ctx.png")
    assert len(fig2.axes) == 2  # twin axis for the context series
    _saved(tmp_path / "util_ctx.png")
    matplotlib.pyplot.close(fig2)

    with pytest.raises(ValueError, match="no column"):
        plot_gate_utilization(trainer, panel, context=ctx, context_col="nope")


def test_gate_utilization_hmm_filtered_path(tmp_path: Path):
    torch.manual_seed(0)
    spec = SyntheticSpec(regime_process="markov", transition_stay=0.95,
                         vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=0)
    panel = SyntheticRegimePanel(spec).generate(n_dates=100, n_entities=6)
    trainer = Trainer(NECModel(small_config(prior_kind="hmm", sigma_init=1.5,
                                            lr=3e-3)))
    trainer.warmstart_experts(panel.full_batch(),
                              sort_key=panel.x_seq[:, :, 0].std(dim=1))
    trainer.fit_sequence(panel.time_sequence(), steps=40, chunk_len=25)
    fig = plot_gate_utilization(trainer, panel, path=tmp_path / "filtered.png")
    _saved(tmp_path / "filtered.png")
    matplotlib.pyplot.close(fig)

    # and the transition heatmap comes from the same model
    fig2 = plot_transition_matrix(trainer, path=tmp_path / "trans.png")
    _saved(tmp_path / "trans.png")
    matplotlib.pyplot.close(fig2)


def test_ic_and_long_short_curves(trained, tmp_path: Path):
    trainer, panel, _ = trained
    trainer.model.eval()
    with torch.no_grad():
        pred = trainer.model(panel.x_seq, panel.x_snap).y_hat
    fig = plot_ic_series(pred, panel.y, panel.date, path=tmp_path / "ic.png")
    _saved(tmp_path / "ic.png")
    matplotlib.pyplot.close(fig)
    fig2 = plot_long_short_curve(pred, panel.y, panel.date, panel.entity,
                                 n_quantiles=4, cost_rate=0.001,
                                 path=tmp_path / "ls.png")
    _saved(tmp_path / "ls.png")
    matplotlib.pyplot.close(fig2)


def test_transition_matrix_rejects_memoryless(trained):
    trainer, _, _ = trained  # soft prior
    with pytest.raises(ValueError, match="memoryless"):
        plot_transition_matrix(trainer)
    # but a raw square tensor is accepted
    fig = plot_transition_matrix(torch.tensor([[0.9, 0.1], [0.2, 0.8]]))
    matplotlib.pyplot.close(fig)


def test_sweep_report_figure(tmp_path: Path):
    report = SweepReport(
        tag="demo",
        arms=(
            ArmSummary("soft", 3, {"mean_ic": 0.4, "nll": 1.0},
                       {"mean_ic": 0.05, "nll": 0.1}, median_p=0.01),
            ArmSummary("uniform", 3, {"mean_ic": 0.01, "nll": 1.4},
                       {"mean_ic": 0.02, "nll": 0.1}, median_p=0.6),
        ),
    )
    fig = plot_sweep_report(report, metric="mean_ic", path=tmp_path / "sweep.png")
    _saved(tmp_path / "sweep.png")
    matplotlib.pyplot.close(fig)
    with pytest.raises(ValueError, match="missing"):
        plot_sweep_report(report, metric="sharpe")
