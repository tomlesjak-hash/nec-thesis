"""Stage C: context-data parsers (offline fixtures) + gate-regime alignment.

The alignment tooling gets a *planted-truth* validation: a model trained on
synthetic vol-regime data is scored against a fake "VIX" built from the true
per-date regime — the report must recover the planted alignment. The live
network test at the bottom is opt-in (``NEC_NETWORK_TESTS=1``).
"""

from __future__ import annotations

import io
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import small_config

from nec_moe import (
    NECModel,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    build_context,
    gate_utilization_by_date,
    load_french_factors,
    load_vix,
    regime_alignment,
)
from nec_moe.context_data import parse_french_csv, parse_vix_csv

# --------------------------------------------------------------------------- #
# Parsers (pure, offline)
# --------------------------------------------------------------------------- #

VIX_FIXTURE = """DATE,OPEN,HIGH,LOW,CLOSE
01/02/2020,13.46,14.55,12.42,12.47
01/03/2020,15.01,16.20,13.13,14.02
01/06/2020,15.45,16.39,13.54,13.85
01/03/2020,15.01,16.20,13.13,14.10
"""

FRENCH_FIXTURE = """This file was created by using the 202605 CRSP database.
Some explanatory preamble text.

,Mkt-RF,SMB,HML,RF
20200102,    0.84,   -0.61,   -0.87,    0.006
20200103,   -0.72,    0.22,   -0.30,    0.006
20200106,    0.30,   -0.40,  -99.99,    0.006

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_parse_vix_csv():
    vix = parse_vix_csv(VIX_FIXTURE)
    assert vix.name == "vix" and vix.index.is_monotonic_increasing
    assert len(vix) == 3  # duplicate date deduped, keep last
    assert vix.loc["2020-01-03"] == pytest.approx(14.10)


def test_parse_french_csv():
    f = parse_french_csv(FRENCH_FIXTURE)
    assert list(f.columns) == ["mkt_rf", "smb", "hml", "rf"]
    assert len(f) == 3
    # percent -> decimal
    assert f.loc["2020-01-02", "mkt_rf"] == pytest.approx(0.0084)
    # the -99.99 missing marker becomes NaN, not a -0.9999 return
    assert math.isnan(f.loc["2020-01-06", "hml"])


def test_load_vix_and_factors_from_cache(tmp_path: Path):
    (tmp_path / "vix_history.csv").write_text(VIX_FIXTURE)

    def _zip_bytes(text: str, name: str) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(name, text)
        return buf.getvalue()

    (tmp_path / "ff_factors_daily.zip").write_bytes(
        _zip_bytes(FRENCH_FIXTURE, "F-F_Research_Data_Factors_daily.csv")
    )
    # the live momentum file carries a trailing comma on every line — the
    # fixture reproduces that exact quirk (',Mom,' header, '...,0.35,' rows)
    mom_text = (
        "This file was created by using the 202605 CRSP database.  It,,\n"
        "Missing data are indicated by -99.99 or -999.,,\n"
        ",,\n"
        ",Mom,\n"
        "20200102,1.20,\n"
        "20200103,-0.55,\n"
        "20200106,0.10,\n"
        ",,\n"
        "Copyright 2026 Eugene F. Fama and Kenneth R. French,,\n"
    )
    (tmp_path / "ff_momentum_daily.zip").write_bytes(
        _zip_bytes(mom_text, "F-F_Momentum_Factor_daily.csv")
    )

    vix = load_vix(tmp_path)  # cache-first: no network touched
    factors = load_french_factors(tmp_path)
    assert "mom" in factors.columns
    assert factors.loc["2020-01-02", "mom"] == pytest.approx(0.012)

    ctx = build_context(vix, factors)
    assert {"vix", "mkt_rf", "abs_mkt", "mkt_vol_20d"} <= set(ctx.columns)
    assert ctx.loc["2020-01-03", "abs_mkt"] == pytest.approx(0.0072)


def test_build_context_requires_input():
    with pytest.raises(ValueError, match="at least one"):
        build_context()


# --------------------------------------------------------------------------- #
# Alignment diagnostics: planted-truth validation
# --------------------------------------------------------------------------- #


def _trained_synthetic(prior_kind: str = "soft"):
    torch.manual_seed(0)
    spec = SyntheticSpec(
        regime_process="markov",
        transition_stay=0.95,
        vol_levels=(0.5, 2.5),
        beta_scale=2.0,
        noise_std=0.4,
        seed=0,
    )
    panel = SyntheticRegimePanel(spec).generate(n_dates=240, n_entities=8)
    cfg = small_config(
        prior_kind=prior_kind, sigma_init=1.5, lr=3e-3, batch_size=256,
        sigma_freeze_steps=30,
    )
    trainer = Trainer(NECModel(cfg))
    trainer.warmstart_experts(
        panel.full_batch(), sort_key=panel.x_seq[:, :, 0].std(dim=1)
    )
    if prior_kind == "hmm":
        trainer.fit_sequence(panel.time_sequence(), steps=150, chunk_len=40)
    else:
        trainer.fit(panel, steps=300)
    return trainer, panel, spec


def test_gate_utilization_shapes_and_simplex():
    trainer, panel, _ = _trained_synthetic()
    dates, util = gate_utilization_by_date(trainer, panel)
    assert util.shape == (len(dates), 2)
    assert torch.allclose(util.sum(dim=1), torch.ones(len(dates)), atol=1e-5)


def test_alignment_recovers_planted_regime():
    """Fake 'VIX' = the true per-date regime vol level. The trained gate's
    utilization must align strongly — one expert positively, one negatively —
    and the high/low-VIX tercile split must reflect it."""
    trainer, panel, spec = _trained_synthetic()
    dates, util = gate_utilization_by_date(trainer, panel)

    regime_by_date = torch.stack(
        [panel.regime[panel.date == d][0] for d in dates]
    )
    vols = torch.tensor(spec.vol_levels)
    fake_vix = pd.DataFrame(
        {"vix": vols[regime_by_date].numpy()}, index=[int(d) for d in dates]
    )

    report = regime_alignment(panel, dates, util, fake_vix)
    corrs = [e.correlations["vix"] for e in report.per_expert]
    # one expert tracks the stressed regime, the other the calm one
    assert max(corrs) > 0.6, corrs
    assert min(corrs) < -0.6, corrs
    assert abs(corrs[0] + corrs[1]) < 1e-4  # K=2: shares sum to 1
    stressed = report.per_expert[int(np.argmax(corrs))]
    assert stressed.high_vix_utilization > stressed.low_vix_utilization + 0.3
    # report table renders
    frame = report.to_frame()
    assert list(frame.index) == ["expert_0", "expert_1"]


def test_alignment_hmm_filtered_path():
    trainer, panel, spec = _trained_synthetic(prior_kind="hmm")
    dates, util = gate_utilization_by_date(trainer, panel)
    regime_by_date = torch.stack(
        [panel.regime[panel.date == d][0] for d in dates]
    )
    fake_vix = pd.DataFrame(
        {"vix": torch.tensor(spec.vol_levels)[regime_by_date].numpy()},
        index=[int(d) for d in dates],
    )
    report = regime_alignment(panel, dates, util, fake_vix)
    corrs = [e.correlations["vix"] for e in report.per_expert]
    assert max(abs(c) for c in corrs) > 0.6, corrs


def test_alignment_no_overlap_raises():
    trainer, panel, _ = _trained_synthetic()
    dates, util = gate_utilization_by_date(trainer, panel)
    ctx = pd.DataFrame({"vix": [1.0, 2.0]}, index=[99991, 99992])
    with pytest.raises(ValueError, match="no overlap"):
        regime_alignment(panel, dates, util, ctx)


# --------------------------------------------------------------------------- #
# Live network check — opt-in only (NEC_NETWORK_TESTS=1)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    os.environ.get("NEC_NETWORK_TESTS", "0") != "1",
    reason="network test; set NEC_NETWORK_TESTS=1 to run",
)
def test_live_context_data(tmp_path: Path):
    vix = load_vix(tmp_path)
    factors = load_french_factors(tmp_path)
    assert len(vix) > 8000 and float(vix.min()) > 0
    assert {"mkt_rf", "smb", "hml", "rf", "mom"} <= set(factors.columns)
    assert factors["mkt_rf"].abs().max() < 0.25  # decimals, not percent
    ctx = build_context(vix, factors)
    assert ctx.loc["2020-03-16", "vix"] > 60  # COVID crash spike sanity
