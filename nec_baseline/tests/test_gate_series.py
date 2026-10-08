"""Brief 10 B (Q23, decided 2026-10-08): the gate's series is the CRSP S&P 500
index's daily total return as a log return.

Fixture tests run on ``crsp_fixture`` (CIZ-format files with invented numbers;
no real CRSP row in any test).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("pyarrow")

import crsp_fixture as fx  # noqa: E402
from conftest import small_config  # noqa: E402

from nec_moe import (  # noqa: E402
    PROVENANCE_KEYS,
    SERIES_REGISTRY,
    CRSPSpec,
    FeatureSchema,
    MarkovGateConfig,
    Panel,
    StageBSpec,
    build_crsp_panel,
    date_level_series,
    extract_crsp,
    load_extract,
    trial_provenance,
)
from nec_moe import markov_gate as markov_gate_module  # noqa: E402
from nec_moe.crsp import MARKET_LOG_RETURN_KEY, market_log_return_by_date  # noqa: E402
from nec_moe.features import FeatureSpec  # noqa: E402

KEY = "crsp_market_log_return"
STAGE_B = StageBSpec(seq_len=10, horizon=5, min_names_per_date=4, target_kind="raw",
                     features=FeatureSpec(feature_set="legacy14"))


def _spec(root: Path, **kw) -> CRSPSpec:
    base = dict(
        crsp_dir=str(root), release=fx.RELEASE, start=fx.START, end=fx.END,
        chunk_bytes=20_000, member_count_min=5, member_count_max=8,
    )
    return CRSPSpec(**(base | kw))


@pytest.fixture(scope="module")
def fixture(tmp_path_factory) -> fx.Fixture:
    return fx.write_fixture(tmp_path_factory.mktemp("crsp") / "Data")


@pytest.fixture(scope="module")
def panel(fixture: fx.Fixture) -> Panel:
    spec = _spec(fixture.root)
    extract_crsp(spec, verbose=False)
    return build_crsp_panel(spec, STAGE_B, extract=load_extract(spec), verbose=False).panel


def _labels(panel: Panel) -> list[str]:
    assert panel.date_labels is not None
    return [panel.date_labels[int(c)] for c in torch.unique(panel.date, sorted=True)]


def _expected(fixture: fx.Fixture, panel: Panel, indno: int = fx.MEMBER_INDNO) -> np.ndarray:
    # the fixture's files carry DlyTotRet to 8 decimals
    ret = fixture.market[indno].round(8).reindex(pd.to_datetime(_labels(panel)))
    return np.log1p(ret.to_numpy(dtype=float))


def _without_stored(panel: Panel, **meta) -> Panel:
    """``panel`` as built before brief 10: no stored series (plus overrides)."""
    kept = {k: v for k, v in panel.metadata.items() if k != MARKET_LOG_RETURN_KEY}
    return dataclasses.replace(panel, metadata=kept | meta)


# --------------------------------------------------------------------------- #
# the series
# --------------------------------------------------------------------------- #


def test_the_series_is_log1p_dlytotret_of_the_market_index_aligned_by_date(fixture, panel):
    cfg = MarkovGateConfig(series=KEY)
    dates, series = date_level_series(panel, cfg)
    assert torch.equal(dates, torch.unique(panel.date, sorted=True))
    np.testing.assert_allclose(series, _expected(fixture, panel), rtol=0, atol=1e-12)
    # the panel's market index (INDNO 1000500), not another index series
    assert not np.allclose(series, _expected(fixture, panel, fx.OTHER_INDNO))
    # stored with the panel at build time, keyed by calendar date
    stored = panel.metadata[MARKET_LOG_RETURN_KEY]
    assert set(stored) == set(_labels(panel))
    assert panel.metadata["market_indno"] == fx.MEMBER_INDNO == 1000500


def test_the_series_survives_subsetting(fixture, panel):
    """The gate reads the series off training blocks and spans, i.e. subsets."""
    codes = torch.unique(panel.date, sorted=True)[10:30]
    sub = panel.subset_dates(codes)
    _, series = date_level_series(sub, MarkovGateConfig(series=KEY))
    np.testing.assert_allclose(series, _expected(fixture, panel)[10:30], rtol=0, atol=1e-12)


def test_a_panel_built_before_brief_10_reads_the_extract(fixture, panel):
    """Fallback: no stored series, so market.parquet of the extract the build
    metadata names."""
    old = _without_stored(panel)
    cfg = MarkovGateConfig(series=KEY, crsp_dir=str(fixture.root))
    _, series = date_level_series(old, cfg)
    np.testing.assert_allclose(series, _expected(fixture, panel), rtol=0, atol=1e-12)


def test_neither_source_raises(fixture, panel):
    cfg = MarkovGateConfig(series=KEY, crsp_dir=str(fixture.root))
    # no extract for the recorded window (nor a default-window one)
    with pytest.raises(FileNotFoundError, match="no complete CRSP extract"):
        date_level_series(_without_stored(panel, start="2019-01-02"), cfg)
    # an extract of another market index does not count
    with pytest.raises(FileNotFoundError, match="market_indno"):
        date_level_series(_without_stored(panel, market_indno=fx.OTHER_INDNO), cfg)
    # no build metadata to locate one with
    with pytest.raises(ValueError, match="build metadata lacks"):
        date_level_series(_without_stored(panel, crsp_release=None), cfg)


def test_a_missing_date_raises_never_forward_filled(panel):
    stored = dict(panel.metadata[MARKET_LOG_RETURN_KEY])
    dropped = _labels(panel)[7]
    del stored[dropped]
    gap = dataclasses.replace(panel, metadata=panel.metadata | {MARKET_LOG_RETURN_KEY: stored})
    with pytest.raises(ValueError, match="no CRSP market return"):
        date_level_series(gap, MarkovGateConfig(series=KEY))
    # a missing value at build time is left out, so it raises the same way
    days = pd.bdate_range("2020-01-02", periods=4)
    ret = pd.Series([0.01, np.nan, -0.02, 0.0], index=days)
    by_date = market_log_return_by_date(np.log1p(ret), [str(d.date()) for d in days])
    assert "2020-01-03" not in by_date and len(by_date) == 3


def test_a_panel_without_calendar_dates_is_refused(panel):
    with pytest.raises(ValueError, match="no date_labels"):
        date_level_series(dataclasses.replace(panel, date_labels=None),
                          MarkovGateConfig(series=KEY))


# --------------------------------------------------------------------------- #
# the default, and the French key kept for the robustness check
# --------------------------------------------------------------------------- #


def test_the_default_is_the_crsp_key_and_the_french_key_still_works(tmp_path, panel):
    import run_experiment as rx

    assert MarkovGateConfig().series == KEY
    assert rx.Experiment().gate_series == KEY
    assert {KEY, "market_excess_return"} <= set(SERIES_REGISTRY)
    labels = _labels(panel)
    rets = np.linspace(-0.03, 0.03, len(labels)).round(6)
    ctx = _french_dir(tmp_path, labels, rets)
    _, french = date_level_series(
        panel, MarkovGateConfig(series="market_excess_return", context_dir=str(ctx))
    )
    np.testing.assert_allclose(french, rets, atol=1e-9)


def test_the_experiment_passes_the_series_to_the_gate():
    import run_experiment as rx
    from nec_moe import SyntheticRegimePanel, SyntheticSpec

    synth = SyntheticRegimePanel(SyntheticSpec(seed=0)).generate(30, 4)
    for series in (KEY, "market_excess_return"):
        exp = rx.Experiment(prior="markov", gate_series=series)
        assert rx._nec_config(exp, synth, 1.0).markov_gate.series == series


def _french_dir(root: Path, labels: list[str], rets: np.ndarray) -> Path:
    """A cache dir with a minimal French daily-factors zip: Mkt-RF in percent,
    as French publishes it."""
    import zipfile

    body = "".join(
        f"{d.replace('-', '')},{r * 100:.4f},0.00,0.00,0.00\n"
        for d, r in zip(labels, rets, strict=True)
    )
    csv = (
        "This file was created by CMPT_ME_BEME_RETS_DAILY\n\n"
        ",Mkt-RF,SMB,HML,RF\n" + body + "\nCopyright 2026 Eugene F. Fama and Kenneth R. French\n"
    )
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "ff_factors_daily.zip", "w") as z:
        z.writestr("F-F_Research_Data_Factors_daily.CSV", csv)
    return root


def test_the_units_match_the_french_path(tmp_path, panel):
    """Both series reach the fit in decimals (0.01 is a 1% day). The same day
    returns through both paths differ only by the log (second order), never
    by a factor of 100."""
    labels = _labels(panel)
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.012, len(labels)).round(6)
    french_cfg = MarkovGateConfig(series="market_excess_return",
                                  context_dir=str(_french_dir(tmp_path, labels, rets)))
    _, french = date_level_series(panel, french_cfg)
    stored = market_log_return_by_date(
        np.log1p(pd.Series(rets, index=pd.to_datetime(labels))), labels
    )
    same = dataclasses.replace(panel, metadata=panel.metadata | {MARKET_LOG_RETURN_KEY: stored})
    _, crsp = date_level_series(same, MarkovGateConfig(series=KEY))
    np.testing.assert_allclose(french, rets, atol=1e-9)          # percent -> decimals
    np.testing.assert_allclose(crsp, np.log1p(rets), atol=1e-12)  # already decimals
    assert np.abs(crsp - french).max() <= (rets**2).max()
    assert crsp.std() / french.std() == pytest.approx(1.0, abs=0.01)


# --------------------------------------------------------------------------- #
# the run settings and the trial registry record the series
# --------------------------------------------------------------------------- #


def test_the_trial_registry_records_the_gate_series(panel):
    assert "gate_series" in PROVENANCE_KEYS
    markov = small_config(prior_kind="markov")
    for series in (KEY, "market_excess_return"):
        cfg = dataclasses.replace(
            markov, markov_gate=dataclasses.replace(markov.markov_gate, series=series)
        )
        assert trial_provenance(panel, cfg)["gate_series"] == series
    # no Hamilton gate, no series
    assert trial_provenance(panel, small_config(prior_kind="soft"))["gate_series"] is None
    assert trial_provenance(panel, None)["gate_series"] is None


def test_the_run_settings_record_the_gate_series(tmp_path):
    """The run store's settings are the whole Experiment, so the series is in
    them under its field name."""
    import run_experiment as rx
    from nec_moe.runstore import RunStore

    exp = rx.Experiment(tag="gs", campaign="unit")
    run = RunStore(tmp_path / "results", tmp_path / "Data" / "derived" / "runs").create(
        campaign=exp.campaign, tag=exp.tag, mode=exp.mode, settings=rx._snapshot(exp),
    )
    settings = json.loads((run.dir / "settings.json").read_text())
    assert settings["gate_series"] == KEY


def test_the_gate_fits_on_the_crsp_series(panel):
    """End to end on the fixture: the default key feeds a Hamilton fit."""
    import warnings

    gate = markov_gate_module.MarkovSwitchingRegimePrior(
        # the fixture has about 125 dates: a 20-day start window, not 60
        2, MarkovGateConfig(series=KEY, start_vol_windows=(20,), start_vol_quantiles=(0.5,)),
        horizon=STAGE_B.horizon,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gate.fit(panel)
    assert gate.fit_result is not None and gate.fitted
    assert torch.equal(gate.covers(torch.unique(panel.date)), torch.ones(
        len(torch.unique(panel.date)), dtype=torch.bool))


def test_schema_free_date_panel_example():
    """The minimal one-row-per-date panel the diagnostic script uses carries
    the series through metadata alone."""
    days = pd.bdate_range("2003-03-03", periods=5)
    labels = tuple(str(d.date()) for d in days)
    ret = pd.Series([0.01, -0.02, 0.003, 0.0, 0.015], index=days)
    p = Panel(
        x_seq=torch.zeros(5, 1, 1), x_snap=torch.zeros(5, 1), y=torch.zeros(5),
        date=torch.arange(5), entity=torch.zeros(5, dtype=torch.long),
        schema=FeatureSchema(sequence_features=("s",), snapshot_features=("x",),
                             target="fwd_ret_1d"),
        date_labels=labels,
        metadata={MARKET_LOG_RETURN_KEY: market_log_return_by_date(np.log1p(ret), labels)},
    )
    _, series = date_level_series(p, MarkovGateConfig(series=KEY))
    np.testing.assert_allclose(series, np.log1p(ret.to_numpy()), atol=1e-15)


def test_the_series_diagnostic_refuses_any_date_after_2008(monkeypatch):
    """Brief 10 B.6: the comparison reads no date after 2008, no override."""
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "gate_series_comparison.py"
    spec = importlib.util.spec_from_file_location("gate_series_comparison", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)  # its dataclass needs it
    spec.loader.exec_module(mod)
    assert mod.Settings().validate().end == "2008-12-31"
    for end in ("2009-01-01", "2009-01-02", "2012-06-30"):
        with pytest.raises(ValueError, match="no date after 2008"):
            mod.Settings(end=end).validate()
    assert not hasattr(mod.Settings(), "allow_later")  # no override field
