"""Brief 07 section A: one test per open audit finding it fixes.

O-1 (quick mode purge), O-3 (purge from the target), E-3 (horizon-consistent
long-short book), B-1 (base RNG), G-2 (autoregressive Hamilton gate), M-5
(HMM baseline on a changing cross-section) and S-3 (the current design end
to end through ``run_experiment.main``). Synthetic panels only.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import small_base_config, small_config

from nec_moe import (
    PROVENANCE_KEYS,
    BaseConfig,
    NECModel,
    Panel,
    StageBSpec,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    TrialRegistry,
    assemble_panel,
    fit_base,
    long_short_book,
    long_short_by_date,
    market_frame,
    portfolio_summary,
    walk_forward_evaluate,
)
from nec_moe.train import DeadParameterWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as rx  # noqa: E402


def _sticky(n_dates: int = 160, n_entities: int = 6, seed: int = 3):
    return SyntheticRegimePanel(
        SyntheticSpec(regime_process="markov", transition_stay=0.96, vol_levels=(0.5, 2.5),
                      beta_scale=2.0, noise_std=0.4, seed=seed)
    ).generate(n_dates, n_entities)


def _with_target(panel: Panel, target: str) -> Panel:
    return dataclasses.replace(panel, schema=dataclasses.replace(panel.schema, target=target))


# --------------------------------------------------------------------------- #
# A.1 O-1 and A.2 O-3: the purge is the target's horizon, quick mode included
# --------------------------------------------------------------------------- #


def test_purge_is_read_from_the_target_and_a_contradiction_raises():
    five_day = _with_target(_sticky(60, 4), "fwd_ret_5d")
    exp = rx.Experiment()
    assert rx.resolve_purge(exp, five_day) == 5  # horizon=None reads the target
    assert rx.resolve_purge(dataclasses.replace(exp, horizon=5), five_day) == 5
    with pytest.raises(ValueError, match="horizon=1 disagrees.*fwd_ret_5d.*horizon 5"):
        rx.resolve_purge(dataclasses.replace(exp, horizon=1), five_day)
    # a target that declares no horizon (synthetic y_synth) is one period
    assert rx.resolve_purge(dataclasses.replace(exp, horizon=5), _sticky(60, 4)) == 1


def test_quick_mode_split_is_purged_by_the_target_horizon():
    panel = _with_target(_sticky(100, 4), "fwd_ret_5d")
    exp = rx.Experiment(quick_train_frac=0.8)
    purge = rx.resolve_purge(exp, panel)
    train, test, fold = rx._quick_split(exp, panel, purge)
    first_test = int(test.date.min())
    # every training label's window (t, t+5] ends before the test block starts
    assert int(train.date.max()) + 5 < first_test
    assert int(train.date.max()) == first_test - purge - 1
    assert len(torch.unique(test.date)) == 100 - 80  # the same test block as before


def test_quick_mode_fits_the_gate_on_the_purged_training_block(tmp_path: Path):
    pytest.importorskip("statsmodels")
    panel = _with_target(_sticky(120, 4), "fwd_ret_5d")
    panel_file = tmp_path / "panel.pt"
    torch.save(panel, panel_file)
    exp = dataclasses.replace(
        _current_design(tmp_path, "quick"), data="panel_file", panel_file=str(panel_file),
        steps=5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        result = rx.main(exp)
    trial = TrialRegistry(result["run_dir"] / "trials.jsonl").trials(exp.tag)[0]
    assert math.isfinite(trial.metrics["nll"])


# --------------------------------------------------------------------------- #
# A.3 E-3: the long-short book is horizon-consistent
# --------------------------------------------------------------------------- #


def _book_panel():
    """4 dates x 4 entities, 2-day target, hand-chosen legs (n_quantiles=2)."""
    pred = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],  # date 0: short {0, 1}, long {2, 3}
        [0.4, 0.3, 0.2, 0.1],  # date 1: short {2, 3}, long {0, 1}
        [0.1, 0.2, 0.3, 0.4],  # date 2: short {0, 1}, long {2, 3}
        [0.1, 0.3, 0.2, 0.4],  # date 3: short {0, 2}, long {1, 3}
    ]).flatten()
    daily = torch.tensor([  # (date, entity) -> (day-1 return, day-2 return)
        [[0.01, 0.02], [0.00, -0.01], [0.03, 0.01], [-0.02, 0.04]],
        [[0.02, -0.03], [0.01, 0.01], [-0.01, 0.02], [0.00, 0.05]],
        [[-0.02, 0.01], [0.04, 0.00], [0.02, -0.01], [0.01, 0.03]],
        [[0.00, 0.02], [0.03, -0.02], [-0.01, 0.01], [0.02, 0.00]],
    ]).reshape(16, 2)
    date = torch.arange(4).repeat_interleave(4)
    entity = torch.arange(4).repeat(4)
    return pred, daily.sum(dim=1), daily, date, entity


def test_both_schemes_equal_the_daily_book_for_a_one_period_target():
    g = torch.Generator().manual_seed(0)
    pred, y = torch.randn(40, generator=g), torch.randn(40, generator=g)
    date, entity = torch.arange(8).repeat_interleave(5), torch.arange(5).repeat(8)
    _, gross, tno = long_short_by_date(pred, y, date, entity, n_quantiles=2)
    for scheme in ("nonoverlapping", "staggered"):
        _, g2, t2 = long_short_book(pred, y, date, entity, n_quantiles=2, horizon=1,
                                    scheme=scheme)
        assert torch.allclose(g2, gross, atol=1e-7) and torch.allclose(t2, tno, atol=1e-7)


def test_nonoverlapping_book_is_the_hand_computed_one():
    pred, y, _, date, entity = _book_panel()
    dates, gross, tno = long_short_book(pred, y, date, entity, n_quantiles=2, horizon=2,
                                        scheme="nonoverlapping")
    yy = y.reshape(4, 4)
    assert dates.tolist() == [0, 2]  # rebalanced every 2 dates
    expected = [(yy[0, 2] + yy[0, 3]) / 2 - (yy[0, 0] + yy[0, 1]) / 2,
                (yy[2, 2] + yy[2, 3]) / 2 - (yy[2, 0] + yy[2, 1]) / 2]
    assert gross.tolist() == pytest.approx([float(e) for e in expected], abs=1e-7)
    assert tno.tolist() == pytest.approx([1.0, 0.0])  # entry, then the same legs


def test_staggered_book_is_the_hand_computed_one():
    """Jegadeesh-Titman, h = 2: cohort j earns its day-(m+1) return on the day
    after date j+m; the book averages the live cohorts at weight 1/2 each."""
    pred, y, daily, date, entity = _book_panel()
    _, gross, tno = long_short_book(pred, y, date, entity, n_quantiles=2, horizon=2,
                                    scheme="staggered", y_daily=daily)
    d = daily.reshape(4, 4, 2)

    def cohort(j: int, m: int, long: tuple[int, int], short: tuple[int, int]) -> float:
        return float((d[j, long[0], m] + d[j, long[1], m]) / 2
                     - (d[j, short[0], m] + d[j, short[1], m]) / 2)

    legs = [((2, 3), (0, 1)), ((0, 1), (2, 3)), ((2, 3), (0, 1)), ((1, 3), (0, 2))]
    c = [[cohort(j, m, *legs[j]) for m in range(2)] for j in range(4)]
    expected = [c[0][0] / 2, (c[1][0] + c[0][1]) / 2,
                (c[2][0] + c[1][1]) / 2, (c[3][0] + c[2][1]) / 2]
    assert gross.tolist() == pytest.approx(expected, abs=1e-7)
    # daily turnover of the combined book: half the book enters on day 0; the
    # date-0 and date-1 cohorts cancel to a flat book; date 2 repeats it; on
    # date 3 the book is {0: -1/2, 3: +1/2}
    assert tno.tolist() == pytest.approx([0.5, 0.5, 0.0, 0.5])


def test_costs_are_charged_once_per_period_of_each_scheme():
    pred, y, daily, date, entity = _book_panel()
    rate = 0.01
    for scheme, periods in (("nonoverlapping", 2), ("staggered", 4)):
        _, gross, tno = long_short_book(pred, y, date, entity, n_quantiles=2, horizon=2,
                                        scheme=scheme, y_daily=daily)
        s = portfolio_summary(gross, tno, rate, scheme=scheme,
                              period_dates=2 if scheme == "nonoverlapping" else 1)
        assert s.n_dates == periods
        assert s.mean_net == pytest.approx(float((gross - rate * 2 * tno).mean()), abs=1e-9)
    with pytest.raises(ValueError, match="y_daily"):
        long_short_book(pred, y, date, entity, n_quantiles=2, horizon=2, scheme="staggered")


def _returns_panel(kind: str):
    g = np.random.default_rng(4)
    idx = pd.bdate_range("2021-01-04", periods=320)
    mkt = pd.Series(g.normal(0, 0.01, 320), index=idx)
    daily = {}
    for i in range(6):
        r = 0.8 * mkt.to_numpy() + g.normal(0, 0.015, 320)
        vol = 1e6 * np.exp(g.normal(0, 0.3, 320))
        daily[str(10001 + i)] = pd.DataFrame(
            {"ret": r, "volume": vol, "dollar_volume": 50 * np.exp(np.cumsum(r)) * vol},
            index=idx,
        )
    spec = StageBSpec(seq_len=10, horizon=3, target_kind=kind, beta_window=60)
    return assemble_panel(daily, market_frame(mkt, spec), spec)


@pytest.mark.parametrize("kind", ["raw", "residual"])
def test_daily_forward_returns_sum_to_the_target(kind: str):
    """``Panel.y_daily`` is built under the target's timing rule: its h daily
    pieces add up to the target, for the raw and the residual target."""
    panel = _returns_panel(kind)
    assert panel.y_daily is not None and panel.y_daily.shape == (len(panel), 3)
    assert torch.allclose(panel.y_daily.sum(dim=1), panel.y, atol=1e-5)
    sub = panel.subset_dates(torch.unique(panel.date)[:5])  # travels with the rows
    assert sub.y_daily is not None and torch.allclose(sub.y_daily.sum(1), sub.y, atol=1e-5)


def test_harness_uses_the_scheme_and_every_row_records_it(tmp_path: Path):
    from nec_moe import nec_arm, run_sweep

    panel = _returns_panel("raw")
    for scheme in ("nonoverlapping", "staggered"):
        reg = TrialRegistry(tmp_path / f"{scheme}.jsonl")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeadParameterWarning)
            cfg = dataclasses.replace(small_config(), data=rx.data_config_from_panel(panel))
            report = run_sweep(
                panel, [nec_arm("soft", cfg)],
                seeds=(0,), registry=reg, tag="t", steps=3, n_folds=2,
                test_dates_per_fold=12, purge_dates=3, backtest_quantiles=2,
                portfolio_scheme=scheme, verbose=False,
            )
        rows = reg.trials("t")
        assert rows and all(r.config["portfolio_scheme"] == scheme for r in rows)
        assert report.arms[0].n_seeds == 1


# --------------------------------------------------------------------------- #
# A.4 B-1: the base leaves the global RNG alone
# --------------------------------------------------------------------------- #


def test_fit_base_never_touches_the_global_rng():
    panel = _sticky(60, 4)
    cfg = dataclasses.replace(small_base_config(), dropout=0.2)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    a = fit_base(panel, cfg, seed=1)
    assert torch.equal(torch.get_rng_state(), before)
    b = fit_base(panel, cfg, seed=1)  # and the fit is a function of its seed
    for k, v in a.model.state_dict().items():
        assert torch.equal(v, b.model.state_dict()[k])
    assert isinstance(cfg, BaseConfig)


# --------------------------------------------------------------------------- #
# A.5 G-2: the autoregressive Hamilton gate runs through the harness
# --------------------------------------------------------------------------- #


def test_autoregressive_gate_leaves_its_first_dates_out_and_covers_every_test_date():
    pytest.importorskip("statsmodels")
    cfg = small_config(prior_kind="markov", freeze_gate=True, sigma_init=1.5)
    cfg = dataclasses.replace(cfg, markov_gate=dataclasses.replace(
        cfg.markov_gate, series="sequence_channel", order=1,
        start_vol_quantiles=(0.6,), start_vol_windows=(10,), start_persistences=(0.95,),
        start_draws_per_centre=1, start_min_history=5, start_min_group_size=5,
    ))
    trainers: list[Trainer] = []

    def make() -> Trainer:
        torch.manual_seed(0)
        trainers.append(Trainer(NECModel(cfg)))
        return trainers[-1]

    panel = _sticky(200, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        res = walk_forward_evaluate(panel, make, n_folds=2, test_dates_per_fold=15,
                                    purge_dates=1, steps=5)
    assert len(res.folds) == len(trainers) == 2
    for fold in res.folds:
        assert fold.gate_excluded_dates == 1  # the AR(1) initial condition
        assert math.isfinite(fold.nll)
    folds = rx.walk_forward_folds(panel.date, n_folds=2, test_dates_per_fold=15,
                                  purge_dates=1)
    for f, trainer in zip(folds, trainers, strict=True):
        prior = trainer.model.prior
        assert bool(prior.covers(f.test_dates).all())  # type: ignore[operator]
        assert not bool(prior.covers(f.train_dates[:1]).any())  # type: ignore[operator]


# --------------------------------------------------------------------------- #
# A.6 M-5: the HMM baseline carries its state by entity
# --------------------------------------------------------------------------- #


def _hmm_trainer() -> Trainer:
    torch.manual_seed(0)
    return Trainer(NECModel(small_config(prior_kind="hmm")))


def _by_entity(panel: Panel, per_date: list[torch.Tensor]) -> dict[tuple[int, int], torch.Tensor]:
    out = {}
    for d, values in zip(torch.unique(panel.date, sorted=True), per_date, strict=True):
        for e, v in zip(panel.entity[panel.date == d], values, strict=True):
            out[(int(d), int(e))] = v
    return out


def test_hmm_runs_when_a_name_drops_and_the_others_keep_their_state():
    panel = SyntheticRegimePanel(SyntheticSpec(regime_process="markov", seed=5)).generate(
        n_dates=10, n_entities=4
    )
    drop = ~((panel.date == 5) & (panel.entity == 2))
    thinned = panel._take(drop)
    trainer = _hmm_trainer()
    full = trainer.evaluate_sequence(panel.time_sequence())
    part = trainer.evaluate_sequence(thinned.time_sequence())  # used to raise (M-5)
    a = _by_entity(panel, list(full.log_filtered))
    b = _by_entity(thinned, list(part.log_filtered))
    for key, value in b.items():
        if key[1] != 2:  # every surviving entity's state is unchanged
            assert torch.allclose(value, a[key], atol=1e-6), key
    # the dropped name restarts from pi_0 on its next date
    pi0 = torch.log_softmax(trainer.model.prior.pi0_logits, 0)  # type: ignore[union-attr]
    prior_b = _by_entity(thinned, list(part.log_prior))
    assert torch.allclose(prior_b[(6, 2)], pi0, atol=1e-6)
    # and the unbalanced panel trains
    trainer.fit_sequence(thinned.time_sequence(), steps=3, chunk_len=4)


def test_hmm_outputs_do_not_depend_on_row_order_within_a_date():
    panel = SyntheticRegimePanel(SyntheticSpec(regime_process="markov", seed=6)).generate(
        n_dates=8, n_entities=5
    )
    g = torch.Generator().manual_seed(1)
    order = torch.cat([
        torch.nonzero(panel.date == d, as_tuple=True)[0][torch.randperm(5, generator=g)]
        for d in torch.unique(panel.date, sorted=True)
    ])
    shuffled = panel._take(order)
    trainer = _hmm_trainer()
    a = trainer.evaluate_sequence(panel.time_sequence())
    b = trainer.evaluate_sequence(shuffled.time_sequence())
    for field in ("log_filtered", "log_prior", "y_hat"):
        fa = _by_entity(panel, list(getattr(a, field)))
        fb = _by_entity(shuffled, list(getattr(b, field)))
        for key in fa:
            assert torch.allclose(fa[key], fb[key], atol=1e-6), (field, key)


# --------------------------------------------------------------------------- #
# A.7 S-3: the current design end to end through run_experiment.main
# --------------------------------------------------------------------------- #


def _current_design(tmp_path: Path, mode: str, **kw) -> rx.Experiment:
    return rx.Experiment(
        tag=f"s3_{mode}", mode=mode, out_dir=str(tmp_path), data="synthetic",
        synth_dates=160, synth_entities=6, synth_regime_process="markov",
        prior="markov", base_enabled=True, correction_mode=True, zero_init_head=True,
        freeze_gate=True, gate_series="sequence_channel", gate_series_channel=0,
        gate_start_vol_quantiles=(0.6,), gate_start_vol_windows=(10,),
        gate_start_persistences=(0.95,), gate_start_draws_per_centre=1,
        gate_start_min_history=5, gate_start_min_group_size=5,
        base_steps=40, base_batch_size=64, expert_hidden_dims=(8, 4), encoder_hidden=8,
        seeds=(0, 1), n_folds=2, test_dates_per_fold=15, include_ridge=False,
        include_mlp=False, backtest_quantiles=3, figures=False, calibration=False,
        **kw,
    )


@pytest.mark.parametrize("mode", ["quick", "evaluate"])
def test_current_design_runs_end_to_end(tmp_path: Path, mode: str):
    pytest.importorskip("statsmodels")
    runs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        # before any expert step the corrections are zero: improvement exactly 0
        runs["zero"] = rx.main(_current_design(tmp_path / "zero", mode, steps=0))
        runs["trained"] = rx.main(_current_design(tmp_path / "trained", mode, steps=15))
    for sub in ("zero", "trained"):
        out = runs[sub]["run_dir"]
        # the run store's layout (brief 07 B)
        assert out.parent.parent == tmp_path / sub and out.parent.name == "dev"
        for name in ("run.json", "settings.json", "status.json", "trials.jsonl",
                     "SUMMARY.md", "logs/run.log"):
            assert (out / name).exists(), name
        for folder in ("metrics", "figures", "checkpoints"):
            assert (out / folder).is_dir(), folder
        assert json.loads((out / "status.json").read_text())["state"] == "completed"
        preds = rx.RunStore().per_security_root / runs[sub]["run_id"]
        assert list(preds.rglob("*_predictions.pt")), "per-security outputs"
        rows = TrialRegistry(out / "trials.jsonl").trials()
        assert rows
        for r in rows:
            assert set(PROVENANCE_KEYS) <= set(r.config), r.tag
        strategy = [r for r in rows if r.tag == f"s3_{mode}"]
        assert strategy
        if sub == "zero":
            assert all(r.metrics["nll_improvement"] == 0.0 for r in strategy)
        assert json.loads((out / "settings.json").read_text())["prior"] == "markov"
