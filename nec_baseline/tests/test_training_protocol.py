"""Brief 09 E: fixed step budget on every day's rows, fresh starts every fold,
seeds paired across arms. Synthetic data only.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import pytest
import torch
from conftest import small_base_config, small_config

from nec_moe import (
    BaseCache,
    NECModel,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    TrialRegistry,
    nec_arm,
    run_sweep,
    walk_forward_evaluate,
    walk_forward_folds,
)
from nec_moe.train import DeadParameterWarning

PURGE = 3


def _panel():
    return SyntheticRegimePanel(SyntheticSpec(
        regime_process="markov", transition_stay=0.96, vol_levels=(0.5, 2.5), seed=5,
    )).generate(160, 6)


def _markov_cfg(prior_kind: str = "markov", **train):
    cfg = small_config(prior_kind=prior_kind, correction_mode=True, base=small_base_config(),
                       sigma_init=1.0, lr=3e-3, batch_size=32, freeze_gate=True,
                       expert_dropout=0.1, **train)
    return dataclasses.replace(cfg, markov_gate=dataclasses.replace(
        cfg.markov_gate, series="sequence_channel", series_channel=0,
        start_vol_quantiles=(0.6,), start_vol_windows=(10,), start_persistences=(0.95,),
        start_draws_per_centre=1, start_min_history=5, start_min_group_size=5,
    ))


def _run(panel, folds, factory, cache=None, seed=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        return walk_forward_evaluate(
            panel, factory, folds=folds, purge_dates=PURGE, steps=8, seed=seed,
            base_cache=cache if cache is not None else BaseCache(),
        )


def test_every_fold_starts_from_fresh_weights():
    """E.2: the weights fold f starts from do not depend on fold f-1's
    trained ones: fold 1 alone and fold 1 after fold 0 start (and end)
    identically, even with a factory that does not seed anything itself."""
    pytest.importorskip("statsmodels")
    panel = _panel()
    folds = walk_forward_folds(panel.date, n_folds=2, test_dates_per_fold=15,
                               purge_dates=PURGE)
    cfg = _markov_cfg()

    def recording(store: list):
        def make():
            t = Trainer(NECModel(cfg))
            store.append({k: v.clone() for k, v in t.model.state_dict().items()})
            return t
        return make

    both: list[dict] = []
    alone: list[dict] = []
    res_both = _run(panel, folds, recording(both))
    torch.manual_seed(12345)  # whatever ran before must not matter
    torch.randn(1000)
    res_alone = _run(panel, folds[1:], recording(alone))
    assert len(both) == 2 and len(alone) == 1
    for key, value in alone[0].items():
        assert torch.equal(value, both[1][key]), key
    assert res_alone.folds[0] == res_both.folds[1]  # the whole fold reproduces


def test_seeds_are_paired_across_arms():
    """E.3: two arms (the Hamilton gate and the uniform no-regime control)
    on the same seed and fold see the same minibatch indices, the same base
    and the same initial experts; another seed changes all three."""
    pytest.importorskip("statsmodels")
    panel = _panel()
    folds = walk_forward_folds(panel.date, n_folds=1, test_dates_per_fold=15,
                               purge_dates=PURGE)
    seen: dict[str, dict] = {}
    original = Trainer._next_fit_index

    def run_arm(name: str, prior_kind: str, seed: int):
        record = {"batches": [], "experts": None, "base": None}

        def spy(self, n_rows):
            idx = original(self, n_rows)
            record["batches"].append(idx.clone())
            return idx

        arm = nec_arm(name, _markov_cfg(prior_kind))
        assert arm.build_trainer is not None

        def make():
            t = arm.build_trainer(seed)
            record["experts"] = {k: v.clone() for k, v in t.model.experts.state_dict().items()}
            return t

        cache = BaseCache()  # separate caches: the seed, not sharing, fixes the base
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Trainer, "_next_fit_index", spy)
            _run(panel, folds, make, cache=cache, seed=seed)
        (fit,) = cache._fits.values()
        record["base"] = {k: v.clone() for k, v in fit.model.state_dict().items()}
        seen[f"{name}_{seed}"] = record

    run_arm("markov", "markov", 0)
    run_arm("control", "uniform", 0)
    run_arm("markov", "markov", 1)
    a, b, c = seen["markov_0"], seen["control_0"], seen["markov_1"]
    assert len(a["batches"]) == len(b["batches"]) == 8
    assert all(torch.equal(x, y) for x, y in zip(a["batches"], b["batches"], strict=True))
    for key in a["base"]:
        assert torch.equal(a["base"][key], b["base"][key]), key
    for key in a["experts"]:
        assert torch.equal(a["experts"][key], b["experts"][key]), key
    # a different seed is a different draw of all three
    assert not torch.equal(a["batches"][0], c["batches"][0])
    assert any(not torch.equal(a["base"][k], c["base"][k]) for k in a["base"])
    assert any(not torch.equal(a["experts"][k], c["experts"][k]) for k in a["experts"])


def test_the_experts_train_on_every_day_of_the_block():
    """E.1: nothing is thinned. With a gate covering every training date (an
    order-0 Hamilton gate), the experts' training panel is the whole
    training block, and the budget is exactly ``steps``."""
    pytest.importorskip("statsmodels")
    panel = _panel()
    folds = walk_forward_folds(panel.date, n_folds=1, test_dates_per_fold=15,
                               purge_dates=PURGE)
    got = {}
    original = Trainer.fit

    def spy(self, data, steps=None, checkpoint_path=None, row_weight=None):
        got["dates"], got["rows"], got["steps"] = torch.unique(data.date), len(data), steps
        return original(self, data, steps, checkpoint_path, row_weight)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Trainer, "fit", spy)
        _run(panel, folds, lambda: Trainer(NECModel(_markov_cfg())))
    train = panel.subset_dates(folds[0].train_dates)
    assert torch.equal(got["dates"], torch.unique(train.date)) and got["rows"] == len(train)
    assert got["steps"] == 8


def test_the_registry_records_the_seed_of_every_run(tmp_path: Path):
    pytest.importorskip("statsmodels")
    panel = _panel()
    reg = TrialRegistry(tmp_path / "t.jsonl")
    arms = [nec_arm("markov", _markov_cfg()), nec_arm("control", _markov_cfg("uniform"))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeadParameterWarning)
        report = run_sweep(panel, arms, seeds=(0, 1), registry=reg, tag="paired", steps=3,
                           n_folds=1, test_dates_per_fold=15, purge_dates=PURGE,
                           verbose=False)
    rows = reg.trials("paired")
    assert sorted((r.config["arm"], r.seed) for r in rows) == [
        ("control", 0), ("control", 1), ("markov", 0), ("markov", 1)]
    # a metric only some seeds report (an empty Q24 bin under each seed's own
    # canonical expert order) is summarised over those seeds, with the count
    # shown, instead of failing the summary or vanishing
    frame = report.to_frame()
    for arm in report.arms:
        for key, n in arm.n_reporting.items():
            assert 1 <= n < arm.n_seeds and frame.loc[arm.arm, f"{key}_n_seeds"] == n
    union = set().union(*(r.metrics for r in rows if r.config["arm"] == "markov"))
    assert union <= set(report.arms[0].mean)
