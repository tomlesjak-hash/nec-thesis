"""Single-model baselines + the thesis's headline comparison, planted.

Ridge and the single MLP are correct on their own terms (exact-recovery and
convergence tests), and the comparison logic is validated both ways:

- on **sign-flip regime data** the pooled cross-sectional relationship averages
  to zero, so single models are *structurally* blind and the NEC must win big;
- on **regime-free data** ridge nails the single true linear relationship —
  the baselines are strong where they should be strong, which is what makes
  losing to the MoE on regime data mean something.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
from conftest import small_config

from nec_moe import (
    MLPBaseline,
    NECModel,
    RidgeBaseline,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    walk_forward_evaluate,
    walk_forward_evaluate_baseline,
)

# --------------------------------------------------------------------------- #
# Ridge
# --------------------------------------------------------------------------- #


def _regime_free_panel(seed: int = 3, n_dates: int = 160):
    """marginal=(1,0): every date sits in regime 0 -> one true linear map."""
    spec = SyntheticSpec(
        marginal=(1.0, 0.0),
        vol_levels=(1.0, 1.0),
        beta_scale=2.0,
        noise_std=0.3,
        seed=seed,
    )
    return SyntheticRegimePanel(spec).generate(n_dates=n_dates, n_entities=8)


def test_ridge_exact_recovery_on_regime_free_data():
    panel = _regime_free_panel()
    train, test = panel.split_by_date(0.8)
    ridge = RidgeBaseline(l2=1e-6)
    ridge.fit(train)
    pred = ridge.predict(test)
    # noiseless conditional mean recovered almost exactly
    corr = torch.corrcoef(torch.stack([pred, test.y_clean]))[0, 1]
    assert float(corr) > 0.999
    assert math.isfinite(ridge.nll(test))


def test_ridge_shrinkage_and_unpenalized_intercept():
    panel = _regime_free_panel()
    train, test = panel.split_by_date(0.8)
    heavy = RidgeBaseline(l2=1e6)
    heavy.fit(train)
    # slope coefficients crushed toward zero...
    assert float(heavy.predict(test).std()) < 0.05
    # ...but the intercept is untouched: a constant shift survives any lambda
    shifted_train = dataclasses.replace(train, y=train.y + 5.0)
    heavy.fit(shifted_train)
    assert float(heavy.predict(test).mean()) == pytest.approx(
        float(shifted_train.y.mean()), abs=0.05
    )
    with pytest.raises(ValueError, match="l2"):
        RidgeBaseline(l2=-1.0)


def test_ridge_requires_fit():
    with pytest.raises(RuntimeError, match="fit"):
        RidgeBaseline().predict(_regime_free_panel(n_dates=20))


# --------------------------------------------------------------------------- #
# Single MLP
# --------------------------------------------------------------------------- #


def test_mlp_baseline_learns_and_is_deterministic():
    panel = _regime_free_panel()
    train, test = panel.split_by_date(0.8)
    mlp = MLPBaseline(input_dim=3, hidden_dims=(16, 8), dropout=0.0, steps=800, seed=0)
    mlp.fit(train)
    corr = torch.corrcoef(torch.stack([mlp.predict(test), test.y_clean]))[0, 1]
    assert float(corr) > 0.9
    # deterministic given the seed
    mlp2 = MLPBaseline(input_dim=3, hidden_dims=(16, 8), dropout=0.0, steps=800, seed=0)
    mlp2.fit(train)
    assert torch.allclose(mlp.predict(test), mlp2.predict(test))
    # sigma moves the right way (Adam walks log-sigma slowly from init=1.0
    # toward the truth, noise_std=0.3) and the fit beats an intercept-only
    # null model on held-out density
    assert float(mlp._log_sigma.exp()) < 0.9
    null = RidgeBaseline(l2=1e6)  # ~intercept-only under heavy shrinkage
    null.fit(train)
    assert mlp.nll(test) < null.nll(test)


# --------------------------------------------------------------------------- #
# The headline comparison, planted both ways
# --------------------------------------------------------------------------- #


def _regime_panel(seed: int = 3):
    spec = SyntheticSpec(
        vol_levels=(0.5, 2.5), beta_scale=2.0, noise_std=0.4, seed=seed
    )
    return SyntheticRegimePanel(spec).generate(n_dates=160, n_entities=8)


def test_moe_beats_single_models_where_regimes_are_real():
    """Sign-flip regimes: the pooled fit is ~zero by construction, so the
    single-model baselines are structurally blind while the NEC routes."""
    panel = _regime_panel()
    common = dict(n_folds=2, test_dates_per_fold=20, purge_dates=5)

    def make_trainer() -> Trainer:
        torch.manual_seed(0)
        return Trainer(NECModel(small_config(sigma_init=1.5, lr=3e-3, batch_size=256)))

    nec = walk_forward_evaluate(
        panel, make_trainer, steps=200,
        warmstart_key=lambda p: p.x_seq[:, :, 0].std(dim=1), **common,
    )
    ridge = walk_forward_evaluate_baseline(
        panel, lambda: RidgeBaseline(l2=1.0), **common
    )
    mlp = walk_forward_evaluate_baseline(
        panel,
        lambda: MLPBaseline(input_dim=3, hidden_dims=(16, 8), dropout=0.0, steps=200),
        **common,
    )
    assert nec.pooled_ic.mean_ic > 0.3, nec.pooled_ic
    assert abs(ridge.pooled_ic.mean_ic) < 0.15, ridge.pooled_ic
    assert abs(mlp.pooled_ic.mean_ic) < 0.15, mlp.pooled_ic
    # the same code path graded all three: baseline folds carry no expert order
    assert ridge.folds[0].expert_order == ()
    assert nec.folds[0].expert_order != ()


def test_ridge_wins_where_regimes_are_absent():
    """Regime-free data: one true linear map — ridge is the right model and
    must score, proving the baselines lose above for structural reasons, not
    because they are broken."""
    panel = _regime_free_panel()
    ridge = walk_forward_evaluate_baseline(
        panel, lambda: RidgeBaseline(l2=1.0),
        n_folds=2, test_dates_per_fold=20, purge_dates=5,
        backtest_quantiles=4, cost_rate=0.001,
    )
    assert ridge.pooled_ic.mean_ic > 0.8, ridge.pooled_ic
    assert ridge.pooled_portfolio is not None
    assert ridge.pooled_portfolio.mean_net > 0
