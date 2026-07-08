"""End-to-end smoke test (design doc §12.4).

The "actually trains and predicts" gate: the default-config model must train on
synthetic regime data, recover the hidden regime, and produce sensible
predictions — before any real data is plugged in.
"""

from __future__ import annotations

import torch

from conftest import small_config
from nec_moe import (
    NECModel,
    SyntheticRegimePanel,
    SyntheticSpec,
    Trainer,
    set_seed,
)
from nec_moe.diagnostics import regime_recovery_auc


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def test_smoke_train_synthetic():
    set_seed(0)
    spec = SyntheticSpec(
        regime_process="iid",
        vol_levels=(0.5, 2.5),  # clearly separable regimes for the gate
        beta_scale=2.0,
        noise_std=0.4,
        seed=0,
    )
    panel = SyntheticRegimePanel(spec).generate(n_dates=300, n_entities=8)
    train, test = panel.split_by_date(0.8)

    cfg = small_config(sigma_init=1.5, lr=3e-3, batch_size=256, sigma_freeze_steps=30)
    model = NECModel(cfg)
    trainer = Trainer(model)

    # 1. initial state sanity: zero-init gate starts exactly uniform (entropy log K)
    model.eval()
    with torch.no_grad():
        init_out = model(test.x_seq, test.x_snap)
    from nec_moe.diagnostics import gate_entropy

    assert abs(
        float(gate_entropy(init_out.prior.log_prior.exp()))
        - torch.log(torch.tensor(2.0)).item()
    ) < 1e-5
    init_nll = trainer.evaluate(test.full_batch())

    # 2. break the symmetric-mixture local optimum on vol-sorted slices
    #    (design doc §4.3). The proxy is the observable channel-0 window vol —
    #    the same regime signal VIX/realized-vol would carry on real data. The
    #    gate is NOT supervised: it must still learn to route afterwards.
    proxy = train.x_seq[:, :, 0].std(dim=1)
    trainer.warmstart_experts(train.full_batch(), sort_key=proxy)

    history = trainer.fit(train, steps=400)
    final_nll = trainer.evaluate(test.full_batch())

    # training reduces held-out NLL by a clear margin
    assert final_nll < init_nll - 0.1, f"init={init_nll:.3f} final={final_nll:.3f}"

    # 3. regime recovery: gate probability separates the latent regime
    model.eval()
    with torch.no_grad():
        out = model(test.x_seq, test.x_snap)
    auc = regime_recovery_auc(out.prior.log_prior.exp()[:, 0], test.regime)
    assert auc > 0.8, f"regime recovery AUC {auc:.3f}"

    # 4. prediction sanity: mixture mean tracks the noiseless conditional mean
    corr = _pearson(out.y_hat, test.y_clean)
    assert corr > 0.5, f"prediction/clean-target correlation {corr:.3f}"

    # 5. monitoring sanity: no dead expert over the run (running-scope alarm)
    assert min(h["min_running_utilization"] for h in history[-50:]) > 0.02


def test_smoke_broken_gradient_path_fails_regime_recovery():
    """Guard-rail: an accidental detach of the gate (the argmax pathology) must
    break regime recovery — proving item 3 above actually has teeth."""

    class BrokenModel(NECModel):
        def forward(self, x_seq, x_snap, ctx=None):
            out = super().forward(x_seq, x_snap, ctx)
            out.prior.log_prior = out.prior.log_prior.detach()  # kill gate learning
            return out

    set_seed(0)
    spec = SyntheticSpec(vol_levels=(0.5, 2.5), noise_std=0.4, seed=0)
    panel = SyntheticRegimePanel(spec).generate(n_dates=300, n_entities=8)
    train, test = panel.split_by_date(0.8)

    model = BrokenModel(small_config(sigma_init=1.5, lr=3e-3, sigma_freeze_steps=30))
    trainer = Trainer(model)
    # even with the expert warm-start, a detached gate cannot learn to route:
    # the experts specialize but the gate never discovers which regime is which
    proxy = train.x_seq[:, :, 0].std(dim=1)
    trainer.warmstart_experts(train.full_batch(), sort_key=proxy)
    trainer.fit(train, steps=400)

    model.eval()
    with torch.no_grad():
        out = model(test.x_seq, test.x_snap)
    auc = regime_recovery_auc(out.prior.log_prior.exp()[:, 0], test.regime)
    assert auc < 0.65, f"detached gate should not recover regimes, got AUC {auc:.3f}"
