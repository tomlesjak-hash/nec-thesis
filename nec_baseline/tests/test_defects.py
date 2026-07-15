"""Defect-regression tests (design doc §12.1).

Every ledger defect gets a test that would have **failed on the prototype**
(`OP model/nec/nec_hybrid_explained.py`).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn as nn
from conftest import D_SEQ, D_SNAP, SEQ_LEN, random_inputs, small_config

from nec_moe import (
    Batch,
    DataConfig,
    EncoderConfig,
    FeatureSchema,
    NECConfig,
    NECModel,
    PriorConfig,
    TrainConfig,
    expert_log_likelihood,
    mixture_nll,
)
from nec_moe.config import ExpertConfig, GateConfig
from nec_moe.gate import GateHead


def _loss(model: NECModel, x_seq, x_snap, y):
    out = model(x_seq, x_snap)
    log_lik = expert_log_likelihood(out.mu, out.log_sigma, y)
    return mixture_nll(out.prior.log_prior, log_lik), out


def _nll_value(model: NECModel, x_seq, x_snap, y) -> float:
    with torch.no_grad():
        return float(_loss(model, x_seq, x_snap, y)[0].nll)


# --------------------------------------------------------------------- defect 1
def test_gate_receives_gradient():
    """Soft routing sends gradient to the gate; hard argmax provably does not."""
    model = NECModel(small_config())
    x_seq, x_snap, y = random_inputs()

    nll_out, _ = _loss(model, x_seq, x_snap, y)
    nll_out.nll.backward()
    grad = model.gate.linear.weight.grad
    assert grad is not None and grad.abs().max() > 0

    # Companion: the prototype's pathology as a test double. One-hot routing
    # built from argmax has no graph edge to the gate -> zero gradient.
    model.zero_grad(set_to_none=True)
    out = model(x_seq, x_snap)
    idx = out.gate_logits.argmax(dim=-1)  # argmax: gradient-dead
    hard_log_prior = torch.full_like(out.gate_logits, float("-inf"))
    hard_log_prior.scatter_(1, idx.unsqueeze(1), 0.0)
    log_lik = expert_log_likelihood(out.mu, out.log_sigma, y)
    mixture_nll(hard_log_prior, log_lik).nll.backward()
    assert model.gate.linear.weight.grad is None or torch.all(
        model.gate.linear.weight.grad == 0
    )


# --------------------------------------------------------------------- defect 2
@pytest.mark.parametrize("k", [2, 3, 5])
def test_all_experts_enter_loss(k: int):
    """No expert is collapsed out of the objective, for any configured K."""
    model = NECModel(small_config(n_experts=k))
    x_seq, x_snap, y = random_inputs()
    base = _nll_value(model, x_seq, x_snap, y)
    for j in range(k):
        with torch.no_grad():
            model.experts.experts[j].l3.bias.add_(1.0)
        perturbed = _nll_value(model, x_seq, x_snap, y)
        assert perturbed != pytest.approx(base, abs=1e-6), f"expert {j} inert"
        with torch.no_grad():
            model.experts.experts[j].l3.bias.sub_(1.0)


# --------------------------------------------------------------------- defect 3
def test_model_emits_logits_not_probs():
    model = NECModel(small_config())
    assert not any(isinstance(m, nn.Softmax) for m in model.modules())

    with torch.no_grad():  # break the zero init so logits are nontrivial
        model.gate.linear.weight.normal_(std=1.0)
        model.gate.linear.bias.normal_(std=1.0)
    x_seq, x_snap, _ = random_inputs()
    z = model(x_seq, x_snap).gate_logits
    assert not torch.allclose(z.sum(dim=-1), torch.ones(len(z)), atol=1e-3)
    assert (z < 0).any(), "logits should be unconstrained, not probabilities"

    # fused path == explicit softmax reference on benign values
    log_lik = torch.randn(16, 2)
    fused = mixture_nll(torch.log_softmax(z, dim=-1), log_lik)
    explicit = (torch.softmax(z, dim=-1) * log_lik.exp()).sum(-1).log().neg()
    assert torch.allclose(fused.per_sample_nll, explicit, atol=1e-5)


# --------------------------------------------------------------------- defect 4
def test_no_cross_batch_coupling():
    """Train-mode output of sample i is independent of the rest of the batch."""
    model = NECModel(small_config())  # expert dropout 0 in test config
    model.train()
    x_seq, x_snap, _ = random_inputs(b=8)
    ref = model(x_seq, x_snap)

    x_seq2, x_snap2 = x_seq.clone(), x_snap.clone()
    x_seq2[1:] = torch.randn_like(x_seq2[1:])
    x_snap2[1:] = torch.randn_like(x_snap2[1:])
    alt = model(x_seq2, x_snap2)

    for name in ("gate_logits", "mu", "y_hat"):
        a, b = getattr(ref, name)[0], getattr(alt, name)[0]
        assert torch.allclose(a, b, atol=1e-6), f"{name} coupled across batch"

    # BatchNorm provably fails the same probe (why the prototype's head leaked)
    bn = nn.BatchNorm1d(4).train()
    x = torch.randn(8, 4)
    x2 = x.clone()
    x2[1:] = torch.randn_like(x2[1:])
    assert not torch.allclose(bn(x)[0], bn(x2)[0], atol=1e-6)


# --------------------------------------------------------------------- defect 5
def test_default_capacity():
    cfg = NECConfig(data=DataConfig(d_seq=10, d_snap=10, seq_len=20))
    model = NECModel(cfg)
    assert isinstance(model.encoder.gru, nn.GRU)
    assert cfg.encoder.num_layers <= 2 and cfg.encoder.hidden_dim <= 64
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 100_000, f"{n_params} params — prototype was ~4M"


# --------------------------------------------------------------------- defect 6
def test_expert_input_modes():
    for mode, extra in (("snapshot", 0), ("snapshot_plus_hidden", 16)):
        cfg = small_config(input_mode=mode)
        assert cfg.expert_input_dim == D_SNAP + extra
        model = NECModel(cfg)
        x_seq, x_snap, _ = random_inputs()
        assert model(x_seq, x_snap).y_hat.shape == (16,)
    # wrong snapshot width fails loudly
    model = NECModel(small_config())
    x_seq, _, _ = random_inputs()
    with pytest.raises(ValueError, match="x_snap"):
        model(x_seq, torch.randn(16, D_SNAP + 1))


# --------------------------------------------------------------------- defect 7
def test_feature_contract():
    model = NECModel(small_config())
    x_seq, x_snap, _ = random_inputs()
    with pytest.raises(ValueError, match="x_seq"):
        model(torch.randn(16, SEQ_LEN, D_SEQ + 1), x_snap)

    # a target column smuggled into the snapshot tensor fails the schema check
    schema = FeatureSchema(
        sequence_features=tuple(f"s{i}" for i in range(D_SEQ)),
        snapshot_features=tuple(f"f{i}" for i in range(D_SNAP)),
        target="y",
    )
    smuggled = Batch(x_seq, torch.randn(16, D_SNAP + 1), torch.randn(16))
    with pytest.raises(ValueError, match="x_snap width"):
        smuggled.validate_schema(schema)
    Batch(x_seq, x_snap, torch.randn(16)).validate_schema(schema)  # clean passes


# --------------------------------------------------------------------- defect 8
def test_config_mismatch_fails_loudly():
    # the prototype's bug is unrepresentable: the gate has no independent dim
    assert not any("dim" in f.name for f in dataclasses.fields(GateConfig))

    # manually mis-wired modules die at forward with both numbers in the message
    gate = GateHead(hidden_dim=64, n_experts=3, cfg=GateConfig())
    with pytest.raises(ValueError, match=r"64.*16|16.*64"):
        gate(torch.randn(4, 16))

    data = DataConfig(d_seq=D_SEQ, d_snap=D_SNAP, seq_len=SEQ_LEN)
    with pytest.raises(ValueError, match="n_experts"):
        NECConfig(data=data, experts=ExpertConfig(n_experts=1)).validate()
    with pytest.raises(ValueError, match="dropout"):
        NECConfig(
            data=data, encoder=EncoderConfig(num_layers=1, dropout=0.3)
        ).validate()
    with pytest.raises(ValueError, match="sequence_ordered"):
        NECConfig(data=data, prior=PriorConfig(kind="hmm")).validate()
    with pytest.raises(NotImplementedError, match="tvtp"):
        NECConfig(
            data=data,
            prior=PriorConfig(kind="hmm", tvtp=True),
            train=TrainConfig(sequence_ordered=True),
        ).validate()
    with pytest.raises(ValueError, match="prior.kind"):
        NECConfig(data=data, prior=PriorConfig(kind="nope")).validate()
