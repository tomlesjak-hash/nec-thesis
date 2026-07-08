import numpy as np
import pytest


def test_lgbm_smoke():
    pytest.importorskip("lightgbm")
    from op_pipeline.models import LGBMModel

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 8)).astype(np.float32)
    # planted signal: y = 0.5 * x0 - 0.3 * x1 + noise
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * rng.normal(size=2000)
    m = LGBMModel(rounds=2, num_boost_round=80, early_stopping=20)
    m.fit(X[:1500], y[:1500], X[1500:], y[1500:])
    rep = m.report(X[1500:], y[1500:])
    assert rep["ic"] > 0.5, f"LGBM IC too low: {rep['ic']}"


def test_mlp_op_smoke():
    pytest.importorskip("torch")
    from op_pipeline.models import MLPOPModel

    rng = np.random.default_rng(0)
    # X: (N, T, F)
    N, T, F = 1500, 8, 5
    X = rng.normal(size=(N, T, F)).astype(np.float32)
    # signal: last-step of first feature plus rolling mean of second
    y = X[:, -1, 0] + X[:, :, 1].mean(axis=1) + 0.1 * rng.normal(size=N)
    y = y.astype(np.float32)
    m = MLPOPModel(
        in_features=F, seq_len=T, window=4, hidden=32,
        epochs=15, batch_size=256, early_stopping=5,
    )
    m.fit(X[:1200], y[:1200], X[1200:], y[1200:])
    pred = m.predict(X[1200:])
    from scipy.stats import pearsonr
    ic = pearsonr(y[1200:], pred)[0]
    assert ic > 0.4, f"MLP-OP IC too low: {ic}"
