# OP Pipeline — runnable quant pipeline built from the OP model framework

A testable, end-to-end factor-prediction pipeline distilled from the OP model
folder (`../OP model/`). The OP folder contains nine separate reconstructed
model files; this project takes the *pipeline structure* common to all of them
(data → features → labels → model → eval → backtest) and makes it actually
run, on synthetic data, with two interchangeable predictors.

## What this is

A clean reimplementation of the OP framework that:

* Generates a realistic synthetic multi-asset OHLCV dataset so the pipeline
  runs without proprietary data dependencies (`dw_data.fastpai`, the
  `/home/jupyterData/...` paths, the truncated feature lists).
* Builds rolling factor features in the spirit of the MLP-OP rolling operators.
* Constructs OP-style labels: continuous forward returns and signed-bucket
  classification labels with downsampling and ± balancing (the LGBM file's
  `create_labels` / `balance_labels` / `downsample_zero_label` logic).
* Trains two models behind a common `BaseModel` interface:
  * `LGBMModel` — gradient-boosted trees, with the OP file's incremental
    training pattern (rounds, learning-rate decay, 20% replay buffer).
  * `MLPOPModel` — PyTorch MLP with rolling-stat operator layers ported from
    `rolling_feature_mlp_explained.py`. The obvious bugs from the original
    (dropout applied to the wrong tensor, unused operators) are fixed.
* Evaluates predictions: RMSE, Pearson IC, Spearman rank-IC, hit rate, and a
  decile long-short backtest with simple turnover/cost accounting.

## Layout

```
OP pipeline/
  op_pipeline/
    __init__.py
    config.py
    data.py            # synthetic multi-asset OHLCV
    features.py        # rolling factor construction
    labels.py          # OP-style label bucketing + balancing
    models/
      __init__.py
      base.py          # BaseModel interface
      lgbm.py          # LGBM with incremental rounds
      mlp_op.py        # rolling-operator MLP (PyTorch)
    train.py           # train/valid splits + training loops
    evaluate.py        # IC, RMSE, hit rate, rank-IC
    backtest.py        # decile long-short with costs
  scripts/
    run_pipeline.py    # demo: data → train → eval → backtest, both models
  tests/
    test_data.py
    test_features.py
    test_labels.py
    test_models.py
    test_pipeline.py   # end-to-end smoke test
  requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py             # end-to-end demo
pytest tests/                              # all tests
```

## What this is NOT

* Not a strategy. The synthetic data has a planted signal so the pipeline
  *can* learn something — this exists to validate plumbing, not to claim alpha.
* Not a backtester with realistic microstructure. Costs are flat bps; no slippage
  model, no borrow costs, no capacity.
* Not a faithful one-to-one port of the OP files — the data layer is replaced
  with synthetic, and several obvious bugs in the originals are fixed (noted
  inline).

## How this connects to the thesis ideas

This is the substrate for the regime-conditional MoE thesis (Idea 1 in
`thesis_candidates.md`). With this pipeline running, the next step is to add a
`MoEModel` that gates between the LGBM and MLP-OP predictors — that's a
swap-in at the `models/` level, not a rewrite.
