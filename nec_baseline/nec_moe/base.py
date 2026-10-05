"""The frozen base predictor ``f0`` of the residual design (brief 02).

    y_hat = f0(x) + sum_k pi_k(t) r_k(x)

``f0`` is an MLP on the characteristic snapshot, fitted on **each fold's
training block** and then frozen. The experts learn a regime-conditional
correction on top of it, so the thesis's headline quantity is the
*improvement over the base* rather than the level.

Why frozen (Q7): the measured quantity is what a regime-conditional
correction adds to a **fixed** baseline. A jointly optimized base would
co-adapt with the experts and any gain could no longer be attributed to
regime structure rather than to the base reorganizing itself around the
mixture. The deliberate contrast is DeepSeekMoE (arXiv:2401.06066), whose
always-on shared expert *is* optimized jointly with the routed experts: that
is the configuration most likely to win on the fitted objective, and it is out
of scope here on purpose. Attribution is bought with performance, and the
docstring says so rather than leaving it to be discovered at a defence.

Scale of the thing being split: Gu, Kelly and Xiu (2020) report a monthly
out-of-sample :math:`R^2` of about 0.4% for the benchmark network class on a
characteristic panel (0.33% for NN1, peaking at 0.40% for NN3). The base is
expected to capture most of that, so the residual left for the experts is
small by construction and a null result is a real possible outcome.

**Leakage discipline.** ``fit_base`` sees one panel — the fold's training
block — and, only when ``val_fraction > 0``, cuts an early-stopping
validation set from the *tail* of that block. No out-of-sample date is read,
and the split is chronological. The default is no tail (brief 09 B.3, audit
B-2): the base trains on the whole block for its fixed step budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from .config import BaseConfig
from .data import Panel
from .networks import MLPBlock

__all__ = ["BaseModel", "BaseFit", "fit_base", "BaseCache", "base_cache_key"]


class BaseModel(nn.Module):
    """``x_snap (B, d) -> y_hat (B,)``: the unconditional predictor.

    Sees only the characteristic snapshot — never the sequence, never the
    gate, never a regime variable. That is what makes it reusable across
    every gate arm of a window (see :class:`BaseCache`) and what makes
    "the base knows nothing about regimes" a structural fact rather than a
    claim.
    """

    def __init__(self, input_dim: int, cfg: BaseConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = MLPBlock(
            input_dim,
            cfg.hidden_dims,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )

    def forward(self, x_snap: Tensor) -> Tensor:
        return self.net(x_snap)

    def freeze(self) -> BaseModel:
        """``requires_grad_(False)`` + ``eval()`` — parameters *and* dropout.

        Both halves matter: without ``eval()`` the base would keep sampling
        dropout masks and its predictions would be a moving target even with
        its weights pinned.
        """
        self.requires_grad_(False)
        self.eval()
        return self


@dataclass(frozen=True)
class BaseFit:
    """A fitted, frozen base plus the trace needed to judge whether it is sane."""

    model: BaseModel
    train_loss: float  # final mean squared error on the fitting slice
    # best validation MSE (the early-stopping criterion); NaN when the base
    # has no validation tail (val_fraction = 0, the default)
    val_loss: float
    steps_run: int  # may be < cfg.steps when early stopping fired
    stopped_early: bool

    def metrics(self) -> dict[str, float]:
        """Registry-safe summary, prefixed so it cannot collide with the mixture's
        (``base_val_mse`` only when there was a validation tail)."""
        out = {
            "base_train_mse": self.train_loss,
            "base_steps_run": float(self.steps_run),
            "base_stopped_early": float(self.stopped_early),
        }
        if math.isfinite(self.val_loss):
            out["base_val_mse"] = self.val_loss
        return out


def fit_base(panel: Panel, cfg: BaseConfig, *, seed: int) -> BaseFit:
    """Fit ``f0`` on a training block and return it **frozen**.

    With ``val_fraction = 0`` (the default) the base trains on all of
    ``panel`` for ``cfg.steps`` steps and has no validation loss. With
    ``val_fraction > 0`` the validation set is the chronological tail of
    ``panel`` (that share of its dates); the returned model is the best
    validation checkpoint when ``early_stopping_patience`` is set, else the
    final one. ``panel`` must be a training block — this function has no way
    to know otherwise, so the caller owns that discipline (the walk-forward
    harness passes ``fold.train_dates`` only).

    **The global RNG is left exactly as it was** (audit finding B-1). The
    weights and the minibatches come from the base's own ``torch.Generator``
    seeded with ``seed`` (the run seed plus ``BaseConfig.seed_offset``, from
    :class:`BaseCache`); construction and dropout, which draw from the global
    RNG, run inside ``torch.random.fork_rng``, reseeded from ``seed``. So a run
    that fits the base and one that reuses it from the cache hand the expert
    stage the same RNG state.
    """
    if cfg.val_fraction == 0.0:
        if cfg.early_stopping_patience is not None:
            raise ValueError(
                "early_stopping_patience needs a validation tail: set "
                "val_fraction > 0, or leave patience None to train the full budget"
            )
        train, val = panel, None
    else:
        train, val = panel.split_by_date(1.0 - cfg.val_fraction)
        if len(train) == 0 or len(val) == 0:
            raise ValueError(
                f"base val_fraction={cfg.val_fraction} leaves an empty split on a "
                f"panel of {len(panel)} rows / "
                f"{int(torch.unique(panel.date).numel())} dates"
            )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)  # dropout inside the fork only
        return _fit_base(panel, train, val, cfg, seed)


def _fit_base(
    panel: Panel, train: Panel, val: Panel | None, cfg: BaseConfig, seed: int
) -> BaseFit:
    model = BaseModel(panel.x_snap.shape[1], cfg)
    model.net.reset_parameters_seeded(seed)  # the base's own generator
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    gen = torch.Generator().manual_seed(seed)
    best_val, best_state, since_best = math.inf, None, 0
    train_loss = math.nan
    steps_run, stopped_early = 0, False

    for step in range(cfg.steps):
        model.train()
        idx = torch.randint(len(train), (min(cfg.batch_size, len(train)),), generator=gen)
        loss = torch.mean((model(train.x_snap[idx]) - train.y[idx]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_loss = float(loss.detach())
        steps_run = step + 1

        if cfg.early_stopping_patience is not None and val is not None:
            model.eval()
            with torch.no_grad():
                v = float(torch.mean((model(val.x_snap) - val.y) ** 2))
            if v < best_val:
                best_val, since_best = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                since_best += 1
                if since_best >= cfg.early_stopping_patience:
                    stopped_early = True
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    val_loss = math.nan
    if val is not None:
        with torch.no_grad():
            final_val = float(torch.mean((model(val.x_snap) - val.y) ** 2))
        val_loss = min(best_val, final_val)
    return BaseFit(
        model=model.freeze(),
        train_loss=train_loss,
        val_loss=val_loss,
        steps_run=steps_run,
        stopped_early=stopped_early,
    )


def base_cache_key(
    cfg: BaseConfig, window: object, seed: int, input_dim: int
) -> tuple:
    """Every field of :class:`BaseConfig`, plus the window, seed and input width.

    Explicit rather than derived from ``id()`` or a hash of the object, so a
    config change can never silently reuse a stale base — the failure the
    amortisation below would otherwise make invisible. ``enabled`` is included
    for completeness even though a disabled base is never fitted.
    """
    return (
        cfg.enabled,
        tuple(cfg.hidden_dims),
        cfg.dropout,
        cfg.activation,
        cfg.lr,
        cfg.weight_decay,
        cfg.steps,
        cfg.batch_size,
        cfg.early_stopping_patience,
        cfg.val_fraction,
        cfg.seed_offset,
        window,
        seed,
        input_dim,
    )


class BaseCache:
    """Fit ``f0`` once per (config, window, seed); reuse it across gate arms.

    The base never sees regime information, so within one window and seed it
    is *identical* for every gate mechanism being compared — fitting it once
    per arm would burn the compute budget on recomputing the same numbers
    (the amortisation Q11's cost arithmetic assumes). Reuse is also the
    stricter comparison: every arm is then measured against literally the same
    floor, not merely a similarly-trained one.

    Not thread-safe, and deliberately in-memory: a cache that survived the
    process would need the staleness guarantees of a checkpoint format, and
    the fits are cheap enough per window not to need it.
    """

    def __init__(self) -> None:
        self._fits: dict[tuple, BaseFit] = {}
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._fits)

    def key(self, panel: Panel, cfg: BaseConfig, *, window: object, seed: int) -> tuple:
        return base_cache_key(cfg, window, seed, panel.x_snap.shape[1])

    def put(self, key: tuple, fit: BaseFit) -> None:
        """Seed the cache with a base restored from a run's checkpoint."""
        self._fits[key] = fit

    def get_or_fit(
        self, panel: Panel, cfg: BaseConfig, *, window: object, seed: int
    ) -> BaseFit:
        key = base_cache_key(cfg, window, seed, panel.x_snap.shape[1])
        hit = self._fits.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        fit = fit_base(panel, cfg, seed=seed + cfg.seed_offset)
        self._fits[key] = fit
        return fit


def save_base_fit(fit: BaseFit, key: tuple, path: str | Path) -> None:
    """Persist a fold's frozen base with its cache key (brief 07 C.2), atomically."""
    from .utils import atomic_torch_save

    atomic_torch_save(
        {
            "key": key,
            "input_dim": fit.model.net.input_dim,
            "state": fit.model.state_dict(),
            "train_loss": fit.train_loss,
            "val_loss": fit.val_loss,
            "steps_run": fit.steps_run,
            "stopped_early": fit.stopped_early,
        },
        path,
    )


def load_base_fit(path: str | Path, cfg: BaseConfig, key: tuple) -> BaseFit | None:
    """The base saved at ``path`` if its cache key equals ``key``, else None.

    Leaves the global RNG untouched (construction runs inside
    ``fork_rng``): a resumed fit has just restored its RNG state from the
    trainer checkpoint, and a module's default initialisation would shift the
    dropout stream away from the uninterrupted run's."""
    payload = torch.load(path, weights_only=False)
    if tuple(payload["key"]) != tuple(key):
        return None
    with torch.random.fork_rng(devices=[]):
        model = BaseModel(payload["input_dim"], cfg)
    model.load_state_dict(payload["state"])
    return BaseFit(
        model=model.freeze(), train_loss=payload["train_loss"],
        val_loss=payload["val_loss"], steps_run=payload["steps_run"],
        stopped_early=payload["stopped_early"],
    )


def base_config_with(cfg: BaseConfig, **overrides: object) -> BaseConfig:
    """``dataclasses.replace`` re-exported for sweeps that vary one base field."""
    return replace(cfg, **overrides)  # type: ignore[arg-type]
