"""The pilot (Q16; brief 09 F): choose every setting once, on 2007-2009 only.

The main study's settings (the three memories, the learning rate, the weight
decay, the batch size, the step budgets) are chosen once, on the validation
years 2007, 2008 and 2009 (training blocks 2000-2006, 2000-2007, 2000-2008),
then frozen and pre-registered (Q15). The pilot never touches a date at or
after the first test year: :func:`run_pilot` slices its panel with
:func:`~nec_moe.evaluation.pilot_slice`, and every fold builder asserts it
again on the panel it receives (B.2).

**Gate memory first, and separately** (F.2). Each candidate half-life is
scored by the gate's own job: the one-step predictive log-likelihood of the
market series on each validation year, ``sum_t log p(m_t | F_{t-1})`` with
the filter run forward through the year at frozen parameters (Nystrup,
Madsen & Lindström 2017), summed over the three folds. The 20 worst days'
contribution is reported separately (those authors' finding holds only
without them), and ties go to the longer memory.

**Then everything else**, on the mixture's **regime-balanced validation
loss**: each validation row's NLL weighted by ``v(s) = sum_k wbar_k(s) /
M_k``, ``wbar`` the window-average gate weight (Q27) and ``M_k`` its mean
over the validation rows, so every regime contributes equal total weight;
averaged over the three folds and the pilot seeds. The search is the full
grid of the pilot config (``search = "factorial"``, the default), or the
documented two-stage alternative (``"staged"``: the optimisation settings at
equal weights first, then the half-lives at the chosen ones). Step budgets
are read at milestones of one training run: with a constant learning rate,
training to 3,000 steps and then on to 10,000 is exactly a 10,000-step run,
so each budget costs nothing extra.

**Training length check.** The best step budget of the chosen setting is
recorded per fold; if it differs by more than ``step_drift_factor`` across
the folds the selection file says so. The fallback (regime-balanced early
stopping) is Tom's call and is not implemented.

**Freezing** (F.3). :func:`write_selection` writes ``pilot_selection.json``
(aggregate: the chosen settings, the grid, per-setting validation summaries)
and its SHA-256. Main-mode runs refuse to start without it, record its hash
in every trial row, and refuse settings that differ from it unless
``force_deviation`` (recorded) - see ``run_experiment.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .base import BaseCache
from .config import FoldConfig, NECConfig
from .data import Panel
from .evaluation import (
    _attach_base,
    _check_protocol_base,
    _gate_training_block,
    assert_pre_test_panel,
    expert_stage_seed,
    fold_decay_weights,
    gate_span_dates,
    pilot_folds,
    pilot_slice,
)
from .markov_gate import MarkovSwitchingRegimePrior, gate_weight_probs
from .model import NECModel
from .train import Trainer

__all__ = [
    "PilotGrid",
    "load_pilot_grid",
    "regime_balanced_loss",
    "gate_memory_selection",
    "run_pilot",
    "write_selection",
    "load_selection",
    "selection_deviations",
    "encode_settings",
    "decode_settings",
    "SELECTION_FILE",
]

#: The selection file's name inside the pilot's output folder.
SELECTION_FILE = "pilot_selection.json"
SEARCHES: tuple[str, ...] = ("factorial", "staged")


def _inf(x: float | None) -> float:
    return math.inf if x is None else float(x)


@dataclass(frozen=True)
class PilotGrid:
    """The pilot's preset grids (brief 09 F.1); every value a documented
    default that the pilot config file can override.

    Half-lives: ``None`` is infinity (equal weights; always in the grid as
    the reference arm of Gu, Kelly & Xiu's undecayed protocol). The experts'
    and the gate's are given as the **calendar years** of the stress regime
    they correspond to and turned into regime-days by
    ``H = stress_share * years * trading_days_per_year``: a regime that holds
    a share ``p`` of days has a calendar half-life of about ``H / p`` days
    (theory notes section 5). ``stress_share = 0.3`` is the share of stress
    days of the 2-state gate on the CRSP S&P 500 index 2015-2024 quoted there;
    10 and 5 years then give 756 and 378 regime-days. The gate's grid is on
    the same scale as the experts'.

    ``lr``, ``weight_decay`` and ``batch_size`` apply to the experts
    (``TrainConfig``); ``steps`` are the experts' step budgets, read as
    milestones of one run; ``base_steps`` are the base's (``BaseConfig.steps``,
    one fit per value). ``seeds`` are the pilot seeds.
    """

    base_half_life_days: tuple[float | None, ...] = (None, 2520.0, 1260.0)
    expert_half_life_years: tuple[float | None, ...] = (None, 10.0, 5.0)
    gate_half_life_years: tuple[float | None, ...] = (None, 10.0, 5.0)
    stress_share: float = 0.3
    trading_days_per_year: int = 252
    lr: tuple[float, ...] = (3e-4, 1e-3)
    weight_decay: tuple[float, ...] = (1e-5, 1e-4)
    batch_size: tuple[int, ...] = (1024, 4096)
    steps: tuple[int, ...] = (3000, 10000, 30000)
    base_steps: tuple[int, ...] = (1000,)
    seeds: tuple[int, ...] = (0, 1, 2)
    search: str = "factorial"
    gate_worst_days: int = 20
    gate_tie_tol: float = 1e-6  # nats: totals this close are a tie
    step_drift_factor: float = 2.0

    def validate(self) -> PilotGrid:
        if self.search not in SEARCHES:
            raise ValueError(f"unknown search {self.search!r}; use one of {SEARCHES}")
        for name in ("lr", "weight_decay", "batch_size", "steps", "base_steps", "seeds",
                     "base_half_life_days", "expert_half_life_years", "gate_half_life_years"):
            if not getattr(self, name):
                raise ValueError(f"pilot grid axis {name!r} is empty")
        if list(self.steps) != sorted(set(self.steps)) or min(self.steps) < 1:
            raise ValueError(f"steps must be distinct, increasing and >= 1, got {self.steps}")
        if not 0 < self.stress_share <= 1 or self.trading_days_per_year < 1:
            raise ValueError("stress_share must be in (0, 1] and trading_days_per_year >= 1")
        if self.gate_worst_days < 0 or self.gate_tie_tol < 0 or self.step_drift_factor < 1:
            raise ValueError("gate_worst_days, gate_tie_tol >= 0 and step_drift_factor >= 1")
        return self

    def regime_days(self, years: float | None) -> float:
        if years is None:
            return math.inf
        return self.stress_share * years * self.trading_days_per_year

    @property
    def gate_half_lives(self) -> tuple[float, ...]:
        return tuple(self.regime_days(y) for y in self.gate_half_life_years)

    @property
    def expert_half_lives(self) -> tuple[float, ...]:
        return tuple(self.regime_days(y) for y in self.expert_half_life_years)

    def _optimisation_axes(self) -> list[dict[str, Any]]:
        return [
            {"lr": lr, "weight_decay": wd, "batch_size": bs, "base_steps": b}
            for lr, wd, bs, b in itertools.product(
                self.lr, self.weight_decay, self.batch_size, self.base_steps
            )
        ]

    def _memory_axes(self) -> list[dict[str, Any]]:
        return [
            {
                "base_decay_half_life_days": hb,
                "expert_decay": "none" if math.isinf(he) else "regime_clock",
                "expert_decay_half_life": he,
            }
            for hb, he in itertools.product(self.base_half_life_days, self.expert_half_lives)
        ]

    def settings(self) -> list[dict[str, Any]]:
        """Every setting of the factorial grid, as ``Experiment`` fields
        (without ``steps``, which are milestones)."""
        return [m | o for m in self._memory_axes() for o in self._optimisation_axes()]


def load_pilot_grid(path: str | Path) -> PilotGrid:
    """A :class:`PilotGrid` from the pilot config file (JSON; lists for the
    axes, ``null`` for an infinite half-life). Unknown keys are refused."""
    raw = json.loads(Path(path).read_text())
    grid_raw = raw.get("grid", raw)
    names = {f.name for f in dataclasses.fields(PilotGrid)}
    unknown = sorted(k for k in grid_raw if k not in names and not k.startswith("_"))
    if unknown:
        raise ValueError(f"unknown pilot grid key(s) {unknown}; known: {sorted(names)}")
    kw: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v
                          for k, v in grid_raw.items() if not k.startswith("_")}
    return PilotGrid(**kw).validate()


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def regime_balanced_loss(per_row: Tensor | np.ndarray, wbar: np.ndarray) -> float:
    """``sum_s v(s) l(s) / sum_s v(s)`` with ``v(s) = sum_k wbar_k(s) / M_k``,
    ``M_k`` the mean of ``wbar_k`` over the rows (F.2): every regime carries
    the same total weight, so a calm-dominated validation year cannot hide a
    poor stress fit. A regime with no weight at all is refused, not dropped."""
    loss = np.asarray(per_row.detach().double() if isinstance(per_row, Tensor) else per_row,
                      dtype=float)
    w = np.asarray(wbar, dtype=float)
    if w.shape != (len(loss), w.shape[1]):
        raise ValueError(f"wbar must be (N, K) with N={len(loss)}, got {w.shape}")
    m = w.mean(axis=0)
    if np.any(m <= 0):
        raise ValueError(
            f"regime(s) {np.flatnonzero(m <= 0).tolist()} carry no gate weight on the "
            "validation rows: the regime-balanced loss is undefined"
        )
    v = (w / m).sum(axis=1)
    return float(np.sum(v * loss) / np.sum(v))


def _window_weights(prior: MarkovSwitchingRegimePrior, dates: Tensor) -> np.ndarray:
    """``wbar`` per date: the window average of the 1..h-step regime
    probabilities (Q27), from the frozen filter, whatever ``gate_weight``
    the model serves."""
    assert prior.fit_result is not None and prior.horizon is not None
    return gate_weight_probs(prior.filtered_probabilities(dates),
                             prior.fit_result.transition, "window", prior.horizon)


@torch.no_grad()
def _validation_loss(trainer: Trainer, val: Panel) -> float:
    """The regime-balanced NLL of the validation rows under the model's
    predictive weights. Run in a forked RNG, so a milestone evaluation
    cannot shift the training stream (the next budget stays exact)."""
    prior = trainer.model.prior
    if not isinstance(prior, MarkovSwitchingRegimePrior):
        raise ValueError("the pilot's regime-balanced loss needs the fitted Hamilton gate")
    with torch.random.fork_rng(devices=[]):
        was = trainer.model.training
        trainer.model.eval()
        try:
            _, nll = trainer._forward_nll(val.full_batch(), train_objective=False)
        finally:
            trainer.model.train(was)
    dates, inverse = torch.unique(val.date, sorted=True, return_inverse=True)
    wbar = _window_weights(prior, dates)[inverse.numpy()]
    return regime_balanced_loss(nll.per_sample_nll, wbar)


# --------------------------------------------------------------------------- #
# Stage 1: the gate's memory
# --------------------------------------------------------------------------- #


def gate_memory_selection(
    panel: Panel,
    folds: list,
    base_cfg: NECConfig,
    grid: PilotGrid,
) -> tuple[dict[str, Any], dict[tuple[int, float], dict]]:
    """Score every gate half-life by the one-step predictive log-likelihood
    of the market series over the validation years (F.2).

    Every candidate is fitted with the native estimator (the regime-clock
    memory needs it; the infinite half-life is the same estimator at full
    memory, so the candidates differ only in memory). For each fold the gate
    is fitted on the training block and its frozen filter run over every date
    from the first training date to the end of the validation year, the purge
    gap included (Q22), exactly the dates the main harness gives the gate;
    only the validation dates' one-step log densities are summed. Returns the
    report and the fitted gate states, keyed by (fold, half-life), for reuse
    in stage 2.
    """
    k = base_cfg.experts.n_experts
    horizon = base_cfg.data.horizon_periods
    results: list[dict[str, Any]] = []
    states: dict[tuple[int, float], dict] = {}
    for h in grid.gate_half_lives:
        mg = dataclasses.replace(
            base_cfg.markov_gate, fit_backend="native",
            memory="full" if math.isinf(h) else "regime_clock", gate_half_life=h,
        )
        per_fold, contributions, settled = [], [], []
        for fold in folds:
            train = panel.subset_dates(fold.train_dates)
            gate = MarkovSwitchingRegimePrior(k, mg, horizon=horizon)
            gate.fit(train)
            # the frozen filter over the contiguous span, the purge gap
            # included (Q22); only the validation dates are scored below
            scored = panel.subset_dates(gate_span_dates(panel, fold))
            gate.apply_causal(scored)
            dates, log_c = gate.predictive_log_density(scored)
            val = log_c[torch.isin(dates, fold.test_dates).numpy()]
            per_fold.append(float(val.sum()))
            contributions.append(val)
            fit = gate.fit_result
            assert fit is not None
            settled.append(fit.weighted_settled)
            states[(fold.fold, h)] = gate.gate_state()
        all_days = np.concatenate(contributions)
        worst = np.sort(all_days)[: grid.gate_worst_days]
        total = float(all_days.sum())
        results.append({
            "half_life": h,
            "memory": "full" if math.isinf(h) else "regime_clock",
            "predictive_loglik": total,
            "worst_days": int(len(worst)),
            "worst_days_loglik": float(worst.sum()),
            "excluding_worst_days": total - float(worst.sum()),
            "per_fold": per_fold,
            "validation_days": int(len(all_days)),
            "weighted_pass_settled": settled,
        })
    totals = [float(r["predictive_loglik"]) for r in results]
    best = max(totals)
    ties = [r for r, t in zip(results, totals, strict=True) if best - t <= grid.gate_tie_tol]
    chosen = max(ties, key=lambda r: float(r["half_life"]))  # ties: the longer memory
    report = {
        "criterion": "one-step predictive log-likelihood of the market series, summed over "
                     "the validation years (frozen filter)",
        "results": results,
        "chosen_half_life": chosen["half_life"],
        "tie_tolerance": grid.gate_tie_tol,
    }
    return report, states


# --------------------------------------------------------------------------- #
# Stage 2: everything else
# --------------------------------------------------------------------------- #


def _train_and_score(
    panel: Panel,
    fold,
    cfg: NECConfig,
    seed: int,
    gate_state: dict,
    base_cache: BaseCache,
    budgets: tuple[int, ...],
    warmstart_key: Callable[[Panel], Tensor] | None,
) -> list[float]:
    """One (setting, fold, seed): the frozen gate (restored from stage 1),
    the base, then the experts trained to each step budget in turn, the
    regime-balanced validation loss read at each. The order is the main
    harness's (gate, base, expert stage seed, gate block, warm-start, decay
    weights, training)."""
    torch.manual_seed(seed)
    seeded = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, seed=seed))
    trainer = Trainer(NECModel(seeded))
    _check_protocol_base(trainer.cfg, [fold])
    train = panel.subset_dates(fold.train_dates)
    val = panel.subset_dates(fold.test_dates)
    trainer.model.prior.load_gate_state(gate_state)  # type: ignore[operator]
    _attach_base(trainer, train, fold, base_cache, seed)
    torch.manual_seed(expert_stage_seed(seed, fold.fold))
    expert_train, _ = _gate_training_block(trainer, train)
    x = trainer.cfg.experts
    if warmstart_key is not None and not (x.correction_mode and x.zero_init_head):
        trainer.warmstart_experts(expert_train.full_batch(), warmstart_key(expert_train))
    row_weight, _ = fold_decay_weights(trainer, train, expert_train)
    losses, done = [], 0
    for budget in budgets:
        trainer.fit(expert_train, steps=budget - done, row_weight=row_weight)
        done = budget
        losses.append(_validation_loss(trainer, val))
    return losses


def _summaries(
    settings: list[dict[str, Any]], scores: dict[int, np.ndarray], budgets: tuple[int, ...]
) -> list[dict[str, Any]]:
    """Per (setting, budget): the loss averaged over folds and seeds, its
    std over the fold-seed runs, and the per-fold means. ``scores[i]`` is
    ``(folds, seeds, budgets)``."""
    out = []
    for i, setting in enumerate(settings):
        arr = scores[i]
        for b, budget in enumerate(budgets):
            runs = arr[:, :, b].ravel()
            out.append({
                "setting": setting,
                "steps": budget,
                "mean_loss": float(runs.mean()),
                "std_loss": float(runs.std(ddof=1)) if runs.size > 1 else 0.0,
                "per_fold": [float(v) for v in arr[:, :, b].mean(axis=1)],
            })
    return out


def run_pilot(
    panel: Panel,
    *,
    make_config: Callable[[dict[str, Any]], NECConfig],
    grid: PilotGrid,
    fold_config: FoldConfig,
    purge_dates: int,
    warmstart_key: Callable[[Panel], Tensor] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the pilot on ``panel`` and return the selection (aggregate only).

    ``make_config(setting)`` builds the model config of a setting (a dict of
    ``run_experiment.Experiment`` fields over the fixed ones). The panel is
    first cut to the pre-test slice (:func:`~nec_moe.evaluation.pilot_slice`),
    and :func:`~nec_moe.evaluation.assert_pre_test_panel` is asserted on it.
    """
    grid = grid.validate()
    fold_config = fold_config.validate()
    first = fold_config.first_test_year
    sliced = pilot_slice(panel, first, purge_dates)
    assert_pre_test_panel(sliced, first)
    folds = pilot_folds(sliced, validation_years=fold_config.pilot_validation_years,
                        first_test_year=first, purge_dates=purge_dates)
    base_cfg = make_config({})
    if base_cfg.prior.kind != "markov":
        raise ValueError("the pilot chooses the Hamilton gate's memory: set prior='markov'")

    def say(msg: str) -> None:
        if verbose:
            print(f"[pilot] {msg}")

    say(f"{len(folds)} folds, validation years {fold_config.pilot_validation_years}; "
        f"gate half-lives {grid.gate_half_lives}")
    gate_report, gate_states = gate_memory_selection(sliced, folds, base_cfg, grid)
    h_gate = gate_report["chosen_half_life"]
    gate_choice = {
        "gate_fit_backend": "native",
        "gate_memory": "full" if math.isinf(h_gate) else "regime_clock",
        "gate_half_life": h_gate,
    }
    say(f"gate memory chosen: {gate_choice}")

    budgets = tuple(grid.steps)
    base_cache = BaseCache()

    def evaluate(settings: list[dict[str, Any]]) -> dict[int, np.ndarray]:
        scores = {}
        for i, setting in enumerate(settings):
            cfg = make_config(gate_choice | setting | {"steps": budgets[-1]})
            arr = np.empty((len(folds), len(grid.seeds), len(budgets)))
            for f, fold in enumerate(folds):
                for s, seed in enumerate(grid.seeds):
                    arr[f, s] = _train_and_score(
                        sliced, fold, cfg, seed, gate_states[(fold.fold, h_gate)],
                        base_cache, budgets, warmstart_key,
                    )
            scores[i] = arr
            say(f"setting {i + 1}/{len(settings)} done")
        return scores

    if grid.search == "factorial":
        settings = grid.settings()
        scores = evaluate(settings)
        summary = _summaries(settings, scores, budgets)
    else:  # staged: optimisation settings at equal weights, then the memories
        equal = {"base_decay_half_life_days": None, "expert_decay": "none",
                 "expert_decay_half_life": math.inf}
        stage_a = [equal | o for o in grid._optimisation_axes()]
        sum_a = _summaries(stage_a, evaluate(stage_a), budgets)
        best_a = min(sum_a, key=lambda r: r["mean_loss"])
        opt = {k: best_a["setting"][k] for k in ("lr", "weight_decay", "batch_size",
                                                 "base_steps")}
        settings = [m | opt for m in grid._memory_axes()]
        scores = evaluate(settings)
        summary = sum_a + _summaries(settings, scores, budgets)

    best = min(summary, key=lambda r: r["mean_loss"])
    i_best = next(i for i, s in enumerate(settings) if s == best["setting"])
    per_fold_curves = scores[i_best].mean(axis=1)  # (folds, budgets)
    best_steps = [budgets[int(np.argmin(curve))] for curve in per_fold_curves]
    ratio = max(best_steps) / min(best_steps)
    step_check = {
        "best_steps_per_fold": dict(zip(
            [str(y) for y in fold_config.pilot_validation_years], best_steps, strict=True)),
        "max_over_min": ratio,
        "factor": grid.step_drift_factor,
        "stable": ratio <= grid.step_drift_factor,
        "note": ("the best step budget differs by more than the factor across the pilot "
                 "folds; the fallback (regime-balanced early stopping) is Tom's decision"
                 if ratio > grid.step_drift_factor else "within the factor"),
    }
    chosen = gate_choice | best["setting"] | {"steps": best["steps"]}
    say(f"chosen: {chosen}; step budget check: {step_check['note']}")
    return {
        "format": 1,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "first_test_year": first,
        "validation_years": list(fold_config.pilot_validation_years),
        "purge_dates": purge_dates,
        "pilot_dates": [sliced.date_labels[int(sliced.date.min())],  # type: ignore[index]
                        sliced.date_labels[int(sliced.date.max())]],  # type: ignore[index]
        "data_source": panel.data_source,
        "grid": dataclasses.asdict(grid),
        "gate_memory": gate_report,
        "settings": summary,
        "chosen": chosen,
        "step_budget_check": step_check,
    }


# --------------------------------------------------------------------------- #
# The selection file (the pre-registration lock, F.3)
# --------------------------------------------------------------------------- #


def encode_settings(obj: Any) -> Any:
    """Strict-JSON form: infinite floats become ``"inf"`` (JSON has none)."""
    if isinstance(obj, float) and math.isinf(obj):
        return "inf" if obj > 0 else "-inf"
    if isinstance(obj, dict):
        return {k: encode_settings(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [encode_settings(v) for v in obj]
    return obj


def decode_settings(obj: Any) -> Any:
    if obj == "inf":
        return math.inf
    if obj == "-inf":
        return -math.inf
    if isinstance(obj, dict):
        return {k: decode_settings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_settings(v) for v in obj]
    return obj


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_selection(selection: dict[str, Any], out_dir: str | Path) -> tuple[Path, str]:
    """Write ``pilot_selection.json`` (strict JSON) and
    ``pilot_selection.json.sha256``; return the path and the hash."""
    from .runstore import atomic_write_text

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / SELECTION_FILE
    text = json.dumps(encode_settings(selection), indent=2, allow_nan=False, sort_keys=True)
    atomic_write_text(path, text + "\n")
    digest = _sha256(path)
    atomic_write_text(path.with_name(SELECTION_FILE + ".sha256"), f"{digest}  {SELECTION_FILE}\n")
    return path, digest


def load_selection(path: str | Path) -> tuple[dict[str, Any], str]:
    """``(selection, sha256 of the file's bytes)``: the hash is recomputed
    from the file, never read from the sidecar."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no pilot selection file at {p}")
    return decode_settings(json.loads(p.read_text())), _sha256(p)


def selection_deviations(
    selection: dict[str, Any], settings: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """``{field: (chosen, given)}`` for every chosen setting that ``settings``
    (an ``Experiment`` as a dict) sets differently."""
    out = {}
    for key, chosen in selection["chosen"].items():
        given = settings.get(key, _MISSING)
        if given is _MISSING or not _same(given, chosen):
            out[key] = (chosen, None if given is _MISSING else given)
    return out


_MISSING = object()


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float | int) and isinstance(b, float | int):
        return float(a) == float(b)
    return bool(a == b)
