"""Run the pilot (brief 09 F; Q16): choose every setting once on 2007-2009.

Reads the pilot config (``configs/pilot_grid.json`` by default): the preset
grids and the Experiment fields fixed for every run. Builds the panel as
``run_experiment.py`` does, cuts it to the dates before the first test year
(and the last ``h`` of those, whose labels reach into it), and then:

1. chooses the gate's memory by the one-step predictive log-likelihood of
   the market series on each validation year (2007, 2008, 2009), the 20 worst
   days reported separately, ties to the longer memory;
2. chooses the other settings (half-lives of the base and the experts,
   learning rate, weight decay, batch size, step budgets) by the mixture's
   regime-balanced validation loss, averaged over the three folds and the
   pilot seeds;
3. writes ``results/pilot/pilot_selection.json`` (aggregate only: the chosen
   settings, the grid, per-setting validation summaries, the step-budget
   check) and its SHA-256. Main-mode runs then need it
   (``Experiment.pilot_selection_file``).

The pilot never reads a date at or after the first test year: the panel is
sliced first, and every fold builder asserts it again.

Usage (from nec_baseline/):
    python3.14 scripts/run_pilot.py [--config configs/pilot_grid.json]
                                    [--out results/pilot] [--dry-run]

``--dry-run`` prints the grid and the number of training runs, and stops.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_experiment as rx  # noqa: E402
from nec_moe import BaseCache, pilot_folds, pilot_slice  # noqa: E402
from nec_moe.pilot import load_pilot_grid, run_pilot, write_selection  # noqa: E402
from nec_moe.runstore import data_fingerprint  # noqa: E402


def fixed_experiment(config_path: Path) -> rx.Experiment:
    """``Experiment()`` with the pilot config's fixed fields."""
    raw = json.loads(config_path.read_text())
    fixed = {k: tuple(v) if isinstance(v, list) else v
             for k, v in raw.get("experiment", {}).items()}
    return dataclasses.replace(rx.Experiment(), **fixed)


def main(config: str, out: str, dry_run: bool) -> None:
    config_path = (ROOT / config) if not Path(config).is_absolute() else Path(config)
    grid = load_pilot_grid(config_path)
    exp = fixed_experiment(config_path)
    n_settings = len(grid.settings()) if grid.search == "factorial" else (
        len(grid._optimisation_axes()) + len(grid._memory_axes()))
    n_folds = len(exp.pilot_validation_years)
    print(f"[pilot] grid from {config_path.name}: search {grid.search}, {n_settings} settings "
          f"x {n_folds} folds x {len(grid.seeds)} seeds = {n_settings * n_folds * len(grid.seeds)} "
          f"expert trainings up to {max(grid.steps):,} steps (budgets {grid.steps} read as "
          f"milestones), plus {len(grid.gate_half_lives)} x {n_folds} gate fits")
    if dry_run:
        return
    panel, purge = rx._build_panel(exp)
    fc = rx.fold_config(exp)
    sliced = pilot_slice(panel, fc.first_test_year, purge)
    first = pilot_folds(sliced, validation_years=fc.pilot_validation_years,
                        first_test_year=fc.first_test_year, purge_dates=purge)[0]
    train0 = sliced.subset_dates(first.train_dates)
    sigma = rx._auto_sigma(exp, sliced, train0, BaseCache(),
                           (int(first.train_dates.min()), int(first.train_dates.max())))

    def make_config(setting: dict):
        return rx._nec_config(dataclasses.replace(exp, **setting), sliced, sigma)

    warm = rx._warm_key(exp)
    selection = run_pilot(panel, make_config=make_config, grid=grid, fold_config=fc,
                          purge_dates=purge, warmstart_key=warm)
    selection["data_fingerprint"] = data_fingerprint(sliced)
    selection["fixed_experiment"] = json.loads(config_path.read_text()).get("experiment", {})
    selection["sigma_init"] = sigma
    path, digest = write_selection(selection, ROOT / out if not Path(out).is_absolute() else out)
    print(f"[pilot] wrote {path} (sha256 {digest})")
    print(f"[pilot] chosen: {selection['chosen']}")
    print(f"[pilot] step-budget check: {selection['step_budget_check']['note']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/pilot_grid.json")
    ap.add_argument("--out", default="results/pilot")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.config, a.out, a.dry_run)
