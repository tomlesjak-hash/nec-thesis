"""The run store: one storage layout for every run (brief 07 section B).

::

    nec_baseline/results/
      INDEX.csv                   one row per run, appended at launch, updated at exit
      <campaign>/<run_id>/        run_id = YYYYMMDD-HHMMSS_<tag>_<8-char settings hash>
        run.json                  identity: campaign, tag, purpose, mode, created,
                                  git commit + dirty flag, data source and fingerprint,
                                  package versions, device, touches_test
        settings.json             the full settings snapshot
        status.json               state, current arm/seed/fold/step, heartbeat, exit reason
        trials.jsonl              the trial registry
        metrics/                  aggregate results (json and csv)
        figures/
        logs/run.log              everything printed, timestamped (gitignored)
        checkpoints/              resume state (gitignored)
        SUMMARY.md                written at exit
    Data/derived/runs/<run_id>/   per-security outputs (licensed; gitignored)

**Licence rule.** ``results/`` holds aggregate outputs only: settings,
per-fold and pooled summaries, date-level series without per-security values,
figures, logs. Every per-security output (predictions, corrections, per-stock
residuals or responsibilities) goes to :attr:`Run.per_security_dir` under
``Data/derived/runs/``, never under ``results/``.

``touches_test`` is true for a run that computed a test-block metric on a real
panel; it feeds the adaptive-overfitting accounting of Q15 and is recorded,
never enforced. The data fingerprint (shape, date range, entity set, target
checksum, CRSP release) and the settings hash let a resume refuse to continue
on changed data or settings (section C).
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, TextIO

__all__ = [
    "RESULTS_DIR",
    "PER_SECURITY_DIR",
    "INDEX_COLUMNS",
    "RunStore",
    "Run",
    "settings_hash",
    "data_fingerprint",
    "atomic_write_text",
]

#: ``nec_baseline/results/``: aggregate outputs only.
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
#: ``Quant Model/Data/derived/runs/``: per-security outputs (licensed, gitignored).
PER_SECURITY_DIR = Path(__file__).resolve().parents[2] / "Data" / "derived" / "runs"

INDEX_COLUMNS: tuple[str, ...] = (
    "run_id", "campaign", "tag", "purpose", "mode", "created", "finished", "status",
    "exit_reason", "git_commit", "dirty", "data_source", "touches_test",
    "settings_hash", "data_fingerprint", "path",
)
#: settings that do not change what a run computes, so they are left out of
#: the hash (asking to resume must not look like a different run)
VOLATILE_SETTINGS: frozenset[str] = frozenset({"resume", "force"})


def atomic_write_text(path: Path, text: str) -> None:
    """Write to a temporary file and rename it into place: a crash mid-write
    never leaves a half-written file (brief 07 C.4)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def settings_hash(settings: Mapping[str, Any]) -> str:
    """8 hex characters of SHA-256 over the canonical JSON of the settings.

    Stable across processes (no Python ``hash()``, keys sorted), and any
    change to a setting that affects the computation changes it; the
    volatile keys (``resume``, ``force``) are left out.
    """
    kept = {k: v for k, v in settings.items() if k not in VOLATILE_SETTINGS}
    return hashlib.sha256(_canonical(kept).encode()).hexdigest()[:8]


def data_fingerprint(panel: Any) -> str:
    """Hash of the panel's shape, date range, entity set and target checksum,
    plus its data source and CRSP release (16 hex characters)."""
    dates = panel.date_labels
    entities = panel.entity_labels
    codes = sorted({int(e) for e in panel.entity.tolist()})
    parts = {
        "shape": [list(panel.x_seq.shape), list(panel.x_snap.shape)],
        "target": panel.schema.target,
        "dates": (
            [dates[int(panel.date.min())], dates[int(panel.date.max())]]
            if dates else [int(panel.date.min()), int(panel.date.max())]
        ),
        "n_dates": int(panel.date.unique().numel()),
        "entities": hashlib.sha256(
            _canonical([entities[c] for c in codes] if entities else codes).encode()
        ).hexdigest(),
        "target_checksum": hashlib.sha256(
            panel.y.detach().contiguous().numpy().tobytes()
        ).hexdigest(),
        "data_source": getattr(panel, "data_source", "unspecified"),
        "crsp_release": (getattr(panel, "metadata", None) or {}).get("crsp_release"),
    }
    return hashlib.sha256(_canonical(parts).encode()).hexdigest()[:16]


def _git_state(repo: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "."],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False
    return commit, dirty


def _versions() -> dict[str, str]:
    import numpy
    import pandas
    import torch

    out = {"python": platform.python_version(), "torch": torch.__version__,
           "numpy": numpy.__version__, "pandas": pandas.__version__}
    try:
        import statsmodels

        out["statsmodels"] = statsmodels.__version__
    except ImportError:  # pragma: no cover - optional
        pass
    return out


def _now() -> dt.datetime:
    return dt.datetime.now().astimezone()


class _Tee(io.TextIOBase):
    """Write to the original stream and to the run log, each line timestamped."""

    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self.stream, self.log, self._at_line_start = stream, log, True

    def write(self, text: str) -> int:
        self.stream.write(text)
        for chunk in text.splitlines(keepends=True):
            if self._at_line_start:
                self.log.write(_now().strftime("%Y-%m-%d %H:%M:%S "))
            self.log.write(chunk)
            self._at_line_start = chunk.endswith("\n")
        self.log.flush()
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()


@dataclasses.dataclass
class Run:
    """One run's folders and files; see the module docstring for the layout."""

    run_id: str
    campaign: str
    dir: Path
    per_security_dir: Path
    store: RunStore

    # ------------------------------------------------------------ paths
    @property
    def run_json(self) -> Path:
        return self.dir / "run.json"

    @property
    def settings_json(self) -> Path:
        return self.dir / "settings.json"

    @property
    def status_json(self) -> Path:
        return self.dir / "status.json"

    @property
    def trials(self) -> Path:
        return self.dir / "trials.jsonl"

    @property
    def metrics_dir(self) -> Path:
        return self.dir / "metrics"

    @property
    def figures_dir(self) -> Path:
        return self.dir / "figures"

    @property
    def log_path(self) -> Path:
        return self.dir / "logs" / "run.log"

    @property
    def checkpoints_dir(self) -> Path:
        return self.dir / "checkpoints"

    @property
    def summary_md(self) -> Path:
        return self.dir / "SUMMARY.md"

    # ------------------------------------------------------------- state
    def info(self) -> dict[str, Any]:
        return json.loads(self.run_json.read_text())

    def settings(self) -> dict[str, Any]:
        return json.loads(self.settings_json.read_text())

    def read_status(self) -> dict[str, Any]:
        return json.loads(self.status_json.read_text()) if self.status_json.exists() else {}

    def status(self, **fields: Any) -> dict[str, Any]:
        """Merge ``fields`` into ``status.json`` and stamp the heartbeat."""
        current = self.read_status() | fields
        current["heartbeat"] = _now().isoformat(timespec="seconds")
        atomic_write_text(self.status_json, json.dumps(current, indent=2, default=str))
        return current

    def record(self, **fields: Any) -> None:
        """Merge ``fields`` into ``run.json`` (e.g. a forced resume)."""
        atomic_write_text(self.run_json, json.dumps(self.info() | fields, indent=2, default=str))

    def write_metrics(self, name: str, data: Mapping[str, Any] | Any) -> Path:
        """An aggregate result as ``metrics/<name>.json`` (a mapping) or
        ``metrics/<name>.csv`` (a DataFrame)."""
        if hasattr(data, "to_csv"):
            path = self.metrics_dir / f"{name}.csv"
            atomic_write_text(path, data.to_csv())
        else:
            path = self.metrics_dir / f"{name}.json"
            atomic_write_text(path, json.dumps(data, indent=2, default=str))
        return path

    @contextlib.contextmanager
    def logging(self) -> Iterator[None]:
        """Everything printed inside, timestamped, also into ``logs/run.log``."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as log:
            out, err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = _Tee(out, log), _Tee(err, log)
            try:
                yield
            finally:
                sys.stdout, sys.stderr = out, err

    def finish(
        self, state: str, exit_reason: str = "", metrics: Any = None
    ) -> None:
        """Final status, ``SUMMARY.md``, and the run's ``INDEX.csv`` row."""
        finished = _now().isoformat(timespec="seconds")
        self.status(state=state, exit_reason=exit_reason, finished=finished)
        atomic_write_text(self.summary_md, self._summary(state, exit_reason, finished, metrics))
        self.store.update_index(
            self.run_id, status=state, finished=finished, exit_reason=exit_reason
        )

    def _summary(self, state: str, reason: str, finished: str, metrics: Any) -> str:
        info, settings = self.info(), self.settings()
        defaults = info.get("default_settings", {})
        lines = [f"# Run {self.run_id}", ""]
        for key in ("campaign", "tag", "purpose", "mode", "created", "git_commit",
                    "dirty", "data_source", "data_fingerprint", "settings_hash",
                    "touches_test"):
            lines.append(f"- **{key}**: {info.get(key)}")
        lines += [f"- **status**: {state}" + (f" ({reason})" if reason else ""),
                  f"- **finished**: {finished}", "",
                  "## Settings that differ from the defaults", ""]
        diff = [(k, v, defaults.get(k)) for k, v in sorted(settings.items())
                if k in defaults and defaults[k] != v]
        if diff:
            lines += ["| setting | value | default |", "|---|---|---|"]
            lines += [f"| {k} | {v} | {d} |" for k, v, d in diff]
        else:
            lines.append("(none)")
        lines += ["", "## Metrics", ""]
        if metrics is None:
            lines.append("(none)")
        elif hasattr(metrics, "to_markdown"):
            try:
                lines.append(metrics.to_markdown())
            except ImportError:  # tabulate not installed
                lines.append("```\n" + metrics.to_string() + "\n```")
        else:
            lines += ["| metric | value |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in sorted(dict(metrics).items())]
        return "\n".join(lines) + "\n"


class RunStore:
    """Creates, finds and indexes runs under ``root`` (default ``results/``)."""

    def __init__(
        self, root: str | Path | None = None, per_security_root: str | Path | None = None
    ) -> None:
        # defaults resolved at call time, so tests can redirect the module's
        # constants and never touch the real results/ or Data/
        self.root = Path(root) if root is not None else RESULTS_DIR
        self.per_security_root = (
            Path(per_security_root) if per_security_root is not None else PER_SECURITY_DIR
        )

    @property
    def index_path(self) -> Path:
        return self.root / "INDEX.csv"

    # ------------------------------------------------------------ index
    def index(self) -> list[dict[str, str]]:
        if not self.index_path.exists():
            return []
        with self.index_path.open(newline="") as fh:
            return list(csv.DictReader(fh))

    def _write_index(self, rows: list[dict[str, Any]]) -> None:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(INDEX_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_text(self.index_path, buf.getvalue())

    def append_index(self, row: Mapping[str, Any]) -> None:
        self._write_index([*self.index(), {c: row.get(c, "") for c in INDEX_COLUMNS}])

    def update_index(self, run_id: str, **fields: Any) -> None:
        rows = self.index()
        for row in rows:
            if row["run_id"] == run_id:
                row.update({k: str(v) for k, v in fields.items()})
        self._write_index(rows)

    # ------------------------------------------------------------- runs
    def create(
        self,
        *,
        campaign: str,
        tag: str,
        mode: str,
        settings: Mapping[str, Any],
        purpose: str = "",
        data_source: str = "unspecified",
        fingerprint: str = "",
        touches_test: bool = False,
        default_settings: Mapping[str, Any] | None = None,
        repo: str | Path | None = None,
    ) -> Run:
        """Create the run's folders and identity files, append its index row."""
        created = _now()
        digest = settings_hash(settings)
        run_id = f"{created:%Y%m%d-%H%M%S}_{tag}_{digest}"
        run_dir = self.root / campaign / run_id
        n = 1
        while run_dir.exists():  # two launches in one second with equal settings
            n += 1
            run_dir = self.root / campaign / f"{run_id}-{n}"
        run_id = run_dir.name
        for sub in ("metrics", "figures", "logs", "checkpoints"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        commit, dirty = _git_state(Path(repo) if repo else Path(__file__).resolve().parents[1])
        info = {
            "run_id": run_id, "campaign": campaign, "tag": tag, "purpose": purpose,
            "mode": mode, "created": created.isoformat(timespec="seconds"),
            "git_commit": commit, "dirty": dirty, "data_source": data_source,
            "data_fingerprint": fingerprint, "settings_hash": digest,
            "touches_test": touches_test, "versions": _versions(), "device": "cpu",
            "per_security_dir": str(self.per_security_root / run_id),
            "default_settings": dict(default_settings or {}),
        }
        atomic_write_text(run_dir / "run.json", json.dumps(info, indent=2, default=str))
        atomic_write_text(run_dir / "settings.json", json.dumps(dict(settings), indent=2,
                                                               default=str))
        run = Run(run_id, campaign, run_dir, self.per_security_root / run_id, self)
        run.status(state="running", exit_reason="")
        self.append_index({
            **info, "status": "running", "finished": "", "exit_reason": "",
            "path": str(run_dir.relative_to(self.root)),
        })
        return run

    def open(self, run_id: str) -> Run:
        for row in self.index():
            if row["run_id"] == run_id:
                run_dir = self.root / row["path"]
                return Run(run_id, row["campaign"], run_dir,
                           self.per_security_root / run_id, self)
        raise KeyError(f"no run {run_id!r} in {self.index_path}")

    def latest(
        self, *, campaign: str | None = None, status: str | None = None,
        tag: str | None = None,
    ) -> str | None:
        """The most recent run id matching the filters: by creation time, ties
        (one-second resolution) broken by index order, which is launch order."""
        rows = [(i, r) for i, r in enumerate(self.index())
                if (campaign is None or r["campaign"] == campaign)
                and (tag is None or r["tag"] == tag)
                and (status is None or r["status"] == status)]
        if not rows:
            return None
        return max(rows, key=lambda ir: (ir[1]["created"], ir[0]))[1]["run_id"]
