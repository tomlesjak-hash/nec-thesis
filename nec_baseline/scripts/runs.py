"""Look at the run store from the command line (brief 07 B.3).

Usage (from ``nec_baseline/``)::

    python3.14 scripts/runs.py list [--campaign C] [--status S]
    python3.14 scripts/runs.py show <run_id>
    python3.14 scripts/runs.py diff <run_id> <run_id>
    python3.14 scripts/runs.py latest [--campaign C] [--status S] [--tag T]

``list`` prints ``results/INDEX.csv`` as a table; ``show`` prints a run's
``SUMMARY.md`` and ``status.json``; ``diff`` the settings that differ between
two runs; ``latest`` the most recent matching run id. No command deletes
anything.

``--root`` points at another run store (default ``results/``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nec_moe import RunStore  # noqa: E402


def _store(args: argparse.Namespace) -> RunStore:
    return RunStore(args.root) if args.root else RunStore()


def cmd_list(args: argparse.Namespace) -> int:
    store = _store(args)
    rows = store.index()
    rows = [r for r in rows
            if (args.campaign is None or r["campaign"] == args.campaign)
            and (args.status is None or r["status"] == args.status)]
    if not rows:
        print("(no runs)")
        return 0
    cols = ("run_id", "campaign", "purpose", "mode", "status", "created",
            "data_source", "touches_test")
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    run = _store(args).open(args.run_id)
    if run.summary_md.exists():
        print(run.summary_md.read_text())
    else:
        print(f"(no SUMMARY.md yet: {run.dir})")
    print("status.json:")
    print(json.dumps(run.read_status(), indent=2))
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    store = _store(args)
    a, b = store.open(args.a).settings(), store.open(args.b).settings()
    keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if not keys:
        print("(identical settings)")
    for k in keys:
        print(f"{k}: {a.get(k, '<absent>')!r} -> {b.get(k, '<absent>')!r}")
    return 0


def cmd_latest(args: argparse.Namespace) -> int:
    run_id = _store(args).latest(campaign=args.campaign, status=args.status, tag=args.tag)
    if run_id is None:
        print("(no matching run)", file=sys.stderr)
        return 1
    print(run_id)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="runs.py", description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=None, help="run store root (default results/)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list")
    p.add_argument("--campaign")
    p.add_argument("--status")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("show")
    p.add_argument("run_id")
    p.set_defaults(func=cmd_show)
    p = sub.add_parser("diff")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_diff)
    p = sub.add_parser("latest")
    p.add_argument("--campaign")
    p.add_argument("--status")
    p.add_argument("--tag")
    p.set_defaults(func=cmd_latest)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
