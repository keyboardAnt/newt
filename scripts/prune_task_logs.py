#!/usr/bin/env python3
"""
Prune tdmpc2/logs/<task>/ by keeping only one run per task.

Keeps:
  - the lexicographically latest run_id (timestamp-based)

Actions on non-kept runs (configurable):
  --delete-run-dirs            : delete whole run directories (DANGEROUS)
  --delete-checkpoints-only    : delete checkpoints/ and checkpoint_latest* symlinks
  --delete-wandb-only          : delete wandb/ directory under the run

Default is dry-run (no changes).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Optional


RUN_RE = re.compile(r"^\d{8}_\d{6}")


def _parse_step_from_name(name: str) -> Optional[int]:
    digits = name.replace("_", "").replace(",", "")
    return int(digits) if digits.isdigit() else None


def _max_ckpt_step(run_dir: Path) -> Optional[int]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    best = None
    for p in ckpt_dir.glob("*.pt"):
        if p.stem.endswith("_trainer"):
            continue
        step = _parse_step_from_name(p.stem)
        if step is None:
            continue
        best = step if best is None else max(best, step)
    return best


def _heartbeat_step(run_dir: Path) -> Optional[int]:
    hb = run_dir / "heartbeat.json"
    if not hb.is_file():
        return None
    try:
        data = json.loads(hb.read_text())
        prog = data.get("progress", {})
        if isinstance(prog, dict):
            step = prog.get("step")
            return step if isinstance(step, int) else None
    except Exception:
        return None
    return None


def _run_score(run_dir: Path) -> tuple[int, str]:
    step = _max_ckpt_step(run_dir)
    if step is None:
        step = _heartbeat_step(run_dir)
    return (step if step is not None else -1, run_dir.name)


def _iter_task_dirs(logs_dir: Path):
    for p in logs_dir.iterdir():
        if p.is_dir() and p.name != "lsf" and not RUN_RE.match(p.name):
            yield p


def _iter_run_dirs(task_dir: Path):
    for p in task_dir.iterdir():
        if p.is_dir() and RUN_RE.match(p.name):
            yield p


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "tdmpc2" / "logs"

    ap = argparse.ArgumentParser(description="Prune tdmpc2/logs/<task> runs.")
    ap.add_argument("--logs-dir", type=Path, default=logs_dir)
    ap.add_argument("--apply", action="store_true", help="Execute deletions (default: dry-run).")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--delete-run-dirs", action="store_true", help="Delete whole run dirs (dangerous).")
    g.add_argument("--delete-checkpoints-only", action="store_true", help="Delete only checkpoints artifacts.")
    g.add_argument("--delete-wandb-only", action="store_true", help="Delete only wandb/ dirs.")
    args = ap.parse_args()

    logs_dir = args.logs_dir.expanduser().resolve()
    dry = not args.apply
    if dry:
        print("DRY-RUN (no changes). Use --apply to execute.\n")

    if not (args.delete_run_dirs or args.delete_checkpoints_only or args.delete_wandb_only):
        # Sensible default: free most space without losing run metadata.
        args.delete_checkpoints_only = True

    for task_dir in sorted(_iter_task_dirs(logs_dir), key=lambda p: p.name):
        runs = list(_iter_run_dirs(task_dir))
        if len(runs) <= 1:
            continue

        keep = max(runs, key=lambda p: p.name)

        for run_dir in sorted(runs, key=lambda p: p.name):
            if run_dir == keep:
                continue

            if args.delete_run_dirs:
                print(f"[DEL-RUN] {run_dir} (keep={keep.name})")
                if not dry:
                    shutil.rmtree(run_dir, ignore_errors=False)
            elif args.delete_wandb_only:
                wdir = run_dir / "wandb"
                print(f"[DEL-WANDB] {wdir} (keep={keep.name})")
                if not dry and wdir.exists():
                    shutil.rmtree(wdir, ignore_errors=False)
            else:
                # delete checkpoints only (default)
                ckpt_dir = run_dir / "checkpoints"
                print(f"[DEL-CKPT] {ckpt_dir} (keep={keep.name})")
                if not dry:
                    if ckpt_dir.exists():
                        shutil.rmtree(ckpt_dir, ignore_errors=False)
                    # remove convenience symlinks if present
                    (run_dir / "checkpoint_latest.pt").unlink(missing_ok=True)
                    (run_dir / "checkpoint_latest_trainer.pt").unlink(missing_ok=True)

    print("\ndone")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Allow piping to head/tail without stack traces
        raise SystemExit(0)


