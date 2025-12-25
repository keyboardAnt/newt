#!/usr/bin/env python3
"""
Migrate TD-MPC2 logs from a legacy run-first layout to the task-first layout.

Moves:
  tdmpc2/logs/<run_id>/...  ->  tdmpc2/logs/<task>/<run_id>/...

The task is inferred primarily from <run_dir>/run_info.yaml (field: 'task').

Safety defaults:
  - Dry-run by default (prints planned moves)
  - Skips "active" runs (recent heartbeat) unless --include-active is set
  - Never overwrites an existing destination directory

Example:
  python scripts/migrate_logs_to_task_first.py                 # dry-run
  python scripts/migrate_logs_to_task_first.py --apply         # execute moves
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import re


def _require_yaml():
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("pyyaml is required. Install with: pip install pyyaml") from exc
    return yaml


def _parse_iso8601(ts: str) -> Optional[datetime]:
    """Parse ISO-8601 timestamps (supports Z suffix)."""
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _sanitize_task_dir(task: str) -> str:
    # Match tdmpc2/config.py behavior: avoid path separators.
    return str(task).strip().replace("/", "-")


def _read_run_info(run_dir: Path) -> Optional[dict]:
    run_info_path = run_dir / "run_info.yaml"
    if not run_info_path.is_file():
        return None
    yaml = _require_yaml()
    try:
        info = yaml.safe_load(run_info_path.read_text()) or {}
        return info if isinstance(info, dict) else None
    except Exception:
        return None


def _infer_task(run_dir: Path, run_info: Optional[dict]) -> Optional[str]:
    if isinstance(run_info, dict):
        task = run_info.get("task")
        if isinstance(task, str) and task.strip():
            return task.strip()

    # Fallback: infer from run dir name for common expert naming.
    name = run_dir.name
    if "_expert_" in name:
        suffix = name.split("_expert_", 1)[1]
        return suffix or None
    return None


def _is_recent_heartbeat(run_dir: Path, threshold_s: float) -> bool:
    hb_path = run_dir / "heartbeat.json"
    if not hb_path.is_file():
        return False

    now = datetime.now(timezone.utc)
    try:
        hb = json.loads(hb_path.read_text())
        status = hb.get("status")
        ts_str = hb.get("timestamp")
        if isinstance(ts_str, str):
            dt = _parse_iso8601(ts_str)
            if dt is not None:
                age_s = (now - dt).total_seconds()
                # Treat any recent heartbeat as "active" regardless of status to avoid
                # racing with still-running/syncing jobs.
                return age_s <= threshold_s
        # If timestamp missing/unparseable, fall back to file mtime.
    except Exception:
        status = None  # noqa: F841 - only for debugging if needed

    try:
        age_s = time.time() - hb_path.stat().st_mtime
        return age_s <= threshold_s
    except OSError:
        return False


@dataclass(frozen=True)
class PlanItem:
    src: Path
    dst: Path
    task: str
    reason: Optional[str] = None  # if skipped


def build_plan(
    logs_dir: Path,
    *,
    include_active: bool,
    active_threshold_s: float,
) -> list[PlanItem]:
    """Plan migration for legacy run-first directories (direct children with run_info.yaml)."""
    logs_dir = logs_dir.expanduser().resolve()
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    # Heuristic: legacy run-first directories start with a timestamp prefix, e.g. 20251225_195659...
    re_run_dir = re.compile(r"^\d{8}_\d{6}")

    items: list[PlanItem] = []
    for child in sorted(logs_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if child.name in {"lsf"}:
            continue

        # Only migrate run-first directories: logs/<run_id>/...
        if not re_run_dir.match(child.name):
            continue
        if not (child / "run_info.yaml").is_file():
            continue

        run_info = _read_run_info(child)
        task = _infer_task(child, run_info)
        if not task:
            items.append(
                PlanItem(
                    src=child,
                    dst=child,
                    task="",
                    reason="skip: could not infer task (missing/invalid run_info.yaml)",
                )
            )
            continue

        if (not include_active) and _is_recent_heartbeat(child, threshold_s=active_threshold_s):
            items.append(
                PlanItem(
                    src=child,
                    dst=child,
                    task=task,
                    reason=f"skip: recent heartbeat (<= {active_threshold_s:.0f}s)",
                )
            )
            continue

        task_dir = _sanitize_task_dir(task)
        dst = logs_dir / task_dir / child.name
        if dst.exists():
            items.append(
                PlanItem(
                    src=child,
                    dst=dst,
                    task=task,
                    reason="skip: destination already exists",
                )
            )
            continue

        items.append(PlanItem(src=child, dst=dst, task=task, reason=None))

    return items


def execute_plan(items: list[PlanItem], *, dry_run: bool, verbose: bool) -> int:
    moved = 0
    skipped = 0
    errors = 0

    for it in items:
        if it.reason:
            skipped += 1
            if verbose:
                print(f"[SKIP] {it.src} -> {it.dst} ({it.reason})")
            continue

        if verbose:
            print(f"[MOVE] {it.src} -> {it.dst}")
        if dry_run:
            continue

        try:
            it.dst.parent.mkdir(parents=True, exist_ok=True)
            # Atomic on same filesystem
            it.src.rename(it.dst)
            moved += 1
        except Exception as e:
            errors += 1
            print(f"[ERROR] Failed to move {it.src} -> {it.dst}: {repr(e)}", file=sys.stderr)

    print()
    print("=== Summary ===")
    print(f"Planned: {len(items)}")
    print(f"Moved:   {moved}{' (dry-run)' if dry_run else ''}")
    print(f"Skipped: {skipped}")
    print(f"Errors:  {errors}")

    return 1 if errors else 0


def main(argv: Optional[list[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_logs_dir = repo_root / "tdmpc2" / "logs"

    p = argparse.ArgumentParser(description="Migrate tdmpc2 logs to task-first layout.")
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=default_logs_dir,
        help=f"Path to logs directory (default: {default_logs_dir})",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Execute moves (default: dry-run only).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final summary (suppresses per-run MOVE/SKIP lines).",
    )
    p.add_argument(
        "--include-active",
        action="store_true",
        help="Also migrate runs with a recent heartbeat (unsafe if jobs are running).",
    )
    p.add_argument(
        "--active-threshold-s",
        type=float,
        default=600.0,
        help="Heartbeat recency threshold (seconds) for considering a run 'active' (default: 600).",
    )

    args = p.parse_args(argv)

    items = build_plan(
        args.logs_dir,
        include_active=bool(args.include_active),
        active_threshold_s=float(args.active_threshold_s),
    )

    dry_run = not bool(args.apply)
    if dry_run:
        print("DRY-RUN mode (no changes). Use --apply to execute.\n")

    return execute_plan(items, dry_run=dry_run, verbose=not bool(args.quiet))


if __name__ == "__main__":
    raise SystemExit(main())


