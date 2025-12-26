#!/usr/bin/env python3
"""
Delete run directories that contain neither checkpoints nor videos.

Definition (per run directory logs/<task>/<run_id>/):
  - checkpoints: run_dir/checkpoints contains at least one agent checkpoint (*.pt not *_trainer)
  - videos: any *.mp4 under:
      - run_dir/wandb/run-*/files/media/videos/**/*.mp4
      - run_dir/videos/*.mp4

Safety:
  - Dry-run by default
  - Skips runs with a recent heartbeat.json unless --include-active
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


RUN_RE = re.compile(r"^\d{8}_\d{6}")


def _is_task_dir(p: Path) -> bool:
    return p.is_dir() and p.name not in {"lsf"} and not RUN_RE.match(p.name)


def _iter_task_dirs(logs_dir: Path) -> Iterable[Path]:
    for p in logs_dir.iterdir():
        if _is_task_dir(p):
            yield p


def _iter_run_dirs(task_dir: Path) -> Iterable[Path]:
    for p in task_dir.iterdir():
        if p.is_dir() and RUN_RE.match(p.name):
            yield p


def _has_checkpoints(run_dir: Path) -> bool:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return False
    for p in ckpt_dir.glob("*.pt"):
        if p.stem.endswith("_trainer"):
            continue
        return True
    return False


def _has_videos(run_dir: Path) -> bool:
    # W&B media
    for p in run_dir.glob("wandb/run-*/files/media/videos/**/*.mp4"):
        if p.is_file():
            return True
    # Direct saves
    for p in (run_dir / "videos").glob("*.mp4"):
        if p.is_file():
            return True
    return False


def _parse_iso8601(ts: str) -> datetime | None:
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _is_recent_heartbeat(run_dir: Path, threshold_s: float) -> bool:
    hb_path = run_dir / "heartbeat.json"
    if not hb_path.is_file():
        return False
    now = datetime.now(timezone.utc)
    try:
        hb = json.loads(hb_path.read_text())
        ts_str = hb.get("timestamp")
        if isinstance(ts_str, str):
            dt = _parse_iso8601(ts_str)
            if dt is not None:
                return (now - dt).total_seconds() <= threshold_s
    except Exception:
        pass
    try:
        return (time.time() - hb_path.stat().st_mtime) <= threshold_s
    except OSError:
        return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_logs_dir = repo_root / "tdmpc2" / "logs"

    ap = argparse.ArgumentParser(description="Prune run dirs with neither checkpoints nor videos.")
    ap.add_argument("--logs-dir", type=Path, default=default_logs_dir)
    ap.add_argument("--apply", action="store_true", help="Actually delete run dirs (default: dry-run).")
    ap.add_argument("--include-active", action="store_true", help="Also delete runs with recent heartbeat (unsafe).")
    ap.add_argument("--active-threshold-s", type=float, default=600.0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logs_dir = args.logs_dir.expanduser().resolve()
    dry = not args.apply
    deleted = 0
    kept = 0
    skipped_active = 0

    if dry and not args.quiet:
        print("DRY-RUN (no changes). Use --apply to delete.\n")

    for task_dir in sorted(_iter_task_dirs(logs_dir), key=lambda p: p.name):
        for run_dir in sorted(_iter_run_dirs(task_dir), key=lambda p: p.name):
            if (not args.include_active) and _is_recent_heartbeat(run_dir, args.active_threshold_s):
                skipped_active += 1
                continue

            has_ckpt = _has_checkpoints(run_dir)
            has_vid = _has_videos(run_dir)
            if has_ckpt or has_vid:
                kept += 1
                continue

            deleted += 1
            if not args.quiet:
                print(f"[DEL]{' (dry-run)' if dry else ''} {run_dir}")
            if not dry:
                shutil.rmtree(run_dir, ignore_errors=False)

    print("\n=== Summary ===")
    print(f"dry_run:       {dry}")
    print(f"deleted_runs:  {deleted}")
    print(f"kept_runs:     {kept}")
    print(f"skipped_active:{skipped_active}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)


