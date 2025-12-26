#!/usr/bin/env python3
"""
Prune evaluation videos under tdmpc2/logs/ so only the latest video per task is kept.

What it does
------------
For each task directory logs/<task>/:
  - Keep only the lexicographically latest run_id (timestamp-based).
  - Delete all *.mp4 videos from other run directories.
  - In the kept run directory, keep only the newest eval video (highest step if parseable),
    delete older eval videos.

Video locations handled
-----------------------
Newer layout (wandb media):
  logs/<task>/<run_id>/wandb/run-*/files/media/videos/**/*.mp4

Legacy layout (older, huge):
  logs/<task>/1/**/eval_video/**/*.mp4

Safety
------
Dry-run by default. Use --apply to actually delete files.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple


RUN_RE = re.compile(r"^\d{8}_\d{6}")
EVAL_VIDEO_STEP_RE = re.compile(r"(?:^|/)eval_video_(\d+)_", re.IGNORECASE)


def _atomic_print(s: str) -> None:
    # Keep prints robust under piping
    try:
        print(s, flush=True)
    except BrokenPipeError:
        raise SystemExit(0)


def _is_task_dir(p: Path) -> bool:
    return p.is_dir() and p.name not in {"lsf"} and not RUN_RE.match(p.name)


def _iter_task_dirs(logs_dir: Path) -> Iterable[Path]:
    for p in logs_dir.iterdir():
        if _is_task_dir(p):
            yield p


def _iter_run_dirs(task_dir: Path) -> list[Path]:
    runs = []
    for p in task_dir.iterdir():
        if p.is_dir() and RUN_RE.match(p.name):
            runs.append(p)
    return sorted(runs, key=lambda p: p.name)


def _resolve_symlink(task_dir: Path, name: str) -> Optional[Path]:
    p = task_dir / name
    if not p.is_symlink():
        return None
    try:
        return (task_dir / os.readlink(p)).resolve()
    except OSError:
        return None


def _pick_kept_run(task_dir: Path, keep: str) -> Optional[Path]:
    assert keep == "latest"
    # Prefer latest symlink if present
    link = _resolve_symlink(task_dir, "latest")
    if link is not None and link.is_dir():
        return link

    runs = _iter_run_dirs(task_dir)
    if not runs:
        return None
    return runs[-1]


def _find_wandb_mp4s(run_dir: Path) -> list[Path]:
    # Do NOT traverse wandb/latest-run symlink; only real run-* dirs.
    vids = []
    vids.extend(run_dir.glob("wandb/run-*/files/media/videos/**/*.mp4"))
    return sorted({p for p in vids if p.is_file()})


def _find_legacy_mp4s(task_dir: Path) -> list[Path]:
    # Legacy: logs/<task>/1/**/eval_video/**/*.mp4
    vids = []
    legacy_root = task_dir / "1"
    if not legacy_root.is_dir():
        return []
    vids.extend(legacy_root.glob("**/eval_video/**/*.mp4"))
    vids.extend(legacy_root.glob("**/eval_video/*.mp4"))
    return sorted({p for p in vids if p.is_file()})


def _find_run_videos(run_dir: Path, *, include_legacy: bool) -> list[Path]:
    vids = []
    vids.extend(_find_wandb_mp4s(run_dir))
    # Sometimes users copy videos into run_dir/videos/
    vids.extend(run_dir.glob("videos/*.mp4"))
    out = sorted({p for p in vids if p.is_file()})
    # Legacy videos are not per-run, handled separately at task level.
    return out


def _parse_eval_video_step(path: Path) -> Optional[int]:
    # Prefer the W&B naming: .../eval_video_<step>_<hash>.mp4
    m = EVAL_VIDEO_STEP_RE.search(str(path).replace("\\", "/"))
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _select_latest_video(videos: list[Path]) -> Tuple[Optional[int], Optional[Path]]:
    if not videos:
        return (None, None)
    # Prefer max step; tie-break by mtime then name.
    best = None
    best_key = (-1, -1.0, "")
    for p in videos:
        step = _parse_eval_video_step(p)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        key = (step if step is not None else -1, mtime, p.name)
        if key > best_key:
            best_key = key
            best = p
    return (best_key[0] if best_key[0] >= 0 else None, best)


@dataclass
class DeleteItem:
    path: Path
    reason: str


def _plan_task(task_dir: Path, *, keep: str, include_legacy: bool) -> tuple[Optional[Path], list[DeleteItem]]:
    kept = _pick_kept_run(task_dir, keep=keep)
    deletes: list[DeleteItem] = []

    # Per-run pruning
    for run_dir in _iter_run_dirs(task_dir):
        vids = _find_run_videos(run_dir, include_legacy=False)
        if run_dir != kept:
            for p in vids:
                deletes.append(DeleteItem(p, "drop: non-kept run"))
        else:
            # keep only latest video in kept run
            step, latest = _select_latest_video(vids)
            for p in vids:
                if latest is not None and p == latest:
                    continue
                deletes.append(DeleteItem(p, "drop: older video in kept run"))

    # Legacy pruning (task-level)
    if include_legacy:
        legacy = _find_legacy_mp4s(task_dir)
        step, latest = _select_latest_video(legacy)
        for p in legacy:
            if latest is not None and p == latest:
                continue
            deletes.append(DeleteItem(p, "drop: legacy older video"))

    return kept, deletes


def _delete_file(p: Path) -> None:
    p.unlink(missing_ok=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_logs_dir = repo_root / "tdmpc2" / "logs"

    ap = argparse.ArgumentParser(description="Prune old eval videos under tdmpc2/logs/<task>/.")
    ap.add_argument("--logs-dir", type=Path, default=default_logs_dir)
    ap.add_argument("--include-legacy", action="store_true", help="Also prune legacy logs/<task>/1/**/eval_video/*.mp4")
    ap.add_argument("--apply", action="store_true", help="Actually delete files (default: dry-run).")
    ap.add_argument("--quiet", action="store_true", help="Only print summary.")
    args = ap.parse_args()

    logs_dir = args.logs_dir.expanduser().resolve()
    dry = not args.apply
    if dry and not args.quiet:
        _atomic_print("DRY-RUN (no changes). Use --apply to delete.\n")

    total_files = 0
    total_bytes = 0

    for task_dir in sorted(_iter_task_dirs(logs_dir), key=lambda p: p.name):
        kept, deletes = _plan_task(task_dir, keep="latest", include_legacy=bool(args.include_legacy))
        if not deletes:
            continue

        if not args.quiet:
            _atomic_print(f"=== {task_dir.name} === kept={kept.name if kept else None} delete={len(deletes)}")

        for item in deletes:
            try:
                size = item.path.stat().st_size
            except OSError:
                size = 0
            total_files += 1
            total_bytes += size
            if not args.quiet:
                _atomic_print(f"[DEL]{' (dry-run)' if dry else ''} {item.path} ({item.reason}, {size/1e6:.1f}MB)")
            if not dry:
                _delete_file(item.path)

    _atomic_print("\n=== Summary ===")
    _atomic_print(f"dry_run: {dry}")
    _atomic_print(f"files:   {total_files}")
    _atomic_print(f"size:    {total_bytes/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)


