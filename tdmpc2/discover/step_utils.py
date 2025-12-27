"""Unified step extraction utilities.

This module is the single source of truth for extracting checkpoint and video
steps from run directories. Use these helpers everywhere to avoid discrepancies.

Definitions:
- ckpt_step: max checkpoint step found on disk (checkpoints/*.pt)
- video_step: step confidently associated with a video, derived ONLY from:
  - run_info.yaml: loaded_step, step, ckpt_step fields, or
  - run_info.yaml: parsed from loaded_checkpoint/checkpoint/ckpt_path path, or
  - video filename: parsed from eval_video_<step>_*.mp4 pattern
  - NEVER from checkpoint files in the same directory (that would inflate video_step)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    pass

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore


def iter_run_info_paths(logs_dir: Path) -> List[Path]:
    """Return candidate run_info.yaml paths across supported on-disk layouts.

    Supported layouts (depth from logs_dir):
      - logs/<run_id>/run_info.yaml                       (legacy run-first)
      - logs/<task>/<run_id>/run_info.yaml                (task-first)
      - logs/<task>/<seed>/<run_id>/run_info.yaml         (older nested)

    We intentionally avoid a full recursive rglob because logs/ can contain very
    large wandb media trees; run_info.yaml is always near the top of each run dir.
    """
    patterns = (
        "*/run_info.yaml",
        "*/*/run_info.yaml",
        "*/*/*/run_info.yaml",
    )
    paths: List[Path] = []
    for pat in patterns:
        paths.extend(logs_dir.glob(pat))
    return sorted(set(paths), key=lambda p: str(p))


def find_run_videos(run_dir: Path) -> List[Path]:
    """Find all video files in a run directory.

    Checks multiple locations where videos might be stored:
    - wandb/run-*/files/media/videos/**/*.mp4 (wandb synced, new structure)
    - */wandb/run-*/files/media/videos/**/*.mp4 (wandb synced, nested under an exp dir)
    - videos/*.mp4 (direct saves)
    """
    videos: List[Path] = []
    videos.extend(run_dir.glob("wandb/run-*/files/media/videos/**/*.mp4"))
    videos.extend(run_dir.glob("*/wandb/run-*/files/media/videos/**/*.mp4"))
    video_dir = run_dir / "videos"
    if video_dir.is_dir():
        videos.extend(video_dir.glob("*.mp4"))
    # De-dup and provide stable ordering.
    return sorted(set(videos), key=lambda p: str(p))


def _dedup_paths(paths: List[Path]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for p in paths:
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def _build_task_run_dir_map(logs_dir: Path, tasks: List[str]) -> Dict[str, List[Path]]:
    """Build task -> candidate run_dir list in a single pass over run_info.yaml.

    This is intentionally linear in the number of run_info.yaml files, to avoid
    O(tasks × runs) behavior.
    """
    wanted = set(str(t) for t in tasks)
    task_to_dirs: Dict[str, List[Path]] = {str(t): [] for t in tasks}

    # 1) Direct scan of logs/<task>/... (catches run dirs without run_info.yaml).
    for t in list(task_to_dirs.keys()):
        task_dir = logs_dir / t
        if task_dir.is_dir():
            try:
                for d in task_dir.iterdir():
                    if d.is_dir():
                        task_to_dirs[t].append(d)
            except Exception:
                pass

    # 2) run_info.yaml mapping (catches run-first layouts, and nested layouts with run_info.yaml).
    if _yaml is not None:
        for run_info_path in iter_run_info_paths(logs_dir):
            run_dir = run_info_path.parent
            try:
                info = _yaml.safe_load(run_info_path.read_text()) or {}
            except Exception:
                continue
            run_tasks = info.get("tasks", [info.get("task")])
            if not isinstance(run_tasks, list):
                run_tasks = [run_tasks]
            for t in run_tasks:
                if t in wanted:
                    task_to_dirs[str(t)].append(run_dir)

    # De-dup while preserving order.
    for t in list(task_to_dirs.keys()):
        task_to_dirs[t] = _dedup_paths(task_to_dirs[t])

    return task_to_dirs


def _pick_best_video_path(videos: List[Path], step: int) -> Optional[Path]:
    if not videos:
        return None
    candidates: List[Path] = []
    if step > 0:
        for v in videos:
            if _step_from_video_filename(v) == step:
                candidates.append(v)
    if not candidates:
        candidates = list(videos)
    try:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    except Exception:
        return candidates[-1]


def compute_task_steps(
    logs_dir: Path,
    tasks: List[str],
) -> Dict[str, Tuple[int, int, bool, Optional[Path]]]:
    """Compute (ckpt_step_max, video_step_max, has_video, best_video_path) per task.

    This is the single source of truth used by:
    - discover tasks/status/domains (via liveness.build_task_progress)
    - discover eval list
    - discover videos collect
    """
    task_to_dirs = _build_task_run_dir_map(logs_dir, tasks)

    result: Dict[str, Tuple[int, int, bool, Optional[Path]]] = {}
    for task in tasks:
        t = str(task)
        ckpt_step_max = 0
        video_step_max = 0
        has_video = False
        best_video_path: Optional[Path] = None

        for run_dir in task_to_dirs.get(t, []):
            ckpt_step_max = max(ckpt_step_max, infer_ckpt_step(run_dir))

            vids = find_run_videos(run_dir)
            if not vids:
                continue
            has_video = True

            vs = infer_video_step(run_dir, videos=vids)
            if vs > video_step_max:
                video_step_max = vs
                best_video_path = _pick_best_video_path(vids, vs)

        result[t] = (ckpt_step_max, video_step_max, has_video, best_video_path)

    return result


def parse_step_from_path(path: Path) -> int:
    """Parse training step from checkpoint filename (e.g., 1_000_000.pt -> 1000000)."""
    try:
        return int(path.stem.replace("_", "").replace(",", ""))
    except ValueError:
        return 0


def infer_ckpt_step(run_dir: Path) -> int:
    """Max checkpoint step on disk for a run directory.
    
    Looks at checkpoints/*.pt files (excluding *_trainer.pt).
    """
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return 0
    best_step = 0
    try:
        for ckpt in ckpt_dir.glob("*.pt"):
            if ckpt.stem.endswith("_trainer"):
                continue
            s = parse_step_from_path(ckpt)
            if s > best_step:
                best_step = s
    except Exception:
        return 0
    return best_step


def _step_from_run_info(run_dir: Path) -> int:
    """Extract step from run_info.yaml (for eval-only runs that load a checkpoint).
    
    Checks:
    - Explicit step fields: loaded_step, step, ckpt_step
    - Parsed from path fields: loaded_checkpoint, checkpoint, ckpt_path
    """
    if _yaml is None:
        return 0
    run_info_path = run_dir / "run_info.yaml"
    if not run_info_path.is_file():
        return 0
    try:
        info = _yaml.safe_load(run_info_path.read_text()) or {}
    except Exception:
        return 0

    # Prefer explicit numeric step
    for k in ("loaded_step", "step", "ckpt_step"):
        v = info.get(k)
        try:
            if v is not None:
                return int(v)
        except Exception:
            pass

    # Fall back to parsing from checkpoint path
    for k in ("loaded_checkpoint", "checkpoint", "ckpt_path"):
        v = info.get(k)
        if not v:
            continue
        try:
            return parse_step_from_path(Path(str(v)))
        except Exception:
            continue
    return 0


def _step_from_video_filename(video_path: Path) -> int:
    """Parse step from video filename (e.g., eval_video_5000000_abc.mp4 -> 5000000)."""
    name = video_path.name
    # Pattern: eval_video_<step>_<hash>.mp4
    m = re.search(r"(?:^|_)eval_video_(\d+)(?:_|\.mp4$)", name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    # Fallback: video_<step>_*.mp4
    m = re.search(r"(?:^|_)video_(\d+)(?:_|\.mp4$)", name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


def infer_video_step(run_dir: Path, videos: Optional[List[Path]] = None) -> int:
    """Step confidently associated with videos in a run directory.
    
    This ONLY uses:
    - run_info.yaml step fields (for eval-only runs)
    - video filename parsing
    
    It does NOT use checkpoint files - a video step should represent what
    the video actually shows, not what checkpoints happen to exist nearby.
    
    Args:
        run_dir: Path to the run directory
        videos: Optional pre-computed list of video paths. If None, will scan.
    
    Returns:
        Best video step, or 0 if none can be determined.
    """
    if videos is None:
        videos = find_run_videos(run_dir)

    # Prefer parsing from filenames when available; this is the most direct evidence.
    parsed = [_step_from_video_filename(v) for v in videos]
    parsed = [s for s in parsed if s > 0]
    if parsed:
        return max(parsed)

    # Fallback: eval-only runs often store the loaded checkpoint step in run_info.yaml.
    ri_step = _step_from_run_info(run_dir)
    return ri_step if ri_step > 0 else 0

