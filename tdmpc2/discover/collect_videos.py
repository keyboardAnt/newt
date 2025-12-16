#!/usr/bin/env python3
"""
Collect videos from trained tasks for presentation/download.

Usage:
    python discover/collect_videos.py --min-progress 0.5 --output ./videos_for_presentation
    python discover/collect_videos.py --copy  # Copy files instead of symlinks
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Add parent to path for imports
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from discover.eval import find_task_videos


def parse_step(path: Path) -> int:
    """Parse step from checkpoint filename like '600_000.pt'."""
    try:
        return int(path.stem.replace("_", ""))
    except ValueError:
        return 0


def find_best_runs(logs_dir: Path, min_step: int) -> List[Dict]:
    """Find the best (highest step) run for each task that meets the threshold."""
    task_best: Dict[str, Dict] = {}
    
    for ckpt in logs_dir.glob("*/*/checkpoints/*.pt"):
        if ckpt.stem.endswith('_trainer'):
            continue  # Skip trainer state files
        run_dir = ckpt.parent.parent
        task_dir = run_dir.parent
        task = task_dir.name
        step = parse_step(ckpt)
        
        if step < min_step:
            continue
        
        if task not in task_best or step > task_best[task]['step']:
            task_best[task] = {
                'task': task,
                'step': step,
                'ckpt_path': str(ckpt),
                'run_dir': str(run_dir),
            }
    
    # Add videos for each task
    for task, info in task_best.items():
        info['videos'] = find_task_videos(task, logs_dir)
    
    return list(task_best.values())


def collect_videos(
    logs_dir: Path,
    output_dir: Path,
    min_progress: float = 0.5,
    target_step: int = 5_000_000,
    use_symlinks: bool = True
) -> list[dict]:
    """Collect videos from tasks that meet the progress threshold."""
    min_step = int(target_step * min_progress)
    
    print(f"Scanning logs in: {logs_dir}")
    print(f"Min step required: {min_step:,} ({min_progress*100:.0f}% of {target_step:,})")
    
    best_runs = find_best_runs(logs_dir, min_step)
    
    if not best_runs:
        print("No runs found meeting the criteria.")
        return []
    
    # Filter to runs with videos
    runs_with_videos = [r for r in best_runs if r['videos']]
    runs_without_videos = [r for r in best_runs if not r['videos']]
    
    print(f"\nFound {len(best_runs)} tasks at ≥{min_progress*100:.0f}%:")
    print(f"  With videos: {len(runs_with_videos)}")
    print(f"  Without videos: {len(runs_without_videos)}")
    
    if runs_without_videos:
        print(f"\nTasks needing eval (no videos):")
        for r in sorted(runs_without_videos, key=lambda x: x['task']):
            print(f"  - {r['task']} (step {r['step']:,})")
    
    if not runs_with_videos:
        print("\nNo videos to collect. Run eval first.")
        return []
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    collected = []
    for run in sorted(runs_with_videos, key=lambda x: x['task']):
        task = run['task']
        step = run['step']
        
        # Use the latest video (last in sorted list)
        src = Path(run['videos'][-1])
        dst_name = f"{task}_{step}.mp4"
        dst = output_dir / dst_name
        
        if use_symlinks:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
            method = "symlinked"
        else:
            shutil.copy2(src, dst)
            method = "copied"
        
        collected.append({
            'task': task,
            'step': step,
            'progress_pct': round(100 * step / target_step, 1),
            'filename': dst_name,
            'source': str(src),
            'dest': str(dst)
        })
        print(f"  {method}: {dst_name}")
    
    # Write manifest
    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps({
        'generated': datetime.now().isoformat(),
        'target_step': target_step,
        'min_progress': min_progress,
        'videos': collected
    }, indent=2))
    
    # Write task list (for reference)
    tasks_path = output_dir / 'tasks.txt'
    tasks_path.write_text('\n'.join(v['task'] for v in collected) + '\n')
    
    print(f"\n✅ Collected {len(collected)} videos to: {output_dir}")
    print(f"   Manifest: {manifest_path}")
    print(f"   Task list: {tasks_path}")
    
    # Print download instructions
    print(f"\n📥 To download to your laptop:")
    print(f"   rsync -avz <server>:{output_dir}/ ./presentation_videos/")
    print(f"   # or")
    print(f"   scp -r <server>:{output_dir} ./presentation_videos/")
    
    return collected


def main():
    parser = argparse.ArgumentParser(
        description="Collect videos from trained tasks for presentation."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=ROOT / "tdmpc2" / "logs",
        help="Path to logs directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "discover" / "videos_for_presentation",
        help="Output directory for collected videos"
    )
    parser.add_argument(
        "--min-progress",
        type=float,
        default=0.5,
        help="Minimum progress fraction (0.5 = 50%%)"
    )
    parser.add_argument(
        "--target-step",
        type=int,
        default=5_000_000,
        help="Target step for 100%% completion"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks"
    )
    
    args = parser.parse_args()
    
    collect_videos(
        logs_dir=args.logs_dir,
        output_dir=args.output,
        min_progress=args.min_progress,
        target_step=args.target_step,
        use_symlinks=not args.copy
    )


if __name__ == "__main__":
    main()
