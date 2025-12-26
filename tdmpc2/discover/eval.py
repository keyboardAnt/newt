"""Evaluation and video management for TD-MPC2 runs."""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd

from .progress import best_step_by_task
from .runs import iter_run_info_paths


def require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required.") from exc
    return pd


def find_run_videos(run_dir: Path) -> List[Path]:
    """Find all video files in a run directory.
    
    Checks multiple locations where videos might be stored:
    - wandb/run-*/files/media/videos/**/*.mp4 (wandb synced, new structure)
    - **/wandb/run-*/files/media/videos/**/*.mp4 (wandb synced, old nested structure)
    - videos/*.mp4 (direct saves)
    """
    videos = []
    # Wandb media directory - new flat structure
    videos.extend(run_dir.glob("wandb/run-*/files/media/videos/**/*.mp4"))
    # Wandb media directory - old nested structure (e.g., task/seed/exp_name/wandb/...)
    videos.extend(run_dir.glob("**/wandb/run-*/files/media/videos/**/*.mp4"))
    # Direct video saves
    video_dir = run_dir / "videos"
    if video_dir.is_dir():
        videos.extend(video_dir.glob("*.mp4"))
    return sorted(set(videos))  # Use set to deduplicate


def find_task_videos(task: str, logs_dir: Path) -> List[str]:
    """Find videos for a task by checking run_info.yaml in each run directory.
    
    Since video filenames don't contain the task name, we use run_info.yaml
    to identify which runs belong to the task, then collect their videos.
    """
    import yaml
    
    if not logs_dir.is_dir():
        return []
    
    videos = []
    for run_info_path in iter_run_info_paths(logs_dir):
        run_dir = run_info_path.parent
        try:
            info = yaml.safe_load(run_info_path.read_text()) or {}
            run_tasks = info.get("tasks", [info.get("task")])
            if task not in run_tasks:
                continue
        except Exception:
            continue
        
        # Found a run for this task - collect its videos
        run_videos = find_run_videos(run_dir)
        videos.extend(str(v) for v in run_videos)
    
    return sorted(set(videos))


def tasks_ready_for_eval(
    df: "pd.DataFrame",
    logs_dir: Path,
    target_step: int = 5_000_000,
    min_progress: float = 0.5
) -> Tuple["pd.DataFrame", List[str], List[str]]:
    """Find tasks that are at least min_progress trained and check video availability.
    
    Returns:
        Tuple of (ready_df, tasks_need_eval, tasks_with_videos)
    """
    pd = require_pandas()
    min_step = int(target_step * min_progress)
    
    best = best_step_by_task(df)
    if best.empty:
        print('No runs found.')
        return pd.DataFrame(), [], []
    
    ready = best[best['max_step'] >= min_step].copy()
    
    ready['video_paths'] = ready['task'].apply(lambda t: find_task_videos(t, logs_dir))
    ready['has_videos'] = ready['video_paths'].apply(lambda x: len(x) > 0)
    ready['progress_pct'] = (100 * ready['max_step'] / target_step).round(1)
    ready = ready.sort_values('max_step', ascending=False)
    
    with_videos = ready[ready['has_videos']]
    without_videos = ready[~ready['has_videos']]
    
    print("=" * 80)
    print(f"{'TASKS READY FOR EVALUATION (≥' + str(int(min_progress*100)) + '% trained)':^80}")
    print("=" * 80)
    print(f"\nTotal tasks at ≥{int(min_progress*100)}%: {len(ready)}")
    print(f"  ✅ With videos:    {len(with_videos)}")
    print(f"  ❌ Without videos: {len(without_videos)}")
    
    print(f"\n{'─' * 80}")
    print("Tasks WITH videos:")
    print("─" * 80)
    if not with_videos.empty:
        for _, row in with_videos.iterrows():
            print(f"  ✅ {row['task']:<45} {int(row['max_step']):>10,} ({row['progress_pct']}%)")
    else:
        print("  (none)")
    
    print(f"\n{'─' * 80}")
    print("Tasks WITHOUT videos (need eval):")
    print("─" * 80)
    if not without_videos.empty:
        for _, row in without_videos.iterrows():
            print(f"  ❌ {row['task']:<45} {int(row['max_step']):>10,} ({row['progress_pct']}%)")
    else:
        print("  (none - all tasks have videos!)")
    
    print("=" * 80)
    
    return ready, without_videos['task'].tolist(), with_videos['task'].tolist()


def generate_eval_script(
    tasks: List[str],
    output_dir: Path,
    project_root: Path,
) -> Optional[Tuple[Path, Path]]:
    """Generate task list and LSF script to run eval on tasks without videos."""
    output_dir = Path(output_dir)
    
    if not tasks:
        print("✅ No tasks need evaluation - all have videos!")
        return None
    
    task_list_path = output_dir / 'tasks_need_eval.txt'
    task_list_path.write_text('\n'.join(tasks) + '\n')
    print(f"✅ Written task list to: {task_list_path}")
    print(f"   ({len(tasks)} tasks)")
    
    lsf_script = f'''#!/bin/bash
# Auto-generated eval script for tasks needing videos
# Generated: {datetime.now().isoformat()}
# Tasks: {len(tasks)}

#BSUB -J newt-eval-videos[1-{len(tasks)}]
#BSUB -q short-gpu
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB,tmp=10240]"  # tmp is MB; 10240 ~= 10 GiB
#BSUB -W 04:00
#BSUB -o {project_root}/logs/lsf/newt-eval-videos.%J.%I.log
#BSUB -e {project_root}/logs/lsf/newt-eval-videos.%J.%I.log

#BSUB -app nvidia-gpu
#BSUB -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.2"

cd {project_root}

pip install -q "wandb[media]"

TASK=$(sed -n "${{LSB_JOBINDEX}}p" jobs/tasks_need_eval.txt)
echo "LSF job index: ${{LSB_JOBINDEX}}, task: ${{TASK}}"

# NOTE: ManiSkill tasks require non-exclusive GPU mode for SAPIEN/Vulkan.
# If this fails for ms-* tasks, resubmit with: -gpu "num=1" instead of exclusive_process

export TASK
eval $(python - <<'PY'
import os, yaml
from pathlib import Path
from discover import parse_step

task = os.environ.get("TASK", "")
logs_dir = Path("logs")
run_info_paths = []
for pat in ("*/run_info.yaml", "*/*/run_info.yaml", "*/*/*/run_info.yaml"):
    run_info_paths.extend(logs_dir.glob(pat))

candidates = []
for run_info_path in run_info_paths:
    run_dir = run_info_path.parent
    info = yaml.safe_load(run_info_path.read_text()) or {{}}
    tasks = info.get("tasks", [info.get("task")])
    if task in tasks:
        for ckpt in (run_dir / "checkpoints").glob("*.pt"):
            if not ckpt.stem.endswith('_trainer'):
                candidates.append(ckpt)
if not candidates:
    print('CKPT=')
    print('RUN_ID=')
else:
    best = max(candidates, key=parse_step)
    run_id = best.parent.parent.name
    print(f'CKPT="{{best}}"')
    print(f'RUN_ID="{{run_id}}"')
PY
)

if [ -z "${{CKPT}}" ]; then
  echo "No checkpoint found for task '${{TASK}}', skipping."
  exit 0
fi

echo "Evaluating checkpoint: ${{CKPT}}"

python train.py \\
  task="${{TASK}}" \\
  model_size=B \\
  checkpoint="${{CKPT}}" \\
  steps=1 \\
  num_envs=2 \\
  use_demos=False \\
  tasks_fp={project_root.parent}/tasks.json \\
  exp_name="eval_${{RUN_ID}}" \\
  save_video=True \\
  env_mode=sync \\
  compile=False  # Keep disabled for eval (compilation overhead > 1-step runtime)
'''
    
    lsf_path = output_dir / 'run_eval_need_videos.lsf'
    lsf_path.write_text(lsf_script)
    print(f"✅ Written LSF script to: {lsf_path}")
    
    print(f"\n📋 To submit the eval jobs:")
    print(f"   make submit-eval")
    
    return task_list_path, lsf_path


def prune_old_videos(output_dir: Path, dry_run: bool = False) -> List[Path]:
    """Remove older checkpoint videos when a newer one exists for the same task.
    
    Args:
        output_dir: Directory containing collected videos
        dry_run: If True, only report what would be removed without deleting
    
    Returns:
        List of paths that were (or would be) removed
    """
    import re
    
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []
    
    # Pattern to match task_step.mp4 filenames
    pattern = re.compile(r'^(.+)_(\d+)\.mp4$')
    
    # Group videos by task
    task_videos: dict[str, List[Tuple[int, Path]]] = {}
    for video_path in output_dir.glob('*.mp4'):
        match = pattern.match(video_path.name)
        if match:
            task = match.group(1)
            step = int(match.group(2))
            task_videos.setdefault(task, []).append((step, video_path))
    
    # Find and remove old videos (keep only the highest step)
    removed = []
    for task, videos in task_videos.items():
        if len(videos) <= 1:
            continue
        videos.sort(key=lambda x: x[0], reverse=True)  # Sort by step, highest first
        max_step = videos[0][0]
        for step, path in videos[1:]:  # Skip the newest
            if dry_run:
                print(f"  [dry-run] Would remove: {path.name} (step {step:,} < {max_step:,})")
            else:
                path.unlink(missing_ok=True)
            removed.append(path)
    
    return removed


def _prune_task_videos(output_dir: Path, task: str, keep_step: int) -> List[Path]:
    """Remove videos for a specific task with steps lower than keep_step.
    
    Internal helper for collect_videos to prune old videos for a single task.
    """
    import re
    
    pattern = re.compile(rf'^{re.escape(task)}_(\d+)\.mp4$')
    removed = []
    
    for video_path in output_dir.glob(f'{task}_*.mp4'):
        match = pattern.match(video_path.name)
        if match:
            step = int(match.group(1))
            if step < keep_step:
                video_path.unlink(missing_ok=True)
                removed.append(video_path)
    
    return removed


def collect_videos(
    df: "pd.DataFrame",
    logs_dir: Path,
    output_dir: Path,
    target_step: int = 5_000_000,
    min_progress: float = 0.5,
    create_symlinks: bool = True,
    prune_old: bool = True,
) -> Optional["pd.DataFrame"]:
    """Collect videos from tasks that are min_progress% trained into a single directory.
    
    Args:
        df: DataFrame with run data
        logs_dir: Directory containing run logs
        output_dir: Output directory for collected videos
        target_step: Target training steps (default 5M)
        min_progress: Minimum progress threshold (default 0.5 = 50%)
        create_symlinks: If True, create symlinks; if False, copy files
        prune_old: If True, remove older checkpoint videos for same task (default True)
    """
    pd = require_pandas()
    
    min_step = int(target_step * min_progress)
    best = best_step_by_task(df)
    ready = best[best['max_step'] >= min_step].copy()
    
    video_info = []
    for _, row in ready.iterrows():
        task = row['task']
        videos = find_task_videos(task, logs_dir)
        if videos:
            video_info.append({
                'task': task,
                'max_step': row['max_step'],
                'progress_pct': 100 * row['max_step'] / target_step,
                'video_path': videos[-1],  # Use latest video
            })
    
    if not video_info:
        print("❌ No videos found in tasks that are 50%+ trained.")
        print("   Run the eval script first to generate videos.")
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    collected = []
    pruned_total = []
    for v in sorted(video_info, key=lambda x: x['task']):
        src = Path(v['video_path'])
        if not src.exists():
            print(f"⚠️  Video not found: {src}")
            continue
        
        task = v['task']
        step = int(v['max_step'])
        dst_name = f"{task}_{step}.mp4"
        dst = output_dir / dst_name
        
        # Prune older videos for this task before creating the new one
        if prune_old:
            pruned = _prune_task_videos(output_dir, task, step)
            pruned_total.extend(pruned)
        
        if create_symlinks:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        
        collected.append({
            'task': task,
            'step': step,
            'progress': v['progress_pct'],
            'filename': dst_name,
            'source_path': str(src),
            'dest_path': str(dst)
        })
    
    print("=" * 80)
    print(f"{'VIDEOS COLLECTED FOR PRESENTATION':^80}")
    print("=" * 80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Total videos: {len(collected)}")
    print(f"   Method: {'symlinks' if create_symlinks else 'copies'}")
    if pruned_total:
        print(f"   🗑️  Pruned old videos: {len(pruned_total)}")
    
    print(f"\n{'─' * 80}")
    print(f"{'Task':<45} {'Step':>12} {'Progress':>10}")
    print("─" * 80)
    for v in collected:
        print(f"  {v['task']:<43} {v['step']:>12,} {v['progress']:>9.1f}%")
    print("─" * 80)
    
    # Save manifest
    manifest_df = pd.DataFrame(collected)
    manifest_path = output_dir / 'video_manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\n📄 Manifest saved to: {manifest_path}")
    
    print(f"\n📥 To download to your laptop:")
    print(f"   rsync -avz <server>:{output_dir}/ ./presentation_videos/")
    print("=" * 80)
    
    return manifest_df


def download_wandb_videos(
    df: "pd.DataFrame",
    output_dir: Path,
    wandb_project: str = "wm-planning/mmbench",
    target_step: int = 5_000_000,
    min_progress: float = 0.5,
) -> Optional["pd.DataFrame"]:
    """Download videos from Wandb for tasks that are min_progress% trained.
    
    Args:
        df: DataFrame with run data (must include 'task', 'wandb_run_id', 'max_step')
        output_dir: Directory to save downloaded videos
        wandb_project: Wandb project path (entity/project)
        target_step: Target training steps (default 5M)
        min_progress: Minimum progress threshold (default 0.5 = 50%)
    
    Returns:
        DataFrame with downloaded video info, or None if no videos found
    """
    pd = require_pandas()
    
    try:
        import wandb
    except ImportError:
        print("❌ wandb is required. Install with: pip install wandb")
        return None
    
    min_step = int(target_step * min_progress)
    best = best_step_by_task(df)
    ready = best[best['max_step'] >= min_step].copy()
    
    if ready.empty:
        print(f"❌ No tasks at ≥{int(min_progress*100)}% progress.")
        return None
    
    # Filter to runs that have wandb_run_id
    wandb_ready = ready[ready['wandb_run_id'].notna()].copy()
    if wandb_ready.empty:
        print("❌ No Wandb runs found for tasks at required progress.")
        return None
    
    print(f"🔍 Checking {len(wandb_ready)} tasks for videos on Wandb...")
    
    api = wandb.Api(timeout=60)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    
    for idx, row in wandb_ready.iterrows():
        task = row['task']
        run_id = row['wandb_run_id']
        max_step = row['max_step']
        
        try:
            run = api.run(f"{wandb_project}/{run_id}")
            
            # Find video files in the run
            files = run.files()
            video_files = [f for f in files if f.name.endswith('.mp4')]
            
            if not video_files:
                sys.stderr.write(f"  {task}: no videos\n")
                continue
            
            # Download the latest/best video (prefer eval_video if exists)
            eval_videos = [f for f in video_files if 'eval' in f.name.lower()]
            video_file = eval_videos[-1] if eval_videos else video_files[-1]
            
            dst_name = f"{task}_{int(max_step)}.mp4"
            dst_path = output_dir / dst_name
            
            sys.stderr.write(f"  {task}: downloading {video_file.name}...")
            sys.stderr.flush()
            
            video_file.download(root=str(output_dir), replace=True)
            downloaded_path = output_dir / video_file.name
            
            # Rename to standardized name
            if downloaded_path.exists() and downloaded_path != dst_path:
                if dst_path.exists():
                    dst_path.unlink()
                downloaded_path.rename(dst_path)
            
            sys.stderr.write(" ✓\n")
            
            downloaded.append({
                'task': task,
                'step': int(max_step),
                'progress': 100 * max_step / target_step,
                'filename': dst_name,
                'wandb_run_id': run_id,
                'source_file': video_file.name,
                'dest_path': str(dst_path),
            })
            
        except Exception as e:
            sys.stderr.write(f"  {task}: error - {e}\n")
            continue
    
    if not downloaded:
        print("\n❌ No videos could be downloaded from Wandb.")
        print("   Videos may not have been logged, or runs may have been deleted.")
        return None
    
    print("\n" + "=" * 80)
    print(f"{'VIDEOS DOWNLOADED FROM WANDB':^80}")
    print("=" * 80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Total videos: {len(downloaded)}")
    
    print(f"\n{'─' * 80}")
    print(f"{'Task':<45} {'Step':>12} {'Progress':>10}")
    print("─" * 80)
    for v in downloaded:
        print(f"  {v['task']:<43} {v['step']:>12,} {v['progress']:>9.1f}%")
    print("─" * 80)
    
    # Save manifest
    manifest_df = pd.DataFrame(downloaded)
    manifest_path = output_dir / 'video_manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\n📄 Manifest saved to: {manifest_path}")
    
    print(f"\n📥 To download to your laptop:")
    print(f"   rsync -avz <server>:{output_dir}/ ./presentation_videos/")
    print("=" * 80)
    
    return manifest_df


def main():
    """CLI to generate eval script for tasks needing videos."""
    import argparse
    from .runs import discover
    from .progress import attach_max_step
    
    parser = argparse.ArgumentParser(description="Generate eval script for tasks needing videos.")
    parser.add_argument("--min-progress", type=float, default=0.5, help="Minimum progress threshold (default: 0.5)")
    parser.add_argument("--target-step", type=int, default=5_000_000, help="Target training steps (default: 5M)")
    parser.add_argument("--limit", type=int, default=None, help="Limit wandb runs fetched")
    args = parser.parse_args()
    
    # Paths
    ROOT = Path(__file__).parents[1]  # tdmpc2/
    logs_dir = ROOT / 'logs'
    jobs_dir = ROOT / 'jobs'
    wandb_project = "wm-planning/mmbench"
    
    print(f"Logs dir: {logs_dir}")
    print(f"Wandb project: {wandb_project}")
    print(f"Min progress: {args.min_progress*100:.0f}%")
    print(f"Target step: {args.target_step:,}\n")
    
    # Discover runs
    df_all = discover(logs_dir, wandb_project, args.limit)
    if df_all.empty:
        print("No runs found.")
        return
    
    df_all = attach_max_step(df_all)
    
    # Find tasks ready for eval
    ready_df, tasks_need_eval, tasks_with_videos = tasks_ready_for_eval(
        df_all, logs_dir, 
        target_step=args.target_step, 
        min_progress=args.min_progress
    )
    
    if not tasks_need_eval:
        print("\n✅ All tasks at required progress already have videos!")
        return
    
    # Generate eval script
    generate_eval_script(tasks_need_eval, output_dir=jobs_dir, project_root=ROOT)
    
    print(f"\n📋 To submit the eval jobs:")
    print(f"   make submit-eval")


def prune_main():
    """CLI to prune old checkpoint videos, keeping only the latest for each task."""
    import argparse
    
    ROOT = Path(__file__).parents[1]  # tdmpc2/
    default_dir = ROOT / 'discover' / 'videos_for_presentation'
    
    parser = argparse.ArgumentParser(
        description="Prune old checkpoint videos, keeping only the latest for each task."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=default_dir,
        help=f"Directory containing videos (default: {default_dir})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"{'PRUNE OLD CHECKPOINT VIDEOS':^60}")
    print("=" * 60)
    print(f"\n📁 Directory: {args.dir}")
    print(f"   Mode: {'dry-run (no files deleted)' if args.dry_run else 'delete old videos'}")
    print()
    
    removed = prune_old_videos(args.dir, dry_run=args.dry_run)
    
    if not removed:
        print("✅ No old videos to prune - directory is clean.")
    else:
        action = "Would remove" if args.dry_run else "Removed"
        print(f"\n🗑️  {action} {len(removed)} old video(s)")
        if args.dry_run:
            print("\n   Run without --dry-run to actually delete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
