"""Evaluation and video management for TD-MPC2 runs."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd

from .analysis import best_step_by_task


def require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required.") from exc
    return pd


def find_task_videos(task: str, logs_dir: Path) -> List[str]:
    """Find all videos for a task by scanning the logs directory."""
    task_dir = logs_dir / task
    if not task_dir.is_dir():
        return []
    videos = list(task_dir.glob("*/wandb/run-*/files/media/videos/**/*.mp4"))
    videos += list(task_dir.glob("*/videos/*.mp4"))
    return sorted(str(v) for v in videos)


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
#BSUB -q long-gpu
#BSUB -n 1
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 04:00
#BSUB -o {project_root}/tdmpc2/logs/lsf/newt-eval-videos.%J.%I.log
#BSUB -e {project_root}/tdmpc2/logs/lsf/newt-eval-videos.%J.%I.log

#BSUB -app nvidia-gpu
#BSUB -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.0"

cd {project_root}/tdmpc2

pip install -q "wandb[media]"

TASK=$(sed -n "${{LSB_JOBINDEX}}p" jobs/tasks_need_eval.txt)
echo "LSF job index: ${{LSB_JOBINDEX}}, task: ${{TASK}}"

export TASK
eval $(python - <<'PY'
import os
from pathlib import Path
from discover import parse_step

task = os.environ.get("TASK", "")
logs_dir = Path("logs") / task
candidates = [p for p in logs_dir.glob("*/checkpoints/*.pt") if not p.stem.endswith('_trainer')]
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
  tasks_fp={project_root}/tasks.json \\
  exp_name="eval_${{RUN_ID}}" \\
  save_video=True \\
  env_mode=sync \\
  compile=False
'''
    
    lsf_path = output_dir / 'run_eval_need_videos.lsf'
    lsf_path.write_text(lsf_script)
    print(f"✅ Written LSF script to: {lsf_path}")
    
    print(f"\n📋 To submit the eval jobs, run:")
    print(f"   cd {output_dir.parent}")
    print(f"   bsub < jobs/run_eval_need_videos.lsf")
    
    return task_list_path, lsf_path


def collect_videos(
    df: "pd.DataFrame",
    logs_dir: Path,
    output_dir: Path,
    target_step: int = 5_000_000,
    min_progress: float = 0.5,
    create_symlinks: bool = True,
) -> Optional["pd.DataFrame"]:
    """Collect videos from tasks that are min_progress% trained into a single directory."""
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
    for v in sorted(video_info, key=lambda x: x['task']):
        src = Path(v['video_path'])
        if not src.exists():
            print(f"⚠️  Video not found: {src}")
            continue
        
        dst_name = f"{v['task']}_{int(v['max_step'])}.mp4"
        dst = output_dir / dst_name
        
        if create_symlinks:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        
        collected.append({
            'task': v['task'],
            'step': int(v['max_step']),
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
