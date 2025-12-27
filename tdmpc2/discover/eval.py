"""Evaluation and video management for TD-MPC2 runs."""

from __future__ import annotations

import shutil
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd

from .progress import best_step_by_task, parse_step
from .step_utils import compute_task_steps, find_run_videos


def require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required.") from exc
    return pd


def tasks_ready_for_eval(
    df: "pd.DataFrame",
    logs_dir: Path,
    target_step: int = 5_000_000,
    min_progress: float = 0.5
) -> Tuple["pd.DataFrame", List[str], List[str]]:
    """Summarize checkpoint vs video availability for tasks and recommend eval-only jobs.

    This is intended to match `discover videos collect` semantics:
    - `ckpt_step`: max checkpoint step found on disk for the task
    - `video_step`: best available video step for the task
      (eval-only runs may have videos but no checkpoints; step is derived from run_info/video name)
    - `needs_eval`: there exists a newer checkpoint than the best available video, or no video at all

    Returns:
        Tuple of (ready_df, tasks_need_eval, tasks_up_to_date)
    """
    pd = require_pandas()
    min_step = int(target_step * min_progress)

    best = best_step_by_task(df)
    if best.empty:
        print("No runs found.")
        return pd.DataFrame(), [], []

    all_tasks = best["task"].dropna().astype(str).unique().tolist()
    print(f"Checking for videos in {len(all_tasks)} tasks...")

    rows: List[dict] = []
    step_info = compute_task_steps(logs_dir, sorted(all_tasks))
    for task in sorted(all_tasks):
        ckpt_step, video_step, has_video, _ = step_info.get(task, (0, 0, False, None))

        # Filter by min_progress based on filesystem checkpoint step (aligns with videos collect)
        if ckpt_step < min_step:
            continue

        progress_pct = round(100 * ckpt_step / target_step, 1) if target_step else 0.0
        is_up_to_date = bool(has_video) and (int(video_step) > 0) and (int(video_step) >= int(ckpt_step)) and (int(ckpt_step) > 0)
        needs_eval = (int(ckpt_step) > 0) and ((not bool(has_video)) or (int(video_step) == 0) or (int(video_step) < int(ckpt_step)))

        rows.append({
            "task": task,
            "ckpt_step": ckpt_step,
            "video_step": video_step,
            "progress_pct": progress_pct,
            "has_video": has_video,
            "is_up_to_date": is_up_to_date,
            "needs_eval": needs_eval,
        })

    ready_df = pd.DataFrame(rows).sort_values(["progress_pct", "task"], ascending=[False, True])

    # Lists used by `discover eval submit`
    tasks_need_eval = ready_df.loc[ready_df["needs_eval"], "task"].tolist()
    tasks_up_to_date = ready_df.loc[ready_df["is_up_to_date"], "task"].tolist()

    n_total = len(ready_df)
    n_has_video = int(ready_df["has_video"].sum()) if not ready_df.empty else 0
    n_up_to_date = int(ready_df["is_up_to_date"].sum()) if not ready_df.empty else 0
    n_need_eval = int(ready_df["needs_eval"].sum()) if not ready_df.empty else 0
    n_no_video = n_total - n_has_video
    n_stale = n_need_eval - n_no_video

    print("=" * 92)
    print(f"{'TASKS READY FOR EVALUATION (checkpoint vs video)':^92}")
    print("=" * 92)
    print(f"\nTotal tasks at ≥{int(min_progress*100)}%: {n_total}")
    print(f"  ✅ Video up-to-date: {n_up_to_date}")
    print(f"  ⚠️  Video stale:     {n_stale}")
    print(f"  ❌ No video:         {n_no_video}")
    print(f"  🧪 Needs eval:       {n_need_eval}")

    print(f"\n{'─' * 120}")
    print(f"{'Task':<45} {'Ckpt':>12} {'Video':>12} {'Progress':>10} {'Needs Eval':>10}")
    print("─" * 120)
    for _, r in ready_df.iterrows():
        needs = "Yes" if bool(r["needs_eval"]) else ""
        ckpt_disp = f"{int(r['ckpt_step']):,}" if int(r["ckpt_step"]) > 0 else "0"
        vid_disp = f"{int(r['video_step']):,}" if int(r["video_step"]) > 0 else "0"
        print(f"  {r['task']:<43} {ckpt_disp:>12} {vid_disp:>12} {r['progress_pct']:>9.1f}% {needs:>10}")
    print("─" * 120)
    print("=" * 92)

    if tasks_need_eval:
        print("\nRecommended eval-only jobs (latest checkpoint missing an updated video):")
        for task in tasks_need_eval:
            row = ready_df[ready_df["task"] == task].iloc[0]
            print(f"  • {task:<45} ckpt={int(row['ckpt_step']):,} video={int(row['video_step']):,}")
        print(f"\nTo generate an LSF script for these tasks:\n  python -m discover eval submit --min-progress {min_progress}\n")

    return ready_df, tasks_need_eval, tasks_up_to_date


def generate_eval_script(
    tasks: List[str],
    output_dir: Path,
    project_root: Path,
) -> Optional[List[Tuple[Path, Path]]]:
    """Generate task list(s) + LSF script(s) to run eval on tasks without videos.

    NOTE: LSF resource requirements (e.g. GPU mode) are fixed per submitted job, so we
    may split the submission into multiple job arrays.
    
    Historically, ManiSkill (SAPIEN/Vulkan) was run in shared GPU mode, but in practice
    shared mode can lead to transient CUDA OOM at init when the node is busy. We now
    request exclusive GPU mode for both arrays for stability; ManiSkill video rendering
    itself is handled robustly in-code (CPU fallback) when Vulkan init fails.
    """
    output_dir = Path(output_dir)
    
    if not tasks:
        print("✅ No tasks need evaluation - all have videos!")
        return None

    ms_tasks = [t for t in tasks if str(t).startswith("ms-")]
    non_ms_tasks = [t for t in tasks if not str(t).startswith("ms-")]

    def _write_one(
        *,
        task_subset: List[str],
        suffix: str,
        gpu_spec: str,
        job_name: str,
    ) -> Optional[Tuple[Path, Path]]:
        if not task_subset:
            return None

        task_list_path = output_dir / f"tasks_need_eval{suffix}.txt"
        task_list_path.write_text("\n".join(sorted(task_subset)) + "\n")
        print(f"✅ Written task list to: {task_list_path}")
        print(f"   ({len(task_subset)} tasks)")

        lsf_script = f'''#!/bin/bash
# Auto-generated eval script for tasks needing videos
# Generated: {datetime.now().isoformat()}
# Tasks: {len(task_subset)}

#BSUB -J {job_name}[1-{len(task_subset)}]
#BSUB -q short-gpu
#BSUB -n 1
#BSUB -gpu "{gpu_spec}"
#BSUB -R "rusage[mem=16GB,tmp=10240]"  # tmp is MB; 10240 ~= 10 GiB
#BSUB -W 04:00
#BSUB -o {project_root}/logs/lsf/{job_name}.%J.%I.log
#BSUB -e {project_root}/logs/lsf/{job_name}.%J.%I.log

#BSUB -app nvidia-gpu
#BSUB -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.2"

cd {project_root}

pip install -q "wandb[media]"

TASK=$(sed -n "${{LSB_JOBINDEX}}p" jobs/{task_list_path.name})
echo "LSF job index: ${{LSB_JOBINDEX}}, task: ${{TASK}}"

export TASK
eval $(python - <<'PY'
import os, yaml
from pathlib import Path
from discover.progress import parse_step

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
    print(f'CKPT="{{best.resolve()}}"')
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
  eval_only=True \\
  num_envs=2 \\
  use_demos=False \\
  tasks_fp={project_root.parent}/tasks.json \\
  exp_name="eval_${{RUN_ID}}" \\
  save_video=True \\
  env_mode=sync \\
  compile=False  # Keep disabled for eval (compilation overhead > 1-step runtime)
'''

        lsf_path = output_dir / f"run_eval_need_videos{suffix}.lsf"
        lsf_path.write_text(lsf_script)
        print(f"✅ Written LSF script to: {lsf_path}")
        return task_list_path, lsf_path

    created: List[Tuple[Path, Path]] = []

    # Non-ManiSkill jobs: keep exclusive GPU mode for stability / avoiding GPU sharing.
    out = _write_one(
        task_subset=non_ms_tasks,
        suffix="",
        gpu_spec="num=1:mode=exclusive_process",
        job_name="newt-eval-videos",
    )
    if out:
        created.append(out)

    # ManiSkill jobs: use exclusive GPU mode for stability (avoid CUDA init OOM on shared nodes).
    out = _write_one(
        task_subset=ms_tasks,
        suffix="_ms",
        gpu_spec="num=1:mode=exclusive_process",
        job_name="newt-eval-videos-ms",
    )
    if out:
        created.append(out)

    print(f"\n📋 To submit the eval jobs:")
    print(f"   make submit-eval")

    return created


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
    create_symlinks: bool = False,
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
    # We'll compute max checkpoint step + best available video purely from the filesystem
    # (under logs/), so this is correct even if the discovery cache is stale.
    # Still use `best` as a convenient list of tasks to consider.
    all_tasks = best['task'].tolist()
    
    print(f"Checking for videos in {len(all_tasks)} tasks...")
    
    video_info = []
    step_info = compute_task_steps(logs_dir, [str(t) for t in all_tasks])

    for task in all_tasks:
        ckpt_step, video_step, found_any_video, best_video_path = step_info.get(str(task), (0, 0, False, None))
        video_path: Optional[str] = str(best_video_path) if best_video_path is not None else None

        if ckpt_step < min_step:
            continue

        missing_latest_video = bool(found_any_video) and (int(ckpt_step) > 0) and ((int(video_step) == 0) or (int(video_step) < int(ckpt_step)))
        no_videos = (not bool(found_any_video))

        # Include the task even if it has no videos, so the summary can surface it.
        # Collection behavior below will prune stale outputs if no videos exist.
        video_info.append({
            'task': task,
            'ckpt_step': ckpt_step,
            'video_step': video_step,
            'missing_latest_video': missing_latest_video,
            'no_videos': no_videos,
            'progress_pct': 100 * ckpt_step / target_step,
            'video_path': video_path,
        })
    
    if not video_info:
        print("❌ No videos found for tasks meeting the progress threshold.")
        print("   Run the eval script first to generate videos.")
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    collected = []
    total_new = 0
    total_updated = 0
    total_pruned = 0
    
    # Summary counters (about videos vs latest checkpoints)
    n_ready = len(video_info)
    n_with_videos = sum(1 for v in video_info if int(v['video_step']) > 0)
    n_need_eval_latest = sum(1 for v in video_info if bool(v.get('missing_latest_video')))
    n_no_videos = sum(1 for v in video_info if bool(v.get('no_videos')))

    for v in sorted(video_info, key=lambda x: x['task']):
        task = v['task']
        ckpt_step = int(v['ckpt_step'])
        video_step = int(v['video_step'])
        video_path_str = v.get('video_path')
        
        # Output should contain the best available video (highest checkpoint step that has a video).
        # IMPORTANT: never delete existing presentation videos for a task unless we can
        # keep or create a valid best-available one (avoid accidental data loss).
        existing_files = list(output_dir.glob(f'{task}_*.mp4'))
        had_any = len(existing_files) > 0

        dst_name: Optional[str] = None
        dst: Optional[Path] = None
        if video_step > 0:
            dst_name = f"{task}_{video_step}.mp4"
            dst = output_dir / dst_name

        # Determine whether we can safely converge this task to the target file.
        # - If we have a target name and it already exists in output_dir, it's safe to prune others.
        # - If we have a target name and a valid source video to link/copy, it's safe to prune others.
        # - Otherwise, do NOT prune anything for this task (keep current state).
        src_path: Optional[Path] = None
        if video_step > 0 and video_path_str:
            candidate = Path(video_path_str)
            if candidate.exists():
                src_path = candidate

        target_exists = False
        if dst_name is not None:
            for p in existing_files:
                if p.name == dst_name:
                    # p may be a symlink; Path.exists() checks the target.
                    # We treat an existing entry as acceptable even if it's a symlink,
                    # but we avoid keeping broken ones by requiring exists().
                    if p.exists():
                        target_exists = True
                    break

        can_converge = (dst_name is not None) and (target_exists or (src_path is not None))

        # Prune everything except the target filename (only when safe)
        n_pruned = 0
        if prune_old and can_converge:
            for p in existing_files:
                if p.name != dst_name:
                    p.unlink(missing_ok=True)
                    n_pruned += 1

        # New/updated status for reporting
        had_exact = (dst_name is not None) and any(p.name == dst_name for p in existing_files)
        is_new = (not had_any) and (video_step > 0) and (src_path is not None)
        is_updated = had_any and (video_step > 0) and (not had_exact) and (src_path is not None)

        # Create/update the target video link/copy if we have one
        if video_step > 0 and dst is not None and src_path is not None:
            if create_symlinks:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src_path.resolve())
                method = "symlinked"
            else:
                # If a previous run created a symlink (or hardlink) at dst pointing to src_path,
                # shutil.copy2 will raise SameFileError. Make copy mode idempotent by replacing
                # any existing dst and treating "same file" as a no-op.
                try:
                    if dst.exists() or dst.is_symlink():
                        # If it's already the same underlying file, just replace it with a real copy
                        # (or keep it if it's already a regular file copy).
                        dst.unlink(missing_ok=True)
                    shutil.copy2(src_path, dst)
                except shutil.SameFileError:
                    # Already the same file; nothing to do.
                    pass
                method = "copied"

        if is_new:
            total_new += 1
        if is_updated:
            total_updated += 1
        total_pruned += n_pruned

        collected.append({
            'task': task,
            'ckpt_step': ckpt_step,
            'video_step': video_step,
            'missing_latest_video': bool(v.get('missing_latest_video')),
            'progress': v['progress_pct'],
            'filename': dst_name or "",
            'source_path': str(src_path) if src_path is not None else "",
            'dest_path': str(dst) if dst is not None else "",
            'is_new': is_new,
            'is_updated': is_updated,
            'pruned_count': n_pruned
        })
    
    print("=" * 80)
    print(f"{'VIDEOS COLLECTED FOR PRESENTATION':^80}")
    print("=" * 80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Total videos: {len(collected)}")
    print(f"   Method: {'symlinks' if create_symlinks else 'copies'}")
    
    print(f"   ✨ New videos: {total_new}")
    print(f"   🔄 Updated videos: {total_updated}")
    print(f"   🗑️  Pruned old videos: {total_pruned}")
    print(f"   ⚠️  Newer checkpoint exists w/o video (needs eval): {n_need_eval_latest}")
    print(f"   ❌ No videos at all (needs eval): {n_no_videos}")
    
    print(f"\n{'─' * 115}")
    print(f"{'Task':<45} {'Ckpt':>12} {'Video':>12} {'Progress':>10} {'Needs Eval':^11} {'New':^6} {'Upd':^6} {'Pruned':>8}")
    print("─" * 115)
    for v in collected:
        needs_eval = "Yes" if v.get('missing_latest_video') or (int(v.get('video_step') or 0) == 0) else ""
        new_mark = "Yes" if v['is_new'] else ""
        upd_mark = "Yes" if v['is_updated'] else ""
        pruned_mark = f"{v['pruned_count']}" if v['pruned_count'] > 0 else ""

        ckpt_disp = f"{int(v.get('ckpt_step') or 0):,}" if int(v.get('ckpt_step') or 0) > 0 else "0"
        vid_disp = f"{int(v.get('video_step') or 0):,}" if int(v.get('video_step') or 0) > 0 else "0"
        print(
            f"  {v['task']:<43} {ckpt_disp:>12} {vid_disp:>12} {v['progress']:>9.1f}% "
            f"{needs_eval:^11} {new_mark:^6} {upd_mark:^6} {pruned_mark:>8}"
        )
    print("─" * 115)
    
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
    
    from .wandb_connector import fetch_run
    
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
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    
    for idx, row in wandb_ready.iterrows():
        task = row['task']
        run_id = row['wandb_run_id']
        max_step = row['max_step']
        
        try:
            run = fetch_run(wandb_project, run_id)
            
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
