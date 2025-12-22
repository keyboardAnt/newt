#!/usr/bin/env python3
"""Unified CLI for TD-MPC2 run discovery and management.

Usage:
    python -m discover <command> [options]

Commands:
    refresh     Refresh cache from local logs and wandb
    status      Show training progress summary
    running     Show currently running tasks
    tasks       List all tasks with progress
    domains     Show progress by domain
    restart     Show/submit jobs for stalled tasks
    eval        List tasks needing eval or submit eval jobs
    videos      Collect or prune videos
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import pandas as pd

from .config import get_logs_dir, get_cache_path, get_wandb_project, get_target_step
from tasks import load_task_list, task_to_index


def require_pandas():
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required. Install with: pip install pandas")
    return pd


# ============================================================================
# Subcommand handlers
# ============================================================================

def cmd_refresh(args) -> int:
    """Refresh cache from local logs and wandb."""
    from .api import load_df_with_meta
    
    print(f"Refreshing cache...")
    print(f"  Logs dir:      {args.logs_dir}")
    print(f"  Wandb project: {args.wandb_project}")
    print(f"  Cache path:    {args.cache_path}")
    print()
    
    df, timestamp, used_cache = load_df_with_meta(
        refresh=True,
        logs_dir=args.logs_dir,
        cache_path=args.cache_path,
        wandb_project=args.wandb_project,
        wandb_limit=args.limit,
    )
    
    print(f"Done. {len(df)} runs cached at {timestamp.isoformat()}")
    return 0


def cmd_status(args) -> int:
    """Show training progress summary."""
    pd = require_pandas()
    from .api import load_df
    from .analysis import attach_max_step, best_step_by_task
    
    df = load_df(
        refresh=args.refresh,
        logs_dir=args.logs_dir,
        cache_path=args.cache_path,
        wandb_project=args.wandb_project,
    )
    
    if df.empty:
        print("No runs found.")
        return 1
    
    # Filter to official tasks only (unless --all)
    if not getattr(args, 'show_all', False):
        official_tasks = set(load_task_list())
        df = df[df['task'].isin(official_tasks)]
    
    df_with_step = attach_max_step(df)
    best = best_step_by_task(df)
    
    steps = best['max_step'].fillna(0)
    n_tasks = len(steps)
    tasks = best['task'].tolist()
    target = args.target_step
    
    # Determine which tasks have active runs (wandb-verified)
    has_wandb = df_with_step['found_in'].isin(['wandb', 'both']) if 'found_in' in df_with_step.columns else pd.Series(True, index=df_with_step.index)
    running_mask = (df_with_step['status'] == 'running') & has_wandb
    tasks_with_running = set(df_with_step[running_mask]['task'].unique())
    
    # Categorize tasks
    completed_mask = steps >= target
    not_started_mask = steps == 0
    in_progress_mask = (steps > 0) & (steps < target)
    
    running_tasks = []
    stalled_tasks = []
    for task, is_in_progress in zip(tasks, in_progress_mask):
        if is_in_progress:
            if task in tasks_with_running:
                running_tasks.append(task)
            else:
                stalled_tasks.append(task)
    
    completed = completed_mask.sum()
    running = len(running_tasks)
    stalled = len(stalled_tasks)
    not_started = not_started_mask.sum()
    
    pct_complete = 100 * completed / n_tasks
    avg_progress = 100 * steps.mean() / target
    median_progress = 100 * steps.median() / target
    
    print("=" * 60)
    print(f"{'TRAINING PROGRESS SUMMARY':^60}")
    print("=" * 60)
    print(f"  Total tasks:         {n_tasks:>6}")
    print(f"  Target step:         {target:>6,}")
    print("-" * 60)
    print(f"  🟢 Completed:        {completed:>6} ({pct_complete:.1f}%)")
    print(f"  🔵 Running:          {running:>6} ({100*running/n_tasks:.1f}%)  <- active in wandb")
    print(f"  🟠 Stalled:          {stalled:>6} ({100*stalled/n_tasks:.1f}%)  <- needs restart")
    print(f"  🔴 Not Started:      {not_started:>6} ({100*not_started/n_tasks:.1f}%)")
    print("-" * 60)
    print(f"  Average progress:    {avg_progress:>6.1f}%")
    print(f"  Median progress:     {median_progress:>6.1f}%")
    print(f"  Min steps:           {int(steps.min()):>6,}")
    print(f"  Max steps:           {int(steps.max()):>6,}")
    print("=" * 60)
    
    return 0


def cmd_running(args) -> int:
    """Show currently running tasks."""
    from .api import load_df
    from .plots import currently_running_tasks
    
    df = load_df(
        refresh=args.refresh,
        logs_dir=args.logs_dir,
        cache_path=args.cache_path,
        wandb_project=args.wandb_project,
    )
    
    if df.empty:
        print("No runs found.")
        return 1
    
    # Filter to official tasks only (unless --all)
    if not getattr(args, 'show_all', False):
        official_tasks = set(load_task_list())
        df = df[df['task'].isin(official_tasks)]
    
    # Show summary + running tasks list (currently_running_tasks calls running_runs_summary internally)
    currently_running_tasks(df, target_step=args.target_step)
    
    return 0


def cmd_tasks(args) -> int:
    """List all tasks with progress."""
    pd = require_pandas()
    from .api import load_df
    from .analysis import attach_max_step, best_step_by_task
    
    df = load_df(
        refresh=args.refresh,
        logs_dir=args.logs_dir,
        cache_path=args.cache_path,
        wandb_project=args.wandb_project,
    )
    
    target = args.target_step
    show_all = getattr(args, 'show_all', False)
    
    # Get official task list (used for filtering and adding missing tasks)
    official_tasks = set(load_task_list()) if not show_all else None
    
    if df.empty:
        if show_all:
            print("No runs found.")
            return 1
        # No runs - show all official tasks as not started
        best = pd.DataFrame({'task': list(official_tasks)})
        best['max_step'] = 0
        best['progress_pct'] = 0.0
        best['running_runs'] = 0
    else:
        # Filter to official tasks only (unless --all)
        if not show_all:
            df = df[df['task'].isin(official_tasks)]
        
        df_with_step = attach_max_step(df)
        best = best_step_by_task(df)
        
        # Determine running status per task
        has_wandb = df_with_step['found_in'].isin(['wandb', 'both']) if 'found_in' in df_with_step.columns else pd.Series(True, index=df_with_step.index)
        running_mask = (df_with_step['status'] == 'running') & has_wandb
        running_counts = df_with_step[running_mask].groupby('task').size()
        
        # Build output table
        best = best.copy()
        best['progress_pct'] = (100 * best['max_step'].fillna(0) / target).clip(upper=100).round(1)
        best['running_runs'] = best['task'].map(running_counts).fillna(0).astype(int)
        
        # Add any official tasks with no runs yet (unless --all)
        if not show_all:
            missing_tasks = official_tasks - set(best['task'])
            if missing_tasks:
                missing_df = pd.DataFrame({'task': list(missing_tasks)})
                missing_df['max_step'] = 0
                missing_df['progress_pct'] = 0.0
                missing_df['running_runs'] = 0
                best = pd.concat([best, missing_df], ignore_index=True)
    
    # Categorize
    best['category'] = 'stalled'
    best.loc[best['max_step'] >= target, 'category'] = 'completed'
    best.loc[best['max_step'].fillna(0) == 0, 'category'] = 'not_started'
    best.loc[(best['running_runs'] > 0) & (best['category'] == 'stalled'), 'category'] = 'running'
    
    # Apply filters
    if args.not_started:
        best = best[best['category'] == 'not_started']
    if args.stalled:
        best = best[best['category'] == 'stalled']
    if args.running:
        best = best[best['category'] == 'running']
    if args.completed:
        best = best[best['category'] == 'completed']
    
    # Sort
    best = best.sort_values(['progress_pct', 'task'], ascending=[True, True])
    
    # Output
    if args.format == 'json':
        import json
        cols = ['task', 'max_step', 'progress_pct', 'running_runs', 'category']
        print(json.dumps(best[cols].to_dict(orient='records'), indent=2))
    elif args.format == 'csv':
        cols = ['task', 'max_step', 'progress_pct', 'running_runs', 'category']
        print(best[cols].to_csv(index=False))
    else:
        # Table format
        print("=" * 80)
        print(f"{'ALL TASKS':^80}")
        print("=" * 80)
        print(f"{'Task':<45} {'Progress':>10} {'Max Step':>12} {'Runs':>6} {'Status':>8}")
        print("-" * 80)
        
        status_emoji = {'completed': '🟢', 'running': '🔵', 'stalled': '🟠', 'not_started': '🔴'}
        for _, row in best.iterrows():
            step_str = f"{int(row['max_step']):,}" if row['max_step'] > 0 else "0"
            emoji = status_emoji.get(row['category'], '')
            print(f"   {row['task']:<42} {row['progress_pct']:>8.1f}% {step_str:>12} {int(row['running_runs']):>6} {emoji:>8}")
        
        print("-" * 80)
        print(f"Total: {len(best)} tasks")
    
    return 0


def cmd_domains(args) -> int:
    """Show progress by domain."""
    pd = require_pandas()
    from .api import load_df
    from .analysis import best_step_by_task
    
    df = load_df(
        refresh=args.refresh,
        logs_dir=args.logs_dir,
        cache_path=args.cache_path,
        wandb_project=args.wandb_project,
    )
    
    if df.empty:
        print("No runs found.")
        return 1
    
    # Filter to official tasks only (unless --all)
    if not getattr(args, 'show_all', False):
        official_tasks = set(load_task_list())
        df = df[df['task'].isin(official_tasks)]
    
    target = args.target_step
    best = best_step_by_task(df)
    
    def get_domain(task: str) -> str:
        for sep in ['-', '_']:
            if sep in task:
                return task.split(sep)[0]
        return task
    
    best_copy = best.copy()
    best_copy['domain'] = best_copy['task'].apply(get_domain)
    best_copy['progress_pct'] = (100 * best_copy['max_step'].fillna(0) / target).clip(upper=100)
    best_copy['is_complete'] = best_copy['max_step'] >= target
    
    domain_stats = best_copy.groupby('domain').agg({
        'task': 'count',
        'max_step': 'mean',
        'progress_pct': 'mean',
        'is_complete': 'sum'
    }).rename(columns={
        'task': 'n_tasks',
        'max_step': 'avg_steps',
        'progress_pct': 'avg_progress',
        'is_complete': 'n_complete'
    })
    domain_stats['completion_rate'] = 100 * domain_stats['n_complete'] / domain_stats['n_tasks']
    domain_stats = domain_stats.sort_values('avg_progress', ascending=False)
    
    if args.format == 'json':
        import json
        print(json.dumps(domain_stats.reset_index().to_dict(orient='records'), indent=2))
    elif args.format == 'csv':
        print(domain_stats.reset_index().to_csv(index=False))
    else:
        print("=" * 80)
        print(f"{'PROGRESS BY DOMAIN':^80}")
        print("=" * 80)
        print(f"{'Domain':<20} {'Tasks':>8} {'Complete':>10} {'Avg Progress':>14} {'Completion %':>14}")
        print("-" * 80)
        for idx, row in domain_stats.iterrows():
            print(f"{idx:<20} {int(row['n_tasks']):>8} {int(row['n_complete']):>10} "
                  f"{row['avg_progress']:>13.1f}% {row['completion_rate']:>13.1f}%")
        print("-" * 80)
    
    return 0


def cmd_restart(args) -> int:
    """Show/submit jobs for stalled tasks."""
    pd = require_pandas()
    from .api import load_df
    from .plots import tasks_needing_restart
    
    df = load_df(
        refresh=args.refresh,
        logs_dir=args.logs_dir,
        cache_path=args.cache_path,
        wandb_project=args.wandb_project,
    )
    
    if df.empty:
        print("No runs found.")
        return 1
    
    # Filter to official tasks only
    official_tasks = set(load_task_list())
    df = df[df['task'].isin(official_tasks)]
    
    # Get tasks needing restart
    needing = tasks_needing_restart(df, target_step=args.target_step)
    
    if needing.empty:
        return 0
    
    task_names = needing['task'].tolist()
    
    # Map tasks to job indices
    try:
        task_indices = [(task, task_to_index(task)) for task in task_names]
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError mapping tasks to job indices: {e}")
        return 1
    
    # Group by queue/GPU-mode (matching submit_expert_array.sh)
    # Tasks from tasks.json sorted alphabetically (225 total after filtering variants):
    # 1-58: long-gpu exclusive (non-ManiSkill)
    # 59-105: long-gpu shared (ManiSkill, needs non-exclusive GPU for SAPIEN/Vulkan)
    # 106-225: short-gpu exclusive (non-ManiSkill)
    groups = {
        'long_exclusive': [],
        'long_shared': [],
        'short_exclusive': [],
    }
    
    for task, idx in task_indices:
        if 1 <= idx <= 58:
            groups['long_exclusive'].append((task, idx))
        elif 59 <= idx <= 105:
            groups['long_shared'].append((task, idx))
        elif 106 <= idx <= 225:
            groups['short_exclusive'].append((task, idx))
    
    # Generate bsub commands
    tdmpc2_dir = get_logs_dir().parent  # tdmpc2/
    
    def format_bsub_cmd(indices: List[int], queue: str, gpu_mode: str, walltime: str) -> str:
        """Format a bsub command for a group of indices."""
        if not indices:
            return ""
        
        # Format indices as comma-separated or ranges
        if len(indices) == 1:
            idx_str = str(indices[0])
        else:
            idx_str = ','.join(str(i) for i in sorted(indices))
        
        mode_opt = 'mode=exclusive_process' if gpu_mode == 'exclusive' else ''
        gpu_spec = f'"num=1:{mode_opt}"' if mode_opt else '"num=1"'
        
        return f'''bsub -J "newt-expert[{idx_str}]" \\
  -q {queue} \\
  -n 1 -gpu {gpu_spec} -R "rusage[mem=32GB]" -W {walltime} -r \\
  -o {tdmpc2_dir}/logs/lsf/newt-expert.%J.%I.log \\
  -e {tdmpc2_dir}/logs/lsf/newt-expert.%J.%I.log \\
  -u "$USER" -N \\
  -app nvidia-gpu \\
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.1" \\
  {tdmpc2_dir}/jobs/run_expert_task.sh'''
    
    commands = []
    
    if groups['long_exclusive']:
        indices = [idx for _, idx in groups['long_exclusive']]
        cmd = format_bsub_cmd(indices, 'long-gpu', 'exclusive', '48:00')
        commands.append(('long-gpu (exclusive, 48h)', cmd, groups['long_exclusive']))
    
    if groups['long_shared']:
        indices = [idx for _, idx in groups['long_shared']]
        cmd = format_bsub_cmd(indices, 'long-gpu', 'shared', '48:00')
        commands.append(('long-gpu (shared/ManiSkill, 48h)', cmd, groups['long_shared']))
    
    if groups['short_exclusive']:
        indices = [idx for _, idx in groups['short_exclusive']]
        cmd = format_bsub_cmd(indices, 'short-gpu', 'exclusive', '5:45')
        commands.append(('short-gpu (exclusive, 6h)', cmd, groups['short_exclusive']))
    
    # Print or execute
    print("\n" + "=" * 80)
    print(f"{'RESTART COMMANDS':^80}")
    print("=" * 80)
    
    for desc, cmd, tasks_in_group in commands:
        print(f"\n# {desc} ({len(tasks_in_group)} tasks)")
        print(f"# Tasks: {', '.join(t for t, _ in tasks_in_group[:5])}", end='')
        if len(tasks_in_group) > 5:
            print(f" ... and {len(tasks_in_group) - 5} more")
        else:
            print()
        print(cmd)
    
    print("\n" + "=" * 80)
    
    if args.submit:
        print("\nSubmitting jobs...")
        for desc, cmd, tasks_in_group in commands:
            print(f"\n  Submitting {len(tasks_in_group)} tasks to {desc.split()[0]}...")
            # Execute the bsub command
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"    ✓ {result.stdout.strip()}")
            else:
                print(f"    ✗ Error: {result.stderr.strip()}")
        print("\nDone. Monitor with: bjobs -J 'newt-expert*'")
    else:
        print("\nDry-run mode. Use --submit to actually run these commands.")
    
    return 0


def cmd_eval(args) -> int:
    """List tasks needing eval or submit eval jobs."""
    from .api import load_df
    from .config import get_logs_dir
    from .eval import tasks_ready_for_eval, generate_eval_script
    
    df = load_df(
        refresh=args.refresh,
        logs_dir=args.logs_dir,
        cache_path=args.cache_path,
        wandb_project=args.wandb_project,
    )
    
    if df.empty:
        print("No runs found.")
        return 1
    
    logs_dir = args.logs_dir or get_logs_dir()
    target = args.target_step
    min_progress = args.min_progress
    
    ready_df, tasks_need_eval, tasks_with_videos = tasks_ready_for_eval(
        df,
        logs_dir=logs_dir,
        target_step=target,
        min_progress=min_progress,
    )
    
    print(f"\nTasks ≥{min_progress*100:.0f}% trained: {len(ready_df)}")
    print(f"  With videos:    {len(tasks_with_videos)}")
    print(f"  Need eval:      {len(tasks_need_eval)}")
    
    if args.action == 'list':
        if tasks_need_eval:
            print("\nTasks needing eval (no videos):")
            for task in tasks_need_eval:
                print(f"  • {task}")
    
    elif args.action == 'submit':
        if not tasks_need_eval:
            print("\n✅ All ready tasks already have videos!")
            return 0
        
        tdmpc2_dir = logs_dir.parent
        output_dir = tdmpc2_dir / 'jobs'
        
        generate_eval_script(
            tasks=tasks_need_eval,
            output_dir=output_dir,
            project_root=tdmpc2_dir,
        )
        
        if args.submit:
            script_path = output_dir / 'run_eval_need_videos.lsf'
            print(f"\nSubmitting eval jobs from {script_path}...")
            result = subprocess.run(
                ['bsub', '<', str(script_path)],
                shell=True,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"  ✓ {result.stdout.strip()}")
            else:
                print(f"  ✗ Error: {result.stderr.strip()}")
        else:
            print("\nScript generated. Use 'make submit-eval' or add --submit to submit.")
    
    return 0


def cmd_videos(args) -> int:
    """Collect or prune videos."""
    from .api import load_df
    from .config import get_logs_dir
    
    logs_dir = args.logs_dir or get_logs_dir()
    target = args.target_step
    
    if args.action == 'collect':
        from .eval import collect_videos
        
        df = load_df(
            refresh=args.refresh,
            logs_dir=args.logs_dir,
            cache_path=args.cache_path,
            wandb_project=args.wandb_project,
        )
        
        if df.empty:
            print("No runs found.")
            return 1
        
        tdmpc2_dir = logs_dir.parent
        output_dir = args.output or (tdmpc2_dir / 'discover' / 'videos_for_presentation')
        
        collect_videos(
            df,
            logs_dir=logs_dir,
            output_dir=output_dir,
            target_step=target,
            min_progress=args.min_progress,
            create_symlinks=not args.copy,
        )
    
    elif args.action == 'prune':
        from .eval import prune_old_videos
        
        tdmpc2_dir = logs_dir.parent
        output_dir = args.output or (tdmpc2_dir / 'discover' / 'videos_for_presentation')
        
        prune_old_videos(output_dir, dry_run=args.dry_run)
    
    return 0


# ============================================================================
# Main CLI
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog='discover',
        description='TD-MPC2 run discovery and management CLI',
    )
    
    # Common arguments
    parser.add_argument('--logs-dir', type=Path, default=None,
                        help='Override logs directory')
    parser.add_argument('--cache-path', type=Path, default=None,
                        help='Override cache file path')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='Override wandb project')
    parser.add_argument('--target-step', type=int, default=None,
                        help='Override target step')
    parser.add_argument('--refresh', action='store_true',
                        help='Force refresh from local logs and wandb')
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # refresh
    p_refresh = subparsers.add_parser('refresh', help='Refresh cache from local logs and wandb')
    p_refresh.add_argument('--limit', type=int, default=None, help='Limit wandb runs fetched')
    
    # status
    p_status = subparsers.add_parser('status', help='Show training progress summary')
    p_status.add_argument('--all', action='store_true', dest='show_all',
                          help='Show all tasks (including non-official)')
    
    # running
    p_running = subparsers.add_parser('running', help='Show currently running tasks')
    p_running.add_argument('--all', action='store_true', dest='show_all',
                           help='Show all tasks (including non-official)')
    
    # tasks
    p_tasks = subparsers.add_parser('tasks', help='List all tasks with progress')
    p_tasks.add_argument('--format', choices=['table', 'json', 'csv'], default='table',
                         help='Output format')
    p_tasks.add_argument('--not-started', action='store_true', help='Filter to not-started tasks')
    p_tasks.add_argument('--stalled', action='store_true', help='Filter to stalled tasks')
    p_tasks.add_argument('--running', action='store_true', help='Filter to running tasks')
    p_tasks.add_argument('--completed', action='store_true', help='Filter to completed tasks')
    p_tasks.add_argument('--all', action='store_true', dest='show_all',
                         help='Show all tasks (including non-official)')
    
    # domains
    p_domains = subparsers.add_parser('domains', help='Show progress by domain')
    p_domains.add_argument('--format', choices=['table', 'json', 'csv'], default='table',
                           help='Output format')
    p_domains.add_argument('--all', action='store_true', dest='show_all',
                           help='Show all tasks (including non-official)')
    
    # restart
    p_restart = subparsers.add_parser('restart', help='Show/submit jobs for stalled tasks')
    p_restart.add_argument('--submit', action='store_true',
                           help='Actually submit bsub commands (default: dry-run)')
    
    # eval
    p_eval = subparsers.add_parser('eval', help='List tasks needing eval or submit eval jobs')
    p_eval.add_argument('action', choices=['list', 'submit'], default='list', nargs='?',
                        help='Action: list tasks or submit eval jobs')
    p_eval.add_argument('--min-progress', type=float, default=0.5,
                        help='Minimum progress to be ready for eval (default: 0.5)')
    p_eval.add_argument('--submit', action='store_true',
                        help='Actually submit the generated eval script')
    
    # videos
    p_videos = subparsers.add_parser('videos', help='Collect or prune videos')
    p_videos.add_argument('action', choices=['collect', 'prune'],
                          help='Action: collect or prune videos')
    p_videos.add_argument('--output', type=Path, default=None,
                          help='Output directory')
    p_videos.add_argument('--min-progress', type=float, default=0.5,
                          help='Minimum progress for video collection (default: 0.5)')
    p_videos.add_argument('--copy', action='store_true',
                          help='Copy files instead of creating symlinks')
    p_videos.add_argument('--dry-run', action='store_true',
                          help='For prune: show what would be removed')
    
    args = parser.parse_args(argv)
    
    # Apply defaults from config
    if args.logs_dir is None:
        args.logs_dir = get_logs_dir()
    if args.cache_path is None:
        args.cache_path = get_cache_path()
    if args.wandb_project is None:
        args.wandb_project = get_wandb_project()
    if args.target_step is None:
        args.target_step = get_target_step()
    
    # Dispatch to handler
    handlers = {
        'refresh': cmd_refresh,
        'status': cmd_status,
        'running': cmd_running,
        'tasks': cmd_tasks,
        'domains': cmd_domains,
        'restart': cmd_restart,
        'eval': cmd_eval,
        'videos': cmd_videos,
    }
    
    return handlers[args.command](args)


if __name__ == '__main__':
    sys.exit(main())

