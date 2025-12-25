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
    from .api import load_df
    from .liveness import build_task_progress, print_unknown_tasks_warning
    
    df = load_df(
        refresh=args.refresh,
        logs_dir=args.logs_dir,
        cache_path=args.cache_path,
        wandb_project=args.wandb_project,
    )
    
    target = args.target_step
    show_all = getattr(args, 'show_all', False)
    
    # Use shared helper aligned to official task list (single source of truth: discover.liveness)
    # If --all, pass task_list=None and include_unknown=True to show everything
    if show_all:
        progress = build_task_progress(df, target_step=target, task_list=None, include_unknown=True)
    else:
        progress = build_task_progress(df, target_step=target)
    
    n_tasks = len(progress)
    steps = progress['max_step']
    
    # Count by category
    completed = (progress['category'] == 'completed').sum()
    running = (progress['category'] == 'running').sum()
    stalled = (progress['category'] == 'stalled').sum()
    not_started = (progress['category'] == 'not_started').sum()
    
    pct_complete = 100 * completed / n_tasks
    avg_progress = progress['progress_pct'].mean()
    median_progress = progress['progress_pct'].median()
    
    print("=" * 60)
    print(f"{'TRAINING PROGRESS SUMMARY':^60}")
    print("=" * 60)
    print(f"  Total tasks:         {n_tasks:>6}  (from tasks.json)")
    print(f"  Target step:         {target:>6,}")
    print("-" * 60)
    print(f"  🟢 Completed:        {completed:>6} ({pct_complete:.1f}%)")
    print(f"  🔵 Running:          {running:>6} ({100*running/n_tasks:.1f}%)  <- active (see discover.liveness)")
    print(f"  🟠 Stalled:          {stalled:>6} ({100*stalled/n_tasks:.1f}%)  <- needs restart")
    print(f"  🔴 Not Started:      {not_started:>6} ({100*not_started/n_tasks:.1f}%)")
    print("-" * 60)
    print(f"  Average progress:    {avg_progress:>6.1f}%")
    print(f"  Median progress:     {median_progress:>6.1f}%")
    print(f"  Min steps:           {int(steps.min()):>6,}")
    print(f"  Max steps:           {int(steps.max()):>6,}")
    print("=" * 60)
    
    # Warn about unknown tasks if any (only when not using --all)
    if not show_all:
        print_unknown_tasks_warning(progress)
    
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
    from .api import load_df
    from .liveness import build_task_progress, print_unknown_tasks_warning
    
    df = load_df(
        refresh=args.refresh,
        logs_dir=args.logs_dir,
        cache_path=args.cache_path,
        wandb_project=args.wandb_project,
    )
    
    target = args.target_step
    show_all = getattr(args, 'show_all', False)
    
    # Use shared helper aligned to tasks.json (single source of truth: discover.liveness)
    if show_all:
        progress = build_task_progress(df, target_step=target, task_list=None, include_unknown=True)
    else:
        progress = build_task_progress(df, target_step=target)
    
    # Apply filters
    if args.not_started:
        progress = progress[progress['category'] == 'not_started']
    if args.stalled:
        progress = progress[progress['category'] == 'stalled']
    if args.running:
        progress = progress[progress['category'] == 'running']
    if args.completed:
        progress = progress[progress['category'] == 'completed']
    
    # Sort (already sorted by progress, but re-sort after filter for consistency)
    progress = progress.sort_values(['progress_pct', 'task'], ascending=[True, True])
    
    # Output
    if args.format == 'json':
        import json
        cols = ['task', 'max_step', 'progress_pct', 'heartbeat_alive_runs', 'wandb_running_runs', 'category']
        print(json.dumps(progress[cols].to_dict(orient='records'), indent=2))
    elif args.format == 'csv':
        cols = ['task', 'max_step', 'progress_pct', 'heartbeat_alive_runs', 'wandb_running_runs', 'category']
        print(progress[cols].to_csv(index=False))
    else:
        # Table format
        print("=" * 90)
        print(f"{'ALL TASKS':^90}")
        print("=" * 90)
        print(f"{'Task':<45} {'Progress':>10} {'Max Step':>12} {'HB':>4} {'WB':>4} {'Status':>15}")
        print("-" * 90)
        
        status_emoji = {'completed': '🟢', 'running': '🔵', 'stalled': '🟠', 'not_started': '🔴'}
        for _, row in progress.iterrows():
            step_str = f"{int(row['max_step']):,}" if row['max_step'] > 0 else "0"
            emoji = status_emoji.get(row['category'], '')
            status_text = row['category'].replace('_', ' ')
            status_display = f"{emoji} {status_text}"
            hb = int(row.get('heartbeat_alive_runs', 0))
            wb = int(row.get('wandb_running_runs', 0))
            print(f"   {row['task']:<42} {row['progress_pct']:>8.1f}% {step_str:>12} {hb:>4} {wb:>4} {status_display:>15}")
        
        print("-" * 90)
        print(f"Total: {len(progress)} tasks")
        
        # Warn about unknown tasks
        if not show_all:
            print_unknown_tasks_warning(progress)
    
    return 0


def cmd_domains(args) -> int:
    """Show progress by domain."""
    from .api import load_df
    from .liveness import build_task_progress, print_unknown_tasks_warning
    
    df = load_df(
        refresh=args.refresh,
        logs_dir=args.logs_dir,
        cache_path=args.cache_path,
        wandb_project=args.wandb_project,
    )
    
    target = args.target_step
    show_all = getattr(args, 'show_all', False)
    
    # Use shared helper aligned to tasks.json (single source of truth: discover.liveness),
    # so domains include not-started tasks.
    # With --all, also include unknown tasks seen in runs.
    if show_all:
        progress = build_task_progress(df, target_step=target, task_list=None, include_unknown=True)
    else:
        progress = build_task_progress(df, target_step=target)
    
    def get_domain(task: str) -> str:
        for sep in ['-', '_']:
            if sep in task:
                return task.split(sep)[0]
        return task
    
    progress_copy = progress.copy()
    progress_copy['domain'] = progress_copy['task'].apply(get_domain)
    progress_copy['is_complete'] = progress_copy['max_step'] >= target
    
    domain_stats = progress_copy.groupby('domain').agg({
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
        
        if not show_all:
            print_unknown_tasks_warning(progress)
    
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
    
    # Get tasks needing restart (uses tasks.json alignment, includes not-started tasks)
    # No need to filter df - tasks_needing_restart uses build_task_progress which is tasks.json-aligned
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
    # 59-105: long-gpu exclusive (ManiSkill)
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
        
        # Always use exclusive GPU mode to avoid CUDA init OOM / GPU contention during restarts.
        gpu_spec = '"num=1:mode=exclusive_process"'
        
        return f'''bsub -J "newt-expert[{idx_str}]" \\
  -q {queue} \\
  -n 1 -gpu {gpu_spec} -R "rusage[mem=32GB]" -W {walltime} -r \\
  -o {tdmpc2_dir}/logs/lsf/newt-expert.%J.%I.log \\
  -e {tdmpc2_dir}/logs/lsf/newt-expert.%J.%I.log \\
  -u "$USER" -N \\
  -app nvidia-gpu \\
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.2" \\
  {tdmpc2_dir}/jobs/run_expert_task.sh'''
    
    commands = []
    
    if groups['long_exclusive']:
        indices = [idx for _, idx in groups['long_exclusive']]
        cmd = format_bsub_cmd(indices, 'long-gpu', 'exclusive', '48:00')
        commands.append(('long-gpu (exclusive, 48h)', cmd, groups['long_exclusive']))
    
    if groups['long_shared']:
        indices = [idx for _, idx in groups['long_shared']]
        cmd = format_bsub_cmd(indices, 'long-gpu', 'exclusive', '48:00')
        commands.append(('long-gpu (exclusive/ManiSkill, 48h)', cmd, groups['long_shared']))
    
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

