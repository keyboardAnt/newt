"""Visualization functions for TD-MPC2 training progress."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pandas as pd

from .analysis import best_step_by_task


def require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required.") from exc
    return pd


def plot_max_steps(df: "pd.DataFrame", target_step: Optional[int] = None) -> None:
    """Plot the max step per task to see training progress.
    
    Args:
        df: DataFrame with all runs
        target_step: Optional target step for color coding (green=complete, orange=in progress, red=not started)
    """
    pd = require_pandas()
    import matplotlib.pyplot as plt
    
    best = best_step_by_task(df)
    if best.empty:
        print('No runs found.')
        return
    if 'max_step' not in best.columns or best['max_step'].isna().all():
        print('No step data available to plot.')
        return

    plot_df = best[['task', 'max_step']].copy().sort_values('max_step', ascending=True)
    
    n_tasks = len(plot_df)
    fig_height = max(8, n_tasks * 0.2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    bars = ax.barh(plot_df['task'], plot_df['max_step'].fillna(0))

    if target_step:
        for bar, step in zip(bars, plot_df['max_step']):
            if pd.isna(step) or step == 0:
                bar.set_color('red')
            elif step >= target_step:
                bar.set_color('green')
            else:
                bar.set_color('orange')
        ax.axvline(x=target_step, color='green', linestyle='--', linewidth=2, label=f'Target: {target_step:,}')
        ax.legend()

    ax.set_xlabel('Max training step')
    ax.set_title('Training progress per task (max step reached)')
    ax.ticklabel_format(style='plain', axis='x')
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.show()


def training_overview(df: "pd.DataFrame", target_step: int = 5_000_000) -> None:
    """Generate overview statistics and visualizations for training progress."""
    pd = require_pandas()
    import matplotlib.pyplot as plt
    
    best = best_step_by_task(df)
    if best.empty:
        print('No runs found.')
        return
    
    steps = best['max_step'].fillna(0)
    n_tasks = len(steps)
    
    completed = (steps >= target_step).sum()
    in_progress = ((steps > 0) & (steps < target_step)).sum()
    not_started = (steps == 0).sum()
    
    pct_complete = 100 * completed / n_tasks
    avg_progress = 100 * steps.mean() / target_step
    median_progress = 100 * steps.median() / target_step
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    ax1 = axes[0]
    sizes = [completed, in_progress, not_started]
    labels = [f'Completed\n({completed})', f'In Progress\n({in_progress})', f'Not Started\n({not_started})']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0.05, 0, 0)
    ax1.pie(sizes, labels=labels, colors=colors, explode=explode,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Task Status Breakdown')
    
    # Cumulative progress
    ax2 = axes[1]
    progress_pct = (100 * steps / target_step).clip(upper=100)
    sorted_progress = progress_pct.sort_values().reset_index(drop=True)
    ax2.fill_between(range(len(sorted_progress)), sorted_progress, alpha=0.3, color='steelblue')
    ax2.plot(sorted_progress.values, color='steelblue', linewidth=2)
    ax2.axhline(y=100, color='green', linestyle='--', linewidth=2, label='100% (Target)')
    ax2.set_xlabel('Task rank (sorted by progress)')
    ax2.set_ylabel('Progress (%)')
    ax2.set_title('Cumulative Progress Distribution')
    ax2.set_ylim(0, 110)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()
    
    print("=" * 60)
    print(f"{'TRAINING PROGRESS SUMMARY':^60}")
    print("=" * 60)
    print(f"  Total tasks:         {n_tasks:>6}")
    print(f"  Target step:         {target_step:>6,}")
    print("-" * 60)
    print(f"  ✅ Completed:        {completed:>6} ({pct_complete:.1f}%)")
    print(f"  🔄 In Progress:      {in_progress:>6} ({100*in_progress/n_tasks:.1f}%)")
    print(f"  ❌ Not Started:      {not_started:>6} ({100*not_started/n_tasks:.1f}%)")
    print("-" * 60)
    print(f"  Average progress:    {avg_progress:>6.1f}%")
    print(f"  Median progress:     {median_progress:>6.1f}%")
    print(f"  Min steps:           {int(steps.min()):>6,}")
    print(f"  Max steps:           {int(steps.max()):>6,}")
    print("=" * 60)


def tasks_needing_attention(df: "pd.DataFrame", target_step: int = 5_000_000, bottom_n: int = 20) -> None:
    """Show tasks that may need attention: not started, crashed, or significantly behind."""
    pd = require_pandas()
    
    best = best_step_by_task(df)
    if best.empty:
        print('No runs found.')
        return
    
    steps = best['max_step'].fillna(0)
    not_started = best[steps == 0][['task', 'max_step']].copy()
    
    in_progress = best[(steps > 0) & (steps < target_step)].copy()
    behind = in_progress.nsmallest(bottom_n, 'max_step')[['task', 'max_step']].copy()
    behind['progress'] = (100 * behind['max_step'] / target_step).round(1).astype(str) + '%'
    
    print("=" * 70)
    print(f"{'TASKS REQUIRING ATTENTION':^70}")
    print("=" * 70)
    
    if not not_started.empty:
        print(f"\n❌ NOT STARTED ({len(not_started)} tasks):")
        print("-" * 70)
        for task in not_started['task'].tolist():
            print(f"   • {task}")
    else:
        print("\n✅ All tasks have started training!")
    
    if not behind.empty:
        print(f"\n🐢 LAGGING TASKS (bottom {len(behind)} by progress):")
        print("-" * 70)
        for _, row in behind.iterrows():
            steps_str = f"{int(row['max_step']):,}"
            print(f"   • {row['task']:<45} {steps_str:>12} ({row['progress']})")
    
    if 'updated_at' in best.columns:
        try:
            best_copy = best.copy()
            best_copy['updated_at_dt'] = pd.to_datetime(best_copy['updated_at'], errors='coerce')
            cutoff = datetime.now() - pd.Timedelta(hours=24)
            recent = best_copy[best_copy['updated_at_dt'] > cutoff]
            if len(recent) > 0:
                print(f"\n📊 RECENT ACTIVITY: {len(recent)} tasks updated in last 24 hours")
        except Exception:
            pass
    
    print("\n" + "=" * 70)


def progress_by_domain(df: "pd.DataFrame", target_step: int = 5_000_000) -> None:
    """Group tasks by domain prefix and show aggregate progress."""
    pd = require_pandas()
    import matplotlib.pyplot as plt
    
    best = best_step_by_task(df)
    if best.empty:
        print('No runs found.')
        return
    
    def get_domain(task: str) -> str:
        for sep in ['-', '_']:
            if sep in task:
                return task.split(sep)[0]
        return task
    
    best_copy = best.copy()
    best_copy['domain'] = best_copy['task'].apply(get_domain)
    best_copy['progress_pct'] = (100 * best_copy['max_step'].fillna(0) / target_step).clip(upper=100)
    best_copy['is_complete'] = best_copy['max_step'] >= target_step
    
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
    domain_stats = domain_stats.sort_values('avg_progress', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(domain_stats) * 0.3)))
    
    ax1 = axes[0]
    colors = ['#2ecc71' if p >= 100 else '#f39c12' if p > 0 else '#e74c3c' 
              for p in domain_stats['avg_progress']]
    ax1.barh(domain_stats.index, domain_stats['avg_progress'], color=colors, edgecolor='black', alpha=0.8)
    ax1.axvline(x=100, color='green', linestyle='--', linewidth=2)
    ax1.set_xlabel('Average Progress (%)')
    ax1.set_title('Average Training Progress by Domain')
    ax1.set_xlim(0, 110)
    
    ax2 = axes[1]
    ax2.barh(domain_stats.index, domain_stats['completion_rate'], color='steelblue', edgecolor='black', alpha=0.8)
    ax2.set_xlabel('Completion Rate (%)')
    ax2.set_title('Task Completion Rate by Domain')
    ax2.set_xlim(0, 110)
    
    for i, (idx, row) in enumerate(domain_stats.iterrows()):
        ax2.text(row['completion_rate'] + 2, i, f"({int(row['n_complete'])}/{int(row['n_tasks'])})", 
                va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    print("\nDomain Summary:")
    print("-" * 80)
    print(f"{'Domain':<20} {'Tasks':>8} {'Complete':>10} {'Avg Progress':>14} {'Completion %':>14}")
    print("-" * 80)
    for idx, row in domain_stats.sort_values('avg_progress', ascending=False).iterrows():
        print(f"{idx:<20} {int(row['n_tasks']):>8} {int(row['n_complete']):>10} "
              f"{row['avg_progress']:>13.1f}% {row['completion_rate']:>13.1f}%")
    print("-" * 80)
