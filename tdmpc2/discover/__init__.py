"""TD-MPC2 Run Discovery and Analysis Tools.

Modules:
    runs: Discover runs from local logs or W&B
    cache: Caching and data loading
    analysis: Data analysis functions
    plots: Visualization functions
    eval: Evaluation and video management
"""

from .runs import discover_local_logs, discover_wandb_runs
from .cache import RunsCache, load_all_runs
from .analysis import best_step_by_task, attach_max_step, attach_runtime, parse_step
from .plots import (
    training_overview, plot_max_steps, tasks_needing_attention, progress_by_domain,
    running_runs_summary, tasks_needing_restart, duplicate_running_runs, duplicate_run_details,
    currently_running_tasks,
)
from .eval import tasks_ready_for_eval, generate_eval_script, collect_videos, download_wandb_videos

__all__ = [
    # runs
    'discover_local_logs',
    'discover_wandb_runs',
    # cache
    'RunsCache',
    'load_all_runs',
    # analysis
    'parse_step',
    'best_step_by_task',
    'attach_max_step',
    'attach_runtime',
    # plots
    'training_overview',
    'plot_max_steps',
    'tasks_needing_attention',
    'progress_by_domain',
    'running_runs_summary',
    'tasks_needing_restart',
    'duplicate_running_runs',
    'duplicate_run_details',
    'currently_running_tasks',
    # eval
    'tasks_ready_for_eval',
    'generate_eval_script',
    'collect_videos',
    'download_wandb_videos',
]
