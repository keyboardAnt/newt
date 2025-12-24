"""TD-MPC2 Run Discovery and Analysis Tools.

Modules:
    api: Simple API for loading run data (primary entry point)
    config: Centralized configuration
    cli: Unified CLI (python -m discover)
    runs: Discover runs from local logs or W&B
    cache: Caching and data loading
    liveness: Central run liveness detection + task aggregation
    analysis: Data analysis functions (delegates to liveness for task progress aggregation)
    plots: Visualization functions
    eval: Evaluation and video management
"""

# Primary API
from .api import load_df, load_df_with_meta
from .config import get_logs_dir, get_cache_path, get_wandb_project, get_target_step
from tasks import load_task_list, task_to_index, index_to_task

# Central liveness + aggregation (single source of truth)
from .liveness import (
    attach_heartbeat, attach_liveness, build_task_progress,
    get_heartbeat_ttl_s, print_unknown_tasks_warning,
)

# Low-level discovery
from .runs import discover_local_logs, discover_wandb_runs
from .cache import RunsCache, load_all_runs
from .progress import best_step_by_task, attach_max_step, attach_runtime, parse_step
from .plots import (
    training_overview, plot_max_steps, tasks_needing_attention, progress_by_domain,
    running_runs_summary, tasks_needing_restart, currently_running_tasks,
    stale_wandb_runs, stale_run_details,
    # Aliases for backwards compatibility
    duplicate_running_runs, duplicate_run_details,
)
from .eval import tasks_ready_for_eval, generate_eval_script, collect_videos, download_wandb_videos, prune_old_videos

__all__ = [
    # api (primary entry point)
    'load_df',
    'load_df_with_meta',
    # config
    'get_logs_dir',
    'get_cache_path',
    'get_wandb_project',
    'get_target_step',
    'load_task_list',
    'task_to_index',
    'index_to_task',
    # liveness (single source of truth)
    'attach_heartbeat',
    'attach_liveness',
    'build_task_progress',
    'get_heartbeat_ttl_s',
    'print_unknown_tasks_warning',
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
    'currently_running_tasks',
    'stale_wandb_runs',
    'stale_run_details',
    # Aliases for backwards compatibility
    'duplicate_running_runs',
    'duplicate_run_details',
    # eval
    'tasks_ready_for_eval',
    'generate_eval_script',
    'collect_videos',
    'download_wandb_videos',
    'prune_old_videos',
]
