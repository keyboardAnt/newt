"""TD-MPC2 Run Discovery and Analysis Tools.

Primary API:
    from discover import load_df
    df = load_df()  # loads from cache; use load_df(refresh=True) to refresh

Submodules (import explicitly):
    discover.api         - load_df, load_df_with_meta
    discover.config      - get_logs_dir, get_cache_path, get_wandb_project, get_target_step
    discover.runs        - discover_local_logs, discover_wandb_runs
    discover.cache       - RunsCache, load_all_runs
    discover.liveness    - attach_liveness, build_task_progress
    discover.progress    - attach_max_step, best_step_by_task, parse_step
    discover.plots       - training_overview, running_runs_summary, etc.
    discover.eval        - tasks_ready_for_eval, collect_videos, etc.
    discover.wandb_connector - get_api, fetch_runs, fetch_runs_by_id, iter_model_collections
    discover.cleanup.model_registry - plan_cleanup_latest_checkpoint_per_expert
"""

from .api import load_df, load_df_with_meta

__all__ = ["load_df", "load_df_with_meta"]
