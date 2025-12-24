"""Analysis functions for TD-MPC2 training runs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    import pandas as pd


def parse_step(path: Union[str, Path]) -> int:
    """Parse step number from checkpoint filename (e.g., '1_000_000.pt' -> 1000000)."""
    stem = Path(path).stem
    return int(stem.replace('_', '').replace(',', ''))


def require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required. Install with `pip install pandas`.") from exc
    return pd


def attach_runtime(df: "pd.DataFrame") -> "pd.DataFrame":
    """Add a 'runtime' column extracted from summary."""
    pd = require_pandas()
    
    if df.empty:
        return df.assign(runtime=pd.Series(dtype='float'))

    out = df.copy()
    if 'summary' not in out.columns:
        out['runtime'] = pd.Series(dtype='float')
        return out

    def _runtime(summary):
        if isinstance(summary, dict):
            return summary.get('runtime', summary.get('_runtime'))
        return None

    out['runtime'] = pd.to_numeric(out['summary'].apply(_runtime), errors='coerce')
    return out


def attach_max_step(df: "pd.DataFrame") -> "pd.DataFrame":
    """Add a 'max_step' column combining ckpt_step (local) and summary['_step'] (wandb)."""
    pd = require_pandas()
    
    if df.empty:
        return df.assign(max_step=pd.Series(dtype='float'))

    out = df.copy()

    def _get_step(row):
        if pd.notna(row.get('ckpt_step')):
            return row['ckpt_step']
        summary = row.get('summary')
        if isinstance(summary, dict):
            return summary.get('_step')
        return None

    out['max_step'] = out.apply(_get_step, axis=1)
    out['max_step'] = pd.to_numeric(out['max_step'], errors='coerce')
    return out


def best_step_by_task(df: "pd.DataFrame") -> "pd.DataFrame":
    """Select the run with the maximum step for each task."""
    pd = require_pandas()
    
    if df.empty:
        return df
    df_with_step = attach_max_step(df)
    filled = df_with_step['max_step'].fillna(-1)
    idx = filled.groupby(df_with_step['task']).idxmax()
    return df_with_step.loc[idx].sort_values('task')


def best_runtime_by_task(df: "pd.DataFrame") -> "pd.DataFrame":
    """Select the run with the maximum runtime for each task."""
    pd = require_pandas()
    
    if df.empty:
        return df
    filled = df['runtime'].fillna(-1)
    idx = filled.groupby(df['task']).idxmax()
    return df.loc[idx].sort_values('task')


def best_runtime_table(df: "pd.DataFrame") -> "pd.DataFrame":
    """Get a table of best runs by runtime with selected columns."""
    df_rt = attach_runtime(df)
    if df_rt.empty:
        return df_rt
    best = best_runtime_by_task(df_rt)
    cols = [
        'task', 'runtime', 'exp_name', 'seed', 'source',
        'ckpt_path', 'ckpt_step', 'url', 'run_dir', 'summary',
    ]
    present = [c for c in cols if c in best.columns]
    return best[present]


def build_task_progress(
    df: "pd.DataFrame",
    target_step: int = 5_000_000,
    task_list: Optional[List[str]] = None,
    include_unknown: bool = False,
) -> "pd.DataFrame":
    """Build a per-task progress table aligned to official task list.
    
    This is the shared helper that ensures both notebook (training_overview)
    and CLI (discover tasks) use the same logic for counting tasks.
    
    Args:
        df: DataFrame with all runs (from load_df())
        target_step: Target step for completion (default 5M)
        task_list: List of official tasks. Defaults to load_task_list().
        include_unknown: If True, include tasks not in task_list.
            If False (default), filter to only official tasks and warn about extras.
    
    Returns:
        DataFrame with columns:
            task, max_step, running_runs, progress_pct, category
        One row per task in task_list (plus unknown tasks if include_unknown=True).
        Category is one of: 'completed', 'running', 'stalled', 'not_started'.
    """
    pd = require_pandas()
    from tasks import load_task_list as _load_task_list
    
    # Default to official task list
    if task_list is None:
        task_list = _load_task_list()
    official_set = set(task_list)
    
    # Handle empty df
    if df.empty:
        result = pd.DataFrame({'task': task_list})
        result['max_step'] = 0
        result['running_runs'] = 0
        result['progress_pct'] = 0.0
        result['category'] = 'not_started'
        return result.sort_values('task').reset_index(drop=True)
    
    df_with_step = attach_max_step(df)
    
    # Identify unknown tasks (present in df but not in task_list)
    tasks_in_df = set(df_with_step['task'].dropna().unique())
    unknown_tasks = tasks_in_df - official_set
    
    # Filter to official tasks only (unless include_unknown)
    if not include_unknown:
        df_with_step = df_with_step[df_with_step['task'].isin(official_set)]
    
    # Compute max_step per task
    max_steps = df_with_step.groupby('task')['max_step'].max()
    
    # Compute running_runs per task (wandb-verified only)
    has_wandb = df_with_step['found_in'].isin(['wandb', 'both']) if 'found_in' in df_with_step.columns else pd.Series(True, index=df_with_step.index)
    running_mask = (df_with_step['status'] == 'running') & has_wandb
    running_counts = df_with_step[running_mask].groupby('task').size()
    
    # Build result aligned to task_list
    if include_unknown:
        all_tasks = sorted(set(task_list) | unknown_tasks)
    else:
        all_tasks = list(task_list)
    
    result = pd.DataFrame({'task': all_tasks})
    result['max_step'] = result['task'].map(max_steps).fillna(0)
    result['running_runs'] = result['task'].map(running_counts).fillna(0).astype(int)
    result['progress_pct'] = (100 * result['max_step'] / target_step).clip(upper=100).round(1)
    
    # Categorize
    result['category'] = 'stalled'  # default for in-progress without running
    result.loc[result['max_step'] >= target_step, 'category'] = 'completed'
    result.loc[result['max_step'] == 0, 'category'] = 'not_started'
    result.loc[(result['running_runs'] > 0) & (result['category'] == 'stalled'), 'category'] = 'running'
    
    # Sort by progress then task name
    result = result.sort_values(['progress_pct', 'task'], ascending=[True, True]).reset_index(drop=True)
    
    # Return unknown task count as an attribute for callers that want to warn
    result.attrs['unknown_tasks'] = sorted(unknown_tasks)
    result.attrs['n_unknown'] = len(unknown_tasks)
    
    return result
