"""Analysis functions for TD-MPC2 training runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


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
