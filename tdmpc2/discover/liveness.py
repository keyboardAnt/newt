"""Heartbeat + wandb liveness detection and task aggregation.

This module is the single source of truth for run liveness (heartbeat first,
wandb fallback) and task-level progress aggregation aligned to tasks.json.

Liveness precedence:
    1. Heartbeat is authoritative if run_dir/heartbeat.json exists:
       - alive if heartbeat is fresh (age <= TTL)
       - otherwise not alive (no W&B fallback when heartbeat exists)
    2. W&B fallback is only used when heartbeat.json does not exist:
       - alive if status == 'running' and found_in in {'wandb', 'both'} (plus freshness)

Usage:
    from discover.liveness import attach_liveness, build_task_progress
    
    df = load_df()
    df = attach_liveness(df)  # adds heartbeat_alive, wandb_alive, is_active, active_source
    
    progress = build_task_progress(df, target_step=5_000_000)  # task-aligned table
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

# Default TTL for heartbeat liveness (seconds)
# A heartbeat is considered "alive" if age <= TTL
HEARTBEAT_TTL_S_DEFAULT = 30

# Default TTL for considering a W&B run "alive" based on recent updates.
# This prevents stale W&B "running" states from showing as active forever.
WANDB_TTL_S_DEFAULT = 3600

def get_heartbeat_ttl_s() -> int:
    """Get heartbeat TTL from env or default."""
    env = os.environ.get("DISCOVER_HEARTBEAT_TTL_S")
    return int(env) if env else HEARTBEAT_TTL_S_DEFAULT


def get_wandb_ttl_s() -> int:
    """Get W&B TTL from env or default."""
    env = os.environ.get("DISCOVER_WANDB_TTL_S")
    return int(env) if env else WANDB_TTL_S_DEFAULT


# =============================================================================
# Helpers
# =============================================================================

def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required. Install with `pip install pandas`.") from exc
    return pd


def _parse_heartbeat_timestamp(ts: str) -> datetime:
    """Parse ISO-8601 timestamp from heartbeat.json, handling Z suffix."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def _parse_updated_at(ts: str) -> Optional[datetime]:
    """Parse ISO-8601 timestamp from W&B updated_at (may be naive or tz-aware)."""
    try:
        # Handle Z suffix
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _read_heartbeat_json(run_dir) -> Optional[dict]:
    """Read and parse heartbeat.json from a run directory. Returns None on error."""
    # Handle None, NaN, empty string
    if run_dir is None or (isinstance(run_dir, float) and run_dir != run_dir) or not run_dir:
        return None
    hb_path = Path(run_dir) / "heartbeat.json"
    if not hb_path.is_file():
        return None
    try:
        with open(hb_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, IOError):
        return None




# =============================================================================
# Run-level liveness attachment
# =============================================================================

def attach_heartbeat(
    df: "pd.DataFrame",
    *,
    ttl_s: Optional[int] = None,
    now: Optional[datetime] = None,
) -> "pd.DataFrame":
    """Attach heartbeat-derived columns to a runs DataFrame.
    
    Reads run_dir/heartbeat.json for each run and extracts:
    - heartbeat_ts: timestamp from heartbeat.json (UTC datetime)
    - heartbeat_age_s: age in seconds (now - heartbeat_ts)
    - heartbeat_alive: True if age <= ttl_s
    - heartbeat_step: progress.step from heartbeat.json
    - heartbeat_status: status field from heartbeat.json
    
    Args:
        df: DataFrame with 'run_dir' column (from discover_local_logs)
        ttl_s: TTL threshold in seconds (default: DISCOVER_HEARTBEAT_TTL_S or HEARTBEAT_TTL_S_DEFAULT)
        now: Current time for age calculation (default: datetime.now(timezone.utc))
    
    Returns:
        DataFrame with heartbeat columns added.
    """
    pd = _require_pandas()
    
    if ttl_s is None:
        ttl_s = get_heartbeat_ttl_s()
    if now is None:
        now = datetime.now(timezone.utc)
    
    if df.empty or 'run_dir' not in df.columns:
        # No run_dir column - return with empty heartbeat columns
        out = df.copy()
        out['heartbeat_exists'] = False
        out['heartbeat_ts'] = pd.NaT
        out['heartbeat_age_s'] = float('nan')
        out['heartbeat_alive'] = False
        out['heartbeat_step'] = pd.NA
        out['heartbeat_status'] = None
        return out
    
    def _extract_heartbeat(run_dir):
        """Extract heartbeat info for a single run."""
        # Existence is based on file presence, even if JSON is unreadable.
        heartbeat_exists = False
        try:
            if run_dir and not (isinstance(run_dir, float) and run_dir != run_dir):
                heartbeat_exists = (Path(run_dir) / "heartbeat.json").is_file()
        except Exception:
            heartbeat_exists = False

        hb = _read_heartbeat_json(run_dir)
        if hb is None:
            return {
                'heartbeat_exists': heartbeat_exists,
                'heartbeat_ts': pd.NaT,
                'heartbeat_age_s': float('nan'),
                'heartbeat_alive': False,
                'heartbeat_step': pd.NA,
                'heartbeat_status': None,
                'heartbeat_job_id': None,
                'heartbeat_host': None,
            }
        
        ts_str = hb.get('timestamp')
        if not ts_str:
            return {
                'heartbeat_exists': heartbeat_exists,
                'heartbeat_ts': pd.NaT,
                'heartbeat_age_s': float('nan'),
                'heartbeat_alive': False,
                'heartbeat_step': hb.get('progress', {}).get('step'),
                'heartbeat_status': hb.get('status'),
                'heartbeat_job_id': (hb.get('job') or {}).get('job_id'),
                'heartbeat_host': (hb.get('host') or {}).get('hostname'),
            }
        
        try:
            ts = _parse_heartbeat_timestamp(ts_str)
            age_s = (now - ts).total_seconds()
            # Guard against clock/timezone skew: a heartbeat timestamp in the future
            # should not mark a run as alive.
            alive = (age_s >= 0) and (age_s <= ttl_s)
        except (ValueError, TypeError):
            ts = pd.NaT
            age_s = float('nan')
            alive = False
        
        return {
            'heartbeat_exists': heartbeat_exists,
            'heartbeat_ts': ts,
            'heartbeat_age_s': age_s,
            'heartbeat_alive': alive,
            'heartbeat_step': hb.get('progress', {}).get('step'),
            'heartbeat_status': hb.get('status'),
            'heartbeat_job_id': (hb.get('job') or {}).get('job_id'),
            'heartbeat_host': (hb.get('host') or {}).get('hostname'),
        }
    
    # Apply to each row
    hb_data = df['run_dir'].apply(_extract_heartbeat)
    hb_df = pd.DataFrame(hb_data.tolist(), index=df.index)
    
    out = df.copy()
    for col in hb_df.columns:
        out[col] = hb_df[col]

    return out


def attach_liveness(
    df: "pd.DataFrame",
    *,
    ttl_s: Optional[int] = None,
    now: Optional[datetime] = None,
) -> "pd.DataFrame":
    """Attach full liveness columns (heartbeat + wandb) to a runs DataFrame.
    
    This is the main entry point for liveness detection. It:
    1. Attaches heartbeat columns via attach_heartbeat()
    2. Computes wandb_alive from status + found_in (only for runs without heartbeat.json)
    3. Chooses liveness: heartbeat if heartbeat.json exists, else W&B
    
    Adds columns:
    - heartbeat_ts, heartbeat_age_s, heartbeat_alive, heartbeat_step, heartbeat_status
    - wandb_alive: True if status == 'running' and found_in in {'wandb', 'both'} (and fresh),
      but only evaluated when heartbeat.json does not exist
    - is_active: heartbeat_alive if heartbeat.json exists, else wandb_alive
    - active_source: 'heartbeat' | 'wandb' | 'none'
    
    Args:
        df: DataFrame with run data (run_dir, status, found_in columns)
        ttl_s: Heartbeat TTL in seconds
        now: Current time for age calculation
    
    Returns:
        DataFrame with all liveness columns added.
    """
    pd = _require_pandas()

    # `now` is used both by attach_heartbeat() and by wandb freshness checks below.
    # attach_heartbeat() will default internally, but we also need a concrete value here.
    if now is None:
        now = datetime.now(timezone.utc)
    
    # First attach heartbeat columns
    out = attach_heartbeat(df, ttl_s=ttl_s, now=now)
    
    # Run liveness policy:
    # - If heartbeat.json exists, rely ONLY on heartbeat liveness (even if stale).
    # - Only if heartbeat.json does not exist, fall back to W&B liveness.
    if 'heartbeat_exists' in out.columns:
        hb_exists_mask = out['heartbeat_exists'].fillna(False).astype(bool)
    else:
        hb_exists_mask = pd.Series(False, index=out.index)

    # Compute wandb_alive only for runs without a heartbeat file.
    out['wandb_fresh'] = False
    out['wandb_alive'] = False
    if (~hb_exists_mask).any() and 'found_in' in out.columns and 'status' in out.columns:
        needs_wandb = ~hb_exists_mask
        has_wandb = out.loc[needs_wandb, 'found_in'].isin(['wandb', 'both'])
        is_running = out.loc[needs_wandb, 'status'] == 'running'

        # Freshness: require updated_at within WANDB_TTL_S to count as alive.
        wandb_ttl_s = get_wandb_ttl_s()
        if 'updated_at' in out.columns:
            def _wandb_fresh(updated_at):
                if not isinstance(updated_at, str) or not updated_at:
                    return False
                dt = _parse_updated_at(updated_at)
                if dt is None:
                    return False
                # Guard against clock/timezone skew: if updated_at is in the future,
                # don't treat it as "fresh".
                age_s = (now - dt).total_seconds()
                return (age_s >= 0) and (age_s <= wandb_ttl_s)

            out.loc[needs_wandb, 'wandb_fresh'] = out.loc[needs_wandb, 'updated_at'].apply(_wandb_fresh)

        out.loc[needs_wandb, 'wandb_alive'] = has_wandb & is_running & out.loc[needs_wandb, 'wandb_fresh']

    # Choose is_active based on presence of heartbeat.json
    hb_alive = out['heartbeat_alive'].fillna(False)
    wb_alive = out['wandb_alive'].fillna(False)
    out['is_active'] = hb_alive
    out.loc[~hb_exists_mask, 'is_active'] = wb_alive.loc[~hb_exists_mask].values

    # Determine active source (only for active runs; heartbeat is authoritative if it exists).
    def _get_source(row):
        if row.get('heartbeat_exists', False):
            return 'heartbeat' if row.get('heartbeat_alive', False) else 'none'
        return 'wandb' if row.get('wandb_alive', False) else 'none'

    out['active_source'] = out.apply(_get_source, axis=1)
    
    return out


# =============================================================================
# Task-level aggregation (tasks.json-aligned)
# =============================================================================

def build_task_progress(
    df: "pd.DataFrame",
    target_step: int = 5_000_000,
    task_list: Optional[List[str]] = None,
    include_unknown: bool = False,
    ttl_s: Optional[int] = None,
    now: Optional[datetime] = None,
    logs_dir: Optional[Path] = None,
) -> "pd.DataFrame":
    """Build a per-task progress table aligned to official task list.
    
    This is the single source of truth for task-level status. Uses:
    - tasks.json as the task universe (via load_task_list())
    - Heartbeat-first liveness (heartbeat OR wandb = active)
    - Step from max(ckpt_step, wandb_summary_step, heartbeat_step)
    - Live filesystem scan for video_step (ensures consistency with eval list)
    
    Args:
        df: DataFrame with all runs (from load_df())
        target_step: Target step for completion (default 5M)
        task_list: List of official tasks. Defaults to load_task_list().
        include_unknown: If True, include tasks not in task_list.
        ttl_s: Heartbeat TTL in seconds
        now: Current time for age calculation
        logs_dir: Path to logs directory for live video scan. If None, uses cached data.
    
    Returns:
        DataFrame with columns:
            task, max_step, progress_pct, 
            wandb_running_runs, heartbeat_alive_runs, is_active, active_source,
            category (completed/running/stalled/not_started)
        One row per task in task_list (plus unknown if include_unknown=True).
        Sorted by progress_pct ascending.
        
        Attrs:
            unknown_tasks: list of tasks in runs but not in task_list
            n_unknown: count of unknown tasks
    """
    pd = _require_pandas()
    from tasks import load_task_list as _load_task_list
    
    # Default to official task list
    if task_list is None:
        task_list = _load_task_list()
    official_set = set(task_list)
    
    # Handle empty df
    if df.empty:
        result = pd.DataFrame({'task': task_list})
        result['max_step'] = 0
        result['ckpt_step_max'] = 0
        result['video_step_max'] = 0
        result['needs_eval_video'] = False
        result['wandb_running_runs'] = 0
        result['heartbeat_alive_runs'] = 0
        result['is_active'] = False
        result['active_source'] = 'none'
        result['progress_pct'] = 0.0
        result['category'] = 'not_started'
        result.attrs['unknown_tasks'] = []
        result.attrs['n_unknown'] = 0
        return result.sort_values('task').reset_index(drop=True)
    
    # Attach liveness columns
    df_live = attach_liveness(df, ttl_s=ttl_s, now=now)
    
    # Attach max_step if not present
    if 'max_step' not in df_live.columns:
        from .progress import attach_max_step
        df_live = attach_max_step(df_live)
    
    # Identify unknown tasks (present in df but not in task_list)
    tasks_in_df = set(df_live['task'].dropna().unique())
    unknown_tasks = sorted(tasks_in_df - official_set)
    
    # Filter to official tasks only (unless include_unknown)
    if not include_unknown:
        df_live = df_live[df_live['task'].isin(official_set)]
    
    # Compute per-task aggregates
    # Step: max of ckpt_step, wandb summary step (in max_step), and heartbeat_step
    def _get_best_step(group):
        """Get best step from all available sources."""
        max_step = group['max_step'].max() if 'max_step' in group.columns else 0
        hb_step = group['heartbeat_step'].max() if 'heartbeat_step' in group.columns else 0
        # Handle NaN
        if pd.isna(max_step):
            max_step = 0
        if pd.isna(hb_step):
            hb_step = 0
        return max(max_step, hb_step)
    
    max_steps = df_live.groupby('task').apply(_get_best_step, include_groups=False)

    # Compute ckpt_step_max and video_step_max via live filesystem scan for consistency.
    # This ensures discover tasks shows the same values as eval list / videos collect.
    if logs_dir is not None:
        from .step_utils import compute_task_steps
        
        # Only scan tasks we care about
        tasks_to_scan = list(official_set) if not include_unknown else list(official_set | set(unknown_tasks))
        step_info = compute_task_steps(logs_dir, tasks_to_scan)
        
        ckpt_step_max = pd.Series({t: v[0] for t, v in step_info.items()})
        video_step_max = pd.Series({t: v[1] for t, v in step_info.items()})
    else:
        # Fall back to cached data from the dataframe
        if 'ckpt_step' in df_live.columns:
            ckpt_steps = pd.to_numeric(df_live['ckpt_step'], errors='coerce').fillna(0)
            ckpt_step_max = ckpt_steps.groupby(df_live['task']).max()
        else:
            ckpt_step_max = pd.Series(dtype="float")

        if 'video_step' in df_live.columns:
            video_steps = pd.to_numeric(df_live['video_step'], errors='coerce').fillna(0)
            video_step_max = video_steps.groupby(df_live['task']).max()
        else:
            video_step_max = pd.Series(dtype="float")
    
    # Wandb running count
    wandb_running_mask = df_live['wandb_alive'] if 'wandb_alive' in df_live.columns else pd.Series(False, index=df_live.index)
    wandb_counts = df_live[wandb_running_mask].groupby('task').size()
    
    # Heartbeat alive count
    hb_alive_mask = df_live['heartbeat_alive'] if 'heartbeat_alive' in df_live.columns else pd.Series(False, index=df_live.index)
    hb_counts = df_live[hb_alive_mask].groupby('task').size()
    
    # Build result aligned to task_list
    if include_unknown:
        all_tasks = sorted(set(task_list) | set(unknown_tasks))
    else:
        all_tasks = list(task_list)
    
    result = pd.DataFrame({'task': all_tasks})
    result['max_step'] = result['task'].map(max_steps).fillna(0)
    result['ckpt_step_max'] = result['task'].map(ckpt_step_max).fillna(0)
    result['video_step_max'] = result['task'].map(video_step_max).fillna(0)
    result['needs_eval_video'] = (result['ckpt_step_max'] > 0) & (result['video_step_max'] < result['ckpt_step_max'])
    result['wandb_running_runs'] = result['task'].map(wandb_counts).fillna(0).astype(int)
    result['heartbeat_alive_runs'] = result['task'].map(hb_counts).fillna(0).astype(int)
    result['is_active'] = (result['wandb_running_runs'] > 0) | (result['heartbeat_alive_runs'] > 0)
    
    # Active source (heartbeat wins)
    def _task_source(row):
        if row['heartbeat_alive_runs'] > 0:
            return 'heartbeat'
        elif row['wandb_running_runs'] > 0:
            return 'wandb'
        return 'none'
    result['active_source'] = result.apply(_task_source, axis=1)
    
    result['progress_pct'] = (100 * result['max_step'] / target_step).clip(upper=100).round(1)
    
    # Categorize
    result['category'] = 'stalled'  # default for in-progress without active
    result.loc[result['max_step'] >= target_step, 'category'] = 'completed'
    result.loc[result['max_step'] == 0, 'category'] = 'not_started'
    result.loc[result['is_active'] & (result['category'].isin(['stalled', 'not_started'])), 'category'] = 'running'
    
    # Sort by progress then task name
    result = result.sort_values(['progress_pct', 'task'], ascending=[True, True]).reset_index(drop=True)
    
    # Store unknown task info as attrs
    result.attrs['unknown_tasks'] = unknown_tasks
    result.attrs['n_unknown'] = len(unknown_tasks)
    
    return result


def print_unknown_tasks_warning(progress: "pd.DataFrame") -> None:
    """Print warning about unknown tasks (not in tasks.json) if any."""
    n_unknown = progress.attrs.get('n_unknown', 0)
    if n_unknown > 0:
        unknown = progress.attrs.get('unknown_tasks', [])
        print(f"\n⚠️  {n_unknown} unknown tasks in runs (not in tasks.json):")
        for t in unknown:
            print(f"  - {t}")

