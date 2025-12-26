"""Caching and data loading for TD-MPC2 runs."""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd

from .runs import discover_local_logs, discover_wandb_runs
from .runs import discover_wandb_runs_by_id


def require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required. Install with `pip install pandas`.") from exc
    return pd


def _with_suffix_or_append(path: Path, suffix: str) -> Path:
    """Like Path.with_suffix, but works even if the path has no suffix."""
    try:
        return path.with_suffix(suffix)
    except ValueError:
        return Path(str(path) + suffix)


def _pkl_path_for(data_path: Path) -> Path:
    """Pickle sidecar path used when parquet engines are unavailable."""
    return _with_suffix_or_append(data_path, ".pkl")


def _is_parquet_path(path: Path) -> bool:
    return path.suffix.lower() in {".parquet", ".pq"}


def combine_runs(*dfs: "pd.DataFrame") -> "pd.DataFrame":
    """Combine multiple DataFrames, ignoring empty ones."""
    pd = require_pandas()
    frames = [df for df in dfs if df is not None and not df.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def latest_timestamp(df: "pd.DataFrame") -> datetime:
    """Get the latest updated_at timestamp from the DataFrame."""
    pd = require_pandas()
    if 'updated_at' not in df.columns or df.empty:
        return datetime.now()
    ts = pd.to_datetime(df['updated_at'], errors='coerce').dropna()
    return ts.max() if not ts.empty else datetime.now()


def normalize_for_save(df: "pd.DataFrame") -> "pd.DataFrame":
    """Normalize DataFrame columns for Parquet serialization."""
    pd = require_pandas()
    
    if df.empty:
        return df
    out = df.copy()
    
    text_cols = [
        'task', 'exp_name', 'seed', 'source', 'ckpt_path', 'run_dir', 'url',
        'wandb_run_id', 'run_id', 'state', 'user', 'config_path'
    ]
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].astype(str)

    def to_json_cell(x):
        if x is None:
            return None
        try:
            if pd.isna(x):
                return None
        except (TypeError, ValueError):
            pass
        return json.dumps(x, default=str)

    list_like = ['videos', 'tags', 'artifacts', 'summary']
    for col in list_like:
        if col in out.columns:
            out[col] = out[col].apply(to_json_cell)

    if 'updated_at' in out.columns:
        out['updated_at'] = out['updated_at'].astype(str)
    return out


def save_cache(df: "pd.DataFrame", data_path: Path, meta_path: Path) -> datetime:
    """Save DataFrame to cache with metadata.

    Prefer Parquet when available; fall back to pickle if Parquet engines
    (pyarrow/fastparquet) are not installed.
    """
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df_norm = normalize_for_save(df)
    try:
        df_norm.to_parquet(data_path, index=False)
    except ImportError:
        # pandas requires optional parquet backends; keep discover usable without them.
        # Store a sidecar pickle cache alongside the configured parquet path.
        pkl_path = _pkl_path_for(data_path)
        df_norm.to_pickle(pkl_path)
        warnings.warn(
            "Parquet engine not available (pyarrow/fastparquet). "
            f"Saved runs cache as pickle instead: {pkl_path}",
            RuntimeWarning,
        )
    ts = latest_timestamp(df)
    meta_path.write_text(ts.isoformat())
    return ts


def load_cache(data_path: Path, meta_path: Path) -> Tuple[Optional["pd.DataFrame"], Optional[datetime]]:
    """Load DataFrame from cache with metadata.

    Reads Parquet if possible; falls back to a sidecar pickle cache if Parquet
    engines are missing.
    """
    pd = require_pandas()
    
    pkl_path = _pkl_path_for(data_path)
    candidates = [p for p in (data_path, pkl_path) if p.is_file()]
    if not candidates:
        return None, None

    # Prefer the newest cache if both exist (avoids stale parquet vs newer pickle).
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = candidates[0]

    df = None
    if _is_parquet_path(chosen):
        try:
            df = pd.read_parquet(chosen)
        except ImportError:
            # No parquet engine in this environment; try pickle fallback.
            if pkl_path.is_file():
                df = pd.read_pickle(pkl_path)
            else:
                warnings.warn(
                    "Parquet cache exists but parquet engine is missing (pyarrow/fastparquet). "
                    "Ignoring cache and refreshing from source.",
                    RuntimeWarning,
                )
                return None, None
    else:
        df = pd.read_pickle(chosen)
    
    # Parse JSON strings back to dicts/lists
    def from_json_cell(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, str):
            try:
                return json.loads(x)
            except (json.JSONDecodeError, TypeError):
                return x
        return x
    
    for col in ['videos', 'tags', 'artifacts', 'summary']:
        if col in df.columns:
            df[col] = df[col].apply(from_json_cell)
    
    ts = None
    if meta_path.is_file():
        try:
            ts = datetime.fromisoformat(meta_path.read_text().strip())
        except Exception:
            ts = None
    return df, ts


def load_all_runs(
    logs_dir: Path,
    *,
    wandb_project: str,
    wandb_limit: Optional[int] = None,
) -> Tuple["pd.DataFrame", datetime]:
    """Load runs from local logs and (optionally) W&B.
    
    Pass an empty string for `wandb_project` to disable W&B fetching (local-only).
    """
    pd = require_pandas()

    local_df = discover_local_logs(logs_dir, limit=None)
    if not wandb_project:
        return local_df, latest_timestamp(local_df)

    # If we have local logs, avoid a full W&B project scan:
    # only fetch W&B runs for local runs that *do not* have a heartbeat file.
    # This aligns with discover.liveness policy: heartbeat presence is authoritative.
    wandb_df = pd.DataFrame()
    if not local_df.empty and "wandb_run_id" in local_df.columns:
        def _hb_exists(run_dir) -> bool:
            try:
                if run_dir is None or (isinstance(run_dir, float) and pd.isna(run_dir)):
                    return False
                p = Path(str(run_dir))
                return (p / "heartbeat.json").is_file()
            except Exception:
                return False

        if "run_dir" in local_df.columns:
            hb_exists = local_df["run_dir"].apply(_hb_exists)
        else:
            hb_exists = pd.Series(False, index=local_df.index)

        ids_to_fetch = local_df.loc[~hb_exists, "wandb_run_id"].dropna().astype(str).unique().tolist()
        if ids_to_fetch:
            wandb_df = discover_wandb_runs_by_id(wandb_project, ids_to_fetch)
    else:
        # No local logs: fall back to full project discovery (keeps wandb-only mode working).
        wandb_df = discover_wandb_runs(wandb_project, limit=wandb_limit)

    if local_df.empty:
        return wandb_df, latest_timestamp(wandb_df)
    if wandb_df.empty:
        return local_df, latest_timestamp(local_df)

    # Merge local and wandb on wandb_run_id (single-row-per-run view).
    merged = pd.merge(
        local_df,
        wandb_df,
        on="wandb_run_id",
        how="outer",
        suffixes=("_local", "_wandb"),
        indicator=True,
    )
    merged["found_in"] = merged["_merge"].map(
        {"both": "both", "left_only": "local", "right_only": "wandb"}
    )
    merged = merged.drop(columns=["_merge"])

    # Consolidate duplicated columns (prefer local for identifiers, prefer wandb for URL/summary).
    for col in ["task", "exp_name", "updated_at", "run_dir", "ckpt_path", "ckpt_step", "steps"]:
        local_col, wandb_col = f"{col}_local", f"{col}_wandb"
        if local_col in merged.columns and wandb_col in merged.columns:
            merged[col] = merged[local_col].combine_first(merged[wandb_col])
            merged = merged.drop(columns=[local_col, wandb_col])

    for col in ["url", "summary"]:
        local_col, wandb_col = f"{col}_local", f"{col}_wandb"
        if local_col in merged.columns and wandb_col in merged.columns:
            merged[col] = merged[wandb_col].combine_first(merged[local_col])
            merged = merged.drop(columns=[local_col, wandb_col])

    # For status, prefer wandb (more accurate real-time state) over local.
    if "status_local" in merged.columns and "status_wandb" in merged.columns:
        merged["status"] = merged["status_wandb"].combine_first(merged["status_local"])
        merged = merged.drop(columns=["status_local", "status_wandb"])

    return merged, latest_timestamp(merged)


class RunsCache:
    """Manager for cached runs data."""
    
    def __init__(
        self,
        logs_dir: Path,
        cache_path: Path,
        wandb_project: str = "",
        wandb_limit: Optional[int] = None,
    ):
        self.logs_dir = Path(logs_dir)
        self.cache_path = Path(cache_path)
        self.cache_meta_path = cache_path.with_suffix('.meta.txt')
        self.wandb_project = wandb_project
        self.wandb_limit = wandb_limit
    
    def load(self, refresh: bool = False) -> Tuple["pd.DataFrame", datetime, bool]:
        """Load runs, using cache if available and fresh.
        
        Returns:
            Tuple of (DataFrame, timestamp, used_cache)
        """
        cached_df, cached_ts = load_cache(self.cache_path, self.cache_meta_path)
        if cached_df is not None and not refresh:
            return cached_df, cached_ts, True

        df_all, new_ts = load_all_runs(
            self.logs_dir,
            wandb_project=self.wandb_project,
            wandb_limit=self.wandb_limit,
        )

        if cached_df is not None and cached_ts is not None and new_ts <= cached_ts:
            return cached_df, cached_ts, True

        saved_ts = save_cache(df_all, self.cache_path, self.cache_meta_path)
        return df_all, saved_ts, False
