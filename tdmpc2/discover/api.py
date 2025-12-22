"""Simple API for loading run data.

This module provides the primary entry point for both CLI and notebook usage.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd

from .config import get_logs_dir, get_cache_path, get_wandb_project, get_target_step
from .cache import RunsCache


def load_df(
    refresh: bool = False,
    logs_dir: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    wandb_project: Optional[str] = None,
    wandb_limit: Optional[int] = None,
) -> "pd.DataFrame":
    """Load merged runs DataFrame from cache or fresh discovery.
    
    Args:
        refresh: Force refresh from local logs and wandb (ignore cache)
        logs_dir: Override logs directory
        cache_path: Override cache file path
        wandb_project: Override wandb project
        wandb_limit: Limit number of wandb runs fetched
        
    Returns:
        DataFrame with all runs (local + wandb merged)
    """
    cache = RunsCache(
        logs_dir=logs_dir or get_logs_dir(),
        cache_path=cache_path or get_cache_path(),
        wandb_project=wandb_project or get_wandb_project(),
        wandb_limit=wandb_limit,
    )
    df, _timestamp, _used_cache = cache.load(refresh=refresh)
    return df


def load_df_with_meta(
    refresh: bool = False,
    logs_dir: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    wandb_project: Optional[str] = None,
    wandb_limit: Optional[int] = None,
) -> Tuple["pd.DataFrame", datetime, bool]:
    """Load merged runs DataFrame with metadata.
    
    Returns:
        Tuple of (DataFrame, timestamp, used_cache)
    """
    cache = RunsCache(
        logs_dir=logs_dir or get_logs_dir(),
        cache_path=cache_path or get_cache_path(),
        wandb_project=wandb_project or get_wandb_project(),
        wandb_limit=wandb_limit,
    )
    return cache.load(refresh=refresh)

