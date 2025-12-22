"""Centralized configuration for the discover module.

All defaults can be overridden via environment variables or kwargs.
"""

from __future__ import annotations

import os
from pathlib import Path

# Compute paths relative to this file
_MODULE_DIR = Path(__file__).parent
_TDMPC2_DIR = _MODULE_DIR.parent

# Defaults
DEFAULT_LOGS_DIR = _TDMPC2_DIR / "logs"
DEFAULT_CACHE_PATH = _MODULE_DIR / "runs_cache.parquet"
DEFAULT_WANDB_PROJECT = "wm-planning/mmbench"
DEFAULT_TARGET_STEP = 5_000_000


def get_logs_dir() -> Path:
    """Get logs directory from env or default."""
    env = os.environ.get("DISCOVER_LOGS_DIR")
    return Path(env) if env else DEFAULT_LOGS_DIR


def get_cache_path() -> Path:
    """Get cache path from env or default."""
    env = os.environ.get("DISCOVER_CACHE_PATH")
    return Path(env) if env else DEFAULT_CACHE_PATH


def get_wandb_project() -> str:
    """Get wandb project from env or default."""
    return os.environ.get("DISCOVER_WANDB_PROJECT", DEFAULT_WANDB_PROJECT)


def get_target_step() -> int:
    """Get target step from env or default."""
    env = os.environ.get("DISCOVER_TARGET_STEP")
    return int(env) if env else DEFAULT_TARGET_STEP

