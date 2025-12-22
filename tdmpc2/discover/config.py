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
DEFAULT_TASKS_FILE = _TDMPC2_DIR / "jobs" / "tasks_soup.txt"


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


def get_tasks_file() -> Path:
    """Get tasks file path from env or default."""
    env = os.environ.get("DISCOVER_TASKS_FILE")
    return Path(env) if env else DEFAULT_TASKS_FILE


def load_task_list() -> list[str]:
    """Load the task list from tasks_soup.txt."""
    tasks_file = get_tasks_file()
    if not tasks_file.is_file():
        raise FileNotFoundError(f"Tasks file not found: {tasks_file}")
    return [line.strip() for line in tasks_file.read_text().splitlines() if line.strip()]


def task_to_index(task: str) -> int:
    """Map task name to 1-based LSB_JOBINDEX (line number in tasks_soup.txt)."""
    tasks = load_task_list()
    try:
        return tasks.index(task) + 1  # LSF uses 1-based indexing
    except ValueError:
        raise ValueError(f"Task '{task}' not found in {get_tasks_file()}")


def index_to_task(index: int) -> str:
    """Map 1-based LSB_JOBINDEX to task name."""
    tasks = load_task_list()
    if index < 1 or index > len(tasks):
        raise ValueError(f"Index {index} out of range [1, {len(tasks)}]")
    return tasks[index - 1]

