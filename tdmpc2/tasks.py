"""Task list management for TD-MPC2 training.

Loads tasks from tasks.json and provides consistent indexing for LSF job arrays.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

# Default path to tasks.json (repo root)
DEFAULT_TASKS_JSON = Path(__file__).parent.parent / "tasks.json"


@lru_cache(maxsize=1)
def load_task_list(tasks_json: Path | None = None) -> list[str]:
    """Load the task list from tasks.json (sorted alphabetically for consistent ordering).
    
    Filters out variant tasks (e.g., task-var2, task-var3) keeping only the base task
    or the first variant (task-var1) if no base exists.
    
    Args:
        tasks_json: Path to tasks.json. Defaults to repo_root/tasks.json.
        
    Returns:
        Sorted list of task names (excluding variants).
    """
    tasks_file = tasks_json or DEFAULT_TASKS_JSON
    if not tasks_file.is_file():
        raise FileNotFoundError(f"Tasks file not found: {tasks_file}")
    
    with open(tasks_file) as f:
        all_tasks = sorted(json.load(f).keys())
    
    # Filter out variant tasks (keep base or var1 only)
    variant_pattern = re.compile(r'^(.+)-var(\d+)$')
    
    # First pass: identify which base tasks exist
    base_exists = set()
    for task in all_tasks:
        if not variant_pattern.match(task):
            base_exists.add(task)
    
    # Second pass: filter
    filtered = []
    for task in all_tasks:
        match = variant_pattern.match(task)
        if match:
            base = match.group(1)
            var_num = int(match.group(2))
            # Skip if base task exists, or if this is not var1
            if base in base_exists or var_num != 1:
                continue
        filtered.append(task)
    
    return filtered


def task_to_index(task: str) -> int:
    """Map task name to 1-based LSB_JOBINDEX."""
    tasks = load_task_list()
    try:
        return tasks.index(task) + 1  # LSF uses 1-based indexing
    except ValueError:
        raise ValueError(f"Task '{task}' not found in task list")


def index_to_task(index: int) -> str:
    """Map 1-based LSB_JOBINDEX to task name."""
    tasks = load_task_list()
    if index < 1 or index > len(tasks):
        raise ValueError(f"Index {index} out of range [1, {len(tasks)}]")
    return tasks[index - 1]


def get_maniskill_indices() -> tuple[int, int]:
    """Get the index range for ManiSkill tasks (ms-*).
    
    Returns:
        Tuple of (first_index, last_index) for ManiSkill tasks.
    """
    tasks = load_task_list()
    ms_indices = [i + 1 for i, t in enumerate(tasks) if t.startswith('ms-')]
    if not ms_indices:
        return (0, 0)
    return (ms_indices[0], ms_indices[-1])


if __name__ == "__main__":
    # Quick test / info dump
    tasks = load_task_list()
    print(f"Total tasks: {len(tasks)}")
    
    ms_start, ms_end = get_maniskill_indices()
    print(f"ManiSkill tasks (ms-*): indices {ms_start}-{ms_end}")
    
    print(f"\nFirst 10 tasks:")
    for i, t in enumerate(tasks[:10], 1):
        print(f"  {i:3}: {t}")
    
    print(f"\nLast 10 tasks:")
    for i, t in enumerate(tasks[-10:], len(tasks) - 9):
        print(f"  {i:3}: {t}")

