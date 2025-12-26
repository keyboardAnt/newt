"""Centralized W&B API connector for the discover module.

This module provides a single point of access to the W&B API, with:
- Lazy API initialization with configurable timeout
- Run discovery (full scan and fast-by-id)
- Artifact/collection iteration for cleanup operations
- Common utilities (state normalization, etc.)

Usage:
    from discover.wandb_connector import (
        get_api,
        fetch_runs,
        fetch_runs_by_id,
        iter_model_collections,
    )
"""

from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Iterable, Iterator, List, Optional

if TYPE_CHECKING:
    import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_API_TIMEOUT = 60
DEFAULT_THREAD_WORKERS = 16

# Status normalization: map wandb state to unified status
WANDB_STATE_TO_STATUS = {
    "finished": "completed",
    "running": "running",
    "crashed": "crashed",
    "failed": "crashed",
    "killed": "crashed",
}


# =============================================================================
# API Access
# =============================================================================

_api_instance = None


def get_api(timeout: int = DEFAULT_API_TIMEOUT):
    """Get or create a cached wandb.Api instance.
    
    Args:
        timeout: API timeout in seconds (only used on first call)
    
    Returns:
        wandb.Api instance
    
    Raises:
        SystemExit if wandb is not installed
    """
    global _api_instance
    if _api_instance is not None:
        return _api_instance
    
    try:
        import wandb
    except ImportError as exc:
        sys.stderr.write("wandb is required. Install with `pip install wandb`.\n")
        raise SystemExit(1) from exc
    
    _api_instance = wandb.Api(timeout=timeout)
    return _api_instance


def reset_api():
    """Reset the cached API instance (useful for testing)."""
    global _api_instance
    _api_instance = None


# =============================================================================
# Run Discovery
# =============================================================================

def _extract_task_from_tags(tags: List[str]) -> Optional[str]:
    """Extract task name from wandb tags (e.g., ['expert-foo', 'foo', 'seed:1'] -> 'foo')."""
    for tag in tags:
        if tag.startswith("seed:") or tag.startswith("expert-") or tag.startswith("eval-"):
            continue
        return tag
    # Fallback: try to extract from expert- tag
    for tag in tags:
        if tag.startswith("expert-"):
            return tag[7:]  # Remove "expert-" prefix
    return None


def _process_run(run) -> dict:
    """Extract data from a single wandb run (called in parallel)."""
    tags = list(run.tags) if run.tags else []
    task = _extract_task_from_tags(tags)
    status = WANDB_STATE_TO_STATUS.get(run.state, run.state)
    
    # Fallback: get task from config if not in tags (slower but needed for older runs)
    if task is None:
        try:
            task = run.config.get("task")
        except Exception:
            pass
    
    # Get step from summary (needed for progress tracking)
    summary = {}
    try:
        summary = dict(run.summary) if run.summary else {}
    except Exception:
        pass

    return {
        "task": task,
        "wandb_run_id": run.id,
        "exp_name": run.name,
        "status": status,
        "updated_at": run.updated_at.isoformat() if getattr(run, "updated_at", None) else None,
        "url": run.url,
        "summary": summary,
    }


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        sys.stderr.write("pandas is required. Install with `pip install pandas`.\n")
        raise SystemExit(1) from exc
    return pd


def fetch_runs(
    project_path: str,
    limit: Optional[int] = None,
    *,
    workers: int = DEFAULT_THREAD_WORKERS,
) -> "pd.DataFrame":
    """Fetch all runs from a W&B project.
    
    Uses parallel processing for faster fetching of run details.
    
    Args:
        project_path: W&B project path (e.g., "entity/project")
        limit: Maximum number of runs to fetch (None for all)
        workers: Number of parallel workers for processing run details
    
    Returns:
        DataFrame with run data (task, wandb_run_id, exp_name, status, updated_at, url, summary)
    """
    pd = _require_pandas()
    api = get_api()
    
    sys.stderr.write(f"Fetching runs from wandb ({project_path})...\n")
    runs_iter = api.runs(project_path, per_page=100)
    
    # Collect run objects first (fast - just the iterator)
    sys.stderr.write("  Listing runs...")
    sys.stderr.flush()
    start_time = time.time()
    run_objects = []
    for run in runs_iter:
        run_objects.append(run)
        if limit is not None and len(run_objects) >= limit:
            break
        if len(run_objects) % 100 == 0:
            sys.stderr.write(f"\r  Listing runs... {len(run_objects)}")
            sys.stderr.flush()
    
    list_time = time.time() - start_time
    sys.stderr.write(f"\r  Listed {len(run_objects)} runs in {list_time:.1f}s\n")
    
    if not run_objects:
        return pd.DataFrame()
    
    # Process run details in parallel (this is the slow part - API calls for config/summary)
    sys.stderr.write("  Fetching run details (parallel)...\n")
    sys.stderr.flush()
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_process_run, run_objects))
    
    elapsed = time.time() - start_time
    rate = len(rows) / elapsed if elapsed > 0 else 0
    sys.stderr.write(f"  Fetched {len(rows)} run details in {elapsed:.1f}s ({rate:.0f}/s)\n")
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df["found_in"] = "wandb"
    return df


def fetch_runs_by_id(
    project_path: str,
    run_ids: Iterable[str],
    *,
    workers: int = DEFAULT_THREAD_WORKERS,
) -> "pd.DataFrame":
    """Fetch specific W&B runs by their IDs (fast path - avoids full project scan).
    
    Args:
        project_path: W&B project path (e.g., "entity/project")
        run_ids: Iterable of W&B run IDs to fetch
        workers: Number of parallel workers for processing run details
    
    Returns:
        DataFrame with run data
    """
    pd = _require_pandas()
    api = get_api()
    
    # Normalize/unique while preserving order
    seen = set()
    ids: List[str] = []
    for rid in run_ids:
        if rid is None:
            continue
        s = str(rid).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        ids.append(s)

    if not ids:
        return pd.DataFrame()

    sys.stderr.write(f"Fetching {len(ids)} runs from wandb by id ({project_path})...\n")

    # Fetch run objects (one call per id). Keep this simple/serial for robustness.
    start_time = time.time()
    run_objects = []
    missing = 0
    for rid in ids:
        try:
            run_objects.append(api.run(f"{project_path}/{rid}"))
        except Exception:
            missing += 1
            continue
    fetch_time = time.time() - start_time
    sys.stderr.write(f"  Resolved {len(run_objects)} runs ({missing} missing) in {fetch_time:.1f}s\n")

    if not run_objects:
        return pd.DataFrame()

    # Process run details (may trigger additional API calls for config/summary)
    sys.stderr.write("  Fetching run details (parallel)...\n")
    sys.stderr.flush()
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_process_run, run_objects))
    elapsed = time.time() - start_time
    rate = len(rows) / elapsed if elapsed > 0 else 0
    sys.stderr.write(f"  Fetched {len(rows)} run details in {elapsed:.1f}s ({rate:.0f}/s)\n")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["found_in"] = "wandb"
    return df


def fetch_run(project_path: str, run_id: str):
    """Fetch a single W&B run object.
    
    Args:
        project_path: W&B project path (e.g., "entity/project")
        run_id: W&B run ID
    
    Returns:
        wandb.Run object
    
    Raises:
        Exception if run not found
    """
    api = get_api()
    return api.run(f"{project_path}/{run_id}")


# =============================================================================
# Artifact / Collection Iteration
# =============================================================================

def iter_model_collections(
    project_path: str,
    *,
    collection_name_regex: Optional[str] = None,
    exact_collections: Optional[Iterable[str]] = None,
    max_collections: Optional[int] = None,
    progress_every: int = 200,
) -> Iterator[object]:
    """Yield W&B ArtifactCollection objects of type 'model' for a given entity/project.

    Notes:
    - `Api.artifact_type(...).collections()` is the most reliable W&B API surface in wandb==0.22.x.
    - If `exact_collections` is provided, we avoid scanning the whole project by fetching each
      collection directly (fast-path).
    
    Args:
        project_path: W&B project path (e.g., "entity/project")
        collection_name_regex: Optional regex to filter collection names
        exact_collections: If provided, fetch only these collections (fast path)
        max_collections: Maximum number of collections to yield
        progress_every: Print progress every N collections scanned
    
    Yields:
        W&B ArtifactCollection objects
    """
    api = get_api()
    name_re = re.compile(collection_name_regex) if collection_name_regex else None
    at = api.artifact_type("model", project_path)

    if exact_collections:
        yielded = 0
        for col_name in exact_collections:
            col_name = str(col_name).strip()
            if not col_name:
                continue
            if name_re and not name_re.search(col_name):
                continue
            try:
                # Full path: "entity/project/collection"
                yield api.artifact_collection("model", f"{project_path}/{col_name}")
                yielded += 1
            except Exception as e:
                sys.stderr.write(f"[wandb-connector] Failed to fetch collection {col_name!r}: {e}\n")
                continue
            if max_collections is not None and yielded >= max_collections:
                break
        return

    scanned = 0
    yielded = 0
    for col in at.collections():
        scanned += 1
        if progress_every and scanned % int(progress_every) == 0:
            sys.stderr.write(f"\rScanning model artifact collections... {scanned}")
            sys.stderr.flush()

        try:
            col_name = str(getattr(col, "name", "") or "")
        except Exception:
            continue

        if name_re and not name_re.search(col_name):
            continue

        yield col
        yielded += 1
        if max_collections is not None and yielded >= max_collections:
            break

    if scanned:
        sys.stderr.write(f"\rScanning model artifact collections... {scanned} ✓\n")
        sys.stderr.flush()


def get_artifact_collection(project_path: str, collection_name: str, artifact_type: str = "model"):
    """Get a specific artifact collection.
    
    Args:
        project_path: W&B project path (e.g., "entity/project")
        collection_name: Name of the artifact collection
        artifact_type: Type of artifact (default: "model")
    
    Returns:
        W&B ArtifactCollection object
    """
    api = get_api()
    return api.artifact_collection(artifact_type, f"{project_path}/{collection_name}")

