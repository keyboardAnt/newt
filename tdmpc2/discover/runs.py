#!/usr/bin/env python3
"""
Discover TD-MPC2 runs from local logs and Weights & Biases.

Usage:
  python runs.py --print                    # Use defaults (local + wandb)
  python runs.py --status completed --print # Filter by status
  python runs.py --local-only --print       # Only local logs
  python runs.py --wandb-only --print       # Only wandb
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import pandas as pd  # type: ignore


# Defaults - edit these if your setup differs
DEFAULT_LOGS_DIR = Path(__file__).parent.parent / "logs"
DEFAULT_WANDB_PROJECT = "wm-planning/mmbench"

# Status normalization: map wandb state to unified status
WANDB_STATE_TO_STATUS = {
    "finished": "completed",
    "running": "running",
    "crashed": "crashed",
    "failed": "crashed",
    "killed": "crashed",
}


def require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        sys.stderr.write("pandas is required. Install with `pip install pandas`.\n")
        raise SystemExit(1) from exc
    return pd


def parse_ckpt_step(path: Path) -> Optional[int]:
    try:
        return int(path.stem.replace("_", ""))
    except ValueError:
        return None


def read_text_if_exists(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def discover_local_logs(logs_dir: Path, limit: Optional[int]) -> "pd.DataFrame":
    pd = require_pandas()
    logs_dir = logs_dir.expanduser().resolve()
    if not logs_dir.is_dir():
        sys.stderr.write(f"Warning: Logs directory not found: {logs_dir}\n")
        return pd.DataFrame()

    rows: List[dict] = []
    for idx, run_info_path in enumerate(sorted(logs_dir.glob("*/run_info.yaml"))):
        if limit is not None and idx >= limit:
            break

        run_dir = run_info_path.parent
        run_info = {}
        try:
            run_info = yaml.safe_load(run_info_path.read_text()) or {}
        except Exception:
            pass

        ckpts = [p for p in (run_dir / "checkpoints").glob("*.pt") 
                 if not p.stem.endswith('_trainer')]
        best_ckpt = max(ckpts, key=parse_ckpt_step) if ckpts else None

        videos = sorted(str(p.resolve()) for p in run_dir.glob("videos/*.mp4"))
        if not videos:
            videos = sorted(str(p.resolve()) for p in run_dir.glob("wandb/run-*/files/media/videos/**/*.mp4"))
        
        config_path = run_dir / "config.yaml"
        wandb_id_path = run_dir / "wandb_run_id.txt"

        rows.append({
            "task": run_info.get("task"),
            "local_run_id": run_dir.name,
            "exp_name": run_info.get("exp_name"),
            "status": run_info.get("status", "unknown"),
            "ckpt_step": parse_ckpt_step(best_ckpt) if best_ckpt else 0,
            "steps": run_info.get("steps"),
            "updated_at": datetime.fromtimestamp(
                best_ckpt.stat().st_mtime if best_ckpt else run_info_path.stat().st_mtime
            ).isoformat(),
            "wandb_run_id": read_text_if_exists(wandb_id_path),
            "run_dir": str(run_dir.resolve()),
            "ckpt_path": str(best_ckpt.resolve()) if best_ckpt else None,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["found_in"] = "local"
    return df


def _extract_task_from_tags(tags: List[str]) -> Optional[str]:
    """Extract task name from wandb tags (e.g., ['expert-foo', 'foo', 'seed:1'] -> 'foo')."""
    for tag in tags:
        if tag.startswith("seed:") or tag.startswith("expert-"):
            continue
        return tag
    return None


def discover_wandb_runs(project_path: str, limit: Optional[int], timeout: Optional[int] = None) -> "pd.DataFrame":
    pd = require_pandas()
    import time
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        sys.stderr.write("wandb is required. Install with `pip install wandb`.\n")
        raise SystemExit(1) from exc

    sys.stderr.write(f"Fetching runs from wandb ({project_path})...\n")
    api = wandb.Api(timeout=timeout or 60)
    runs = api.runs(project_path, per_page=100)
    rows: List[dict] = []
    start_time = time.time()

    for run in runs:
        if limit is not None and len(rows) >= limit:
            break
        if timeout and (time.time() - start_time) > timeout:
            sys.stderr.write(f"  Timeout after {timeout}s, returning {len(rows)} runs\n")
            break

        tags = list(run.tags) if run.tags else []
        task = _extract_task_from_tags(tags)
        status = WANDB_STATE_TO_STATUS.get(run.state, run.state)

        rows.append({
            "task": task,
            "wandb_run_id": run.id,
            "exp_name": run.name,
            "status": status,
            "updated_at": run.updated_at.isoformat() if getattr(run, "updated_at", None) else None,
            "url": run.url,
        })
        
        if len(rows) % 500 == 0:
            sys.stderr.write(f"  ...{len(rows)} runs\n")

    elapsed = time.time() - start_time
    sys.stderr.write(f"  {len(rows)} runs fetched in {elapsed:.1f}s\n")
    df = pd.DataFrame(rows)
    if not df.empty:
        df["found_in"] = "wandb"
    return df


def discover(logs_dir: Optional[Path], wandb_project: Optional[str], limit: Optional[int], timeout: Optional[int] = None) -> "pd.DataFrame":
    """Discover runs, joining local and wandb if both provided."""
    pd = require_pandas()
    
    local_df = discover_local_logs(logs_dir, limit) if logs_dir else pd.DataFrame()
    wandb_df = discover_wandb_runs(wandb_project, limit, timeout) if wandb_project else pd.DataFrame()
    
    if local_df.empty:
        return wandb_df
    if wandb_df.empty:
        return local_df
    
    # Outer join on wandb_run_id
    merged = pd.merge(local_df, wandb_df, on="wandb_run_id", how="outer",
                      suffixes=("_local", "_wandb"), indicator=True)
    
    merged["found_in"] = merged["_merge"].map({
        "both": "both", "left_only": "local", "right_only": "wandb"
    })
    merged = merged.drop(columns=["_merge"])
    
    # Consolidate duplicated columns (prefer local)
    for col in ["task", "exp_name", "status", "updated_at"]:
        local_col, wandb_col = f"{col}_local", f"{col}_wandb"
        if local_col in merged.columns and wandb_col in merged.columns:
            merged[col] = merged[local_col].combine_first(merged[wandb_col])
            merged = merged.drop(columns=[local_col, wandb_col])
    
    return merged


def print_summary(df, max_rows: int = 20) -> None:
    print(f"Discovered {len(df)} rows")
    if df.empty:
        return
    cols = [c for c in ["found_in", "task", "exp_name", "status", "ckpt_step", "updated_at"] if c in df.columns]
    print(df[cols].head(max_rows).to_string(index=False))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Discover TD-MPC2 runs.")
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS_DIR, help="Logs directory")
    parser.add_argument("--wandb", type=str, default=DEFAULT_WANDB_PROJECT, help="Wandb project")
    parser.add_argument("--local-only", action="store_true", help="Skip wandb")
    parser.add_argument("--wandb-only", action="store_true", help="Skip local logs")
    parser.add_argument("--limit", type=int, help="Limit runs per source")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds for wandb (default: 120)")
    parser.add_argument("--status", type=str, help="Filter by status")
    parser.add_argument("--found-in", type=str, choices=["both", "local", "wandb"], help="Filter by source")
    parser.add_argument("--save", type=Path, help="Save to file (csv/parquet)")
    parser.add_argument("--print", dest="do_print", action="store_true", help="Print summary")
    
    args = parser.parse_args(list(argv) if argv is not None else None)
    
    logs_dir = None if args.wandb_only else args.logs
    wandb_project = None if args.local_only else args.wandb
    
    df = discover(logs_dir, wandb_project, args.limit, args.timeout)
    
    if args.status and not df.empty:
        df = df[df["status"] == args.status]
    if args.found_in and not df.empty:
        df = df[df["found_in"] == args.found_in]
    
    if args.save:
        path = args.save.expanduser()
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        print(f"Saved to {path}")
    
    if args.do_print:
        print_summary(df)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
