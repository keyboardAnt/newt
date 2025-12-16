#!/usr/bin/env python3
"""
Unified CLI to discover TD-MPC2 runs from local logs and/or Weights & Biases.

Usage:
  python runs.py --print                              # Auto-detect sources from env vars
  python runs.py --logs ./logs --print                # Local logs only
  python runs.py --wandb entity/project --print       # Wandb only
  python runs.py --logs ./logs --wandb entity/project --print  # Both (joined)

Environment variables:
  LOGS_DIR       Default path to local logs directory
  WANDB_PROJECT  Default wandb project (entity/project)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import pandas as pd  # type: ignore


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
        sys.stderr.write(
            "pandas is required. Install with `pip install pandas`.\n"
        )
        raise SystemExit(1) from exc
    return pd


def parse_ckpt_step(path: Path) -> Optional[int]:
    """Return integer step from checkpoint filename like '600_000.pt'."""
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

        videos = sorted(
            str(p.resolve()) for p in run_dir.glob("videos/*.mp4")
        )
        if not videos:
            videos = sorted(
                str(p.resolve()) for p in run_dir.glob("wandb/run-*/files/media/videos/**/*.mp4")
            )
        config_path = run_dir / "config.yaml"
        wandb_id_path = run_dir / "wandb_run_id.txt"

        rows.append(
            {
                "task": run_info.get("task"),
                "tasks": run_info.get("tasks", []),
                "num_tasks": run_info.get("num_tasks", 1),
                "local_run_id": run_dir.name,
                "seed": run_info.get("seed"),
                "exp_name": run_info.get("exp_name"),
                "status": run_info.get("status", "unknown"),
                "steps": run_info.get("steps"),
                "final_step": run_info.get("final_step"),
                "error": run_info.get("error"),
                "run_dir": str(run_dir.resolve()),
                "ckpt_path": str(best_ckpt.resolve()) if best_ckpt else None,
                "ckpt_step": parse_ckpt_step(best_ckpt) if best_ckpt else 0,
                "updated_at": datetime.fromtimestamp(
                    best_ckpt.stat().st_mtime if best_ckpt else run_info_path.stat().st_mtime
                ).isoformat(),
                "videos": videos,
                "config_path": str(config_path.resolve())
                if config_path.is_file()
                else None,
                "wandb_run_id": read_text_if_exists(wandb_id_path),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["found_in"] = "local"
    return df


def discover_wandb_runs(
    project_path: str, limit: Optional[int], include_artifacts: bool
) -> "pd.DataFrame":
    pd = require_pandas()
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        sys.stderr.write(
            "wandb is required for remote discovery. Install with `pip install wandb`.\n"
        )
        raise SystemExit(1) from exc

    api = wandb.Api()
    runs = api.runs(project_path)
    rows: List[dict] = []

    for run in runs:
        if limit is not None and len(rows) >= limit:
            break

        summary = {}
        try:
            summary = dict(run.summary)
        except Exception:
            summary = {}

        artifacts = None
        if include_artifacts:
            artifacts = []
            try:
                for art in run.logged_artifacts():
                    artifacts.append(
                        {
                            "name": art.name,
                            "type": art.type,
                            "version": art.version,
                            "state": getattr(art, "state", None),
                        }
                    )
            except Exception as exc:
                warnings.warn(f"Could not list artifacts for run {run.id}: {exc}")

        # Normalize wandb state to unified status
        wandb_state = run.state
        status = WANDB_STATE_TO_STATUS.get(wandb_state, wandb_state)

        rows.append(
            {
                "task": run.config.get("task"),
                "seed": run.config.get("seed"),
                "exp_name": run.config.get("exp_name") or run.name,
                "wandb_run_id": run.id,
                "status": status,
                "tags": list(run.tags) if run.tags else [],
                "user": getattr(run.user, "username", None),
                "url": run.url,
                "updated_at": run.updated_at.isoformat()
                if getattr(run, "updated_at", None)
                else None,
                "summary": summary,
                "artifacts": artifacts,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["found_in"] = "wandb"
    return df


def discover(
    logs_dir: Optional[Path],
    wandb_project: Optional[str],
    limit: Optional[int],
    include_artifacts: bool,
) -> "pd.DataFrame":
    """Discover runs from available sources, joining if both are provided."""
    pd = require_pandas()
    
    local_df = pd.DataFrame()
    wandb_df = pd.DataFrame()
    
    if logs_dir:
        local_df = discover_local_logs(logs_dir, limit)
    
    if wandb_project:
        wandb_df = discover_wandb_runs(wandb_project, limit, include_artifacts)
    
    # Single source - return as-is
    if local_df.empty:
        return wandb_df
    if wandb_df.empty:
        return local_df
    
    # Both sources - outer join on wandb_run_id
    merged = pd.merge(
        local_df,
        wandb_df,
        on="wandb_run_id",
        how="outer",
        suffixes=("_local", "_wandb"),
        indicator=True,
    )
    
    # Create found_in column from merge indicator
    merged["found_in"] = merged["_merge"].map({
        "both": "both",
        "left_only": "local",
        "right_only": "wandb",
    })
    merged = merged.drop(columns=["_merge"])
    
    # Consolidate common columns (prefer local value, fall back to wandb)
    for col in ["task", "seed", "exp_name", "status", "updated_at"]:
        local_col = f"{col}_local"
        wandb_col = f"{col}_wandb"
        if local_col in merged.columns and wandb_col in merged.columns:
            merged[col] = merged[local_col].combine_first(merged[wandb_col])
            merged = merged.drop(columns=[local_col, wandb_col])
    
    return merged


def print_summary(df, max_rows: int = 20) -> None:
    if df is None:
        return
    print(f"Discovered {len(df)} rows")
    if df.empty:
        return

    preview_cols = [
        col
        for col in (
            "found_in",
            "task",
            "local_run_id",
            "wandb_run_id",
            "exp_name",
            "status",
            "ckpt_step",
            "steps",
            "updated_at",
        )
        if col in df.columns
    ]
    print(df[preview_cols].head(max_rows).to_string(index=False))


def save_df(df, path: Path) -> None:
    path = path.expanduser()
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"Saved to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover TD-MPC2 runs from local logs and/or Weights & Biases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  LOGS_DIR       Default path to local logs directory
  WANDB_PROJECT  Default wandb project (entity/project)

Examples:
  %(prog)s --print                                    # Use env var defaults
  %(prog)s --logs ./logs --print                      # Local only
  %(prog)s --wandb wm-planning/mmbench --print        # Wandb only
  %(prog)s --logs ./logs --wandb wm-planning/mmbench  # Both (joined)
  %(prog)s --status completed --print                 # Filter by status
  %(prog)s --found-in local --print                   # Only local-only runs
""",
    )
    
    parser.add_argument(
        "--logs",
        type=Path,
        default=None,
        help="Path to local logs directory. Default: $LOGS_DIR env var.",
    )
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="W&B project path (entity/project). Default: $WANDB_PROJECT env var.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of runs fetched from each source.",
    )
    parser.add_argument(
        "--artifacts",
        action="store_true",
        help="Include logged artifacts from wandb.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save dataframe to file (csv or parquet).",
    )
    parser.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="Print a compact summary to stdout.",
    )
    parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="Filter by status (completed, running, crashed, preempted, unknown).",
    )
    parser.add_argument(
        "--found-in",
        type=str,
        default=None,
        choices=["both", "local", "wandb"],
        help="Filter by where run was found.",
    )

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Get sources from args or environment
    logs_dir = args.logs or (Path(os.environ["LOGS_DIR"]) if "LOGS_DIR" in os.environ else None)
    wandb_project = args.wandb or os.environ.get("WANDB_PROJECT")
    
    if not logs_dir and not wandb_project:
        parser.error("No sources specified. Use --logs, --wandb, or set LOGS_DIR/WANDB_PROJECT env vars.")
        return 1
    
    # Discover runs
    df = discover(logs_dir, wandb_project, args.limit, args.artifacts)
    
    # Apply filters
    if args.status and not df.empty:
        df = df[df["status"] == args.status]
    if args.found_in and not df.empty:
        df = df[df["found_in"] == args.found_in]

    # Output
    if args.save:
        save_df(df, args.save)
    if args.do_print:
        print_summary(df)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
