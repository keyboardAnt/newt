#!/usr/bin/env python3
"""
Lightweight CLI to discover TD-MPC2 runs from local logs or Weights & Biases.

Usage:
  python discover/runs.py logs tdmpc2/logs --print
  python discover/runs.py wandb <entity/project> --print
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
        raise SystemExit(f"Logs directory not found: {logs_dir}")

    rows: List[dict] = []
    for idx, ckpt in enumerate(logs_dir.glob("*/*/checkpoints/*.pt")):
        if limit is not None and idx >= limit:
            break

        run_dir = ckpt.parent.parent
        task_dir = run_dir.parent

        # Load metadata from run_info.yaml if available
        run_info_path = run_dir / "run_info.yaml"
        run_info = {}
        if run_info_path.is_file():
            try:
                run_info = yaml.safe_load(run_info_path.read_text()) or {}
            except Exception:
                pass

        # Look for videos in both standard location and wandb media directory
        videos = sorted(
            str(p.resolve()) for p in run_dir.glob("videos/*.mp4")
        )
        if not videos:
            # Also check wandb media directory
            videos = sorted(
                str(p.resolve()) for p in run_dir.glob("wandb/run-*/files/media/videos/**/*.mp4")
            )
        config_path = run_dir / "config.yaml"
        wandb_id_path = run_dir / "wandb_run_id.txt"

        rows.append(
            {
                "source": "local",
                "task": task_dir.name,
                "run_id": run_dir.name,  # Timestamp-based run identifier
                "seed": run_info.get("seed"),
                "exp_name": run_info.get("exp_name"),
                "run_dir": str(run_dir.resolve()),
                "ckpt_path": str(ckpt.resolve()),
                "ckpt_step": parse_ckpt_step(ckpt),
                "updated_at": datetime.fromtimestamp(
                    ckpt.stat().st_mtime
                ).isoformat(),
                "videos": videos,
                "config_path": str(config_path.resolve())
                if config_path.is_file()
                else None,
                "wandb_run_id": read_text_if_exists(wandb_id_path),
            }
        )

    return pd.DataFrame(rows)


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

        rows.append(
            {
                "source": "wandb",
                "task": run.config.get("task"),
                "seed": run.config.get("seed"),
                "exp_name": run.config.get("exp_name") or run.name,
                "run_id": run.id,
                "state": run.state,
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

    return pd.DataFrame(rows)


def print_summary(df, max_rows: int = 20) -> None:
    if df is None:
        return
    print(f"Discovered {len(df)} rows")
    if df.empty:
        return

    preview_cols = [
        col
        for col in (
            "source",
            "task",
            "run_id",
            "exp_name",
            "seed",
            "ckpt_step",
            "state",
            "updated_at",
            "ckpt_path",
            "url",
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
        description="Discover TD-MPC2 runs from local logs or Weights & Biases."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    logs_parser = subparsers.add_parser(
        "logs", help="Scan local logs directory (logs/<task>/<run_id>/...)."
    )
    logs_parser.add_argument("logs_dir", type=Path, help="Path to logs directory.")
    logs_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of checkpoints scanned (for speed).",
    )
    logs_parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save dataframe (csv or parquet).",
    )
    logs_parser.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="Print a compact summary to stdout.",
    )

    wandb_parser = subparsers.add_parser(
        "wandb", help="List runs from a W&B project (entity/project)."
    )
    wandb_parser.add_argument("project_path", help="W&B path like entity/project.")
    wandb_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of runs fetched.",
    )
    wandb_parser.add_argument(
        "--artifacts",
        action="store_true",
        help="Also list logged artifacts (names only).",
    )
    wandb_parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save dataframe (csv or parquet).",
    )
    wandb_parser.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="Print a compact summary to stdout.",
    )

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "logs":
        df = discover_local_logs(args.logs_dir, args.limit)
    elif args.command == "wandb":
        df = discover_wandb_runs(args.project_path, args.limit, args.artifacts)
    else:
        parser.error("Unknown command")
        return 2

    if args.save:
        save_df(df, args.save)
    if getattr(args, "do_print", False):
        print_summary(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
