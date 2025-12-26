#!/usr/bin/env python3
"""
Generate per-task indexes and convenience symlinks under tdmpc2/logs/.

Writes:
  logs/<task>/index.jsonl   (one JSON object per run dir)
Creates/updates:
  logs/<task>/latest -> logs/<task>/<run_id>
  logs/<task>/best   -> logs/<task>/<run_id>  (highest step seen)

This is safe to run any time. By default it *writes the index* but does not
touch symlinks unless --update-links is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _require_yaml():
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("pyyaml is required. Install with: pip install pyyaml") from exc
    return yaml


RUN_RE = re.compile(r"^\d{8}_\d{6}")


def _parse_step_from_name(name: str) -> Optional[int]:
    # "1_100_000.pt" -> 1100000
    digits = name.replace("_", "").replace(",", "")
    return int(digits) if digits.isdigit() else None


def _max_ckpt_step(run_dir: Path) -> Optional[int]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    best = None
    for p in ckpt_dir.glob("*.pt"):
        if p.stem.endswith("_trainer"):
            continue
        step = _parse_step_from_name(p.stem)
        if step is None:
            continue
        best = step if best is None else max(best, step)
    return best


def _read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _read_yaml(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    yaml = _require_yaml()
    try:
        data = yaml.safe_load(path.read_text()) or {}
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _atomic_symlink(link_path: Path, target: Path) -> None:
    tmp = link_path.with_name(link_path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target)
    os.replace(tmp, link_path)


@dataclass(frozen=True)
class RunRow:
    task: str
    run_id: str
    run_dir: str
    status: Optional[str]
    max_step: Optional[int]
    heartbeat_step: Optional[int]
    heartbeat_status: Optional[str]
    checkpoint_path: Optional[str]
    wandb_run_id: Optional[str]


def build_rows_for_task(task_dir: Path) -> list[RunRow]:
    task = task_dir.name
    rows: list[RunRow] = []
    for run_dir in sorted(task_dir.iterdir(), key=lambda p: p.name):
        if not run_dir.is_dir():
            continue
        if run_dir.name in {"latest", "best", "_legacy"}:
            continue
        if not RUN_RE.match(run_dir.name):
            continue

        run_info = _read_yaml(run_dir / "run_info.yaml") or {}
        hb = _read_json(run_dir / "heartbeat.json") or {}

        hb_step = hb.get("progress", {}).get("step") if isinstance(hb.get("progress"), dict) else None
        hb_status = hb.get("status")
        ckpt_path = None
        if isinstance(hb.get("checkpoint"), dict):
            ckpt_path = hb["checkpoint"].get("path")

        ckpt_step = _max_ckpt_step(run_dir)
        max_step = ckpt_step if ckpt_step is not None else (hb_step if isinstance(hb_step, int) else None)

        wandb_run_id = None
        try:
            wid_path = run_dir / "wandb_run_id.txt"
            if wid_path.is_file():
                wandb_run_id = wid_path.read_text().strip() or None
        except Exception:
            pass

        rows.append(
            RunRow(
                task=task,
                run_id=run_dir.name,
                run_dir=str(run_dir.resolve()),
                status=run_info.get("status"),
                max_step=max_step,
                heartbeat_step=hb_step if isinstance(hb_step, int) else None,
                heartbeat_status=hb_status if isinstance(hb_status, str) else None,
                checkpoint_path=ckpt_path if isinstance(ckpt_path, str) else None,
                wandb_run_id=wandb_run_id,
            )
        )
    return rows


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "tdmpc2" / "logs"

    ap = argparse.ArgumentParser(description="Index tdmpc2/logs/<task> run directories.")
    ap.add_argument("--logs-dir", type=Path, default=logs_dir)
    ap.add_argument("--update-links", action="store_true", help="Update latest/best symlinks.")
    args = ap.parse_args()

    logs_dir = args.logs_dir.expanduser().resolve()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    print(f"logs_dir: {logs_dir}")

    tasks = [p for p in logs_dir.iterdir() if p.is_dir() and not RUN_RE.match(p.name) and p.name != "lsf"]
    print(f"tasks: {len(tasks)}")

    for task_dir in sorted(tasks, key=lambda p: p.name):
        rows = build_rows_for_task(task_dir)
        index_path = task_dir / "index.jsonl"
        with open(index_path, "w") as f:
            for r in rows:
                f.write(json.dumps({**r.__dict__, "indexed_at": now}) + "\n")

        if not rows:
            continue

        # latest: lexicographically newest run_id (timestamp prefix)
        latest = max(rows, key=lambda r: r.run_id)

        # best: highest max_step, tie-break by run_id
        def _best_key(r: RunRow):
            return (r.max_step if r.max_step is not None else -1, r.run_id)

        best = max(rows, key=_best_key)

        if args.update_links:
            _atomic_symlink(task_dir / "latest", Path(latest.run_id))
            _atomic_symlink(task_dir / "best", Path(best.run_id))

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


