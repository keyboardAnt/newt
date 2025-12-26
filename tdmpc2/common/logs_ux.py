"""Small utilities to improve local log directory UX.

We keep these best-effort and robust: failures should never crash training.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_LOG = logging.getLogger(__name__)


def _atomic_symlink(link_path: Path, target: Path) -> None:
    """Atomically update a symlink (best-effort)."""
    try:
        link_path = Path(link_path)
        target = Path(target)
        tmp = link_path.with_name(link_path.name + ".tmp")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        tmp.symlink_to(target)
        os.replace(tmp, link_path)
    except Exception:
        _LOG.exception("[logs-ux] Failed to update symlink %s -> %s", link_path, target)


def ensure_task_latest(task_dir: Path, run_dir: Path) -> None:
    """Set logs/<task>/latest -> <run_id>."""
    task_dir = Path(task_dir)
    run_dir = Path(run_dir)
    _atomic_symlink(task_dir / "latest", run_dir)


def maybe_update_task_best(task_dir: Path, run_dir: Path, step: int) -> None:
    """Update logs/<task>/best symlink if this run is the highest-step seen so far.

    Stores metadata in logs/<task>/best.json for fast comparisons.
    """
    task_dir = Path(task_dir)
    run_dir = Path(run_dir)
    meta_path = task_dir / "best.json"

    prev_step: Optional[int] = None
    try:
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            prev_step = int(meta.get("step")) if meta.get("step") is not None else None
    except Exception:
        _LOG.exception("[logs-ux] Failed to read %s (continuing)", meta_path)

    if prev_step is not None and step <= prev_step:
        return

    try:
        meta = {
            "run_id": run_dir.name,
            "step": int(step),
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2) + "\n")
        os.replace(tmp, meta_path)
    except Exception:
        _LOG.exception("[logs-ux] Failed to write %s (continuing)", meta_path)

    _atomic_symlink(task_dir / "best", run_dir)


def update_run_checkpoint_symlinks(work_dir: Path, identifier: str) -> None:
    """Create/update work_dir/checkpoint_latest*.pt -> checkpoints/<identifier>*.pt."""
    try:
        work_dir = Path(work_dir)
        ckpt_dir = work_dir / "checkpoints"
        agent = ckpt_dir / f"{identifier}.pt"
        trainer = ckpt_dir / f"{identifier}_trainer.pt"

        if agent.exists():
            _atomic_symlink(work_dir / "checkpoint_latest.pt", agent)
        if trainer.exists():
            _atomic_symlink(work_dir / "checkpoint_latest_trainer.pt", trainer)
    except Exception:
        _LOG.exception("[logs-ux] Failed to update checkpoint_latest symlinks (continuing)")


