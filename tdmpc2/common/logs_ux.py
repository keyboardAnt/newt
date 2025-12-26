"""Small utilities to improve local log directory UX."""

from __future__ import annotations

import logging
import os
from pathlib import Path

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


