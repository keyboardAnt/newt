"""
Utilities for writing/updating run_info.yaml (the per-run metadata contract).

This module centralizes all writes to run_info.yaml so that the "contract"
(what fields exist, how they're updated) is easy to audit in one place.
"""
from pathlib import Path

import yaml


def extract_parent_run_id(checkpoint_path: Path) -> str | None:
    """
    Extract parent_run_id from a checkpoint path.

    Checkpoint paths follow the convention:
      - logs/<task>/<run_id>/checkpoints/<step>.pt (task-first)
      - logs/<run_id>/checkpoints/<step>.pt        (legacy run-first)

    The run_id is the directory name containing 'checkpoints' (i.e., the parent
    directory of the 'checkpoints' folder).

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        The parent run_id if extractable, else None.
    """
    try:
        ckpt_parent = checkpoint_path.resolve().parent
        if ckpt_parent.name == 'checkpoints':
            return ckpt_parent.parent.name
    except OSError:
        # If the path cannot be resolved (e.g. missing file, permission error),
        # treat it as "no parent run id".
        return None
    return None


def update_run_info_resume(
    work_dir: Path,
    loaded_checkpoint: Path | str,
    loaded_step: int,
    parent_run_id: str | None = None,
) -> None:
    """
    Update run_info.yaml with resume lineage after loading a checkpoint.

    This records the provenance of resumed runs, enabling:
    - Tracking which checkpoint was used to resume
    - Building lineage chains across multiple resumes
    - Distinguishing fresh runs from resumed ones

    Args:
        work_dir: Path to the run's working directory.
        loaded_checkpoint: Path to the checkpoint file that was loaded.
        loaded_step: Training step from which the run resumed.
        parent_run_id: The run_id of the parent run (if known). If None, it
            will be auto-extracted from loaded_checkpoint using the standard
            path convention.
    """
    loaded_checkpoint = Path(loaded_checkpoint)
    run_info_path = Path(work_dir) / 'run_info.yaml'
    if not run_info_path.exists():
        return

    # Auto-extract parent_run_id if not provided
    if parent_run_id is None:
        parent_run_id = extract_parent_run_id(loaded_checkpoint)

    info = yaml.safe_load(run_info_path.read_text())
    if not isinstance(info, dict):
        raise RuntimeError(
            f"Invalid run_info.yaml (expected a YAML mapping/dict): {run_info_path}"
        )

    info['loaded_checkpoint'] = str(loaded_checkpoint)
    info['loaded_step'] = loaded_step
    if parent_run_id is not None:
        info['parent_run_id'] = parent_run_id
    run_info_path.write_text(yaml.dump(info, default_flow_style=False, sort_keys=False))
