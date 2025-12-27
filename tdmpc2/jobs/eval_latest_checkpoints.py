#!/usr/bin/env python
"""
Evaluate the latest checkpoint per task listed in jobs/tasks_soup.txt.

For each task:
- scan all run directories under logs/
- find runs containing the task (via run_info.yaml)
- pick the checkpoint with the largest step number
- run a short eval (steps=1) via train.py

Tasks without any checkpoints are skipped with a log message.
"""

import subprocess
import yaml
from pathlib import Path

from discover.progress import parse_step


def find_best_checkpoint_for_task(logs_dir: Path, task: str) -> Path | None:
    """Find the best checkpoint for a given task across all runs."""
    candidates = []
    run_info_paths = []
    for pat in ("*/run_info.yaml", "*/*/run_info.yaml", "*/*/*/run_info.yaml"):
        run_info_paths.extend(logs_dir.glob(pat))
    for run_info_path in run_info_paths:
        run_dir = run_info_path.parent
        info = yaml.safe_load(run_info_path.read_text()) or {}
        tasks = info.get("tasks", [info.get("task")])
        if task in tasks:
            for ckpt in (run_dir / "checkpoints").glob("*.pt"):
                if not ckpt.stem.endswith('_trainer'):
                    candidates.append(ckpt)
    return max(candidates, key=parse_step) if candidates else None


def main() -> None:
    root = Path(__file__).resolve().parents[1]  # .../newt/tdmpc2
    logs_dir = root / "logs"
    tasks_file = root / "jobs" / "tasks_soup.txt"
    tasks_fp = root.parent / "tasks.json"

    if not tasks_file.is_file():
        raise FileNotFoundError(f"Task list not found: {tasks_file}")

    with tasks_file.open("r") as f:
        tasks = [line.strip() for line in f if line.strip()]

    for task in tasks:
        best_ckpt = find_best_checkpoint_for_task(logs_dir, task)
        if not best_ckpt:
            print(f"[SKIP] No checkpoints found for task '{task}'")
            continue

        run_id = best_ckpt.parent.parent.name
        step_stem = best_ckpt.stem

        eval_exp_name = f"eval_{run_id}_{step_stem}"
        print(f"[EVAL] task={task}, ckpt={best_ckpt}, exp_name={eval_exp_name}")

        cmd = [
            "python",
            "train.py",
            f"task={task}",
            "model_size=B",
            f"checkpoint={str(best_ckpt)}",
            "steps=1",
            "eval_only=True",
            "num_envs=2",
            "use_demos=False",
            f"tasks_fp={str(tasks_fp)}",
            f"exp_name={eval_exp_name}",
            "save_video=True",
            "env_mode=sync",
            "compile=False",  # Keep disabled for eval (compilation overhead > 1-step runtime)
        ]

        # Run eval; raise if it fails so you see the error immediately.
        subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()


