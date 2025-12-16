#!/usr/bin/env python
"""
Evaluate the latest checkpoint per task listed in jobs/tasks_soup.txt.

For each task:
- find all *.pt under logs/<task>/<run_id>/checkpoints/
- pick the checkpoint with the largest step number from the filename
- run a short eval (steps=1) via train.py

Tasks without any checkpoints are skipped with a log message.
"""

import subprocess
from pathlib import Path

from discover import parse_step


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
        task_logs_dir = logs_dir / task
        candidates = [p for p in task_logs_dir.glob("*/checkpoints/*.pt") if not p.stem.endswith('_trainer')]
        if not candidates:
            print(f"[SKIP] No checkpoints found for task '{task}' under {task_logs_dir}")
            continue

        # Pick the checkpoint with the largest step number
        best_ckpt = max(candidates, key=parse_step)
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
            "num_envs=2",
            "use_demos=False",
            f"tasks_fp={str(tasks_fp)}",
            f"exp_name={eval_exp_name}",
            "save_video=True",
            "env_mode=sync",
            "compile=False",
        ]

        # Run eval; raise if it fails so you see the error immediately.
        subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()


