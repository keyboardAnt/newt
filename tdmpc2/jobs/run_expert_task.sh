#!/bin/bash
# Task execution script for expert training (called by submit_expert_array.sh)

cd /home/projects/dharel/nadavt/repos/newt/tdmpc2

# Ensure video deps for wandb.Video (moviepy, imageio-ffmpeg)
# Avoid doing network installs on every cluster job; only install if missing.
# Keep this pinned for reproducibility (matches docker/environment.yaml).
python - <<'PY' || pip install -q "wandb[media]==0.22.1"
import wandb  # noqa: F401
import moviepy  # noqa: F401
import imageio_ffmpeg  # noqa: F401
PY

# Get task name from tasks.py (ensures consistent filtering of variant tasks)
TASK=$(python -c "from tasks import index_to_task; print(index_to_task(${LSB_JOBINDEX}))")
echo "LSF job index: ${LSB_JOBINDEX}, task: ${TASK}"

# Auto-resume from the latest available checkpoint for this task (if any).
# This makes restarts robust even if the submit script doesn't explicitly pass checkpoint=...
CKPT=$(TASK="${TASK}" python - <<'PY'
import os
from pathlib import Path

task = os.environ["TASK"]
logs_dir = Path("/home/projects/dharel/nadavt/repos/newt/tdmpc2/logs")

def parse_step(p: Path):
    stem = p.stem
    if stem.endswith("_trainer"):
        return None
    digits = stem.replace("_", "")
    if not digits.isdigit():
        return None
    return int(digits)

candidates = []

# Run-first layout: logs/<timestamp>_expert_<task>/checkpoints/<step>.pt
for run_dir in logs_dir.glob(f"*expert_{task}*"):
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        continue
    for p in ckpt_dir.glob("*.pt"):
        step = parse_step(p)
        if step is not None:
            candidates.append((step, p.resolve()))

# Legacy layout: logs/<task>/**/checkpoints/<step>.pt
task_dir = logs_dir / task
if task_dir.is_dir():
    for p in task_dir.glob("**/checkpoints/*.pt"):
        step = parse_step(p)
        if step is not None:
            candidates.append((step, p.resolve()))

if candidates:
    _step, path = max(candidates, key=lambda x: x[0])
    print(str(path))
PY
)

if [[ -n "${CKPT}" ]]; then
  echo "Auto-resume: found checkpoint for ${TASK}: ${CKPT}"
else
  echo "Auto-resume: no checkpoint found for ${TASK} (starting from scratch)"
fi

ARGS=(
  task="${TASK}"
  model_size=B
  steps=5000000
  num_envs=2
  use_demos=False
  tasks_fp=/home/projects/dharel/nadavt/repos/newt/tasks.json
  exp_name="expert_${TASK}"
  save_video=True
  compile=False
)

if [[ -n "${CKPT}" ]]; then
  ARGS+=(checkpoint="${CKPT}")
fi

python train.py "${ARGS[@]}"
