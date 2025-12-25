#!/bin/bash
# Task execution script for expert training (called by submit_expert_array.sh)

set -euo pipefail

cd /home/projects/dharel/nadavt/repos/newt/tdmpc2

# =============================================================================
# GPU BINDING: Ensure only the LSF-assigned GPU is visible to CUDA/PyTorch.
# LSF sets CUDA_VISIBLE_DEVICES when using -gpu with exclusive mode, but inside
# containers this can be lost or overwritten. Re-export from the LSF env var.
# =============================================================================
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "GPU binding: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (from LSF)"
elif [[ -n "${LSB_GPU_ALLOC:-}" ]]; then
  # Fallback: parse GPU index from LSF's allocation string (e.g., "gpu000/0" -> "0")
  GPU_ID=$(echo "$LSB_GPU_ALLOC" | grep -oP '\d+$' | head -1)
  if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "GPU binding: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (parsed from LSB_GPU_ALLOC=${LSB_GPU_ALLOC})"
  fi
fi

# Diagnostics: show GPU environment (helps debug "device busy" errors)
echo "=== GPU environment ==="
echo "hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L 2>/dev/null || echo "(nvidia-smi not available)"
echo "======================="

# If LSF gave us a specific physical GPU index, wait until it's not running compute workloads.
# This avoids fail-fast startup when GPUs are in EXCLUSIVE_PROCESS mode but still occupied.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  PHYS_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
  if [[ "${PHYS_GPU}" =~ ^[0-9]+$ ]] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU preflight: waiting for GPU ${PHYS_GPU} to be free..."
    # Wait up to ~30 minutes (1800s) before giving up.
    DEADLINE=$(( $(date +%s) + 1800 ))
    while true; do
      # List compute PIDs on the target GPU; empty output means no compute clients.
      PIDS="$(nvidia-smi -i "${PHYS_GPU}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ',' || true)"
      # Also check used memory to catch non-compute contexts that still occupy VRAM.
      MEM_USED="$(nvidia-smi -i "${PHYS_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -1 || true)"
      if [[ -z "${PIDS}" ]]; then
        if [[ "${MEM_USED}" =~ ^[0-9]+$ ]] && (( MEM_USED > 1024 )); then
          echo "GPU preflight: GPU ${PHYS_GPU} has ${MEM_USED} MiB in use -> sleeping 30s"
          sleep 30
          continue
        fi
        echo "GPU preflight: GPU ${PHYS_GPU} appears free."
        break
      fi
      NOW=$(date +%s)
      if (( NOW >= DEADLINE )); then
        echo "GPU preflight: timeout waiting for GPU ${PHYS_GPU} (pids=${PIDS}, mem_used=${MEM_USED}MiB). Proceeding anyway."
        break
      fi
      echo "GPU preflight: GPU ${PHYS_GPU} busy (pids=${PIDS}, mem_used=${MEM_USED}MiB) -> sleeping 30s"
      sleep 30
    done
  fi
fi

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

# Task-first layout: logs/<task>/<run_id>/checkpoints/<step>.pt
task_dir = logs_dir / task
if task_dir.is_dir():
    for p in task_dir.glob("*/checkpoints/*.pt"):
        step = parse_step(p)
        if step is not None:
            candidates.append((step, p.resolve()))
    # Older nested: logs/<task>/<seed>/<run_id>/checkpoints/<step>.pt
    for p in task_dir.glob("*/*/checkpoints/*.pt"):
        step = parse_step(p)
        if step is not None:
            candidates.append((step, p.resolve()))

# Legacy run-first layout: logs/<timestamp>_expert_<task>/checkpoints/<step>.pt
# Match exact task to avoid accidental prefix matches
# (e.g. mw-plate-slide vs mw-plate-slide-back-side, mw-button-press-topdown vs mw-button-press-topdown-wall).
for run_dir in logs_dir.glob("*_expert_*"):
    name = run_dir.name
    if "_expert_" not in name:
        continue
    suffix = name.split("_expert_", 1)[1]
    if suffix != task:
        continue
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        continue
    for p in ckpt_dir.glob("*.pt"):
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
  env_mode=sync
  use_demos=False
  tasks_fp=/home/projects/dharel/nadavt/repos/newt/tasks.json
  exp_name="expert_${TASK}"
  save_video=False
  compile=False
)

if [[ -n "${CKPT}" ]]; then
  ARGS+=(checkpoint="${CKPT}")
fi

python train.py "${ARGS[@]}"
