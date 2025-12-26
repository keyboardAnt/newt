#!/bin/bash
# Task execution script for expert training (called by submit_expert_array.sh)

set -euo pipefail

cd /home/projects/dharel/nadavt/repos/newt/tdmpc2

# Get task name from tasks.py (ensures consistent filtering of variant tasks)
TASK=$(python -c "from tasks import index_to_task; print(index_to_task(${LSB_JOBINDEX}))")

# =============================================================================
# SIGNAL HANDLING: Propagate SIGTERM to child processes
# This prevents orphaned python processes when LSF preempts/kills the job.
# =============================================================================
CURRENT_PID=""
_term() { 
  echo "Caught SIGTERM/SIGINT signal!" 
  if [[ -n "${CURRENT_PID}" ]]; then
    echo "Forwarding signal to child PID ${CURRENT_PID}..."
    kill -TERM "${CURRENT_PID}" 2>/dev/null
    wait "${CURRENT_PID}"
  fi
  exit 143
}
trap _term SIGTERM SIGINT

# =============================================================================
# RETRY POLICY
# =============================================================================
# Note: `bsub -r` does NOT reliably retry application-level crashes on many LSF setups
# (it is typically for rerunning after certain system failures / requeues).
# We therefore implement a simple in-job retry loop for transient failures.
#
# Controls:
# - NEWT_TRAIN_RETRIES: number of retries after the first attempt (default: 1)
# - NEWT_TRAIN_RETRY_SLEEP_S: sleep between retries in seconds (default: 60)
NEWT_TRAIN_RETRIES="${NEWT_TRAIN_RETRIES:-1}"
NEWT_TRAIN_RETRY_SLEEP_S="${NEWT_TRAIN_RETRY_SLEEP_S:-60}"

# Create a deterministic work_dir for this run up-front so we can capture LSF stdout/stderr there.
# We pass run_id/work_dir into Hydra so train.py uses exactly this directory.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="expert_${TASK}"
RUN_ID="${RUN_TS}_${EXP_NAME}"
WORK_DIR="/home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/${TASK}/${RUN_ID}"
mkdir -p "${WORK_DIR}"

# =============================================================================
# LSF LOG BROWSING UX:
# - Keep the canonical run layout: logs/<task>/<run_id>/...
# - Also create a run-centric directory under logs/lsf/<run_id>/ that can be used
#   to browse LSF-related artifacts across tasks.
# - Symlink logs/<task>/<run_id>/lsf_logs -> logs/lsf/<run_id>
#
# Note: the raw LSF stdout/stderr file is configured at submission time as:
#   logs/lsf/newt-expert.%J.%I.log
# We can't redirect that to <run_id> because <run_id> is computed here, but we can
# create a convenient symlink from logs/lsf/<run_id>/ to that raw file.
# =============================================================================
LSF_ROOT="/home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf"
LSF_RUN_DIR="${LSF_ROOT}/${RUN_ID}"
mkdir -p "${LSF_RUN_DIR}" || true

# Link the canonical run dir to the LSF view (best effort; never fail the job on this).
ln -sfn "${LSF_RUN_DIR}" "${WORK_DIR}/lsf_logs" 2>/dev/null || true

# Best-effort: link the raw LSF combined stdout/stderr file (if it exists) into the run LSF dir.
# This file is typically written even for very early failures (e.g., container startup errors),
# as long as the parent directory exists at submission time.
RAW_LSF_FILE="${LSF_ROOT}/newt-expert.${LSB_JOBID}.${LSB_JOBINDEX}.log"
if [[ -n "${LSB_JOBID:-}" && -n "${LSB_JOBINDEX:-}" ]]; then
  ln -sfn "${RAW_LSF_FILE}" "${LSF_RUN_DIR}/newt-expert.log" 2>/dev/null || true
fi

# Capture everything from this point onward into the run directory as well.
# (LSF will still write its own -o/-e file; this makes per-run debugging much easier.)
exec > >(tee -a "${WORK_DIR}/lsf.log") 2>&1

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

echo "LSF job index: ${LSB_JOBINDEX}, task: ${TASK}"
echo "Work dir: ${WORK_DIR}"

find_latest_ckpt() {
  # Auto-resume from the latest available checkpoint for this task (if any).
  # This makes restarts robust even if the submit script doesn't explicitly pass checkpoint=...
  TASK="${TASK}" python - <<'PY'
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
}

ARGS=(
  task="${TASK}"
  model_size=B
  steps=5000000
  num_envs=2
  env_mode=sync
  use_demos=False
  tasks_fp=/home/projects/dharel/nadavt/repos/newt/tasks.json
  exp_name="${EXP_NAME}"
  run_id="${RUN_ID}"
  work_dir="${WORK_DIR}"
  save_video=False
  compile=False
)

MAX_ATTEMPTS=$((NEWT_TRAIN_RETRIES + 1))
ATTEMPT=1
while (( ATTEMPT <= MAX_ATTEMPTS )); do
  echo "=== train.py attempt ${ATTEMPT}/${MAX_ATTEMPTS} (task=${TASK}, run_id=${RUN_ID}) ==="

  CKPT="$(find_latest_ckpt || true)"
  if [[ -n "${CKPT}" ]]; then
    echo "Auto-resume: found checkpoint for ${TASK}: ${CKPT}"
  else
    echo "Auto-resume: no checkpoint found for ${TASK} (starting from scratch)"
  fi

  # Build per-attempt args (avoid accidentally accumulating checkpoint= across retries).
  ARGS_ATTEMPT=("${ARGS[@]}")
  if [[ -n "${CKPT}" ]]; then
    ARGS_ATTEMPT+=(checkpoint="${CKPT}")
  fi

  # Run in background to allow signal trapping
  python train.py "${ARGS_ATTEMPT[@]}" &
  CURRENT_PID=$!
  wait "${CURRENT_PID}"
  EXIT_CODE=$?
  CURRENT_PID=""

  if (( EXIT_CODE == 0 )); then
    echo "train.py completed successfully."
    break
  fi

  echo "train.py crashed/failed with exit code ${EXIT_CODE}."

  if (( ATTEMPT >= MAX_ATTEMPTS )); then
    echo "No retries left (NEWT_TRAIN_RETRIES=${NEWT_TRAIN_RETRIES}). Exiting."
    exit "${EXIT_CODE}"
  fi

  echo "Retrying after ${NEWT_TRAIN_RETRY_SLEEP_S}s... (set NEWT_TRAIN_RETRIES/NEWT_TRAIN_RETRY_SLEEP_S to control)"
  sleep "${NEWT_TRAIN_RETRY_SLEEP_S}"
  ATTEMPT=$((ATTEMPT + 1))
done
