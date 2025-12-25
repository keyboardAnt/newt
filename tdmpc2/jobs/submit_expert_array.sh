#!/bin/bash
# Submit expert training jobs across multiple queues to maximize parallelism.
# Tasks are loaded from tasks.json via tdmpc2/tasks.py (225 tasks after filtering variants).
# - long-gpu: 70 GPU limit, 48h walltime
# - short-gpu: 180 GPU limit, 6h walltime
# Note: we run with shared/default GPU mode because EXCLUSIVE_PROCESS has caused
# frequent CUDA init failures ("device(s) busy or unavailable") on this cluster.

cd /home/projects/dharel/nadavt/repos/newt/tdmpc2

# Ensure LSF log directory exists
mkdir -p logs/lsf

# ============ NON-MANISKILL TASKS (shared/default GPU mode) ============
echo "Submitting non-ManiSkill jobs 1-58 to long-gpu (48h walltime, shared GPU)..."
bsub -J "newt-expert[1-58]" \
  -q long-gpu \
  -n 1 -gpu "num=1" -R "rusage[mem=32GB]" -W 48:00 -r \
  -o /home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf/newt-expert.%J.%I.log \
  -e /home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf/newt-expert.%J.%I.log \
  -u "$USER" -N \
  -app nvidia-gpu \
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.2" \
  jobs/run_expert_task.sh

echo "Submitting non-ManiSkill jobs 106-225 to short-gpu (6h walltime, shared GPU)..."
bsub -J "newt-expert[106-225]" \
  -q short-gpu \
  -n 1 -gpu "num=1" -R "rusage[mem=32GB]" -W 5:45 -r \
  -o /home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf/newt-expert.%J.%I.log \
  -e /home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf/newt-expert.%J.%I.log \
  -u "$USER" -N \
  -app nvidia-gpu \
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.2" \
  jobs/run_expert_task.sh

# ============ MANISKILL TASKS (shared/default GPU mode) ============
echo "Submitting ManiSkill jobs 59-105 to long-gpu (48h walltime, shared GPU)..."
bsub -J "newt-expert[59-105]" \
  -q long-gpu \
  -n 1 -gpu "num=1" -R "rusage[mem=32GB]" -W 48:00 -r \
  -o /home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf/newt-expert.%J.%I.log \
  -e /home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf/newt-expert.%J.%I.log \
  -u "$USER" -N \
  -app nvidia-gpu \
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.2" \
  jobs/run_expert_task.sh

echo "Done. Monitor with: bjobs -J 'newt-expert*'"
