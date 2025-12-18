#!/bin/bash
# Submit expert training jobs across multiple queues to maximize parallelism.
# - long-gpu: 70 GPU limit, 48h walltime
# - short-gpu: 180 GPU limit, 6h walltime
# ManiSkill tasks (87-122) use non-exclusive GPU mode due to SAPIEN/Vulkan requirements.

cd /home/projects/dharel/nadavt/repos/newt/tdmpc2

# Ensure LSF log directory exists
mkdir -p logs/lsf

# ============ NON-MANISKILL TASKS (exclusive GPU mode) ============
echo "Submitting non-ManiSkill jobs 1-70 to long-gpu (48h walltime, exclusive GPU)..."
bsub -J "newt-expert[1-70]" \
  -q long-gpu \
  -n 1 -gpu "num=1:mode=exclusive_process" -R "rusage[mem=32GB]" -W 48:00 -r \
  -o /home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf/newt-expert.%J.%I.log \
  -e /home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf/newt-expert.%J.%I.log \
  -u "$USER" -N \
  -app nvidia-gpu \
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.1" \
  jobs/run_expert_task.sh

echo "Submitting non-ManiSkill jobs 71-86 to short-gpu (6h walltime, exclusive GPU)..."
bsub -J "newt-expert[71-86]" \
  -q short-gpu \
  -n 1 -gpu "num=1:mode=exclusive_process" -R "rusage[mem=32GB]" -W 5:45 -r \
  -o /home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf/newt-expert.%J.%I.log \
  -e /home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf/newt-expert.%J.%I.log \
  -u "$USER" -N \
  -app nvidia-gpu \
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.1" \
  jobs/run_expert_task.sh

echo "Submitting non-ManiSkill jobs 123-200 to short-gpu (6h walltime, exclusive GPU)..."
bsub -J "newt-expert[123-200]" \
  -q short-gpu \
  -n 1 -gpu "num=1:mode=exclusive_process" -R "rusage[mem=32GB]" -W 5:45 -r \
  -o /home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf/newt-expert.%J.%I.log \
  -e /home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf/newt-expert.%J.%I.log \
  -u "$USER" -N \
  -app nvidia-gpu \
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.1" \
  jobs/run_expert_task.sh

# ============ MANISKILL TASKS (non-exclusive GPU for SAPIEN/Vulkan) ============
echo "Submitting ManiSkill jobs 87-122 to long-gpu (48h walltime, shared GPU)..."
bsub -J "newt-expert[87-122]" \
  -q long-gpu \
  -n 1 -gpu "num=1" -R "rusage[mem=32GB]" -W 48:00 -r \
  -o /home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf/newt-expert.%J.%I.log \
  -e /home/projects/dharel/nadavt/repos/newt/tdmpc2/logs/lsf/newt-expert.%J.%I.log \
  -u "$USER" -N \
  -app nvidia-gpu \
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.1" \
  jobs/run_expert_task.sh

echo "Done. Monitor with: bjobs -J 'newt-expert*'"
