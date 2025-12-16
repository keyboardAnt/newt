#!/bin/bash
# Submit expert training jobs across multiple queues to maximize parallelism.
# Priority: long-gpu first (fewer interruptions with 48h walltime), then short-gpu for overflow.
# - long-gpu: 70 GPU limit, 48h walltime (jobs 1-70)
# - short-gpu: 180 GPU limit, 6h walltime (jobs 71-200)

cd /home/projects/dharel/nadavt/repos/newt/tdmpc2

# Ensure LSF log directory exists
mkdir -p logs/lsf

echo "Submitting jobs 1-70 to long-gpu (48h walltime)..."
bsub -J "newt-expert[1-70]" \
  -q long-gpu \
  -n 1 -gpu "num=1" -R "rusage[mem=32GB]" -W 48:00 -r \
  -o logs/lsf/newt-expert.%J.%I.log \
  -e logs/lsf/newt-expert.%J.%I.log \
  -u "$USER" -N \
  -app nvidia-gpu \
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.0" \
  jobs/run_expert_task.sh

echo "Submitting jobs 71-200 to short-gpu (6h walltime)..."
bsub -J "newt-expert[71-200]" \
  -q short-gpu \
  -n 1 -gpu "num=1" -R "rusage[mem=32GB]" -W 5:45 -r \
  -o logs/lsf/newt-expert.%J.%I.log \
  -e logs/lsf/newt-expert.%J.%I.log \
  -u "$USER" -N \
  -app nvidia-gpu \
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.0" \
  jobs/run_expert_task.sh

echo "Done. Monitor with: bjobs -u \$USER | grep newt-expert"
