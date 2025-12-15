#!/bin/bash
# Submit expert training jobs across multiple queues to maximize parallelism.
# - short-gpu: 180 GPU limit (jobs 1-180)
# - long-gpu: 70 GPU limit (jobs 181-200)

cd /home/projects/dharel/nadavt/repos/newt/tdmpc2

echo "Submitting jobs 1-180 to short-gpu..."
bsub -J "newt-expert[1-180]" \
  -q short-gpu \
  -n 1 -gpu "num=1" -R "rusage[mem=32GB]" -W 5:45 -r \
  -o logs/lsf/newt-expert.%J.%I.log \
  -e logs/lsf/newt-expert.%J.%I.log \
  -app nvidia-gpu \
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.0" \
  jobs/run_expert_task.sh

echo "Submitting jobs 181-200 to long-gpu..."
bsub -J "newt-expert[181-200]" \
  -q long-gpu \
  -n 1 -gpu "num=1" -R "rusage[mem=32GB]" -W 5:45 -r \
  -o logs/lsf/newt-expert.%J.%I.log \
  -e logs/lsf/newt-expert.%J.%I.log \
  -app nvidia-gpu \
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.0" \
  jobs/run_expert_task.sh

echo "Done. Monitor with: bjobs -u \$USER | grep newt-expert"
