#!/bin/bash
# Task execution script for expert training (called by submit_expert_array.sh)

cd /home/projects/dharel/nadavt/repos/newt/tdmpc2

# Ensure video deps for wandb.Video (moviepy, imageio-ffmpeg)
pip install -q "wandb[media]"

# Select this array job's task from the 200-task list (1-based index)
TASK=$(sed -n "${LSB_JOBINDEX}p" jobs/tasks_soup.txt)
echo "LSF job index: ${LSB_JOBINDEX}, task: ${TASK}"

python train.py \
  task="${TASK}" \
  model_size=B \
  steps=5000000 \
  num_envs=2 \
  use_demos=False \
  tasks_fp=/home/projects/dharel/nadavt/repos/newt/tasks.json \
  exp_name="expert_${TASK}" \
  save_video=True \
  compile=False
