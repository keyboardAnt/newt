#!/bin/bash
# Launch an interactive GPU session WITHOUT exclusive mode.
# Required for ManiSkill tasks (SAPIEN/Vulkan needs shared GPU access).
# Usage: ./jobs/interactive_nonexclusive.sh

bsub -q interactive-gpu \
  -gpu "num=1" \
  -R "rusage[tmp=10240]" \
  -app nvidia-gpu-interactive \
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.2" \
  -Is /bin/bash

