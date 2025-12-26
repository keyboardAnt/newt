#!/bin/bash
# Launch an interactive GPU session with the newt Docker container.
# Usage: ./jobs/interactive.sh

bsub -q interactive-gpu \
  -gpu "num=1:mode=exclusive_process" \
  -R "rusage[tmp=10240]" \
  -app nvidia-gpu-interactive \
  -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.2" \
  -Is /bin/bash
