#!/bin/bash
# Quick sanity check for imports after code changes.
# Usage: make test-sanity (from repo root, inside Docker container)
#    or: bsub -q short-gpu -gpu num=1 -W 0:10 -app nvidia-gpu \
#          -env "LSB_CONTAINER_IMAGE=ops:5000/newt:1.0.2" \
#          -o test.log -e test.log tdmpc2/tests/local/test_imports.sh

set -e
cd /home/projects/dharel/nadavt/repos/newt/tdmpc2

echo "=== Testing imports ==="

echo -n "1. parse_step from discover.progress: "
python -c "from discover.progress import parse_step; print(parse_step('1_000_000.pt'))"

echo -n "2. Trainer import: "
python -c "from trainer import Trainer; print('OK')"

echo -n "3. train.py imports: "
python -c "import train; print('OK')"

echo ""
echo "=== All imports successful ==="
