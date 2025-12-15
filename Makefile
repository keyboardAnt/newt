PYTHON ?= python
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT := $(MAKEFILE_DIR)
DISCOVER := $(ROOT)/tdmpc2/discover/runs.py

# Default args are lower-case for convenience; override as needed.
# Works from any cwd if you pass -C $(ROOT) or run at repo root.
#   make -C $(ROOT) discover-logs opts="--print"
#   make -C $(ROOT) discover-wandb wandb=entity/project opts="--print --limit 10"
logs ?= $(ROOT)/tdmpc2/logs
wandb ?=
opts ?=

.PHONY: help discover-logs discover-wandb test-sanity

help:
	@echo "Targets:"
	@echo "  test-sanity     Run import sanity checks (inside Docker container)"
	@echo "  discover-logs   [logs=<path>] [opts=\"...\"]"
	@echo "  discover-wandb  wandb=<entity/project> [opts=\"...\"]"
	@echo ""
	@echo "Example: python tdmpc2/discover/runs.py logs tdmpc2/logs --print"

discover-logs:
	@test -n "$(logs)" || (echo "Set logs=<path to logs dir>"; exit 1)
	$(PYTHON) $(DISCOVER) logs $(logs) $(opts)

discover-wandb:
	@test -n "$(wandb)" || (echo "Set wandb=<entity/project>"; exit 1)
	$(PYTHON) $(DISCOVER) wandb $(wandb) $(opts)

test-sanity:
	@echo "Running sanity checks (requires Docker container with torch)..."
	@cd $(ROOT)/tdmpc2 && ./tests/test_imports.sh
