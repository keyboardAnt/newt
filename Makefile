PYTHON ?= python
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT := $(MAKEFILE_DIR)
DISCOVER := $(ROOT)/tdmpc2/discover/runs.py

# Configuration - set via environment or command line
#   export LOGS_DIR=./logs WANDB_PROJECT=wm-planning/mmbench
#   make discover
#   make list-completed
logs ?= $(ROOT)/tdmpc2/logs
wandb ?=
opts ?=

# Export for runs.py to pick up
export LOGS_DIR := $(logs)
ifdef wandb
export WANDB_PROJECT := $(wandb)
endif

.PHONY: help interactive submit-expert test-sanity \
        discover list-completed list-running list-crashed list-local-only list-wandb-only

help:
	@echo "Targets:"
	@echo "  interactive     Launch interactive GPU session (Docker container)"
	@echo "  submit-expert   Submit 200 expert training jobs (multi-queue)"
	@echo "  test-sanity     Run import sanity checks (inside Docker container)"
	@echo ""
	@echo "Discovery (set logs= and/or wandb= for sources):"
	@echo "  discover        Discover all runs from configured sources"
	@echo "  list-completed  Filter: status=completed"
	@echo "  list-running    Filter: status=running"
	@echo "  list-crashed    Filter: status=crashed"
	@echo "  list-local-only Filter: found_in=local (not synced to wandb)"
	@echo "  list-wandb-only Filter: found_in=wandb (no local logs)"
	@echo ""
	@echo "Configuration:"
	@echo "  logs=<path>           Local logs directory (default: tdmpc2/logs)"
	@echo "  wandb=<entity/proj>   Wandb project (optional)"
	@echo "  opts=\"...\"            Additional options for runs.py"
	@echo ""
	@echo "Examples:"
	@echo "  make discover                                # Local logs only"
	@echo "  make discover wandb=wm-planning/mmbench      # Both sources"
	@echo "  make list-completed wandb=wm-planning/mmbench"
	@echo "  make list-local-only wandb=wm-planning/mmbench"

interactive:
	@cd $(ROOT)/tdmpc2 && ./jobs/interactive.sh

submit-expert:
	@cd $(ROOT)/tdmpc2 && ./jobs/submit_expert_array.sh

test-sanity:
	@echo "Running sanity checks (requires Docker container with torch)..."
	@cd $(ROOT)/tdmpc2 && ./tests/test_imports.sh

# Discovery targets - all use the unified discover command
discover:
	@$(PYTHON) $(DISCOVER) --print $(opts)

list-completed:
	@$(PYTHON) $(DISCOVER) --status completed --print $(opts)

list-running:
	@$(PYTHON) $(DISCOVER) --status running --print $(opts)

list-crashed:
	@$(PYTHON) $(DISCOVER) --status crashed --print $(opts)

list-local-only:
	@$(PYTHON) $(DISCOVER) --found-in local --print $(opts)

list-wandb-only:
	@$(PYTHON) $(DISCOVER) --found-in wandb --print $(opts)
