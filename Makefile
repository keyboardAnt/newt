PYTHON ?= python
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
DISCOVER := $(ROOT)/tdmpc2/discover/runs.py

.PHONY: help interactive submit-expert test-sanity \
        discover list-completed list-running list-crashed list-local-only list-wandb-only

help:
	@echo "Targets:"
	@echo "  discover        All runs (local + wandb)"
	@echo "  list-completed  Completed runs"
	@echo "  list-running    Running runs"
	@echo "  list-crashed    Crashed runs"
	@echo "  list-local-only Runs not synced to wandb"
	@echo "  list-wandb-only Runs only on wandb"
	@echo ""
	@echo "  interactive     Launch interactive GPU session"
	@echo "  submit-expert   Submit expert training jobs"

interactive:
	@cd $(ROOT)/tdmpc2 && ./jobs/interactive.sh

submit-expert:
	@cd $(ROOT)/tdmpc2 && ./jobs/submit_expert_array.sh

test-sanity:
	@cd $(ROOT)/tdmpc2 && ./tests/test_imports.sh

discover:
	@$(PYTHON) $(DISCOVER) --print

list-completed:
	@$(PYTHON) $(DISCOVER) --status completed --print

list-running:
	@$(PYTHON) $(DISCOVER) --status running --print

list-crashed:
	@$(PYTHON) $(DISCOVER) --status crashed --print

list-local-only:
	@$(PYTHON) $(DISCOVER) --found-in local --print

list-wandb-only:
	@$(PYTHON) $(DISCOVER) --found-in wandb --print
