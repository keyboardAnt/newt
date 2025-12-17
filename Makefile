PYTHON ?= python
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
DISCOVER := $(ROOT)/tdmpc2/discover/runs.py
STATUS := $(ROOT)/tdmpc2/discover/status.py

.PHONY: help interactive submit-expert submit-eval test-sanity \
        status status-debug discover list-completed list-running list-crashed list-local-only list-wandb-only

help:
	@echo "Targets:"
	@echo "  status          Training progress overview"
	@echo "  status-debug    Training status with debug info"
	@echo ""
	@echo "  discover        All runs (local + wandb)"
	@echo "  list-completed  Completed runs"
	@echo "  list-running    Running runs"
	@echo "  list-crashed    Crashed runs"
	@echo "  list-local-only Runs not synced to wandb"
	@echo "  list-wandb-only Runs only on wandb"
	@echo ""
	@echo "  interactive     Launch interactive GPU session"
	@echo "  submit-expert   Submit expert training jobs"
	@echo "  submit-eval     Submit eval job for tasks needing videos"

interactive:
	@cd $(ROOT)/tdmpc2 && ./jobs/interactive.sh

submit-expert:
	@cd $(ROOT)/tdmpc2 && ./jobs/submit_expert_array.sh

submit-eval:
	@cd $(ROOT)/tdmpc2 && bsub < jobs/run_eval_need_videos.lsf

test-sanity:
	@cd $(ROOT)/tdmpc2 && ./tests/test_imports.sh

status:
	@cd $(ROOT)/tdmpc2 && $(PYTHON) discover/status.py

status-debug:
	@cd $(ROOT)/tdmpc2 && $(PYTHON) discover/status.py --debug

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
