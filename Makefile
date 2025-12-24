PYTHON ?= python
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
TDMPC2 := $(ROOT)/tdmpc2

.PHONY: help interactive interactive-nonexclusive submit-expert test-sanity \
        status running tasks restart domains \
        gen-eval submit-eval videos-collect videos-prune videos-prune-dry refresh \
        test-heartbeat test-heartbeat-lsf

help:
	@echo "=== Observability ==="
	@echo "  status          Training progress overview (completed/running/stalled/not-started)"
	@echo "  running         Currently running tasks (wandb-verified)"
	@echo "  tasks           List all tasks with progress"
	@echo "  domains         Progress grouped by domain"
	@echo "  refresh         Force refresh cache from local logs + wandb"
	@echo ""
	@echo "=== Job Management ==="
	@echo "  restart         Show bsub commands to restart stalled tasks (dry-run)"
	@echo "  restart-submit  Actually submit restart jobs"
	@echo "  submit-expert   Submit all expert training jobs (full 200 tasks)"
	@echo ""
	@echo "=== Evaluation & Videos ==="
	@echo "  gen-eval        List tasks needing eval (no videos)"
	@echo "  submit-eval     Generate & submit eval jobs for tasks without videos"
	@echo "  videos-collect  Collect videos for presentation"
	@echo "  videos-prune    Remove old checkpoint videos (keep only latest)"
	@echo "  videos-prune-dry Preview what videos-prune would remove"
	@echo ""
	@echo "=== Interactive Sessions ==="
	@echo "  interactive              Launch interactive GPU session (exclusive mode)"
	@echo "  interactive-nonexclusive Launch interactive GPU session (shared mode, for ManiSkill)"
	@echo ""
	@echo "=== Development ==="
	@echo "  test-sanity        Run basic import tests"
	@echo "  test-heartbeat     Run heartbeat unit tests"
	@echo "  test-heartbeat-lsf Submit heartbeat e2e test to LSF cluster"

# === Observability ===

status:
	@cd $(TDMPC2) && $(PYTHON) -m discover status

running:
	@cd $(TDMPC2) && $(PYTHON) -m discover running

tasks:
	@cd $(TDMPC2) && $(PYTHON) -m discover tasks

domains:
	@cd $(TDMPC2) && $(PYTHON) -m discover domains

refresh:
	@cd $(TDMPC2) && $(PYTHON) -m discover refresh

# === Job Management ===

restart:
	@cd $(TDMPC2) && $(PYTHON) -m discover restart

restart-submit:
	@cd $(TDMPC2) && $(PYTHON) -m discover restart --submit

submit-expert:
	@cd $(TDMPC2) && ./jobs/submit_expert_array.sh

# === Evaluation & Videos ===

gen-eval:
	@cd $(TDMPC2) && $(PYTHON) -m discover eval list

submit-eval:
	@cd $(TDMPC2) && $(PYTHON) -m discover eval submit --submit

videos-collect:
	@cd $(TDMPC2) && $(PYTHON) -m discover videos collect

videos-prune:
	@cd $(TDMPC2) && $(PYTHON) -m discover videos prune

videos-prune-dry:
	@cd $(TDMPC2) && $(PYTHON) -m discover videos prune --dry-run

# === Interactive Sessions ===

interactive:
	@cd $(TDMPC2) && ./jobs/interactive.sh

interactive-nonexclusive:
	@cd $(TDMPC2) && ./jobs/interactive_nonexclusive.sh

# === Development ===

test-sanity:
	@cd $(TDMPC2) && ./tests/local/test_imports.sh

test-heartbeat:
	@cd $(TDMPC2) && $(PYTHON) -m unittest tests.local.test_heartbeat -v

test-heartbeat-lsf:
	@cd $(TDMPC2) && bsub < jobs/test_heartbeat_e2e.lsf
