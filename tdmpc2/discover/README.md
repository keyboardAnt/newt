# Discover: Training Monitoring & Analysis Tools

Tools for discovering, analyzing, and visualizing TD-MPC2 training runs from local logs and Weights & Biases.

## Quick Start

### CLI (Primary Interface)

```bash
# From repo root:
make status              # Training progress overview
make running             # Currently running tasks
make tasks               # List all tasks with progress
make restart             # Show bsub commands to restart stalled tasks
make restart-submit      # Actually submit restart jobs

# Or directly:
cd tdmpc2
python -m discover status
python -m discover running
python -m discover tasks
python -m discover restart --submit
```

### Python API

```python
from discover import load_df
from discover.plots import training_overview, plot_max_steps

# Load runs (uses cache)
df = load_df(refresh=False)

# Visualize progress
training_overview(df, target_step=5_000_000)
plot_max_steps(df, target_step=5_000_000)
```

### Interactive Notebook

Open `discover/browse_runs.ipynb` for visualizations and CLI output logging.

## CLI Commands

All commands are available via `python -m discover <command>` or `make <target>`.

| Command | Make Target | Description |
|---------|-------------|-------------|
| `status` | `make status` | Training progress overview (completed/running/stalled/not-started) |
| `running` | `make running` | Currently running tasks (uses central liveness) |
| `tasks` | `make tasks` | List all tasks with progress |
| `domains` | `make domains` | Progress grouped by domain |
| `refresh` | `make refresh` | Force refresh cache from local logs + wandb |
| `restart` | `make restart` | Show bsub commands for stalled tasks (dry-run) |
| `restart --submit` | `make restart-submit` | Actually submit restart jobs |
| `eval list` | `make gen-eval` | List tasks needing eval (no videos) |
| `eval submit --submit` | `make submit-eval` | Generate & submit eval jobs |
| `videos collect` | `make videos-collect` | Collect videos for presentation |
| `videos prune` | `make videos-prune` | Remove old checkpoint videos |
| `cleanup-models` | *(none)* | W&B cleanup: keep only latest checkpoint per expert (dry-run by default) |

### Common Options

```bash
python -m discover status --refresh         # Force refresh before showing status
python -m discover tasks --format json      # Output as JSON (also: csv, table)
python -m discover tasks --not-started      # Filter to not-started tasks
python -m discover tasks --stalled          # Filter to stalled tasks
python -m discover tasks --all              # Include non-official tasks (e.g., smoke-test)
python -m discover status --no-wandb        # Local-only (disable W&B)
python -m discover --help                   # Show all options
```

**Note:** By default, commands only show official tasks from `tasks.json` (225 tasks).
Use `--all` to include all runs in the cache (including test/debug runs like `smoke-test`, `default`, etc.).

## W&B storage cleanup: keep only the latest checkpoint per expert

Training logs checkpoints to W&B as artifacts of type `model`, with names that include the step:
`<task>-<exp_name>-<seed>-<step_with_underscores>`. This means a long run creates many large artifact
entries. The `cleanup-models` command keeps only the **max-step** checkpoint per
`<task>-<exp_name>-<seed>` and deletes the rest.

Tip: if you want to target a specific checkpoint name without scanning the whole project, use
`--artifact-name <name>` (the artifact base name without `:vN`). This is much faster than regex scans.

- **Dry-run**:

```bash
python -m discover --wandb-project wm-planning/mmbench cleanup-models
```

- **Apply deletions** (with a safety cap):

```bash
python -m discover --wandb-project wm-planning/mmbench cleanup-models --apply --max-delete 5000
```

- **Optional filtering** (only delete names matching regex):

```bash
python -m discover --wandb-project wm-planning/mmbench cleanup-models --name-regex 'expert-' --apply
```

## Log Directory Structure

All logs use a **task-first** structure where each task gets its own directory, and each run gets a unique timestamp-based subdirectory.

```
logs/<task>/<YYYYMMDD_HHMMSS>[_exp_name]/
├── run_info.yaml      # Metadata: task(s), seed, exp_name, LSF job ID, git commit
├── checkpoints/       # Model checkpoints
├── videos/            # Evaluation videos
└── wandb/             # Wandb sync data
```

Legacy layouts that are still supported by `discover/` tooling:

```
logs/<run_id>/run_info.yaml                 # older run-first
logs/<task>/<seed>/<run_id>/run_info.yaml   # older nested
```

## Module Structure

```
discover/
├── cli.py              # Unified CLI (python -m discover)
├── api.py              # Simple API: load_df()
├── config.py           # Centralized configuration
├── wandb_connector.py  # Centralized W&B API access
├── runs.py             # Run discovery (local logs + W&B)
├── cache.py            # Caching and data loading
├── liveness.py         # Heartbeat + W&B liveness detection
├── progress.py         # Progress helpers (attach_max_step, etc.)
├── plots.py            # Visualization functions
├── eval.py             # Evaluation and video management
├── cleanup/
│   └── model_registry.py  # W&B artifact cleanup
└── browse_runs.ipynb   # Interactive notebook
```

## Understanding Run Status

**Status normalization:** Wandb's `state` is mapped to unified `status`:
- `finished` → `completed`
- `running` → `running`
- `crashed`, `failed`, `killed` → `crashed`

**Task categories in `discover status`:**
- 🟢 **Completed**: Reached target step (5M)
- 🔵 **Running**: Incomplete but has active runs in wandb
- 🟠 **Stalled**: Incomplete, has progress, but no active runs (needs restart)
- 🔴 **Not Started**: 0 steps

**Understanding "running" vs actual LSF state:**
- **Wandb "running"**: Runs that wandb thinks are active. May include LSF-suspended (SSUSP) jobs.
- **LSF RUN**: Jobs actually using CPU right now.
- **LSF SSUSP**: Jobs suspended by cluster (process paused, wandb still says "running").
- **Local-only "running"**: Unreliable - likely crashed runs with stale `run_info.yaml`.

### Local-only mode (no W&B)

You can run `discover` without any W&B dependency:

- **CLI**: `python -m discover --no-wandb <command>`
- **Env**: set `DISCOVER_WANDB_PROJECT` to an empty string (`DISCOVER_WANDB_PROJECT=""`)

Tradeoffs:
- **Running detection**: becomes **heartbeat-only** (requires `run_dir/heartbeat.json` to be written).
- **Progress**: derived from local checkpoints (`checkpoints/*.pt`). If you rely on W&B summary `_step`
  being ahead of the last checkpoint, local-only progress may lag.
- **Downloading videos from W&B**: not available (but collecting already-synced videos under `logs/**/wandb/...`
  still works).

## Restart Workflow

When tasks are stalled (incomplete, no active wandb runs), restart them:

```bash
# 1. See what needs restart
make running

# 2. Preview restart commands (dry-run)
make restart

# 3. Actually submit
make restart-submit

# 4. Monitor
bjobs -J 'newt-expert*'
```

Notes:
- `python -m discover restart --submit` is **not a daemon**: it submits jobs based on the current snapshot, and then exits.
  If you want continuous “auto-restart”, run it periodically (e.g., cron) or keep it running in a loop.
- Job-level retries on transient crashes are handled in `jobs/run_expert_task.sh` via `NEWT_TRAIN_RETRIES`
  (default: 1) and `NEWT_TRAIN_RETRY_SLEEP_S` (default: 60).

The restart command:
- Maps task names to indices via `tdmpc2/tasks.py` (loaded from `tasks.json`)
- Groups by queue/GPU-mode (matching `jobs/submit_expert_array.sh`)
- Generates proper `bsub` commands with correct queue, walltime, GPU mode

## Python API Reference

### Primary API (`discover.api`)

```python
from discover import load_df

# Load runs from cache
df = load_df(refresh=False)

# Force refresh from local logs + wandb
df = load_df(refresh=True)
```

### Configuration (`discover.config`)

```python
from discover.config import get_logs_dir, get_target_step, get_wandb_project

# All have environment variable overrides:
# DISCOVER_LOGS_DIR, DISCOVER_TARGET_STEP, DISCOVER_WANDB_PROJECT
```

### Visualization (`discover.plots`)

| Function | Description |
|----------|-------------|
| `training_overview(df, target_step)` | Pie chart + cumulative distribution + summary stats |
| `plot_max_steps(df, target_step)` | Horizontal bar chart of per-task progress |
| `progress_by_domain(df, target_step)` | Aggregate progress by domain with visualization |

### Progress helpers (`discover.progress`)

| Function | Description |
|----------|-------------|
| `attach_max_step(df)` | Add `max_step` column from the best available progress fields |
| `best_step_by_task(df)` | Get run with highest step per task |

### Eval (`discover.eval`)

| Function | Description |
|----------|-------------|
| `collect_videos(df, logs_dir, output_dir, ...)` | Collect local videos into single directory |
| `tasks_ready_for_eval(df, logs_dir, ...)` | Find tasks ≥50% trained, check video status |
| `generate_eval_script(tasks, output_dir, ...)` | Generate LSF script for tasks missing videos |

## Interactive Notebook

The `browse_runs.ipynb` notebook provides:

1. **Data Loading** - Load from cache via `load_df()`
2. **CLI Output Logging** - Run CLI commands and capture output in notebook
3. **Visualizations** - `training_overview()`, `plot_max_steps()`, `progress_by_domain()`

Most text-based reports have moved to the CLI. The notebook focuses on visualizations.

## Workflow Example

```bash
# 1. Check overall status
make status

# 2. Refresh if needed
make refresh

# 3. See what's running / stalled
make running

# 4. Restart stalled tasks
make restart            # Preview
make restart-submit     # Submit

# 5. Check eval status and collect videos
make gen-eval
make videos-collect

# 6. Download videos to local machine
rsync -avz server:tdmpc2/discover/videos_for_presentation/ ./videos/
   ```
