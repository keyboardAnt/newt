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
from discover import load_df, training_overview, plot_max_steps

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
| `running` | `make running` | Currently running tasks (wandb-verified) |
| `tasks` | `make tasks` | List all tasks with progress |
| `domains` | `make domains` | Progress grouped by domain |
| `refresh` | `make refresh` | Force refresh cache from local logs + wandb |
| `restart` | `make restart` | Show bsub commands for stalled tasks (dry-run) |
| `restart --submit` | `make restart-submit` | Actually submit restart jobs |
| `eval list` | `make gen-eval` | List tasks needing eval (no videos) |
| `eval submit --submit` | `make submit-eval` | Generate & submit eval jobs |
| `videos collect` | `make videos-collect` | Collect videos for presentation |
| `videos prune` | `make videos-prune` | Remove old checkpoint videos |

### Common Options

```bash
python -m discover status --refresh         # Force refresh before showing status
python -m discover tasks --format json      # Output as JSON (also: csv, table)
python -m discover tasks --not-started      # Filter to not-started tasks
python -m discover tasks --stalled          # Filter to stalled tasks
python -m discover tasks --all              # Include non-official tasks (e.g., smoke-test)
python -m discover --help                   # Show all options
```

**Note:** By default, commands only show official tasks from `tasks.json` (225 tasks).
Use `--all` to include all runs in the cache (including test/debug runs like `smoke-test`, `default`, etc.).

## Log Directory Structure

All logs use a **run-first** structure where each run gets a unique timestamp-based directory.

```
logs/<YYYYMMDD_HHMMSS>[_exp_name]/
├── run_info.yaml      # Metadata: task(s), seed, exp_name, LSF job ID, git commit
├── checkpoints/       # Model checkpoints
├── videos/            # Evaluation videos
└── wandb/             # Wandb sync data
```

## Module Structure

```
discover/
├── cli.py           # Unified CLI (python -m discover)
├── api.py           # Simple API: load_df()
├── config.py        # Centralized configuration
├── runs.py          # Run discovery (local logs + W&B)
├── cache.py         # Caching and data loading
├── analysis.py      # Data analysis functions
├── plots.py         # Visualization functions
├── eval.py          # Evaluation and video management
└── browse_runs.ipynb    # Interactive notebook (visualizations only)
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
from discover import get_logs_dir, get_target_step, get_wandb_project

# All have environment variable overrides:
# DISCOVER_LOGS_DIR, DISCOVER_TARGET_STEP, DISCOVER_WANDB_PROJECT
```

### Visualization (`discover.plots`)

| Function | Description |
|----------|-------------|
| `training_overview(df, target_step)` | Pie chart + cumulative distribution + summary stats |
| `plot_max_steps(df, target_step)` | Horizontal bar chart of per-task progress |
| `progress_by_domain(df, target_step)` | Aggregate progress by domain with visualization |

### Analysis (`discover.analysis`)

| Function | Description |
|----------|-------------|
| `attach_max_step(df)` | Add `max_step` column from checkpoint or W&B summary |
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
