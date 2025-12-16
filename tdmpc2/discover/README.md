# Discover: Training Monitoring & Analysis Tools

Tools for discovering, analyzing, and visualizing TD-MPC2 training runs from local logs and Weights & Biases.

## Log Directory Structure

All logs are consolidated in a single timestamp-based directory structure (Hydra's default `outputs/` directory is disabled):

```
logs/<task>/<YYYYMMDD_HHMMSS>[_exp_name]/
├── run_info.yaml      # Metadata: task, seed, exp_name, LSF job ID, git commit, etc.
├── checkpoints/       # Model checkpoints
├── videos/            # Evaluation videos
└── wandb/             # Wandb sync data
```

**Example:**
```
logs/walker-walk/
├── 20251215_234251/                   # Run with default exp_name
├── 20251216_091500_ablation_lr/       # Run with custom exp_name suffix
└── 20251216_143000_fix_async/         # Another run with exp_name suffix
```

Common operations:
- List all runs for a task: `ls logs/walker-walk/`
- Find latest run: `ls logs/walker-walk/ | tail -1`
- Find today's runs: `ls logs/walker-walk/ | grep ^$(date +%Y%m%d)`
- Find failed runs: `grep -l "status: failed" logs/*/*/run_info.yaml`

## Installation

The module requires `pandas` and `matplotlib`. These are included in the main project's conda environment.

## Quick Start

```python
from pathlib import Path
from discover import RunsCache, training_overview, plot_max_steps

# Load runs (uses cache if available)
cache = RunsCache(
    logs_dir=Path('tdmpc2/logs'),
    cache_path=Path('discover/runs_cache.parquet'),
    wandb_project='wm-planning/mmbench',
)
df, timestamp, used_cache = cache.load(refresh=False)

# Visualize progress
training_overview(df, target_step=5_000_000)
plot_max_steps(df, target_step=5_000_000)
```

Or use the interactive notebook: `discover/browse_runs.ipynb`

## Module Structure

```
discover/
├── runs.py          # Run discovery (local logs + W&B)
├── cache.py         # Caching and data loading
├── analysis.py      # Data analysis functions
├── plots.py         # Visualization functions
├── eval.py          # Evaluation and video management
├── collect_videos.py    # Standalone CLI for video collection
└── browse_runs.ipynb    # Interactive notebook
```

## CLI Usage

### Discover Runs

```bash
# Scan local logs and print summary
python discover/runs.py logs tdmpc2/logs --print

# Fetch runs from Weights & Biases
python discover/runs.py wandb wm-planning/mmbench --print --limit 100

# Save to parquet file
python discover/runs.py logs tdmpc2/logs --save runs.parquet
```

### Collect Videos

```bash
# Collect videos from tasks ≥50% trained (symlinks)
python discover/collect_videos.py --min-progress 0.5

# Copy files instead of symlinks
python discover/collect_videos.py --copy --output ./my_videos

# Download to local machine
rsync -avz <server>:discover/videos_for_presentation/ ./presentation_videos/
```

## API Reference

### Cache (`discover.cache`)

**`RunsCache`**: Main class for loading and caching run data.

```python
cache = RunsCache(
    logs_dir=Path('tdmpc2/logs'),
    cache_path=Path('discover/runs_cache.parquet'),
    wandb_project='wm-planning/mmbench',
    wandb_limit=None,  # Optional limit on W&B runs
)
df, timestamp, used_cache = cache.load(refresh=False)
```

### Analysis (`discover.analysis`)

| Function | Description |
|----------|-------------|
| `attach_max_step(df)` | Add `max_step` column from checkpoint or W&B summary |
| `attach_runtime(df)` | Add `runtime` column from W&B summary |
| `best_step_by_task(df)` | Get run with highest step per task |
| `best_runtime_by_task(df)` | Get run with highest runtime per task |

### Plots (`discover.plots`)

| Function | Description |
|----------|-------------|
| `training_overview(df, target_step)` | Pie chart + cumulative distribution + summary stats |
| `plot_max_steps(df, target_step)` | Horizontal bar chart of per-task progress |
| `tasks_needing_attention(df, target_step)` | List not-started and lagging tasks |
| `progress_by_domain(df, target_step)` | Aggregate progress by task domain prefix |

### Eval (`discover.eval`)

| Function | Description |
|----------|-------------|
| `tasks_ready_for_eval(df, logs_dir, ...)` | Find tasks ≥50% trained, check video status |
| `generate_eval_script(tasks, output_dir, project_root)` | Generate LSF script for video eval jobs |
| `collect_videos(df, logs_dir, output_dir, ...)` | Collect videos into single directory |

## Interactive Notebook

The `browse_runs.ipynb` notebook provides:

1. **Training Progress Overview** - Pie chart showing completed/in-progress/not-started breakdown
2. **Per-Task Progress** - Bar chart with color coding (green=done, orange=progress, red=not started)
3. **Tasks Requiring Attention** - Lists of not-started and lagging tasks
4. **Progress by Domain** - Aggregate statistics grouped by task prefix (e.g., `walker-*`)
5. **Evaluation Management** - Find tasks needing video generation, create LSF job scripts
6. **Video Collection** - Gather videos from completed tasks for presentation

## Workflow Example

1. **Monitor training progress**:
   ```python
   df, _, _ = cache.load()
   training_overview(df, target_step=5_000_000)
   ```

2. **Identify tasks needing attention**:
   ```python
   tasks_needing_attention(df, target_step=5_000_000)
   ```

3. **Generate videos for trained tasks**:
   ```python
   from discover.eval import tasks_ready_for_eval, generate_eval_script
   
   ready_df, tasks_need_eval, _ = tasks_ready_for_eval(df, logs_dir, min_progress=0.5)
   generate_eval_script(tasks_need_eval, output_dir, project_root)
   # Then: bsub < tdmpc2/jobs/run_eval_need_videos.lsf
   ```

4. **Collect videos for presentation**:
   ```bash
   python discover/collect_videos.py --min-progress 0.5
   rsync -avz server:discover/videos_for_presentation/ ./videos/
   ```
