<h1>MMBench & Newt</span></h1>

Official code repository for the paper

[Learning Massively Multitask World Models for Continuous Control](https://www.nicklashansen.com/NewtWM)

[Nicklas Hansen](https://nicklashansen.github.io), [Hao Su](https://cseweb.ucsd.edu/~haosu)\*, [Xiaolong Wang](https://xiaolonw.github.io)\* (UC San Diego)</br>

<img src="assets/0.gif" width="12.5%"><img src="assets/1.gif" width="12.5%"><img src="assets/2.gif" width="12.5%"><img src="assets/3.gif" width="12.5%"><img src="assets/4.gif" width="12.5%"><img src="assets/5.gif" width="12.5%"><img src="assets/6.gif" width="12.5%"><img src="assets/7.gif" width="12.5%"></br>

[[Website]](https://www.nicklashansen.com/NewtWM) [[Paper]](https://www.nicklashansen.com/NewtWM/newt.pdf) [[Models]](https://huggingface.co/nicklashansen/newt) [[Dataset]](https://huggingface.co/datasets/nicklashansen/mmbench)

----

**Early access (Nov 2025):** This is an early code release; we will continue to add features and code improvements in the coming months, but wanted to make the code available to the public as soon as possible. Please let us know if you have any questions or issues by opening an issue on GitHub!

----


## MMBench

MMBench contains a total of **200** unique continuous control tasks for training of massively multitask RL policies. The task suite consists of 159 existing tasks proposed in previous work, 22 new tasks and task variants for these existing domains, as well as 19 entirely new arcade-style tasks that we dub *MiniArcade*. MMBench tasks span multiple domains and embodiments, and each task comes with language instructions, demonstrations, and optionally image observations, enabling research on both multitask pretraining, offline-to-online RL, and RL from scratch.

<img src="assets/0.png" width="100%" style="max-width: 640px"><br/>


## Newt

Newt is a language-conditioned multitask world model based on [TD-MPC2](https://www.tdmpc2.com). We train Newt by first pretraining on demonstrations to acquire task-aware representations and action priors, and then jointly optimizing with online interaction across all tasks. To extend TD-MPC2 to the massively multitask online setting, we propose a series of algorithmic improvements including a refined architecture, model-based pretraining on the available demonstrations, additional action supervision in RL policy updates, and a drastically accelerated training pipeline.

<img src="assets/1.png" width="100%" style="max-width: 640px"><br/>

----

## Getting started

We provide a `Dockerfile` for easy installation. You can build the docker image by running

```
cd docker && docker build . -t <user>/newt:1.0.1
```

This docker image contains all dependencies needed for running MMBench and Newt.

----

## Example usage

Agents can trained by running the `train.py` script. Below are some example commands:

```
$ python train.py    # <-- a 20M parameter agent trained on all 200 MMBench tasks
$ python train.py model_size=XL    # <-- a 80M parameter agent
$ python train.py model_size=B task=walker-walk   # <-- a 5M parameter single-task agent
$ python train.py obs=rgb    # <-- a 20M parameter agent trained with state+RGB observations
```

We recommend using default hyperparameters, including the default model size of 20M parameters (`model_size=L`). See `config.py` for a full list of arguments.

----

## Resumable Training

Training automatically supports checkpoint-based resumption. If a job is interrupted (e.g., by cluster preemption), it will automatically resume from the latest checkpoint when restarted.

**How it works:**
- Each run gets a unique timestamp-based directory: `logs/<YYYYMMDD_HHMMSS>[_exp_name]/`
- Checkpoints are saved periodically to `logs/<run_id>/checkpoints/`
- Each run directory contains a `run_info.yaml` with metadata (tasks, seed, exp_name, LSF job ID, git commit, status, etc.)
- The run-first structure supports multi-task training (e.g., "soup" with 200 tasks) with shared checkpoints
- On startup, the trainer automatically finds and loads the latest checkpoint if one exists
- Signal handlers (SIGTERM, SIGUSR2) save a checkpoint before the job exits
- Optimizer states are preserved for seamless training continuation

**Run status tracking:**

Each run's `run_info.yaml` is automatically updated with its final status:

| Status | Description |
|--------|-------------|
| `running` | Training in progress |
| `completed` | Training finished successfully |
| `crashed` | Training failed with an error (error message saved) |
| `preempted` | Job was preempted by cluster (checkpoint saved) |
| `interrupted` | User interrupted with Ctrl+C (checkpoint saved) |

The `final_step` field shows how far training progressed, useful for debugging partial runs.

**LSF cluster usage:**

```bash
make interactive      # Launch interactive GPU session
make submit-expert    # Submit all 200 expert training jobs
make submit-eval      # Submit eval jobs to generate videos
make test-sanity      # Verify imports (run inside container)
```

`submit-expert` splits jobs across queues (long-gpu first for fewer interruptions):
- Jobs 1-70 → `long-gpu` (70 GPU limit, 48h walltime)
- Jobs 71-200 → `short-gpu` (130 jobs, 6h walltime with auto-resume)

Jobs that get preempted are automatically requeued and resume from their last checkpoint.

----

## Monitoring Training Progress

The `discover/` module provides tools for monitoring training runs, tracking progress, and collecting results. See [`tdmpc2/discover/README.md`](tdmpc2/discover/README.md) for full documentation.

**Quick start:**

```python
from pathlib import Path
from discover import RunsCache, training_overview

cache = RunsCache(
    logs_dir=Path('logs'),
    cache_path=Path('discover/runs_cache.parquet'),
    wandb_project='<entity/project>',
)
df, _, _ = cache.load()
training_overview(df, target_step=5_000_000)
```

Or use the interactive notebook: `tdmpc2/discover/browse_runs.ipynb`

**CLI tools:**

```bash
make status           # Training progress overview
make status-debug     # Status with detailed debug info

make discover         # All runs (local + wandb)
make list-completed   # Completed runs
make list-running     # Running runs
make list-crashed     # Crashed runs
make list-local-only  # Not synced to wandb
make list-wandb-only  # Only on wandb (no local logs)

# Collect videos from trained tasks (videos are generated during training)
cd tdmpc2 && python discover/collect_videos.py --min-progress 0.5
```

----

## Make Targets

Run `make help` to see all available targets:

| Target | Description |
|--------|-------------|
| `make status` | Training progress overview |
| `make status-debug` | Status with detailed debug info |
| `make discover` | All runs (local + wandb) |
| `make list-completed` | Completed runs |
| `make list-running` | Running runs |
| `make list-crashed` | Crashed runs |
| `make list-local-only` | Runs not synced to wandb |
| `make list-wandb-only` | Runs only on wandb |
| `make interactive` | Launch interactive GPU session |
| `make submit-expert` | Submit expert training jobs |
| `make gen-eval` | Generate eval task list (for tasks missing videos) |
| `make submit-eval` | Submit eval jobs (rarely needed - videos generated during training) |

----

## Citation

If you find our work useful, please consider citing our paper as follows:

```
@misc{Hansen2025Newt,
	title={Learning Massively Multitask World Models for Continuous Control}, 
	author={Nicklas Hansen and Hao Su and Xiaolong Wang},
	booktitle={Preprint},
	url={https://www.nicklashansen.com/NewtWM},
	year={2025}
}
```

----

## Contributing

You are very welcome to contribute to this project, but please understand that we will not be able to respond to any pull requests or issues while the submission is under review. Feel free to open an issue or pull request if you have any suggestions or bug reports, but please review our [guidelines](CONTRIBUTING.md) first.

----

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.
