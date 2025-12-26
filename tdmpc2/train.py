import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import signal
import socket
import subprocess
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List

import yaml

import torch
import torch.nn as nn
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from termcolor import colored
from tensordict import TensorDict

from common import barrier, set_seed
from common.buffer import Buffer, EnsembleBuffer
from common.logger import Logger
from common.logs_ux import ensure_task_latest
from common.world_model import WorldModel
from config import Config, split_by_rank, parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2
from trainer import Trainer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

_LOG = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


def setup(rank, world_size, port):
	os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
	os.environ["MASTER_PORT"] = port
	torch.distributed.init_process_group(
		backend="nccl",
		rank=rank,
		world_size=world_size
	)
	return port


def write_run_info(cfg, work_dir: Path):
	"""Write run metadata to run_info.yaml for easier debugging and log discovery."""
	# Get unique tasks (handles both single-task and multi-task/soup runs)
	unique_tasks = list(dict.fromkeys(cfg.global_tasks)) if cfg.global_tasks else [cfg.task]
	
	info = {
		'task': cfg.task,  # Primary task identifier (e.g., 'soup' or 'walker-walk')
		'tasks': unique_tasks,  # Full list of unique tasks being trained
		'num_tasks': len(unique_tasks),
		'run_id': cfg.run_id,
		'exp_name': cfg.exp_name,
		'seed': cfg.seed,
		'started_at': datetime.now().isoformat(),
		'lsf_job_id': os.environ.get('LSB_JOBID'),
		'lsf_array_idx': os.environ.get('LSB_JOBINDEX'),
		'hostname': socket.gethostname(),
		'git_commit': subprocess.getoutput('git rev-parse --short HEAD 2>/dev/null') or None,
		'status': 'running',
		'steps': cfg.steps,
		'model_size': cfg.model_size,
		'checkpoint': cfg.checkpoint,
		# UTD config for reproducibility
		'utd': cfg.utd,
		'auto_utd': cfg.get('auto_utd', False),
		'auto_utd_dry_run': cfg.get('auto_utd_dry_run', False),
		'auto_utd_max': cfg.get('auto_utd_max', 0.5),
	}
	work_dir = Path(work_dir)
	work_dir.mkdir(parents=True, exist_ok=True)
	(work_dir / 'run_info.yaml').write_text(yaml.dump(info, default_flow_style=False, sort_keys=False))


def update_run_info(work_dir: Path, status: str, final_step: int = None, error: str = None):
    """Update run_info.yaml with final status. Called on completion, crash, or interruption."""
    run_info_path = Path(work_dir) / 'run_info.yaml'
    if not run_info_path.exists():
        return

    info = yaml.safe_load(run_info_path.read_text())
    info['status'] = status
    info['finished_at'] = datetime.now().isoformat()
    if final_step is not None:
        info['final_step'] = final_step
    if error:
        info['error'] = error[:500]  # Truncate long errors
    run_info_path.write_text(yaml.dump(info, default_flow_style=False, sort_keys=False))


def write_hydra_snapshot(cfg_composed, work_dir: Path) -> None:
	"""Persist Hydra-composed config + overrides into <work_dir>/.hydra/.

	We keep Hydra output dirs disabled (no tdmpc2/outputs/...), but still want the
	exact config used for each run colocated with run artifacts.
	"""
	work_dir = Path(work_dir)
	hydra_dir = work_dir / ".hydra"
	try:
		hydra_dir.mkdir(parents=True, exist_ok=True)
	except Exception:
		_LOG.exception("[hydra-snapshot] Failed to create %s", hydra_dir)
		return

	# 1) Composed job config (what train.py received from Hydra)
	try:
		config_path = hydra_dir / "config.yaml"
		if OmegaConf.is_config(cfg_composed):
			config_path.write_text(OmegaConf.to_yaml(cfg_composed))
		else:
			config_path.write_text(yaml.dump(cfg_composed, sort_keys=False))
	except Exception:
		_LOG.exception("[hydra-snapshot] Failed to write composed config to %s", hydra_dir / "config.yaml")

	# 2) Hydra runtime config (hydra.yaml) + overrides list (overrides.yaml)
	try:
		hcfg = HydraConfig.get()
	except Exception:
		# This can happen if HydraConfig isn't initialized yet (should be rare under @hydra.main).
		_LOG.exception("[hydra-snapshot] Failed to access HydraConfig.get(); skipping hydra.yaml/overrides.yaml")
		return

	try:
		(hydra_dir / "hydra.yaml").write_text(OmegaConf.to_yaml(hcfg))
	except Exception:
		_LOG.exception("[hydra-snapshot] Failed to write hydra runtime config to %s", hydra_dir / "hydra.yaml")

	try:
		overrides = list(getattr(hcfg.overrides, "task", []) or [])
		(hydra_dir / "overrides.yaml").write_text(
			yaml.dump(overrides, default_flow_style=False, sort_keys=False)
		)
	except Exception:
		_LOG.exception("[hydra-snapshot] Failed to write overrides to %s", hydra_dir / "overrides.yaml")


class DDPWrapper(nn.Module):
	def __init__(self, module: nn.Module):
		super().__init__()
		self._module = module  # Can be plain or DDP-wrapped

	def forward(self, *args, **kwargs):
		return self._module(*args, **kwargs)

	def __getattr__(self, name):
		if name == '_module':
			return super().__getattr__(name)
		try:
			return getattr(self._module, name)
		except AttributeError:
			# Try to unwrap once if wrapped by DDP
			if hasattr(self._module, 'module'):
				return getattr(self._module.module, name)
			raise

	def __setattr__(self, name, value):
		if name == '_module':
			super().__setattr__(name, value)
		else:
			setattr(self._module, name, value)

	def state_dict(self, *args, **kwargs):
		return self._module.state_dict(*args, **kwargs)

	def load_state_dict(self, *args, **kwargs):
		return self._module.load_state_dict(*args, **kwargs)


def train(rank: int, cfg: dict, buffer: Buffer):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.
	See config.yaml for a full list of args.
	"""
	if cfg.world_size > 1:
		setup(rank, cfg.world_size, cfg.port)
		print(colored('Rank:', 'yellow', attrs=['bold']), rank)
	set_seed(cfg.seed + rank)
	cfg.rank = rank

	# -------------------------------------------------------------------------
	# CUDA init can occasionally fail transiently on the cluster (e.g. bad/blocked GPU
	# assignment). Retry a few times before giving up.
	# -------------------------------------------------------------------------
	def _init_cuda_device_with_retries(max_retries: int = 5, base_sleep_s: float = 5.0):
		last_exc = None
		for attempt in range(1, max_retries + 1):
			try:
				torch.cuda.set_device(rank)
				# Force CUDA context init early to fail fast here (and be retried)
				_ = torch.empty(1, device=f"cuda:{rank}")
				return
			except Exception as e:
				last_exc = e
				msg = str(e)
				retriable = (
					"CUDA-capable device(s) is/are busy or unavailable" in msg
					or "CUDA error" in msg
				)
				if not retriable or attempt == max_retries:
					raise
				sleep_s = min(60.0, base_sleep_s * attempt)
				print(
					colored(
						f"[Rank {cfg.rank}] CUDA init failed (attempt {attempt}/{max_retries}): {msg}. "
						f"Retrying in {sleep_s:.0f}s...",
						"yellow",
						attrs=["bold"],
					)
				)
				time.sleep(sleep_s)
		if last_exc is not None:
			raise last_exc

	_init_cuda_device_with_retries()

	# split tasks across processes by rank
	if cfg.task == 'soup':
		assert cfg.num_tasks % cfg.world_size == 0, \
			'Number of tasks must be divisible by number of GPUs.'
		cfg.tasks = split_by_rank(cfg.tasks, rank, cfg.world_size)
		print(f'[Rank {rank}] Tasks: {cfg.tasks}')
		cfg.num_tasks = len(cfg.tasks)
		cfg.num_envs = len(cfg.tasks)

	def make_agent(cfg):
		model = WorldModel(cfg).to(f"cuda:{cfg.rank}")
		if cfg.world_size > 1:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.rank])
			model = DDPWrapper(model)
		agent = TDMPC2(model, cfg)
		return agent

	# Build components with best-effort crash reporting to run_info.yaml.
	# (This is helpful for early failures before the main training try/except.)
	try:
		env = make_env(cfg)
		logger = Logger(cfg)
		agent = make_agent(cfg)
		trainer = Trainer(
			cfg=cfg,
			env=env,
			agent=agent,
			buffer=buffer,
			logger=logger,
		)
	except Exception as e:
		print(colored(f'[Rank {cfg.rank}] Training init crashed with exception: {repr(e)}', 'red', attrs=['bold']))
		if cfg.rank == 0:
			update_run_info(cfg.work_dir, 'crashed', final_step=0, error=repr(e))
		raise

	# Register signal handlers for graceful preemption (LSF uses SIGTERM/SIGUSR2)
	def handle_preemption(signum, frame):
		sig_name = signal.Signals(signum).name
		print(colored(f'[Rank {cfg.rank}] Received {sig_name}, saving checkpoint...', 'yellow', attrs=['bold']))
		trainer.save_checkpoint()
		if cfg.rank == 0:
			update_run_info(cfg.work_dir, 'preempted', final_step=trainer._step)
		print(colored(f'[Rank {cfg.rank}] Checkpoint saved, exiting.', 'yellow', attrs=['bold']))
		sys.exit(0)

	signal.signal(signal.SIGTERM, handle_preemption)
	signal.signal(signal.SIGUSR2, handle_preemption)  # LSF preemption signal

	barrier()  # Ensure all processes are ready before starting training
	try:
		trainer.train()
		if cfg.rank == 0:
			update_run_info(cfg.work_dir, 'completed', final_step=trainer._step)
			print('\nTraining completed successfully')
	except KeyboardInterrupt:
		print(colored(f'[Rank {cfg.rank}] Training interrupted by user (Ctrl+C)', 'red', attrs=['bold']))
		trainer.save_checkpoint()
		if cfg.rank == 0:
			update_run_info(cfg.work_dir, 'interrupted', final_step=trainer._step)
		raise
	except Exception as e:
		print(colored(f'[Rank {cfg.rank}] Training crashed with exception: {repr(e)}', 'red', attrs=['bold']))
		# Best-effort: persist a checkpoint for debugging/resume before propagating.
		try:
			trainer.save_checkpoint()
		except Exception as ckpt_e:
			print(colored(f'[Rank {cfg.rank}] Failed to save crash checkpoint: {repr(ckpt_e)}', 'red', attrs=['bold']))
		if cfg.rank == 0:
			update_run_info(cfg.work_dir, 'crashed', final_step=trainer._step, error=repr(e))
		raise
	finally:
		if torch.distributed.is_initialized():
			torch.distributed.destroy_process_group()


def load_demos(
		cfg: dict,
		buffers: List[Buffer] = [],
		expected_obs_dim: int = 128,
		expected_action_dim: int = 16,
	):
	"""
	Load demonstrations into the buffer.
	"""
	tds = []
	num_eps = 0
	tasks = cfg.global_tasks if cfg.task == 'soup' else [cfg.task]
	for i, task in enumerate(tasks):
		demo_path = f'{cfg.data_dir}/{task}.pt'
		if not os.path.exists(demo_path):
			print(f'No demonstrations found for task {task}, skipping.')
			continue
		td = torch.load(demo_path, weights_only=False)
		
		# Load image observations if specified
		if cfg.obs == 'rgb':
			if 'feat' not in td:
				print(f'Warning: no visual features found in demonstrations for task {task}, skipping.')
				continue
			td['obs'] = TensorDict({'state': td['obs'], 'rgb': td['feat']})  # Non-stacked features
		try:
			del td['feat']
			del td['feat-stacked']
		except:
			pass
		td['task'] = torch.full_like(td['reward'], i, dtype=torch.int32)
		num_new_eps = td['episode'].max().item() + 1
		td['episode'] = td['episode'] + num_eps
		if task.startswith('ms-'): # Limit to 20 episodes for maniskill3 tasks
			td = td[td['episode'] < num_eps + 20]
			num_new_eps = 20
		num_eps += num_new_eps
		tds.append(td)
		print(f'Loaded {num_new_eps} episodes for task {task}')
	assert len(tds) > 0, 'No demonstrations found for any task.'
	tds = torch.cat(tds, dim=0)
	for buffer in buffers:
		buffer.load_demos(tds)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def launch(cfg: Config):
	# Ensure exceptions from helper utilities are visible even if the environment
	# doesn't configure logging by default.
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	# Keep the Hydra-composed config around so we can snapshot it into work_dir/.hydra/
	cfg_composed = cfg
	cfg = parse_cfg(cfg)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
	# Persist Hydra config/overrides into the run directory for reproducibility.
	write_hydra_snapshot(cfg_composed, cfg.work_dir)

	# Convenience: logs/<task>/latest -> this run directory (rank 0 only).
	try:
		task_dir = Path(cfg.work_dir).parent
		ensure_task_latest(task_dir, Path(cfg.work_dir))
	except Exception:
		_LOG.exception("[logs-ux] Failed to update task latest symlink (continuing).")
	
	# Write run metadata for easier debugging and log discovery
	write_run_info(cfg, cfg.work_dir)

	# Set batch size
	cfg.world_size = torch.cuda.device_count() if cfg.multiproc else 1
	if cfg.world_size > 1:
		print(colored(f'Using {cfg.world_size} GPUs', 'green', attrs=['bold']))
		assert cfg.batch_size % cfg.world_size == 0, \
			'Batch size must be divisible by number of GPUs.'
		print(colored('Effective batch size:', 'yellow', attrs=['bold']), cfg.batch_size)
		cfg.batch_size = cfg.batch_size // cfg.world_size
		print(colored('Per-GPU batch size:', 'yellow', attrs=['bold']), cfg.batch_size)

	# Create buffer
	# Note: Buffer sampler compilation is disabled because episode count grows
	# dynamically, causing shape changes that trigger recompilation conflicts
	# with TDMPC2's compiled methods. TDMPC2 methods are still compiled.
	buffer_args = {
		'capacity': cfg.buffer_size,
		'batch_size': cfg.batch_size,
		'horizon': cfg.horizon,
		'multiproc': cfg.multiproc,
		'compile': False,  # Disabled - dynamic shapes cause recompilation issues
	}
	if cfg.use_demos and cfg.no_demo_buffer:
		buffer = Buffer(**buffer_args)
		load_demos(cfg, [buffer])
		print('Warning: using a single buffer for both demos and experience!')
	if cfg.use_demos:
		# Create demonstration buffer
		demo_buffer_args = deepcopy(buffer_args)
		demo_buffer_args['capacity'] = 1_900_000 if cfg.task == 'soup' else 50_000
		demo_buffer_args['batch_size'] = demo_buffer_args['batch_size'] // 2
		demo_buffer_args['cache_values'] = True
		buffer = EnsembleBuffer(Buffer(**demo_buffer_args), **buffer_args)
		load_demos(cfg, [buffer._offline, buffer])
	else:
		# Default to regular buffer
		buffer = Buffer(**buffer_args)

	if cfg.world_size > 1:
		cfg.port = os.getenv("MASTER_PORT", str(12355 + int(os.getpid()) % 1000))
		torch.multiprocessing.spawn(
			train,
			args=(cfg, buffer),
			nprocs=cfg.world_size,
			join=True,
		)
	else:
		train(0, cfg, buffer)


if __name__ == '__main__':
	launch()
