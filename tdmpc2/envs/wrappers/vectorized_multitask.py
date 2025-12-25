from copy import deepcopy

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from envs.wrappers.embedding import EmbeddingWrapper


MAX_OBS_DIM = 128
MAX_ACTION_DIM = 16


class VecWrapper(gym.Wrapper):
	"""
	A simple wrapper that unifies observation and action spaces across vectorized environments.
 	This is useful for environments that have different observation/action spaces per task.
	"""
	def __init__(self, env):
		super().__init__(env)
		self.orig_observation_space = env.observation_space
		self.orig_action_space = env.action_space
		if isinstance(env.observation_space, gym.spaces.Dict):  # State + RGB
			self.observation_space['state'] = gym.spaces.Box(
				low=-np.inf, high=np.inf, shape=(MAX_OBS_DIM,), dtype=np.float32)
		elif len(env.observation_space.shape) == 1:  # State
			self.observation_space = gym.spaces.Box(
				low=-np.inf, high=np.inf, shape=(MAX_OBS_DIM,), dtype=np.float32)
		else:  # RGB
			self.observation_space = env.observation_space
		self.action_space = gym.spaces.Box(
			low=-1, high=1, shape=(MAX_ACTION_DIM,), dtype=np.float32)

	def _pad_obs(self, obs):
		if isinstance(self.observation_space, gym.spaces.Dict):  # State + RGB
			if len(obs['state'].shape) == 1 and obs['state'].shape != self.observation_space['state'].shape:
				pad = np.zeros(self.observation_space['state'].shape[0] - obs['state'].shape[0], dtype=obs['state'].dtype)
				obs['state'] = np.concatenate((obs['state'], pad), axis=-1)
		elif len(self.observation_space.shape) == 1 and obs.shape != self.observation_space.shape:
			pad = np.zeros(self.observation_space.shape[0] - obs.shape[0], dtype=obs.dtype)
			obs = np.concatenate((obs, pad), axis=-1)
		return obs

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		return self._pad_obs(obs), info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action[:self.orig_action_space.shape[0]])
		return self._pad_obs(obs), reward, terminated, truncated, info

	def render(self, *args, **kwargs):
		return self.env.render(*args, **kwargs).copy()

	def close(self):
		return self.env.close()


class VectorizedMultitaskWrapper(gym.Wrapper):
	"""
	Wrapper for vectorized multi-task environments.
	Each environment in the vectorized setup may have different observation and action spaces,
	and corresponds to different tasks. Creates one environment per task.
	"""

	def __init__(self, cfg, make_fn):
		self.cfg = deepcopy(cfg)
		self.cfg.task_embeddings = []  # Saves memory by not storing copies of embeddings
		
		# Create configs
		cfgs = [deepcopy(self.cfg) for _ in range(self.cfg.num_envs)]
		for i in range(len(cfgs)):
			cfgs[i].task = cfgs[i].tasks[i%self.cfg.num_tasks]
			cfgs[i].tasks = [cfgs[i].task]
			cfgs[i].num_envs = 1
			cfgs[i].child_env = True
			cfgs[i].seed = cfgs[i].seed + np.random.randint(1000)
		
		# Create environment
		wrapper = AsyncVectorEnv if self.cfg.get('env_mode', 'sync') == 'async' else SyncVectorEnv
		env_fns = [lambda c=cfgs[i]: VecWrapper(make_fn(c)) for i in range(self.cfg.num_envs)]
		self.env = wrapper(env_fns)
		super().__init__(self.env)
		
		# Define obs and action spaces
		if self.cfg.obs == 'state':
			self.observation_space = gym.spaces.Box(
				low=-np.inf, high=np.inf, shape=(MAX_OBS_DIM,), dtype=np.float32)
		else:
			self.observation_space = gym.spaces.Dict({
				'state': gym.spaces.Box(
					low=-np.inf, high=np.inf, shape=(MAX_OBS_DIM,), dtype=np.float32),
				'rgb': gym.spaces.Box(
					low=0, high=255, shape=(3, self.cfg.render_size, self.cfg.render_size), dtype=np.uint8),
			})
		self.action_space = gym.spaces.Box(
			low=-1, high=1, shape=(MAX_ACTION_DIM,), dtype=np.float32)
		assert self.action_space.shape[0] == self.env.action_space.shape[1], \
			"Action space mismatch between multitask wrapper and individual envs."		
		self._max_episode_steps = self.cfg.episode_lengths
		self.max_episode_steps = max(self._max_episode_steps)
		if self.cfg.rank == 0:
			print('Episode lengths:', self._max_episode_steps)
			print('Action dims:', self.cfg.action_dims)

	def rand_act(self):
		return torch.rand((self.cfg.num_envs, *self.action_space.shape)) * 2 - 1
	
	def _preprocess_obs(self, obs):
		if self.cfg.obs == 'state':
			obs = torch.tensor(obs, dtype=torch.float32)
		else:  # State + RGB
			obs = {
				'state': torch.tensor(obs['state'], dtype=torch.float32),
				'rgb': torch.tensor(obs['rgb'], dtype=torch.uint8),
			}
		return obs

	def reset(self):
		obs, info = self.env.reset()
		return self._preprocess_obs(obs), self._preprocess_info(info, obs=obs, done=None)
	
	def _preprocess_info(self, info, obs=None, done=None):
		"""
		Preprocess vector-env info dict into torch-friendly structures.
		
		Critical invariants expected by `Trainer`:
		- `info['success']` is a float tensor of shape [num_envs]
		- when any env is done, `info['final_info']` is a dict of float tensors of shape [num_envs]
		  and must at least include keys: 'success', 'score'
		- when any env is done, `info['final_observation']` is a tensor stacked in env-index order
		  with shape [num_done, ...] so that `_obs[done] = info['final_observation']` works.
		"""
		num_envs = self.cfg.num_envs

		# Ensure success exists and is vector-shaped
		if 'success' not in info:
			info['success'] = np.full((num_envs,), np.nan, dtype=np.float32)
		success_np = np.asarray(info['success'], dtype=np.float32).reshape(num_envs)
		info['success'] = torch.from_numpy(success_np)

		# Normalize done mask (numpy bool array shape [num_envs]) if provided
		if done is not None:
			done = np.asarray(done, dtype=bool).reshape(num_envs)
		else:
			done = None

		# Helper: convert numpy arrays / dicts to torch/tensordict
		fp64_to_fp32 = lambda x: x.astype(np.float32) if isinstance(x, np.ndarray) and x.dtype == np.float64 else x
		def np_to_torch(x):
			if isinstance(x, np.ndarray):
				return torch.from_numpy(fp64_to_fp32(x))
			if isinstance(x, dict):
				return TensorDict({k: np_to_torch(v) for k, v in x.items()})
			return x

		# Handle final transitions robustly (Async/SyncVectorEnv may provide final_* fields inconsistently)
		if done is not None and done.any():
			# --- final_observation: fill missing done entries with current obs[i] ---
			final_obs = info.get('final_observation', None)
			if final_obs is None:
				final_obs_list = [None] * num_envs
			elif isinstance(final_obs, (list, tuple)):
				final_obs_list = list(final_obs)
				if len(final_obs_list) != num_envs:
					# Unexpected shape; fall back to empty list and use current obs
					final_obs_list = [None] * num_envs
			elif isinstance(final_obs, np.ndarray):
				# Assume batched obs
				final_obs_list = [final_obs[i] for i in range(min(num_envs, final_obs.shape[0]))]
				if len(final_obs_list) != num_envs:
					final_obs_list = [None] * num_envs
			else:
				final_obs_list = [None] * num_envs

			# current obs per env (only needed for done entries)
			def obs_i(i):
				if obs is None:
					return None
				if isinstance(obs, dict):
					return {k: v[i] for k, v in obs.items()}
				return obs[i]

			for i in range(num_envs):
				if done[i] and final_obs_list[i] is None:
					final_obs_list[i] = obs_i(i)

			# Stack final obs in env-index order of done=True
			final_obs_stacked = []
			for i in range(num_envs):
				if done[i]:
					x = final_obs_list[i]
					if x is None:
						# As a last resort, just skip (should be rare); trainer assignment will still work
						# as long as counts match, so we avoid introducing mismatches.
						continue
					final_obs_stacked.append(np_to_torch(x))
			if len(final_obs_stacked) > 0:
				info['final_observation'] = torch.stack(final_obs_stacked)
			else:
				# If we can't construct it, don't set the key (prevents mismatched assignment)
				info.pop('final_observation', None)

			# --- final_info: ensure at least success/score tensors of shape [num_envs] ---
			score_src = info.get('score', None)
			if score_src is None:
				score_np = success_np.copy()
			else:
				score_np = np.asarray(score_src, dtype=np.float32).reshape(num_envs)

			final_info_list = info.get('final_info', None)
			if not isinstance(final_info_list, (list, tuple)) or len(final_info_list) != num_envs:
				final_info_list = [None] * num_envs

			# Fill done entries with minimal dicts if missing
			for i in range(num_envs):
				if done[i] and final_info_list[i] is None:
					final_info_list[i] = {'success': success_np[i], 'score': score_np[i]}

			final_info_out = {}
			for k in ('success', 'score'):
				arr = np.full((num_envs,), np.nan, dtype=np.float32)
				for i in range(num_envs):
					d = final_info_list[i]
					if d is None:
						continue
					v = d.get(k, None)
					if v is None:
						continue
					arr[i] = float(np.asarray(v, dtype=np.float32))
				final_info_out[k] = torch.from_numpy(arr)
			info['final_info'] = final_info_out

			# final_frame/frame are optional; only convert if present and well-formed
			if 'final_frame' in info and isinstance(info['final_frame'], (list, tuple)):
				try:
					info['final_frame'] = torch.stack([torch.from_numpy(d) for d in info['final_frame'] if d is not None])
				except Exception:
					# Leave as-is if conversion fails
					pass

		if 'frame' in info:
			try:
				info['frame'] = torch.stack([torch.from_numpy(d) for d in info['frame']])
			except Exception:
				pass
		return info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.numpy())
		done = np.logical_or(terminated, truncated)
		return self._preprocess_obs(obs), \
			   torch.tensor(reward, dtype=torch.float32), \
			   torch.tensor(terminated, dtype=torch.bool), \
			   torch.tensor(truncated, dtype=torch.bool), \
			   self._preprocess_info(info, obs=obs, done=done)
	
	def render(self, *args, **kwargs):
		if isinstance(self.env, AsyncVectorEnv):
			# AsyncVectorEnv runs envs in subprocesses - use call() to invoke render
			frames = self.env.call("render", *args, **kwargs)
			frames = [torch.from_numpy(frame) for frame in frames if frame is not None]
		else:
			# SyncVectorEnv has direct access to .envs
			frames = []
			for env in self.env.envs:
				frame = env.render(*args, **kwargs)
				if frame is not None:
					frames.append(torch.from_numpy(frame))
		return torch.cat(frames, dim=1) if len(frames) > 0 else None


def make_vectorized_multitask_env(cfg, make_fn):
	"""Make a vectorized multi-task environment for Newt experiments."""
	print(f'[Rank {cfg.rank}] Creating multi-task environment with tasks: {cfg.tasks}')
	env = VectorizedMultitaskWrapper(cfg, make_fn)
	if cfg.obs == 'rgb':
		env = EmbeddingWrapper(env)
	return env
