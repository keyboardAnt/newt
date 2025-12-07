import os
import sys

import numpy as np
import pygame


# Ensure we can import the local envs package when running from the repo root.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TD_MPC_DIR = os.path.join(ROOT_DIR, "tdmpc2")
if TD_MPC_DIR not in sys.path:
	sys.path.insert(0, TD_MPC_DIR)

from envs.tasks.pygame import CowboyEnv


def main():
	# Instantiate the environment directly (bypassing Newt wrappers for simplicity).
	env = CowboyEnv(max_episode_steps=500)
	_obs, _info = env.reset()

	pygame.init()
	width, height = 224, 224
	screen = pygame.display.set_mode((width, height))
	pygame.display.set_caption("CowboyEnv (keyboard control)")
	clock = pygame.time.Clock()

	running = True
	while running:
		# Default to zero action every frame.
		action = np.zeros(env.action_space.shape, dtype=np.float32)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

		keys = pygame.key.get_pressed()

		# Horizontal control (left/right arrows) for first action dimension.
		if keys[pygame.K_LEFT]:
			action[0] = -1.0
		elif keys[pygame.K_RIGHT]:
			action[0] = 1.0

		# Optional vertical control (up/down arrows) for second action dimension.
		if env.action_space.shape[0] > 1:
			if keys[pygame.K_UP]:
				action[1] = 1.0
			elif keys[pygame.K_DOWN]:
				action[1] = -1.0

		# Escape to quit.
		if keys[pygame.K_ESCAPE]:
			running = False

		_obs, _reward, terminated, truncated, _info = env.step(action)

		# Render current frame and display in the window.
		frame = env.render()  # H x W x 3 uint8
		surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))  # Convert to W x H x 3
		screen.blit(surface, (0, 0))
		pygame.display.flip()

		# If the episode ended, reset the environment.
		if terminated or truncated:
			_obs, _info = env.reset()

		clock.tick(30)  # Limit to ~30 FPS.

	env.close()
	pygame.quit()


if __name__ == "__main__":
	main()


