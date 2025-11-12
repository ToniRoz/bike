import gymnasium as gym
import numpy as np
from wheel_env import WheelEnv  # Make sure this imports your actual class

class WheelGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, len_theta=360, n_spokes=36, render=False, logging=True, scaling=0):
        super().__init__()
        self.inner_env = WheelEnv(
            len_theta=len_theta,
            n_spokes=n_spokes,
            render=render,
            logging=logging,
            scaling=scaling
        )

        self.observation_space = self.inner_env.observation_space
        self.action_space = self.inner_env.action_space

    def reset(self, *, seed=None, options=None):
        # Optional: Reset randomness with seed
        if seed is not None:
            np.random.seed(seed)
        obs, reward = self.inner_env.reset(seed=seed, options=options)
        info = {}  # Could include reward, etc.
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.inner_env.step(action)
        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.inner_env.render(mode=mode)

    def close(self):
        self.inner_env.close()
