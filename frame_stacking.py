"""
Frame Stacking Wrapper with Action History

Provides temporal context to feedforward networks by:
1. Stacking N previous observations
2. Including N-1 previous actions (one-hot encoded)
3. Zero-padding at episode start
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Optional, Tuple, List


class FrameStackWrapper(gym.Wrapper):
    """
    Wrapper that stacks observations and includes action history.
    
    For stack_size=4 with discrete actions:
    - Stacks 4 observations: [obs_{t-3}, obs_{t-2}, obs_{t-1}, obs_t]
    - Includes 3 previous actions (one-hot): [a_{t-3}, a_{t-2}, a_{t-1}]
    
    Final obs dim = obs_dim * stack_size + num_actions * (stack_size - 1)
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        stack_size: int = 4,
        include_actions: bool = True
    ):
        super().__init__(env)
        
        self.stack_size = stack_size
        self.include_actions = include_actions
        
        # Original observation dimension
        if isinstance(env.observation_space, spaces.Box):
            self.obs_dim = int(np.prod(env.observation_space.shape))
        else:
            raise ValueError(f"Unsupported observation space: {type(env.observation_space)}")
        
        # Action space (discrete only for action history)
        if isinstance(env.action_space, spaces.Discrete):
            self.num_actions = env.action_space.n
        else:
            self.include_actions = False
            self.num_actions = 0
            print("[FrameStackWrapper] Non-discrete action space: action history disabled")
        
        # Calculate new observation dimension
        self.stacked_obs_dim = self.obs_dim * stack_size
        if self.include_actions:
            self.action_history_dim = self.num_actions * (stack_size - 1)
        else:
            self.action_history_dim = 0
        self.total_obs_dim = self.stacked_obs_dim + self.action_history_dim
        
        # New observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.total_obs_dim,), 
            dtype=np.float32
        )
        
        # Buffers
        self.obs_buffer = deque(maxlen=stack_size)
        self.action_buffer = deque(maxlen=stack_size - 1)
        
        print(f"[FrameStackWrapper] stack_size={stack_size}, "
              f"obs_dim={self.obs_dim} -> {self.total_obs_dim}")
    
    def _get_stacked_obs(self) -> np.ndarray:
        """Build stacked observation with action history."""
        # Pad observations if needed
        obs_list = list(self.obs_buffer)
        while len(obs_list) < self.stack_size:
            obs_list.insert(0, np.zeros(self.obs_dim, dtype=np.float32))
        stacked_obs = np.concatenate(obs_list)
        
        if not self.include_actions:
            return stacked_obs.astype(np.float32)
        
        # Pad action history if needed
        action_list = list(self.action_buffer)
        while len(action_list) < self.stack_size - 1:
            action_list.insert(0, np.zeros(self.num_actions, dtype=np.float32))
        action_history = np.concatenate(action_list)
        
        return np.concatenate([stacked_obs, action_history]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.obs_buffer.append(obs.flatten().astype(np.float32))
        return self._get_stacked_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add action to history (one-hot)
        if self.include_actions:
            one_hot = np.zeros(self.num_actions, dtype=np.float32)
            one_hot[action] = 1.0
            self.action_buffer.append(one_hot)
        
        self.obs_buffer.append(obs.flatten().astype(np.float32))
        return self._get_stacked_obs(), reward, terminated, truncated, info


class VecFrameStackWrapper:
    """Frame stacking for vectorized environments."""
    
    def __init__(self, vec_env, stack_size: int = 4, include_actions: bool = True):
        self.vec_env = vec_env
        self.n_envs = len(vec_env)
        self.stack_size = stack_size
        self.include_actions = include_actions
        
        # Preserve vec_env attributes
        self.single_observation_space = vec_env.single_observation_space
        self.single_action_space = vec_env.single_action_space
        self.action_space = vec_env.action_space
        
        # Dimensions
        self.obs_dim = int(np.prod(self.single_observation_space.shape))
        
        if isinstance(self.single_action_space, spaces.Discrete):
            self.num_actions = self.single_action_space.n
        else:
            self.include_actions = False
            self.num_actions = 0
        
        self.stacked_obs_dim = self.obs_dim * stack_size
        self.action_history_dim = self.num_actions * (stack_size - 1) if self.include_actions else 0
        self.total_obs_dim = self.stacked_obs_dim + self.action_history_dim
        
        # New observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.total_obs_dim,),
            dtype=np.float32
        )
        
        # Per-environment buffers
        self.obs_buffers = [deque(maxlen=stack_size) for _ in range(self.n_envs)]
        self.action_buffers = [deque(maxlen=stack_size - 1) for _ in range(self.n_envs)]
        
        print(f"[VecFrameStackWrapper] n_envs={self.n_envs}, "
              f"obs_dim={self.obs_dim} -> {self.total_obs_dim}")
    
    def __len__(self):
        return self.n_envs
    
    def _get_stacked_obs(self, i: int) -> np.ndarray:
        obs_list = list(self.obs_buffers[i])
        while len(obs_list) < self.stack_size:
            obs_list.insert(0, np.zeros(self.obs_dim, dtype=np.float32))
        stacked = np.concatenate(obs_list)
        
        if not self.include_actions:
            return stacked.astype(np.float32)
        
        act_list = list(self.action_buffers[i])
        while len(act_list) < self.stack_size - 1:
            act_list.insert(0, np.zeros(self.num_actions, dtype=np.float32))
        act_hist = np.concatenate(act_list)
        
        return np.concatenate([stacked, act_hist]).astype(np.float32)
    
    def reset(self, seed=None):
        obs, infos = self.vec_env.reset(seed=seed)
        for i in range(self.n_envs):
            self.obs_buffers[i].clear()
            self.action_buffers[i].clear()
            self.obs_buffers[i].append(obs[i].flatten().astype(np.float32))
        return np.array([self._get_stacked_obs(i) for i in range(self.n_envs)]), infos
    
    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.vec_env.step(actions)
        for i in range(self.n_envs):
            if self.include_actions:
                one_hot = np.zeros(self.num_actions, dtype=np.float32)
                one_hot[int(actions[i])] = 1.0
                self.action_buffers[i].append(one_hot)
            
            if terms[i] or truncs[i]:
                self.obs_buffers[i].clear()
                self.action_buffers[i].clear()
            
            self.obs_buffers[i].append(obs[i].flatten().astype(np.float32))
        
        return np.array([self._get_stacked_obs(i) for i in range(self.n_envs)]), rewards, terms, truncs, infos
    
    def close(self):
        self.vec_env.close()