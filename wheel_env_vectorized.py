

import gymnasium as gym
import numpy as np
from typing import List, Tuple, Optional, Callable
import multiprocessing as mp
from functools import partial
import cloudpickle


def _worker(remote, parent_remote, env_fn_wrapper):
    """Worker process for SubprocVecEnv."""
    parent_remote.close()
    env = env_fn_wrapper.fn()
    
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, terminated, truncated, info = env.step(data)
                # Auto-reset on done
                if terminated or truncated:
                    final_obs = obs
                    final_info = info
                    obs, reset_info = env.reset()
                    info = {'final_obs': final_obs, 'final_info': final_info, **reset_info}
                remote.send((obs, reward, terminated, truncated, info))
            elif cmd == 'reset':
                obs, info = env.reset(**data if data else {})
                remote.send((obs, info))
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
        except EOFError:
            break


class CloudpickleWrapper:
    """Wrapper that uses cloudpickle for serialization."""
    def __init__(self, fn):
        self.fn = fn
    
    def __getstate__(self):
        return cloudpickle.dumps(self.fn)
    
    def __setstate__(self, fn):
        self.fn = cloudpickle.loads(fn)


class SubprocVecEnv:

    
    def __init__(self, env_fns: List[Callable], start_method: str = 'spawn'):
        self.n_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        
        # Use spawn to avoid issues with CUDA
        ctx = mp.get_context(start_method)
        
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_envs)])
        self.processes = []
        
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        
        # Get observation and action spaces from first env
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        
        # For compatibility with gym.vector API
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
    
    def step_async(self, actions):
        """Send step commands to all environments."""
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
    
    def step_wait(self):
        """Wait for step results from all environments."""
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rewards, terminateds, truncateds, infos = zip(*results)
        return (
            np.stack(obs),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            list(infos)
        )
    
    def step(self, actions):
        """Step all environments with given actions."""
        self.step_async(actions)
        return self.step_wait()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset all environments."""
        for i, remote in enumerate(self.remotes):
            env_seed = seed + i if seed is not None else None
            remote.send(('reset', {'seed': env_seed, 'options': options}))
        
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), list(infos)
    
    def close(self):
        """Close all environments."""
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True
    
    def get_attr(self, attr_name: str):
        """Get attribute from all environments."""
        for remote in self.remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in self.remotes]
    
    def __len__(self):
        return self.n_envs


class DummyVecEnv:

    
    def __init__(self, env_fns: List[Callable]):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
    
    def step(self, actions):
        obs_list, rewards, terminateds, truncateds, infos = [], [], [], [], []
        
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                final_obs = obs
                final_info = info
                obs, reset_info = env.reset()
                info = {'final_obs': final_obs, 'final_info': final_info, **reset_info}
            
            obs_list.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        return (
            np.stack(obs_list),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs_list, infos = [], []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed, options=options)
            obs_list.append(obs)
            infos.append(info)
        return np.stack(obs_list), infos
    
    def close(self):
        for env in self.envs:
            env.close()
    
    def __len__(self):
        return self.n_envs




class BatchedWheelEnv:

    
    def __init__(
        self,
        n_envs: int,
        len_theta: int = 360,
        n_spokes: int = 36,
        random_spoke_n: int = 5,
        random_spoke_turns_max: float = 1.0,
        max_tension_penalty: bool = True,
        max_tension_threshold: float = 1000,
        include_tan_displacement: bool = False,
        reward_func: str = "raw",
        action_space_selection: str = "discrete",
        state_space_selection: str = "rimpoints",
        n_harmonics: int = 20,
        # Wheel parameters
        hub_width: float = 0.05,
        hub_diameter: float = 0.04,
        rim_radius: float = 0.3,
        rim_area: float = 100e-6,
        rim_I_lat: float = 1500e-12,
        rim_I_rad: float = 3000e-12,
        rim_J_tor: float = 500e-12,
        rim_young_mod: float = 69e9,
        rim_shear_mod: float = 26e9,
        rim_I_warp: float = 0.0,
        spokes_crossings: int = 3,
        spokes_diameter: float = 2.0e-3,
        spokes_young_mod: float = 210e9,
        number_modes: int = 40,
        init_tension: float = 800.0,
    ):
        self.n_envs = n_envs
        self.n_spokes = n_spokes
        self.len_theta = len_theta
        self.reward_func = reward_func
        self.action_space_selection = action_space_selection
        self.state_space_selection = state_space_selection
        self.include_tan_displacement = include_tan_displacement
        self.max_tension_penalty = max_tension_penalty
        self.max_tension_threshold = max_tension_threshold
        self.random_spoke_n = random_spoke_n
        self.random_spoke_turns_max = random_spoke_turns_max
        self.init_tension = init_tension
        self.n_harmonics = n_harmonics
        self.adjustment_per_turn = 25.4 / 56 / 1000
        
        self.stacksize = 3 if include_tan_displacement else 2
        self.theta = np.linspace(-np.pi, np.pi, len_theta)
        
        # Import here to avoid issues if bikewheelcalc not installed
        from bikewheelcalc import BicycleWheel, Rim, Hub, ModeMatrix
        
        # Create single wheel for shared matrices (they're the same for all envs)
        self.wheel = BicycleWheel()
        self.wheel.hub = Hub(width=hub_width, diameter=hub_diameter)
        self.wheel.rim = Rim(
            radius=rim_radius, area=rim_area,
            I_lat=rim_I_lat, I_rad=rim_I_rad, J_tor=rim_J_tor, I_warp=rim_I_warp,
            young_mod=rim_young_mod, shear_mod=rim_shear_mod
        )
        self.wheel.lace_cross(
            n_spokes=n_spokes, n_cross=spokes_crossings,
            diameter=spokes_diameter, young_mod=spokes_young_mod
        )
        
        self.mm = ModeMatrix(self.wheel, N=number_modes)
        self.B_lat = self.mm.B_theta(self.theta, 0)
        self.B_rad = self.mm.B_theta(self.theta, 1)
        self.B_tan = self.mm.B_theta(self.theta, 2)
        
        self.wheel.apply_tension(init_tension)
        self.K = self.mm.K_rim(tension=True) + self.mm.K_spk(smeared_spokes=False, tension=True)
        self.F_matrix = self.mm.A_adj()
        
        self._prepare_numba_spoke_arrays()
        
        # Pre-compute LU decomposition for faster solves
        from scipy.linalg import lu_factor
        self.K_lu = lu_factor(self.K)
        
        # Batched state arrays
        self.spoke_turns = np.zeros((n_envs, n_spokes), dtype=np.float64)
        self.previous_turns = np.zeros((n_envs, n_spokes), dtype=np.float64)
        self.episode_counters = np.zeros(n_envs, dtype=np.int32)
        self.last_state_norms = np.zeros(n_envs, dtype=np.float64)
        self.first_state_norms = np.zeros(n_envs, dtype=np.float64)
        
        # Define spaces
        self._setup_spaces()
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        if self.state_space_selection == "rimpoints":
            obs_shape = (self.len_theta * self.stacksize,)
            self.observation_space = gym.spaces.Box(-50.0, 50.0, shape=obs_shape, dtype=np.float32)
        elif self.state_space_selection == "spoketensions":
            obs_shape = (self.n_spokes,)
            self.observation_space = gym.spaces.Box(400.0, 1200.0, shape=obs_shape, dtype=np.float32)
        elif self.state_space_selection == "rimandspokes":
            obs_shape = (self.len_theta * self.stacksize + self.n_spokes,)
            self.observation_space = gym.spaces.Box(-50.0, 1200.0, shape=obs_shape, dtype=np.float32)
        elif self.state_space_selection == "fourier":
            n_features = 2 + 4 * self.n_harmonics
            obs_shape = (n_features,)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        elif self.state_space_selection == "fourier_and_spokes":
            n_features = 2 + 4 * self.n_harmonics + self.n_spokes
            obs_shape = (n_features,)
            self.observation_space = gym.spaces.Box(-np.inf, 1200.0, shape=obs_shape, dtype=np.float32)
        
        if self.action_space_selection == "discrete":
            self.action_space = gym.spaces.Discrete(72)
        elif self.action_space_selection == "continous":
            self.action_space = gym.spaces.Box(
                low=np.array([0.0, -1.0]),
                high=np.array([float(self.n_spokes - 1), 1.0]),
                dtype=np.float32
            )
        elif self.action_space_selection == "all_spokes":
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.n_spokes,), dtype=np.float32)
        
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
    
    def _prepare_numba_spoke_arrays(self):
        """Prepare spoke arrays for numba acceleration."""
        spokes = self.wheel.spokes
        n = len(spokes)
        
        self.n_vec = np.zeros((n, 3), dtype=np.float64)
        self.b_vec = np.zeros((n, 3), dtype=np.float64)
        self.EA = np.zeros(n, dtype=np.float64)
        self.lengths = np.zeros(n, dtype=np.float64)
        
        dof = 4 + 8 * self.mm.n_modes
        self.B_spk = np.zeros((n, 4, dof), dtype=np.float64)
        
        for i, s in enumerate(spokes):
            self.n_vec[i] = s.n
            self.b_vec[i] = s.b
            self.EA[i] = s.EA
            self.lengths[i] = s.length
            theta_i = s.rim_pt[1]
            self.B_spk[i, :, :] = self.mm.B_theta(theta_i)
    
    def _wheel_calc_batched(self, tensionchanges_batch):
        """
        Calculate wheel displacement for a batch of tension changes.
        
        Args:
            tensionchanges_batch: (n_envs, n_spokes) array
        
        Returns:
            displacements: (n_envs, len_theta * stacksize)
            tensions: (n_envs, n_spokes)
        """
        from scipy.linalg import lu_solve
        from wheel_env import fast_wheel_calc_with_tension
        
        n_envs = tensionchanges_batch.shape[0]
        displacements = np.zeros((n_envs, self.len_theta * self.stacksize), dtype=np.float32)
        tensions = np.zeros((n_envs, self.n_spokes), dtype=np.float64)
        
        # Unfortunately np.linalg.solve doesn't batch well, so we loop
        # But this is still fast because the loop is in numpy, not Python
        for i in range(n_envs):
            disp, tens = fast_wheel_calc_with_tension(
                self.K, self.F_matrix,
                self.B_rad, self.B_lat, self.B_tan,
                tensionchanges_batch[i].astype(np.float64),
                self.n_vec, self.b_vec, self.EA, self.lengths,
                self.B_spk, self.include_tan_displacement
            )
            displacements[i] = disp
            tensions[i] = tens + self.init_tension
        
        return displacements, tensions
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset all environments."""
        if seed is not None:
            np.random.seed(seed)
        
        self.episode_counters[:] = 0
        self.spoke_turns[:] = 0
        
        # Randomize spoke turns for each environment
        for i in range(self.n_envs):
            n_random = min(self.random_spoke_n, self.n_spokes)
            random_indices = np.random.choice(self.n_spokes, size=n_random, replace=False)
            self.spoke_turns[i, random_indices] = (
                np.random.rand(n_random) * self.random_spoke_turns_max - 
                (self.random_spoke_turns_max / 2)
            )
        
        self.previous_turns[:] = self.spoke_turns
        tensionchanges = self.spoke_turns * self.adjustment_per_turn
        
        displacements, tensions = self._wheel_calc_batched(tensionchanges)
        
        # Compute state norms
        for i in range(self.n_envs):
            stacked = displacements[i].reshape(-1, self.stacksize)
            self.last_state_norms[i] = np.sqrt(np.trapz(np.sum(stacked**2, axis=1), self.theta))
        self.first_state_norms[:] = self.last_state_norms
        
        obs = self._get_observations(displacements, tensions)
        
        infos = [{'spoke turns': self.spoke_turns[i], 
                  'raw state norm': self.last_state_norms[i]} 
                 for i in range(self.n_envs)]
        
        return obs, infos
    
    def _get_observations(self, displacements, tensions):
        """Convert raw state to observations based on state_space_selection."""
        if self.state_space_selection == "rimpoints":
            return displacements.astype(np.float32)
        elif self.state_space_selection == "spoketensions":
            return (tensions / self.init_tension).astype(np.float32)
        elif self.state_space_selection == "rimandspokes":
            return np.concatenate([displacements, tensions / self.init_tension], axis=1).astype(np.float32)
        # Add fourier options as needed
        return displacements.astype(np.float32)
    
    def step(self, actions):
        """Step all environments."""
        # Process actions
        if self.action_space_selection == "discrete":
            spoke_indices = actions // 2
            adjustments = np.where(actions % 2 == 0, -0.1, 0.1)
            self.previous_turns[:] = self.spoke_turns
            for i in range(self.n_envs):
                self.spoke_turns[i, spoke_indices[i]] += adjustments[i]
        
        elif self.action_space_selection == "continous":
            spoke_indices = np.clip(np.round(actions[:, 0]).astype(int), 0, self.n_spokes - 1)
            deltas = np.clip(actions[:, 1], -1.0, 1.0)
            self.previous_turns[:] = self.spoke_turns
            for i in range(self.n_envs):
                self.spoke_turns[i, spoke_indices[i]] += deltas[i]
        
        elif self.action_space_selection == "all_spokes":
            self.spoke_turns += actions
        
        tensionchanges = self.spoke_turns * self.adjustment_per_turn
        displacements, tensions = self._wheel_calc_batched(tensionchanges)
        
        # Compute rewards and done flags
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        terminateds = np.zeros(self.n_envs, dtype=bool)
        truncateds = np.zeros(self.n_envs, dtype=bool)
        
        for i in range(self.n_envs):
            stacked = displacements[i].reshape(-1, self.stacksize)
            state_norm = np.sqrt(np.trapz(np.sum(stacked**2, axis=1), self.theta))
            
            if self.reward_func == "raw":
                rewards[i] = -state_norm
            elif self.reward_func == "normalized":
                step_improvement = 100 * (self.last_state_norms[i] - state_norm) / (abs(self.last_state_norms[i]) + 1e-6)
                rewards[i] = 0 if step_improvement > 0 else -1.0
            elif self.reward_func == "spokes":
                if np.all(np.abs(self.previous_turns[i]) >= np.abs(self.spoke_turns[i])):
                    rewards[i] = 0
                else:
                    rewards[i] = -1
            elif self.reward_func == "combined":
                rewards[i] = -(state_norm + np.sum(np.abs((tensions[i] - self.init_tension) / 400)) / self.n_spokes)
            
            self.last_state_norms[i] = state_norm
            self.episode_counters[i] += 1
            
            truncateds[i] = self.episode_counters[i] > 40
            
            if self.max_tension_penalty and np.any(tensions[i] > self.max_tension_threshold):
                rewards[i] -= 2
            
            # Check termination
            if (np.all(np.abs(displacements[i]) < 0.3) and 
                np.all((tensions[i] >= 700) & (tensions[i] <= 900)) and
                np.ptp(tensions[i]) <= 0.1 * np.mean(tensions[i])):
                terminateds[i] = True
                rewards[i] = 20
        
        obs = self._get_observations(displacements, tensions)
        
        infos = [{'spoke turns': self.spoke_turns[i],
                  'raw state norm': self.last_state_norms[i],
                  'tensions delta': tensions[i] - self.init_tension}
                 for i in range(self.n_envs)]
        
        # Auto-reset done environments
        done_mask = terminateds | truncateds
        if np.any(done_mask):
            done_indices = np.where(done_mask)[0]
            for idx in done_indices:
                infos[idx]['final_obs'] = obs[idx].copy()
                infos[idx]['final_info'] = {
                    'spoke turns': self.spoke_turns[idx].copy(),
                    'raw state norm': self.last_state_norms[idx]
                }
            
            # Reset done environments
            self.episode_counters[done_mask] = 0
            self.spoke_turns[done_mask] = 0
            
            for idx in done_indices:
                n_random = min(self.random_spoke_n, self.n_spokes)
                random_indices = np.random.choice(self.n_spokes, size=n_random, replace=False)
                self.spoke_turns[idx, random_indices] = (
                    np.random.rand(n_random) * self.random_spoke_turns_max -
                    (self.random_spoke_turns_max / 2)
                )
            
            tensionchanges_reset = self.spoke_turns[done_mask] * self.adjustment_per_turn
            displacements_reset, tensions_reset = self._wheel_calc_batched(tensionchanges_reset)
            
            for j, idx in enumerate(done_indices):
                stacked = displacements_reset[j].reshape(-1, self.stacksize)
                self.last_state_norms[idx] = np.sqrt(np.trapz(np.sum(stacked**2, axis=1), self.theta))
                self.first_state_norms[idx] = self.last_state_norms[idx]
            
            obs_reset = self._get_observations(displacements_reset, tensions_reset)
            for j, idx in enumerate(done_indices):
                obs[idx] = obs_reset[j]
        
        return obs, rewards, terminateds, truncateds, infos
    
    def close(self):
        pass
    
    def __len__(self):
        return self.n_envs


# =============================================================================
# Helper function to create vectorized environments
# =============================================================================

def make_vec_env(
    n_envs: int = 8,
    use_subproc: bool = True,
    **env_kwargs
) -> SubprocVecEnv | BatchedWheelEnv | DummyVecEnv:

    if use_subproc:
        # Import here to avoid circular imports
        from wheel_env import WheelEnv
        
        def make_env(seed):
            def _init():
                env = WheelEnv(**env_kwargs)
                return env
            return _init
        
        env_fns = [make_env(i) for i in range(n_envs)]
        return SubprocVecEnv(env_fns)
    else:
        return BatchedWheelEnv(n_envs=n_envs, **env_kwargs)


# =============================================================================
# Example usage and testing
# =============================================================================

if __name__ == "__main__":
    # Test with DummyVecEnv first (no multiprocessing issues)
    print("Testing DummyVecEnv...")
    
    from wheel_env import WheelEnv
    
    def make_env(seed):
        def _init():
            return WheelEnv(
                reward_func='raw',
                action_space_selection='discrete',
                state_space_selection='rimpoints'
            )
        return _init
    
    n_envs = 4
    vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")
    
    obs, infos = vec_env.reset(seed=42)
    print(f"Reset obs shape: {obs.shape}")
    
    # Take a few steps
    for step in range(5):
        actions = np.array([vec_env.action_space.sample() for _ in range(n_envs)])
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        print(f"Step {step}: obs shape={obs.shape}, rewards={rewards}, dones={terminateds | truncateds}")
    
    vec_env.close()
    print("DummyVecEnv test passed!")
