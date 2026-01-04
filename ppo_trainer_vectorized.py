"""
PPO Trainer with Vectorized Environment Support

This trainer is designed to work with vectorized environments for better GPU utilization.
Key changes from the original:
1. Collects rollouts from multiple environments in parallel
2. Better batching for GPU operations
3. Proper handling of episode boundaries in vectorized setting
"""

import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import deque
from datetime import datetime
from tqdm import trange
from typing import Optional, Tuple

# Assuming these imports work in your setup
from Agents import PPOAgent
from wheel_env_vectorized import make_vec_env, SubprocVecEnv, DummyVecEnv


class VectorizedRolloutBuffer:
    """
    Rollout buffer for vectorized environments.
    Stores experiences from multiple environments simultaneously.
    """
    
    def __init__(self, buffer_size: int, n_envs: int, obs_dim: int, action_dim: int, device: str = 'cpu'):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Pre-allocate buffers
        self.observations = np.zeros((buffer_size, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        self.pos = 0
        self.full = False
    
    def add(self, obs, action, reward, done, value, log_prob):
        """Add a transition from all environments."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action if len(action.shape) > 1 else action.reshape(-1, 1)
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    def get(self, last_values: np.ndarray, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Get all data and compute returns/advantages using GAE.
        
        Args:
            last_values: Value estimates for the last observation (n_envs,)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        
        Returns:
            Dictionary of tensors ready for PPO update
        """
        # Compute advantages using GAE
        advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        last_gae_lam = 0
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            
            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + self.values[:self.buffer_size]
        
        # Flatten: (buffer_size, n_envs, ...) -> (buffer_size * n_envs, ...)
        return {
            'observations': torch.tensor(self.observations[:self.buffer_size].reshape(-1, self.obs_dim), 
                                        dtype=torch.float32, device=self.device),
            'actions': torch.tensor(self.actions[:self.buffer_size].reshape(-1, self.action_dim),
                                   dtype=torch.float32, device=self.device),
            'old_log_probs': torch.tensor(self.log_probs[:self.buffer_size].reshape(-1),
                                         dtype=torch.float32, device=self.device),
            'advantages': torch.tensor(advantages.reshape(-1),
                                      dtype=torch.float32, device=self.device),
            'returns': torch.tensor(returns.reshape(-1),
                                   dtype=torch.float32, device=self.device),
        }
    
    def reset(self):
        """Reset the buffer."""
        self.pos = 0
        self.full = False


class VectorizedPPOTrainer:
    """
    PPO Trainer optimized for vectorized environments.
    
    Key improvements:
    1. Parallel rollout collection from multiple environments
    2. Larger effective batch sizes for better GPU utilization
    3. GAE (Generalized Advantage Estimation) for better variance reduction
    """
    
    def __init__(
        self,
        config,
        vec_env,
        writer=None,
        output_dir: Optional[str] = None
    ):
        self.config = config
        self.vec_env = vec_env
        self.writer = writer
        self.n_envs = len(vec_env)
        
        # Setup directories
        if output_dir is None:
            output_dir = f"outputs/ppo_vec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = Path(output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Get observation and action dimensions
        obs_space = vec_env.single_observation_space
        act_space = vec_env.single_action_space
        
        self.obs_dim = np.prod(obs_space.shape)
        
        # Determine action space type from config
        action_space_type = getattr(config.env, 'action_space_selection', 'discrete')
        n_spokes = getattr(config.env, 'n_spokes', 36)
        
        # Set action_dim based on action space type
        if action_space_type == 'discrete':
            self.action_dim = 1
            self.continuous = False
        elif action_space_type in ['continous', 'hybrid']:
            # Hybrid: [spoke_idx, delta] = 2 dimensions
            self.action_dim = 2
            self.continuous = True  # treated as continuous for buffer storage
        elif action_space_type == 'all_spokes':
            self.action_dim = n_spokes
            self.continuous = True
        else:
            # Fallback to gym space detection
            if hasattr(act_space, 'n'):
                self.action_dim = 1
                self.continuous = False
            else:
                self.action_dim = np.prod(act_space.shape)
                self.continuous = True
        
        print(f"[VecPPOTrainer] Observation dim: {self.obs_dim}")
        print(f"[VecPPOTrainer] Action dim: {self.action_dim}")
        print(f"[VecPPOTrainer] Action space type: {action_space_type}")
        print(f"[VecPPOTrainer] Num envs: {self.n_envs}")
        
        # Use factory to create appropriate agent
        from ppo_agent_factory import create_ppo_agent
        
        self.agent = create_ppo_agent(
            obs_dim=self.obs_dim,
            action_space=act_space,
            action_space_type=action_space_type,
            n_spokes=n_spokes,
            hidden_dim=config.hidden_dim,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            gamma=config.gamma,
            num_epochs=config.num_epochs,
            eps_clip=config.eps_clip,
            action_std_init=getattr(config, 'action_std_init', 0.1),
            entropy_coef=config.entropy_coef,
            value_loss_coef=config.value_loss_coef,
            batch_size=config.batch_size,
            max_grad_norm=config.max_grad_norm,
            device=config.device,
            use_recurrent=getattr(config, 'use_recurrent', False),
            recurrent_type=getattr(config, 'recurrent_type', 'lstm'),
            recurrent_hidden_dim=getattr(config, 'recurrent_hidden_dim', 128),
            recurrent_layers=getattr(config, 'recurrent_layers', 1),
            recurrent_sequence_length=getattr(config, 'recurrent_sequence_length', 16),
            recurrent_dropout=getattr(config, 'recurrent_dropout', 0.0),
        )
        
        # Rollout buffer
        self.rollout_length = config.update_interval // self.n_envs
        print(f"[VecPPOTrainer] Rollout length per env: {self.rollout_length}")
        print(f"[VecPPOTrainer] Total samples per update: {self.rollout_length * self.n_envs}")
        
        # Persistent episode counters (must survive across rollout collections)
        self.current_episode_rewards = np.zeros(self.n_envs)
        self.current_episode_lengths = np.zeros(self.n_envs)
        
        self.buffer = VectorizedRolloutBuffer(
            buffer_size=self.rollout_length,
            n_envs=self.n_envs,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=config.device
        )
        
        # GAE parameters
        self.gamma = config.gamma
        self.gae_lambda = getattr(config, 'gae_lambda', 0.95)
    
    def log(self, message: str):
        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        print(f"[{timestamp}] {message}")
    
    def collect_rollouts(self, obs: np.ndarray, first_state_norms: np.ndarray, 
                         first_tensions: np.ndarray, first_turns: np.ndarray,
                         total_steps: int) -> Tuple[np.ndarray, dict, int]:
        """
        Collect rollouts from vectorized environment.
        
        Args:
            obs: Current observations (n_envs, obs_dim)
            first_state_norms: Initial state norms for each env
            first_tensions: Initial tensions for each env
            first_turns: Initial turns for each env
            total_steps: Current total step count for logging
        
        Returns:
            next_obs: Observations after rollout
            metrics: Dictionary of collected metrics
            total_steps: Updated step count
        """
        episode_rewards = []
        episode_lengths = []
        
        for step in range(self.rollout_length):
            # Get actions from policy
            with torch.no_grad():
                actions = []
                log_probs = []
                values = []
                
                for i in range(self.n_envs):
                    action, log_prob, value = self.agent.select_action(obs[i])
                    actions.append(action)
                    log_probs.append(log_prob)
                    values.append(value)
                
                actions = np.array(actions)
                log_probs = np.array(log_probs).flatten()
                values = np.array(values).flatten()
            
            # Step environments
            next_obs, rewards, terminateds, truncateds, infos = self.vec_env.step(actions)
            dones = terminateds | truncateds
            
            # Track episode statistics
            self.current_episode_rewards += rewards
            self.current_episode_lengths += 1
            
            for i, done in enumerate(dones):
                if done:
                    total_steps_for_logging = total_steps + step * self.n_envs + i
                    
                    episode_rewards.append(self.current_episode_rewards[i])
                    episode_lengths.append(self.current_episode_lengths[i])
                    
                    # Get final info (may be in 'final_info' due to auto-reset)
                    info = infos[i].get('final_info', infos[i])
                    
                    # Log all wheel-specific metrics for this episode
                    if self.writer and 'raw state norm' in info:
                        current_norm = info['raw state norm']
                        current_tensions_max = np.max(np.abs(info['tensions delta']))
                        current_turns_max = np.max(np.abs(info['spoke turns']))
                        current_tension = np.sum(np.abs(info['tensions delta']))
                        current_turns = np.sum(np.abs(info['spoke turns']))
                        max_displacement = info.get('max disp', 0)
                        terminated = info.get('terminated', False)
                        
                        # Calculate improvements
                        wheel_change = 100 * (first_state_norms[i] - current_norm) / max(abs(first_state_norms[i]), 1e-15)
                        turn_change = 100 * (first_turns[i] - current_turns) / max(abs(first_turns[i]), 1e-15)
                        tension_change = 100 * (first_tensions[i] - current_tension) / max(abs(first_tensions[i]), 1e-15)
                        
                        # Log everything
                        self.writer.add_scalar('episode/return', self.current_episode_rewards[i], total_steps_for_logging)
                        self.writer.add_scalar('episode/length', self.current_episode_lengths[i], total_steps_for_logging)
                        self.writer.add_scalar('environment/wheel improvement', wheel_change, total_steps_for_logging)
                        self.writer.add_scalar('environment/turn improvement', turn_change, total_steps_for_logging)
                        self.writer.add_scalar('environment/tension improvement', tension_change, total_steps_for_logging)
                        self.writer.add_scalar('environment/final state norm', current_norm, total_steps_for_logging)
                        self.writer.add_scalar('environment/final tension deltas max', current_tensions_max, total_steps_for_logging)
                        self.writer.add_scalar('environment/final tension deltas sum', current_tension, total_steps_for_logging)
                        self.writer.add_scalar('environment/final turns max', current_turns_max, total_steps_for_logging)
                        self.writer.add_scalar('environment/final turns sum', current_turns, total_steps_for_logging)
                        self.writer.add_scalar('environment/wheel max', max_displacement, total_steps_for_logging)
                        self.writer.add_scalar('episode/terminated', 1 if terminated else 0, total_steps_for_logging)
                    
                    # Update initial state for this env from reset info
                    reset_info = infos[i]  # After auto-reset, this has the new episode's info
                    if 'raw state norm' in reset_info:
                        first_state_norms[i] = reset_info['raw state norm']
                    if 'tensions delta' in reset_info:
                        first_tensions[i] = np.sum(np.abs(reset_info['tensions delta']))
                    if 'spoke turns' in reset_info:
                        first_turns[i] = np.sum(np.abs(reset_info['spoke turns']))
                    
                    # Reset counters for this env
                    self.current_episode_rewards[i] = 0
                    self.current_episode_lengths[i] = 0
            
            # Store transition
            self.buffer.add(obs, actions, rewards, dones, values, log_probs)
            obs = next_obs
        
        metrics = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
        }
        
        return obs, metrics, first_state_norms, first_tensions, first_turns
    
    def update_policy(self, last_obs: np.ndarray):
        """
        Update policy using collected rollouts.
        
        Args:
            last_obs: Final observations for computing last values
        """
        # Compute last values for GAE
        with torch.no_grad():
            last_values = []
            for i in range(self.n_envs):
                _, _, value = self.agent.select_action(last_obs[i])
                last_values.append(value)
            last_values = np.array(last_values)
        
        # Get data from buffer
        data = self.buffer.get(last_values, self.gamma, self.gae_lambda)
        
        # Normalize advantages
        advantages = data['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        data['advantages'] = advantages
        
        # PPO update
        total_samples = len(data['observations'])
        indices = np.arange(total_samples)
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(self.config.num_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, total_samples, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = data['observations'][batch_indices]
                batch_actions = data['actions'][batch_indices]
                batch_old_log_probs = data['old_log_probs'][batch_indices]
                batch_advantages = data['advantages'][batch_indices]
                batch_returns = data['returns'][batch_indices]
                
                # Evaluate actions
                values, log_probs, entropy = self.agent.policy.evaluate_actions(
                    batch_obs, batch_actions.squeeze(-1) if self.action_dim == 1 else batch_actions
                )
                
                # Policy loss (clipped surrogate)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * ((values.squeeze() - batch_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss + 
                    self.config.value_loss_coef * value_loss + 
                    self.config.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.agent.optimizer.zero_grad()
                loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), self.config.max_grad_norm)
                self.agent.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Reset buffer
        self.buffer.reset()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
        }
    
    def train(self):
        """Main training loop with detailed TensorBoard logging."""
        self.log(f"Starting vectorized PPO training")
        self.log(f"  Total steps: {self.config.num_train_steps}")
        self.log(f"  Num envs: {self.n_envs}")
        self.log(f"  Steps per update: {self.rollout_length * self.n_envs}")
        
        # Reset environments and get initial info
        obs, initial_infos = self.vec_env.reset(seed=self.config.random_seed)
        
        # Track initial state norms for each env
        first_state_norms = np.zeros(self.n_envs)
        first_tensions = np.zeros(self.n_envs)
        first_turns = np.zeros(self.n_envs)
        
        for i, info in enumerate(initial_infos):
            if 'raw state norm' in info:
                first_state_norms[i] = info['raw state norm']
            if 'tensions delta' in info:
                first_tensions[i] = np.sum(np.abs(info['tensions delta']))
            if 'spoke turns' in info:
                first_turns[i] = np.sum(np.abs(info['spoke turns']))
            
            # Log initial state for each env
            if self.writer:
                self.writer.add_scalar('environment/initial state norm', first_state_norms[i], i)
                self.writer.add_scalar('environment/initial tension deltas sum', first_tensions[i], i)
                if 'tensions delta' in info:
                    self.writer.add_scalar('environment/initial tension deltas max', np.max(np.abs(info['tensions delta'])), i)
                self.writer.add_scalar('environment/initial turns sum', first_turns[i], i)
                if 'spoke turns' in info:
                    self.writer.add_scalar('environment/initial turns max', np.max(np.abs(info['spoke turns'])), i)
        
        # Reset agent episode state
        self.agent.reset_episode()
        
        total_steps = 0
        num_updates = 0
        episode_rewards_buffer = deque(maxlen=100)
        episode_lengths_buffer = deque(maxlen=100)
        
        pbar = trange(0, self.config.num_train_steps, self.rollout_length * self.n_envs)
        
        for _ in pbar:
            # Collect rollouts (logging happens inside)
            obs, metrics, first_state_norms, first_tensions, first_turns = self.collect_rollouts(
                obs, first_state_norms, first_tensions, first_turns, total_steps
            )
            total_steps += self.rollout_length * self.n_envs
            
            # Track episode stats for progress bar
            episode_rewards_buffer.extend(metrics['episode_rewards'])
            episode_lengths_buffer.extend(metrics['episode_lengths'])
            
            # Update policy
            update_metrics = self.update_policy(obs)
            num_updates += 1
            
            # Progress bar update
            if len(episode_rewards_buffer) > 0:
                mean_reward = np.mean(episode_rewards_buffer)
                mean_length = np.mean(episode_lengths_buffer)
                
                pbar.set_postfix({
                    'reward': f'{mean_reward:.2f}',
                    'length': f'{mean_length:.1f}',
                    'p_loss': f'{update_metrics["policy_loss"]:.4f}'
                })
                
                # Log training metrics
                if self.writer:
                    self.writer.add_scalar('train/policy_loss', update_metrics['policy_loss'], total_steps)
                    self.writer.add_scalar('train/value_loss', update_metrics['value_loss'], total_steps)
                    self.writer.add_scalar('train/entropy', -update_metrics['entropy_loss'], total_steps)
            
            # Save checkpoint
            if total_steps % self.config.save_interval == 0:
                self.save_checkpoint(total_steps)
        
        self.log("Training completed!")
        self.save_checkpoint(total_steps, final=True)
    
    def save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint."""
        suffix = "final" if final else f"step_{step}"
        path = self.checkpoint_dir / f"ppo_{suffix}.pt"
        self.agent.save(str(path))
        self.log(f"Saved checkpoint: {path}")



