"""
Todo:
    add dynamic state space from env (rainbow)
    add lstm support (rainbow) (ppo) should take the action with the last states
    add logging for tensions and turns 
"""
import os
import bz2
import pickle
from datetime import datetime
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import torch
from tqdm import trange, tqdm
import jax

from config import RainbowConfig, PPOConfig
from Agents import RainbowAgent, PPOAgent
from Memory import ReplayMemory

import gymnasium as gym

from omegaconf import OmegaConf

from pathlib import Path
import hydra

from pathlib import Path
from hydra.core.hydra_config import HydraConfig  # ✅ this import is required
import os

class BaseTrainer:
    def __init__(self, config, env, writer=None, output_dir=None):
        self.config = config
        self.env = env
        self.writer = writer

        # --- Hydra-safe output directory setup ---
        if output_dir is not None:
            # Use the provided output_dir (from experiment script)
            self.run_dir = Path(output_dir)
        else:
            try:
                # Works when running under a Hydra-managed script (e.g. train_rainbow.py with @hydra.main)
                self.run_dir = Path(HydraConfig.get().runtime.output_dir)
            except Exception:
                # Fallback for non-Hydra contexts (e.g. debugging manually)
                self.run_dir = Path("/tmp/rainbow_fallback")
                print(f"[BaseTrainer] Warning: HydraConfig not initialized, using {self.run_dir}")

        # --- Make sure directories exist ---
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.results_dir = self.run_dir / "results"

        for d in [self.checkpoint_dir, self.tensorboard_dir, self.results_dir]:
            d.mkdir(exist_ok=True)

        # Optional: print for debugging
        print(f"[BaseTrainer] Run directory: {self.run_dir}")
        print(f"[BaseTrainer] Checkpoints -> {self.checkpoint_dir}")
        print(f"[BaseTrainer] TensorBoard  -> {self.tensorboard_dir}")
        print(f"[BaseTrainer] Results      -> {self.results_dir}")

    def log(self, message):
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        print(f"[{timestamp}] {message}")
    
    def save_checkpoint(self, path, epoch):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def evaluate(self):
        raise NotImplementedError


class RainbowTrainer(BaseTrainer):
    """Trainer for Rainbow DQN"""
    
    def __init__(self, config, env, writer=None, output_dir=None):
        # Pass output_dir to parent class
        super().__init__(config, env, writer, output_dir=output_dir)
        
        # Rest of your initialization code...
        self.env = env
        
        # ========== ADD THIS SECTION (INFER STATE SHAPE) ==========
        # Infer state shape from environment
        if isinstance(env.observation_space, gym.spaces.Box):
            state_shape = env.observation_space.shape
        else:
            raise ValueError(f"Unsupported observation space: {type(env.observation_space)}")
        
        self.log(f"[RainbowTrainer] State shape: {state_shape}")
        # ========== END OF NEW SECTION ==========
        
        # Create agent
        from Agents import RainbowAgent
        self.agent = RainbowAgent(config, self.env)
        
        # Setup memory - NOW WITH state_shape PARAMETER
        from Memory import ReplayMemory
        if config.model_path and config.memory_path and os.path.exists(config.memory_path):
            self.memory = self._load_memory(config.memory_path)
            self.log("Loaded memory from checkpoint")
        else:
            # ========== FIX: Add state_shape parameter ==========
            self.memory = ReplayMemory(config, config.memory_capacity, state_shape)
            #                                                          ^^^^^^^^^^^^ ADD THIS
        
        # Validation memory - NOW WITH state_shape PARAMETER
        # ========== FIX: Pass state_shape to method ==========
        self.val_memory = self._create_validation_memory(state_shape)
        #                                                 ^^^^^^^^^^^^ ADD THIS
        
        # Training metrics (rest stays the same)
        self.metrics = {
            'steps': [],
            'rewards': [],
            'Qs': [],
            'best_avg_reward': -float('inf')
        }
        
        # Priority weight annealing
        self.priority_weight_increase = (
            (1 - config.priority_weight) / (config.T_max - config.learn_start)
        )

    
    def _create_validation_memory(self, state_shape):  # ← Add state_shape parameter
        """Create validation memory"""
        # ========== FIX: Pass state_shape to ReplayMemory ==========
        val_mem = ReplayMemory(self.config, self.config.evaluation_size, state_shape)
        #                                                                 ^^^^^^^^^^^^ ADD THIS
        T, done = 0, True
        
        while T < self.config.evaluation_size:
            if done:
                state, _ = self.env.reset()
            
            # Use proper action sampling (no more hardcoded values!)
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            val_mem.append(state, -1, 0.0, done)
            state = next_state
            done = terminated or truncated
            T += 1
        
        return val_mem
    
    def _load_memory(self, path):
        """Load memory from file"""
        with bz2.open(path, 'rb') as f:
            return pickle.load(f)
    
    def _save_memory(self, path):
        """Save memory to file"""
        with bz2.open(path, 'wb') as f:
            pickle.dump(self.memory, f)
    
    def train(self):
        """Main training loop with debug logging"""
        self.log(f"Starting Rainbow training for {self.config.T_max} steps")
        
        self.agent.train()
        done = False
        episode_reward = 0
        episode_counter = 0
        step_counter = 0
        glob_step = 0
        first_state_norm = -1000
        first_tensions = 10
        first_turns = 10

        try:
            state, _ = self.env.reset()
        except Exception as e:
            self.log(f"Error during env.reset(): {e}")
            return

        for T in trange(1, self.config.T_max + 1):
            try:
                # Episode management
                if done:
                    try:
                        if self.writer and state is not None:
                            # Only compute wheel_change if info exists
                            if 'raw state norm' in info:
                                current_norm = info['raw state norm']
                                current_tension = np.linalg.norm(info['spoke tensions']-800)
                                current_turns = np.sum(abs(info['spoke turns']))
                                wheel_change = -100 * (current_norm - first_state_norm) / max(abs(first_state_norm), 1e-15)
                                turn_change = -100 * (current_turns - first_turns) / max(abs(first_turns), 1e-15)
                                tension_change = -100 * (current_tension - first_tensions) / max(abs(first_tensions), 1e-15)
                                self.writer.add_scalar(f'episode/return', episode_reward, glob_step)
                                self.writer.add_scalar(f'episode/length', step_counter, glob_step)
                                self.writer.add_scalar(f'environment/wheel improvement', wheel_change, glob_step)
                                self.writer.add_scalar(f'environment/tension improvement', tension_change, glob_step)
                                self.writer.add_scalar(f'environment/turn improvement', turn_change, glob_step)
                            else:
                                self.log(f"Warning: 'raw state norm' missing in info at episode {episode_counter}")
                    except Exception as e:
                        self.log(f"Error during episode logging: {e}")

                    episode_counter += 1
                    episode_reward = 0
                    step_counter = 0
                    state, info = self.env.reset()
                    first_state_norm = info.get('raw state norm', -1000)
                    first_tensions = np.linalg.norm(info.get('spoke tensions', np.zeros(1))-800)
                    first_turns = np.sum(abs(info.get('spoke turns', np.zeros(1))))
                    done = False 

                # Reset noise
                if T % self.config.replay_frequency == 0:
                    self.agent.reset_noise()

                # Take action
                action = self.agent.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                if terminated or truncated:
                    done = True
                step_counter += 1
                glob_step += 1

                # Clip rewards if needed
                if self.config.reward_clip > 0:
                    reward = max(min(reward, self.config.reward_clip), -self.config.reward_clip)

                episode_reward += reward
                self.memory.append(state, action, reward, done)

                # Training
                if T >= self.config.learn_start:
                    # Anneal importance sampling weight
                    self.memory.priority_weight = min(
                        self.memory.priority_weight + self.priority_weight_increase, 1
                    )

                    # Learn
                    if T % self.config.replay_frequency == 0:
                        try:
                            self.agent.learn(self.memory)
                            # Optional: check for NaNs in network
                            if hasattr(self.agent, 'q_values') and torch.isnan(self.agent.q_values).any():
                                self.log(f"NaNs detected in Q-values at step {T}")
                                break
                        except Exception as e:
                            self.log(f"Error during agent.learn(): {e}")
                            break

                    # Evaluate
                    if self.config.evaluation_interval and T % self.config.evaluation_interval == 0:
                        try:
                            self._evaluate_agent(T)
                        except Exception as e:
                            self.log(f"Error during evaluation at step {T}: {e}")

                    # Update target network
                    if T % self.config.target_update == 0:
                        try:
                            self.agent.update_target_net()
                        except Exception as e:
                            self.log(f"Error during target net update at step {T}: {e}")

                    # Checkpoint
                    if self.config.checkpoint_interval and T % self.config.checkpoint_interval == 0:
                        try:
                            self.save_checkpoint(self.checkpoint_dir, T)
                        except Exception as e:
                            self.log(f"Error saving checkpoint at step {T}: {e}")

                state = next_state

            except Exception as e:
                self.log(f"Unexpected error at step {T}: {e}")
                self.log(f"Action: {action}")
                self.log(f"Reward: {reward}")
                break

        self.log("Training completed!")

    
    def _evaluate_agent(self, step):
        """Evaluate agent"""
        self.agent.eval()
        
        # Run evaluation episodes
        total_reward = 0
        total_q = 0
        
        for _ in range(self.config.evaluation_episodes):
            state, info = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.agent.act_e_greedy(state)
                state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                total_q += self.agent.evaluate_q(state)
                if terminated or truncated: 
                    done = True
            
            total_reward += episode_reward
        
        avg_reward = total_reward / self.config.evaluation_episodes
        avg_q = total_q / self.config.evaluation_episodes
        
        # Log metrics
        self.log(f'Step {step}/{self.config.T_max} | Avg Reward: {avg_reward:.2f} | Avg Q: {avg_q:.2f}')
        
        if self.writer:
            self.writer.add_scalar('Eval/avg_reward', avg_reward, step)
            self.writer.add_scalar('Eval/avg_q', avg_q, step)
        
        # Save best model
        if avg_reward > self.metrics['best_avg_reward']:
            self.metrics['best_avg_reward'] = avg_reward
            self.agent.save(str(self.results_dir), 'best_model.pth')
            self.log(f"New best model saved! Reward: {avg_reward:.2f}")
        
        self.agent.train()
        
        # Save memory if path provided
        if self.config.memory_path:
            self._save_memory(self.config.memory_path)
    
    def save_checkpoint(self, path, step):
        self.agent.save(path)
        self.log(f"Checkpoint saved locally at {path}")

    
    def evaluate(self):
        """Run evaluation only"""
        self.log("Running evaluation...")
        self.agent.eval()
        self._evaluate_agent(0)

class PPOTrainer(BaseTrainer):
    """Trainer for PPO"""
    
    def __init__(self, config: PPOConfig, env, writer=None):
        super().__init__(config, env, writer)
        
        # Setup checkpoint directory
        #Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)
        
        # Get environment dimensions
        obs_dim = env.observation_space.shape[0]
        action_dim = (env.action_space.shape[0] if config.continuous_action_space 
                     else env.action_space.n)
        
        # Create agent
        self.agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            continuous_action_space=config.continuous_action_space,
            num_epochs=config.num_epochs,
            eps_clip=config.eps_clip,
            action_std_init=config.action_std_init,
            gamma=config.gamma,
            entropy_coef=config.entropy_coef,
            value_loss_coef=config.value_loss_coef,
            batch_size=config.batch_size,
            max_grad_norm=config.max_grad_norm,
            device=config.device,
        )
    
    def train(self):
        """Main training loop"""
        self.log(f"Starting PPO training for {self.config.num_train_steps} steps")
        
        t_so_far = 0
        eps_so_far = 0
        running_eps_reward = 0
        running_eps_length = 0
        
        while t_so_far < self.config.num_train_steps:
            # Reset environment
            obs, info = self.env.reset(seed=self.config.random_seed)
            first_state_norm = info['raw state norm']
            #print("reset:",first_state_norm)
            eps_reward = 0
            eps_length = 0
            step_counter = 0
            
            # Collect trajectory
            for step in range(1, self.config.max_eps_steps + 1):
                # Select action
                action, logprob, value = self.agent.policy.select_action(obs)
                
                # Take step
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                if done or truncated:
                    done = True
                
                eps_reward += reward
                step_counter += 1
                t_so_far += 1
                eps_length += 1
                
                # Store transition
                self.agent.buffer.store_transition(obs, action, logprob, reward, done, value)
                
                # Update policy
                if t_so_far % self.config.update_interval == 0:
                    self.agent.update_policy()
                
                # Save checkpoint
                if t_so_far % self.config.save_interval == 0:
                    self.save_checkpoint(self.checkpoint_dir, t_so_far)
                
                obs = next_obs
                
                if done:
                    current_norm = info['raw state norm']
                    #print("first:",first_state_norm)
                    #print("last:", current_norm)
                    wheel_change = 100 * (current_norm - first_state_norm) / abs(first_state_norm)
                    #print("change:", wheel_change)
                    self.writer.add_scalar(f'episode/return', eps_reward, t_so_far)
                    self.writer.add_scalar(f'episode/length', step_counter, t_so_far)
                    self.writer.add_scalar(f'environment/wheel improvement', wheel_change, t_so_far)
                    
                    break
            
            running_eps_reward += eps_reward
            running_eps_length += eps_length
            eps_so_far += 1
            
            # Periodic logging
            if eps_so_far % 10 == 0:
                avg_reward = running_eps_reward / 10
                avg_length = running_eps_length / 10
                self.log(f"Episode {eps_so_far} | Steps: {t_so_far} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f}")
                running_eps_reward = 0
                running_eps_length = 0
        
        self.log("Training completed!")
    
    def save_checkpoint(self, path, step):
        """Save checkpoint"""
        ckpt_path = os.path.join(path, f"{self.config.env_name}_step_{step}.pt")
        torch.save(self.agent.policy.state_dict(), ckpt_path)
        self.log(f"Checkpoint saved at step {step}")
    
    def evaluate(self):
        """Run evaluation"""
        self.log("Running PPO evaluation...")
        
        # Load checkpoint
        ckpt_path = os.path.join(self.config.ckpt_dir, self.config.eval_ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location=self.config.device)
        self.agent.policy.load_state_dict(ckpt)
        self.agent.policy.eval()
        
        metrics = {
            'eps_rewards': [],
            'eps_lengths': [],
            'min_reward': float('inf'),
            'max_reward': float('-inf'),
        }
        
        recent_rewards = deque(maxlen=10)
        
        with torch.no_grad():
            for ep in range(self.config.num_eval_eps):
                obs, _ = self.env.reset()
                eps_reward = 0
                eps_length = 0
                
                for _ in range(self.config.max_eps_steps):
                    if self.config.render_mode:
                        self.env.render()
                    
                    action, _, _ = self.agent.policy.select_action(obs)
                    obs, reward, done, _, _ = self.env.step(action)
                    
                    eps_reward += reward
                    eps_length += 1
                    
                    if done:
                        break
                
                metrics['eps_rewards'].append(eps_reward)
                metrics['eps_lengths'].append(eps_length)
                metrics['min_reward'] = min(metrics['min_reward'], eps_reward)
                metrics['max_reward'] = max(metrics['max_reward'], eps_reward)
                recent_rewards.append(eps_reward)
                
                self.log(f"Episode {ep+1}/{self.config.num_eval_eps} | Reward: {eps_reward:.2f} | Length: {eps_length}")
        
        # Summary statistics
        metrics['mean_reward'] = np.mean(metrics['eps_rewards'])
        metrics['std_reward'] = np.std(metrics['eps_rewards'])
        metrics['mean_eps_length'] = np.mean(metrics['eps_lengths'])
        
        self.log("\n" + "="*50)
        self.log("Evaluation Results:")
        self.log(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        self.log(f"Min/Max Reward: {metrics['min_reward']:.2f} / {metrics['max_reward']:.2f}")
        self.log(f"Mean Episode Length: {metrics['mean_eps_length']:.1f}")
        self.log("="*50)
        
        return metrics

