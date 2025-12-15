# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
import gymnasium as gym
from collections import deque

from Models import DQN, RecurrentDQN

"""
Todo:
    add lstm option to rainbow and ppo 
"""



class RainbowAgent:
    """Rainbow DQN agent with dynamic state and action space inference and optional recurrent support"""
    
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.device = config.device
        
        # ========== INFER ACTION SPACE ==========
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_space = env.action_space.n
            self.action_type = 'discrete'
        elif isinstance(env.action_space, gym.spaces.Box):
            self.action_space = np.prod(env.action_space.shape)
            self.action_type = 'continuous'
            print(f"[RainbowAgent] Warning: Box action space detected. "
                  f"Using product of shape: {self.action_space}")
        else:
            raise ValueError(f"Unsupported action space type: {type(env.action_space)}")
        
        # ========== INFER STATE DIMENSION ==========
        if isinstance(env.observation_space, gym.spaces.Box):
            self.state_shape = env.observation_space.shape
            self.state_dim = np.prod(self.state_shape)
        else:
            raise ValueError(f"Unsupported observation space type: {type(env.observation_space)}")
        
        print(f"[RainbowAgent] Inferred action_space: {self.action_space} (type: {self.action_type})")
        print(f"[RainbowAgent] Inferred state_dim: {self.state_dim}, state_shape: {self.state_shape}")
        
        # ========== RECURRENT NETWORK SUPPORT ==========
        self.use_recurrent = getattr(config, 'use_recurrent', False)
        if self.use_recurrent:
            self.recurrent_seq_len = config.recurrent_sequence_length
            print(f"[RainbowAgent] Using recurrent network with sequence length: {self.recurrent_seq_len}")
            
            # Initialize action history buffer
            self.action_history = deque(maxlen=self.recurrent_seq_len)
            self.state_history = deque(maxlen=self.recurrent_seq_len)
        
        # ========== CREATE NETWORKS ==========
        if self.use_recurrent:
            self.online_net = RecurrentDQN(config, self.action_space, self.state_dim).to(self.device)
            self.target_net = RecurrentDQN(config, self.action_space, self.state_dim).to(self.device)
        else:
            self.online_net = DQN(config, self.action_space, self.state_dim).to(self.device)
            self.target_net = DQN(config, self.action_space, self.state_dim).to(self.device)
        
        # Initialize target network with same weights
        self.update_target_net()
        
        # Set target network to eval mode
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=config.learning_rate,
            eps=config.adam_eps
        )
        
        # Number of quantiles for training and evaluation
        self.num_quantiles_train = config.num_quantiles
        self.num_quantiles_eval = config.num_quantiles_eval if hasattr(config, 'num_quantiles_eval') else 32
        
        # Huber loss parameter
        self.kappa = config.kappa if hasattr(config, 'kappa') else 1.0
        
    def reset_noise(self):
        """Reset noise in noisy layers"""
        self.online_net.reset_noise()
    
    def reset_episode(self):
        """Reset episode-specific state (e.g., action history for recurrent networks)"""
        if self.use_recurrent:
            self.action_history.clear()
            self.state_history.clear()
            self.online_net.reset_hidden_state()
    
    def _construct_input_sequence(self, state):
        """Construct input sequence for recurrent network from current state and history"""
        # Add current state to history
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(self.device)
        else:
            state_tensor = state.to(self.device)
        
        # Build state sequence (pad with zeros if necessary)
        state_seq = []
        action_seq = []
        
        # Add historical states and actions
        for s, a in zip(self.state_history, self.action_history):
            state_seq.append(s)
            action_seq.append(a)
        
        # Add current state
        state_seq.append(state_tensor)
        
        # Pad if we don't have enough history yet
        while len(state_seq) < self.recurrent_seq_len:
            state_seq.insert(0, torch.zeros_like(state_tensor))
            action_seq.insert(0, 0)  # Pad with action 0
        
        # Convert to tensors
        # Stack states: (seq_len, *state_shape)
        state_seq_tensor = torch.stack(state_seq[-self.recurrent_seq_len:])
        
        # One-hot encode actions: (seq_len, action_space)
        action_seq_tensor = torch.zeros(self.recurrent_seq_len, self.action_space, device=self.device)
        for i, a in enumerate(action_seq[-self.recurrent_seq_len:]):
            action_seq_tensor[i, a] = 1.0
        
        # Flatten state if necessary and concatenate with one-hot actions
        # state_seq_tensor: (seq_len, *state_shape) -> (seq_len, state_dim)
        state_seq_flat = state_seq_tensor.reshape(self.recurrent_seq_len, -1)
        
        # Concatenate state and action: (seq_len, state_dim + action_space)
        input_seq = torch.cat([state_seq_flat, action_seq_tensor], dim=1)
        
        # Add batch dimension: (1, seq_len, state_dim + action_space)
        input_seq = input_seq.unsqueeze(0)
        
        return input_seq
    
    def act(self, state):
        """Select action using the online network (for training with noise)"""
        with torch.no_grad():
            if self.use_recurrent:
                # Construct input sequence
                input_seq = self._construct_input_sequence(state)
                
                # Get Q-values from recurrent network
                q_values, _ = self.online_net(input_seq, self.num_quantiles_eval)
                q_values = q_values.mean(dim=2)  # Average over quantiles
                
                # Select action with highest Q-value
                action = q_values.argmax(dim=1).item()
                
                # Update history
                if isinstance(state, np.ndarray):
                    state_tensor = torch.FloatTensor(state).to(self.device)
                else:
                    state_tensor = state.to(self.device)
                
                self.state_history.append(state_tensor)
                self.action_history.append(action)
                
            else:
                # Standard DQN behavior
                # Convert state to tensor
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state).to(self.device)
                
                # Add batch dimension if needed
                if state.dim() == len(self.state_shape):
                    state = state.unsqueeze(0)
                
                # Get Q-values (averaged over quantiles)
                q_values, _ = self.online_net(state, self.num_quantiles_eval)
                q_values = q_values.mean(dim=2)  # Average over quantiles
                
                # Select action with highest Q-value
                action = q_values.argmax(dim=1).item()
            
            # Store for potential debugging
            self.q_values = q_values
            
            return action
    
    def act_e_greedy(self, state, epsilon=0.001):
        """Select action using epsilon-greedy policy (for evaluation)"""
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return self.act(state)
    
    def learn(self, memory):
        """Perform one step of learning"""
        # Sample batch from memory
        if self.use_recurrent:
            (idxs, states_seq, actions_seq, masks, actions, returns, 
             next_states_seq, next_actions_seq, next_masks, nonterminals, weights) = memory.sample(
                self.config.batch_size
            )
            return self._learn_recurrent(
                memory, idxs, states_seq, actions_seq, masks, actions, returns,
                next_states_seq, next_actions_seq, next_masks, nonterminals, weights
            )
        else:
            idxs, states, actions, returns, next_states, nonterminals, weights = memory.sample(
                self.config.batch_size
            )
            return self._learn_standard(memory, idxs, states, actions, returns, next_states, nonterminals, weights)
    
    def _learn_standard(self, memory, idxs, states, actions, returns, next_states, nonterminals, weights):
        """Standard (non-recurrent) learning step"""
        # Reshape states for network input
        batch_size = states.shape[0]
        # states shape: (batch_size, history_length, *state_shape)
        # Flatten to (batch_size, history_length * state_dim)
        states = states.reshape(batch_size, -1)
        next_states = next_states.reshape(batch_size, -1)
        
        # Get current Q-value distributions
        current_quantiles, taus = self.online_net(states, self.num_quantiles_train)
        # current_quantiles: (batch_size, action_space, num_quantiles)
        
        # Select the quantiles for the taken actions
        # actions: (batch_size,) -> (batch_size, 1, 1)
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).expand(
            batch_size, 1, self.num_quantiles_train
        )
        current_quantiles = current_quantiles.gather(1, actions_expanded).squeeze(1)
        # current_quantiles: (batch_size, num_quantiles)
        
        with torch.no_grad():
            # Double DQN: Use online network to select actions
            next_q_values, _ = self.online_net(next_states, self.num_quantiles_eval)
            next_q_values = next_q_values.mean(dim=2)  # Average over quantiles
            next_actions = next_q_values.argmax(dim=1)
            
            # Use target network to evaluate the selected actions
            next_quantiles, _ = self.target_net(next_states, self.num_quantiles_train)
            next_actions_expanded = next_actions.unsqueeze(1).unsqueeze(2).expand(
                batch_size, 1, self.num_quantiles_train
            )
            next_quantiles = next_quantiles.gather(1, next_actions_expanded).squeeze(1)
            # next_quantiles: (batch_size, num_quantiles)
            
            # Compute target quantiles
            # returns: (batch_size,) -> (batch_size, 1)
            # nonterminals: (batch_size, 1)
            target_quantiles = returns.unsqueeze(1) + nonterminals * self.config.discount * next_quantiles
        
        # Compute quantile Huber loss
        # Expand dimensions for broadcasting
        # current_quantiles: (batch_size, num_quantiles, 1)
        # target_quantiles: (batch_size, 1, num_quantiles)
        current_quantiles_exp = current_quantiles.unsqueeze(2)
        target_quantiles_exp = target_quantiles.unsqueeze(1)
        
        # TD errors
        td_errors = target_quantiles_exp - current_quantiles_exp
        # td_errors: (batch_size, num_quantiles, num_quantiles)
        
        # Huber loss
        huber_loss = torch.where(
            td_errors.abs() <= self.kappa,
            0.5 * td_errors.pow(2),
            self.kappa * (td_errors.abs() - 0.5 * self.kappa)
        )
        
        # Quantile weights
        taus_exp = taus.unsqueeze(2)  # (batch_size, num_quantiles, 1)
        quantile_weights = torch.abs(taus_exp - (td_errors < 0).float())
        
        # Quantile Huber loss
        quantile_huber_loss = quantile_weights * huber_loss
        loss = quantile_huber_loss.sum(dim=2).mean(dim=1)  # (batch_size,)
        
        # Apply importance sampling weights
        loss = (weights * loss).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        
        # Update priorities in memory
        priorities = quantile_huber_loss.sum(dim=2).mean(dim=1).detach().cpu().numpy()
        memory.update_priorities(idxs, priorities)
        
        return loss.item()
    
    def _learn_recurrent(self, memory, idxs, states_seq, actions_seq, masks, actions, returns,
                         next_states_seq, next_actions_seq, next_masks, nonterminals, weights):
        """Recurrent network learning step"""
        batch_size = states_seq.shape[0]
        seq_len = states_seq.shape[1]
        
        # Prepare input sequences: concatenate states and one-hot actions
        # states_seq: (batch, seq_len, *state_shape)
        # actions_seq: (batch, seq_len)
        
        # Flatten states: (batch, seq_len, state_dim)
        states_seq_flat = states_seq.reshape(batch_size, seq_len, -1)
        next_states_seq_flat = next_states_seq.reshape(batch_size, seq_len, -1)
        
        # One-hot encode actions: (batch, seq_len, action_space)
        actions_one_hot = torch.zeros(batch_size, seq_len, self.action_space, device=self.device)
        actions_one_hot.scatter_(2, actions_seq.unsqueeze(2), 1.0)
        
        next_actions_one_hot = torch.zeros(batch_size, seq_len, self.action_space, device=self.device)
        next_actions_one_hot.scatter_(2, next_actions_seq.unsqueeze(2), 1.0)
        
        # Concatenate: (batch, seq_len, state_dim + action_space)
        input_seq = torch.cat([states_seq_flat, actions_one_hot], dim=2)
        next_input_seq = torch.cat([next_states_seq_flat, next_actions_one_hot], dim=2)
        
        # Get current Q-value distributions from recurrent network
        current_quantiles, taus = self.online_net(input_seq, self.num_quantiles_train, mask=masks)
        # current_quantiles: (batch_size, action_space, num_quantiles)
        
        # Select the quantiles for the taken actions
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).expand(
            batch_size, 1, self.num_quantiles_train
        )
        current_quantiles = current_quantiles.gather(1, actions_expanded).squeeze(1)
        # current_quantiles: (batch_size, num_quantiles)
        
        with torch.no_grad():
            # Double DQN: Use online network to select actions
            next_q_values, _ = self.online_net(next_input_seq, self.num_quantiles_eval, mask=next_masks)
            next_q_values = next_q_values.mean(dim=2)  # Average over quantiles
            next_actions = next_q_values.argmax(dim=1)
            
            # Use target network to evaluate the selected actions
            next_quantiles, _ = self.target_net(next_input_seq, self.num_quantiles_train, mask=next_masks)
            next_actions_expanded = next_actions.unsqueeze(1).unsqueeze(2).expand(
                batch_size, 1, self.num_quantiles_train
            )
            next_quantiles = next_quantiles.gather(1, next_actions_expanded).squeeze(1)
            # next_quantiles: (batch_size, num_quantiles)
            
            # Compute target quantiles
            target_quantiles = returns.unsqueeze(1) + nonterminals * self.config.discount * next_quantiles
        
        # Compute quantile Huber loss
        current_quantiles_exp = current_quantiles.unsqueeze(2)
        target_quantiles_exp = target_quantiles.unsqueeze(1)
        
        # TD errors
        td_errors = target_quantiles_exp - current_quantiles_exp
        
        # Huber loss
        huber_loss = torch.where(
            td_errors.abs() <= self.kappa,
            0.5 * td_errors.pow(2),
            self.kappa * (td_errors.abs() - 0.5 * self.kappa)
        )
        
        # Quantile weights
        taus_exp = taus.unsqueeze(2)
        quantile_weights = torch.abs(taus_exp - (td_errors < 0).float())
        
        # Quantile Huber loss
        quantile_huber_loss = quantile_weights * huber_loss
        loss = quantile_huber_loss.sum(dim=2).mean(dim=1)  # (batch_size,)
        
        # Apply importance sampling weights
        loss = (weights * loss).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        if hasattr(self.config, 'grad_clip') and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        
        # Update priorities in memory
        priorities = quantile_huber_loss.sum(dim=2).mean(dim=1).detach().cpu().numpy()
        memory.update_priorities(idxs, priorities)
        
        return loss.item()
    
    def update_target_net(self):
        """Update target network with online network weights"""
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    def train(self):
        """Set network to training mode"""
        self.online_net.train()
    
    def eval(self):
        """Set network to evaluation mode"""
        self.online_net.eval()
    
    def save(self, path):
        """Save model checkpoint with recurrent state"""
        checkpoint = {
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': {
                'use_recurrent': self.use_recurrent,
                'action_space': self.action_space,
                'state_dim': self.state_dim,
                'state_shape': self.state_shape,
            }
        }
    
        # Save recurrent-specific state if applicable
        if self.use_recurrent:
            checkpoint['recurrent_state'] = {
                'action_history': list(self.action_history),
                'state_history': [s.cpu() for s in self.state_history],
                'seq_len': self.recurrent_seq_len
            }
        
        torch.save(checkpoint, path)
        print(f"[RainbowAgent] Checkpoint saved to {path}")
    
    def load(self, path, load_optimizer=True, load_recurrent_state=False):
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state (set False for evaluation)
            load_recurrent_state: Whether to restore action/state history (usually False)
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        
        if load_optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Optionally restore recurrent state (usually not needed)
        if load_recurrent_state and self.use_recurrent and 'recurrent_state' in checkpoint:
            recurrent_state = checkpoint['recurrent_state']
            self.action_history = deque(recurrent_state['action_history'], 
                                        maxlen=self.recurrent_seq_len)
            self.state_history = deque([s.to(self.device) for s in recurrent_state['state_history']], 
                                    maxlen=self.recurrent_seq_len)
        
        print(f"[RainbowAgent] Checkpoint loaded from {path}")


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.state_values = []

    def add(self, state, action, logprob, reward, done, state_value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.state_values.append(state_value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.state_values.clear()


class ActorCritic(nn.Module):
    """
    Actor-Critic network with optional recurrent support
    """
    def __init__(
        self, 
        obs_dim, 
        action_dim, 
        hidden_dim, 
        continuous_action_space=False, 
        action_std_init=0.6,
        device='cpu',
        # NEW: Recurrent parameters
        use_recurrent=False,
        recurrent_type='lstm',
        recurrent_hidden_dim=128,
        recurrent_layers=1,
        recurrent_dropout=0.0
    ):
        super(ActorCritic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space
        self.device = device
        
        # NEW: Recurrent parameters
        self.use_recurrent = use_recurrent
        self.recurrent_type = recurrent_type
        self.recurrent_hidden_dim = recurrent_hidden_dim
        self.recurrent_layers = recurrent_layers

        if use_recurrent:
            print(f"[ActorCritic] Creating RECURRENT network:")
            print(f"      - Recurrent type: {recurrent_type}")
            print(f"      - Recurrent hidden dim: {recurrent_hidden_dim}")
            print(f"      - Recurrent layers: {recurrent_layers}")
            
            # Create recurrent layer
            if recurrent_type == 'lstm':
                self.recurrent = nn.LSTM(
                    input_size=obs_dim,
                    hidden_size=recurrent_hidden_dim,
                    num_layers=recurrent_layers,
                    batch_first=True,
                    dropout=recurrent_dropout if recurrent_layers > 1 else 0.0
                )
            elif recurrent_type == 'gru':
                self.recurrent = nn.GRU(
                    input_size=obs_dim,
                    hidden_size=recurrent_hidden_dim,
                    num_layers=recurrent_layers,
                    batch_first=True,
                    dropout=recurrent_dropout if recurrent_layers > 1 else 0.0
                )
            else:
                raise ValueError(f"Unknown recurrent type: {recurrent_type}")
            
            # Feature extractor takes recurrent output
            feature_input_dim = recurrent_hidden_dim
            self.hidden_state = None
        else:
            print(f"[ActorCritic] Creating STANDARD network")
            feature_input_dim = obs_dim

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_input_dim, hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.Tanh()
        ).to(device)

        # Actor head
        if continuous_action_space:
            self.action_var = nn.Parameter(torch.full(size=(action_dim,), fill_value=action_std_init * action_std_init)).to(device)
            self.actor_head = nn.Linear(hidden_dim, action_dim, dtype=torch.float32).to(device)
        else:
            self.actor_head = nn.Sequential(
                nn.Linear(hidden_dim, action_dim, dtype=torch.float32),
                nn.Softmax(dim=-1)
            ).to(device)

        # Critic head
        self.critic_head = nn.Linear(hidden_dim, 1).to(device)

    def forward(self, obs, hidden_state=None, mask=None):
        """
        Forward pass - handles both recurrent and non-recurrent cases
        
        Args:
            obs: Observations (batch_size, obs_dim) or (batch_size, seq_len, obs_dim)
            hidden_state: Optional hidden state for recurrent networks
            mask: Optional mask for variable length sequences
        """
        batch_size = obs.size(0)
        
        if self.use_recurrent:
            # Add sequence dimension if needed
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)  # (batch_size, 1, obs_dim)
            
            # Process through recurrent layer
            if self.recurrent_type == 'lstm':
                if hidden_state is None:
                    h0 = torch.zeros(self.recurrent_layers, batch_size, self.recurrent_hidden_dim).to(self.device)
                    c0 = torch.zeros(self.recurrent_layers, batch_size, self.recurrent_hidden_dim).to(self.device)
                    hidden_state = (h0, c0)
                rnn_out, new_hidden_state = self.recurrent(obs, hidden_state)
            else:  # GRU
                if hidden_state is None:
                    hidden_state = torch.zeros(self.recurrent_layers, batch_size, self.recurrent_hidden_dim).to(self.device)
                rnn_out, new_hidden_state = self.recurrent(obs, hidden_state)
            
            # Get last timestep output
            if mask is not None:
                lengths = mask.sum(dim=1).long() - 1
                lengths = lengths.clamp(min=0)
                idx = lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.recurrent_hidden_dim)
                features_input = rnn_out.gather(1, idx).squeeze(1)
            else:
                features_input = rnn_out[:, -1, :]
            
            # Process through feature extractor
            features = self.feature_extractor(features_input)
        else:
            # Standard forward pass
            features = self.feature_extractor(obs)
            new_hidden_state = None
        
        # Actor and critic outputs
        actor_out = self.actor_head(features)
        critic_out = self.critic_head(features)
        
        return actor_out, critic_out, new_hidden_state

    def select_action(self, obs, hidden_state=None, deterministic=False):
        """Select action - handles both recurrent and non-recurrent"""
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            action_out, value, new_hidden_state = self.forward(obs, hidden_state)

            if self.continuous_action_space:
                if deterministic:
                    action = action_out
                    action_cov = torch.diag(self.action_var)
                    dist = MultivariateNormal(action_out, action_cov)
                    action_logprob = dist.log_prob(action)
                else:
                    action_cov = torch.diag(self.action_var)
                    dist = MultivariateNormal(action_out, action_cov)
                    action = dist.sample()
                    action_logprob = dist.log_prob(action)

                if action.dim() == 2 and action.shape[0] == 1:
                    action = action.squeeze(0).cpu().numpy()
            else:
                dist = Categorical(action_out)
                if deterministic:
                    action = action_out.argmax(dim=-1)
                else:
                    action = dist.sample()
                action_logprob = dist.log_prob(action)
                action = action.item()

        if self.use_recurrent:
            return action, action_logprob.cpu().numpy(), value.item(), new_hidden_state
        else:
            return action, action_logprob.cpu().numpy(), value.item()

    def evaluate_actions(self, states, actions, hidden_states=None, masks=None):
        """Evaluate actions for PPO update"""
        action_out, values, _ = self.forward(states, hidden_states, masks)

        if self.continuous_action_space:
            action_cov = torch.diag(self.action_var)
            dist = MultivariateNormal(action_out, action_cov)
            action_logprobs = dist.log_prob(actions)
        else:
            dist = Categorical(action_out)
            action_logprobs = dist.log_prob(actions.squeeze(-1).long())
        
        dist_entropy = dist.entropy()

        return values.squeeze(), action_logprobs, dist_entropy
    
    def reset_hidden_state(self, batch_size=1):
        """Reset hidden state for recurrent networks"""
        if not self.use_recurrent:
            return None
            
        if self.recurrent_type == 'lstm':
            h0 = torch.zeros(self.recurrent_layers, batch_size, self.recurrent_hidden_dim).to(self.device)
            c0 = torch.zeros(self.recurrent_layers, batch_size, self.recurrent_hidden_dim).to(self.device)
            self.hidden_state = (h0, c0)
        else:  # GRU
            self.hidden_state = torch.zeros(self.recurrent_layers, batch_size, self.recurrent_hidden_dim).to(self.device)
        
        return self.hidden_state



import torch
import torch.nn as nn
import numpy as np
from collections import deque

class PPOAgent:
    def __init__(
        self, 
        obs_dim, 
        action_dim, 
        hidden_dim, 
        lr_actor, 
        lr_critic, 
        continuous_action_space=False, 
        num_epochs=10, 
        eps_clip=0.2, 
        action_std_init=0.6, 
        gamma=0.99,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        batch_size=64,
        max_grad_norm=0.5,
        device='cpu',
        # Recurrent parameters
        use_recurrent=False,
        recurrent_type='lstm',
        recurrent_hidden_dim=128,
        recurrent_layers=1,
        recurrent_sequence_length=16,
        recurrent_dropout=0.0
    ):
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_std_init = action_std_init
        self.continuous_action_space = continuous_action_space
        self.device = device

        # Recurrent settings
        self.use_recurrent = use_recurrent
        self.recurrent_sequence_length = recurrent_sequence_length
        self.recurrent_type = recurrent_type

        print(f"\n[PPOAgent] Initializing {'RECURRENT' if use_recurrent else 'STANDARD'} PPO agent")
        if use_recurrent:
            print(f"      - Recurrent type: {recurrent_type}")
            print(f"      - Sequence length: {recurrent_sequence_length}")
            print(f"      - Recurrent hidden dim: {recurrent_hidden_dim}")
            print(f"      - Recurrent layers: {recurrent_layers}")
            
            # Observation history buffer
            self.obs_history = deque(maxlen=recurrent_sequence_length)
            self.current_hidden_state = None

        # Create policy network
        self.policy = ActorCritic(
            obs_dim,
            action_dim,
            hidden_dim,
            continuous_action_space=continuous_action_space,
            action_std_init=action_std_init,
            device=device,
            use_recurrent=use_recurrent,
            recurrent_type=recurrent_type,
            recurrent_hidden_dim=recurrent_hidden_dim,
            recurrent_layers=recurrent_layers,
            recurrent_dropout=recurrent_dropout
        ).to(device)

        # Optimizer
        optimizer_params = [
            {'params': self.policy.feature_extractor.parameters()},
            {'params': self.policy.actor_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic}
        ]
        if use_recurrent:
            optimizer_params.append({'params': self.policy.recurrent.parameters()})
        
        self.optimizer = torch.optim.Adam(optimizer_params)
        self.buffer = RolloutBuffer()
        self.mse_loss = nn.MSELoss()

    def reset_episode(self):
        """Reset recurrent states"""
        if self.use_recurrent:
            self.obs_history.clear()
            self.current_hidden_state = self.policy.reset_hidden_state(batch_size=1).to(self.device)

    def select_action(self, obs, deterministic=False):
        """Select action (handles recurrent vs standard)"""
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        else:
            obs_tensor = obs.to(self.device)

        if self.use_recurrent:
            self.obs_history.append(obs_tensor)
            while len(self.obs_history) < self.recurrent_sequence_length:
                self.obs_history.insert(0, torch.zeros_like(obs_tensor))
            obs_seq = torch.stack(list(self.obs_history)).unsqueeze(0)  # (1, seq_len, obs_dim)

            result = self.policy.select_action(obs_seq, self.current_hidden_state, deterministic)
            if len(result) == 4:
                action, log_prob, value, self.current_hidden_state = result
            else:
                action, log_prob, value = result
        else:
            action, log_prob, value = self.policy.select_action(obs_tensor, deterministic=deterministic)
        
        return action, log_prob, value

    def compute_returns(self):
        """Compute discounted rewards"""
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        return returns

    def update_policy(self):
        """PPO update (recurrent or standard)"""
        rewards_to_go = self.compute_returns()
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor(np.array(self.buffer.logprobs), dtype=torch.float32).to(self.device)
        state_vals = torch.tensor(np.array(self.buffer.state_values), dtype=torch.float32).to(self.device)

        advantages = rewards_to_go - state_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        for _ in range(self.num_epochs):
            indices = np.random.permutation(len(states))
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_idx = indices[start_idx:end_idx]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_rewards_to_go = rewards_to_go[batch_idx]

                if self.use_recurrent:
                    # Add sequence dimension
                    state_values, logprobs, dist_entropy = self.policy.evaluate_actions(
                        batch_states.unsqueeze(1), batch_actions
                    )
                else:
                    state_values, logprobs, dist_entropy = self.policy.evaluate_actions(
                        batch_states, batch_actions
                    )

                ratios = torch.exp(logprobs - batch_old_logprobs.squeeze(-1))
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * self.mse_loss(state_values.squeeze(), batch_rewards_to_go)
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * dist_entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.buffer.clear()

    def save(self, path):
        checkpoint = {
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': {
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'use_recurrent': self.use_recurrent,
                'continuous_action_space': self.continuous_action_space,
            }
        }
        if self.use_recurrent:
            checkpoint['config'].update({
                'recurrent_type': self.recurrent_type,
                'recurrent_sequence_length': self.recurrent_sequence_length,
            })
        torch.save(checkpoint, path)
        print(f"[PPOAgent] Model saved to {path}")

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        if load_optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"[PPOAgent] Model loaded from {path}")

