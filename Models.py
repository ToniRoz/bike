# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import gymnasium as gym
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np





# Factorised NoisyLinear layer with bias 
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
        
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
        
    def reset_noise(self):
        device = self.weight_mu.device 
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in).to(device))
        self.bias_epsilon.copy_(epsilon_out.to(device))
        
    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, 
                          self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


# Quantile Embedding Network for IQN 
class QuantileEmbedding(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(QuantileEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, taus):
        """
        Args:
            taus: Tensor of shape (batch_size, num_quantiles) with values in [0, 1]
        Returns:
            Embeddings of shape (batch_size * num_quantiles, output_dim)
        """
        batch_size = taus.shape[0]
        num_quantiles = taus.shape[1]
        
        i_pi = torch.arange(0, self.embedding_dim, dtype=torch.float32, device=taus.device) * math.pi
        taus_expanded = taus.unsqueeze(2)
        cos_embedding = torch.cos(taus_expanded * i_pi)
        
        cos_embedding = cos_embedding.view(batch_size * num_quantiles, self.embedding_dim)
        embedding = F.relu(self.fc(cos_embedding))
        
        return embedding


class DQN(nn.Module):
    """
    Rainbow DQN with IQN (Implicit Quantile Network)
    Now with DYNAMIC input dimension based on environment
    """
    def __init__(self, args, action_space, state_dim):
        super(DQN, self).__init__()
        self.action_space = action_space
        self.state_dim = state_dim
        
        # Calculate input dimension: history_length * state_dim
        self.input_dim = args.history_length * state_dim
        self.hidden_size = args.hidden_size
        self.embedding_dim = args.embedding_dim
        
        print(f"[DQN] Creating network with:")
        print(f"      - Input dim: {self.input_dim} (history={args.history_length} × state_dim={state_dim})")
        print(f"      - Hidden size: {self.hidden_size}")
        print(f"      - Action space: {self.action_space}")
        print(f"      - Embedding dim: {self.embedding_dim}")
        
        # State feature extraction layers with DYNAMIC input_dim
        self.fc_h_v = NoisyLinear(self.input_dim, self.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(self.input_dim, self.hidden_size, std_init=args.noisy_std)
        
        # Quantile embedding network
        self.quantile_embedding = QuantileEmbedding(self.embedding_dim, self.hidden_size)
        
        # Output layers
        self.fc_z_v = NoisyLinear(self.hidden_size, 1, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(self.hidden_size, action_space, std_init=args.noisy_std)
        
    def forward(self, x, num_quantiles):
        """
        Args:
            x: State tensor of shape (batch_size, history_length * state_dim)
            num_quantiles: Number of quantiles to sample (N for training, K for evaluation)
        Returns:
            quantile_values: Tensor of shape (batch_size, action_space, num_quantiles)
            taus: Sampled quantile fractions of shape (batch_size, num_quantiles)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Flatten input (should already be flat, but ensure)
        x = x.view(batch_size, -1)
        
        # Sample quantile fractions uniformly from [0, 1]
        taus = torch.rand(batch_size, num_quantiles, device=device)
        
        # Get quantile embeddings: (batch_size * num_quantiles, hidden_size)
        quantile_embed = self.quantile_embedding(taus)
        
        # Extract state features for value and advantage streams
        state_v = F.relu(self.fc_h_v(x))  # (batch_size, hidden_size)
        state_a = F.relu(self.fc_h_a(x))  # (batch_size, hidden_size)
        
        # Replicate state features for each quantile
        state_v_expanded = state_v.unsqueeze(1).repeat(1, num_quantiles, 1)
        state_a_expanded = state_a.unsqueeze(1).repeat(1, num_quantiles, 1)
        
        # Reshape to (batch_size * num_quantiles, hidden_size)
        state_v_flat = state_v_expanded.view(batch_size * num_quantiles, self.hidden_size)
        state_a_flat = state_a_expanded.view(batch_size * num_quantiles, self.hidden_size)
        
        # Element-wise multiplication with quantile embeddings
        combined_v = state_v_flat * quantile_embed
        combined_a = state_a_flat * quantile_embed
        
        # Pass through output layers
        v = self.fc_z_v(combined_v)  # (batch_size * num_quantiles, 1)
        a = self.fc_z_a(combined_a)  # (batch_size * num_quantiles, action_space)
        
        # Reshape back
        v = v.view(batch_size, num_quantiles, 1)
        a = a.view(batch_size, num_quantiles, self.action_space)
        
        # Transpose to (batch_size, 1, num_quantiles) and (batch_size, action_space, num_quantiles)
        v = v.transpose(1, 2)
        a = a.transpose(1, 2)
        
        # Dueling architecture: Q = V + (A - mean(A))
        q = v + a - a.mean(1, keepdim=True)
        
        # Output shape: (batch_size, action_space, num_quantiles)
        return q, taus
        
    def reset_noise(self):
        """Reset noise in all noisy layers"""
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()


class RecurrentDQN(nn.Module):
    """
    Recurrent Rainbow DQN with IQN (Implicit Quantile Network)
    Takes sequences of (state, action) pairs as input
    """
    def __init__(self, args, action_space, state_dim):
        super(RecurrentDQN, self).__init__()
        self.action_space = action_space
        self.state_dim = state_dim
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_type = args.recurrent_type
        self.recurrent_layers = args.recurrent_layers
        self.hidden_size = args.hidden_size
        self.embedding_dim = args.embedding_dim
        
        # Input: state_dim + action_space (one-hot encoded action)
        self.input_dim = state_dim + action_space
        
        print(f"[RecurrentDQN] Creating recurrent network with:")
        print(f"      - Input dim per timestep: {self.input_dim} (state={state_dim} + action={action_space})")
        print(f"      - Recurrent type: {self.recurrent_type}")
        print(f"      - Recurrent hidden size: {self.recurrent_hidden_size}")
        print(f"      - Recurrent layers: {self.recurrent_layers}")
        print(f"      - DQN hidden size: {self.hidden_size}")
        print(f"      - Action space: {self.action_space}")
        
        # Recurrent layer
        if self.recurrent_type == "lstm":
            self.recurrent = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.recurrent_hidden_size,
                num_layers=self.recurrent_layers,
                batch_first=True,
                dropout=args.recurrent_dropout if self.recurrent_layers > 1 else 0.0
            )
        elif self.recurrent_type == "gru":
            self.recurrent = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.recurrent_hidden_size,
                num_layers=self.recurrent_layers,
                batch_first=True,
                dropout=args.recurrent_dropout if self.recurrent_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Unknown recurrent type: {self.recurrent_type}")
        
        # Feature extraction layers (take recurrent output)
        self.fc_h_v = NoisyLinear(self.recurrent_hidden_size, self.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(self.recurrent_hidden_size, self.hidden_size, std_init=args.noisy_std)
        
        # Quantile embedding network
        self.quantile_embedding = QuantileEmbedding(self.embedding_dim, self.hidden_size)
        
        # Output layers
        self.fc_z_v = NoisyLinear(self.hidden_size, 1, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(self.hidden_size, action_space, std_init=args.noisy_std)
        
        # Hidden state buffer (for stateful mode if needed)
        self.hidden_state = None
        
    def forward(self, x, num_quantiles, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, state_dim + action_space)
            num_quantiles: Number of quantiles to sample
            mask: Optional mask of shape (batch_size, seq_len) for padded timesteps
        Returns:
            quantile_values: Tensor of shape (batch_size, action_space, num_quantiles)
            taus: Sampled quantile fractions of shape (batch_size, num_quantiles)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        device = x.device
        
        # Process sequence through recurrent layer
        # x: (batch_size, seq_len, input_dim)
        if self.recurrent_type == "lstm":
            rnn_out, (h_n, c_n) = self.recurrent(x)
        else:  # GRU
            rnn_out, h_n = self.recurrent(x)
        
        # Take the output from the last timestep
        # If mask is provided, use the last valid timestep for each sequence
        if mask is not None:
            # mask: (batch_size, seq_len)
            # Find the last valid timestep for each sequence
            lengths = mask.sum(dim=1).long() - 1  # (batch_size,)
            lengths = lengths.clamp(min=0)
            # Gather the output at the last valid timestep
            idx = lengths.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.recurrent_hidden_size)
            rnn_output = rnn_out.gather(1, idx).squeeze(1)  # (batch_size, recurrent_hidden_size)
        else:
            # Use the last timestep
            rnn_output = rnn_out[:, -1, :]  # (batch_size, recurrent_hidden_size)
        
        # Sample quantile fractions uniformly from [0, 1]
        taus = torch.rand(batch_size, num_quantiles, device=device)
        
        # Get quantile embeddings: (batch_size * num_quantiles, hidden_size)
        quantile_embed = self.quantile_embedding(taus)
        
        # Extract state features for value and advantage streams
        state_v = F.relu(self.fc_h_v(rnn_output))  # (batch_size, hidden_size)
        state_a = F.relu(self.fc_h_a(rnn_output))  # (batch_size, hidden_size)
        
        # Replicate state features for each quantile
        state_v_expanded = state_v.unsqueeze(1).repeat(1, num_quantiles, 1)
        state_a_expanded = state_a.unsqueeze(1).repeat(1, num_quantiles, 1)
        
        # Reshape to (batch_size * num_quantiles, hidden_size)
        state_v_flat = state_v_expanded.view(batch_size * num_quantiles, self.hidden_size)
        state_a_flat = state_a_expanded.view(batch_size * num_quantiles, self.hidden_size)
        
        # Element-wise multiplication with quantile embeddings
        combined_v = state_v_flat * quantile_embed
        combined_a = state_a_flat * quantile_embed
        
        # Pass through output layers
        v = self.fc_z_v(combined_v)  # (batch_size * num_quantiles, 1)
        a = self.fc_z_a(combined_a)  # (batch_size * num_quantiles, action_space)
        
        # Reshape back
        v = v.view(batch_size, num_quantiles, 1)
        a = a.view(batch_size, num_quantiles, self.action_space)
        
        # Transpose to (batch_size, 1, num_quantiles) and (batch_size, action_space, num_quantiles)
        v = v.transpose(1, 2)
        a = a.transpose(1, 2)
        
        # Dueling architecture: Q = V + (A - mean(A))
        q = v + a - a.mean(1, keepdim=True)
        
        # Output shape: (batch_size, action_space, num_quantiles)
        return q, taus
        
    def reset_noise(self):
        """Reset noise in all noisy layers"""
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()
    
    def reset_hidden_state(self):
        """Reset hidden state (for stateful mode)"""
        self.hidden_state = None




#### PPO #####

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


class FeedForwardNN(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_size=64):
        super(FeedForwardNN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(inp_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_dim),
        )

    def forward(self, obs):
        return self.net(obs)


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

class HybridActorCritic(nn.Module):
    """
    Actor-Critic with hybrid action space:
    - Discrete: which spoke to adjust (Categorical over n_spokes)
    - Continuous: how much to adjust (Normal distribution)
    
    Supports optional recurrent layers (LSTM/GRU).
    """
    
    def __init__(
        self,
        obs_dim,
        n_spokes,
        hidden_dim,
        delta_std_init=0.3,
        device='cpu',
        # Recurrent parameters
        use_recurrent=False,
        recurrent_type='lstm',
        recurrent_hidden_dim=128,
        recurrent_layers=1,
        recurrent_dropout=0.0
    ):
        super(HybridActorCritic, self).__init__()
        
        self.obs_dim = obs_dim
        self.n_spokes = n_spokes
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Recurrent settings
        self.use_recurrent = use_recurrent
        self.recurrent_type = recurrent_type
        self.recurrent_hidden_dim = recurrent_hidden_dim
        self.recurrent_layers = recurrent_layers
        
        print(f"[HybridActorCritic] Creating HYBRID network:")
        print(f"      - Observation dim: {obs_dim}")
        print(f"      - Number of spokes: {n_spokes}")
        print(f"      - Hidden dim: {hidden_dim}")
        print(f"      - Delta std init: {delta_std_init}")
        
        if use_recurrent:
            print(f"      - Recurrent type: {recurrent_type}")
            print(f"      - Recurrent hidden dim: {recurrent_hidden_dim}")
            print(f"      - Recurrent layers: {recurrent_layers}")
            
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
            
            feature_input_dim = recurrent_hidden_dim
            self.hidden_state = None
        else:
            feature_input_dim = obs_dim
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        ).to(device)
        
        # Spoke selection head (discrete) - outputs logits
        self.spoke_head = nn.Linear(hidden_dim, n_spokes).to(device)
        
        # Delta head (continuous) - outputs mean, learnable log_std
        self.delta_mean_head = nn.Linear(hidden_dim, 1).to(device)
        self.delta_log_std = nn.Parameter(torch.tensor([np.log(delta_std_init)], device=device))
        
        # Critic head
        self.critic_head = nn.Linear(hidden_dim, 1).to(device)
    
    def _get_features(self, obs, hidden_state=None):
        """Extract features, handling recurrent case."""
        batch_size = obs.size(0)
        
        if self.use_recurrent:
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)
            
            if self.recurrent_type == 'lstm':
                if hidden_state is None:
                    h0 = torch.zeros(self.recurrent_layers, batch_size, self.recurrent_hidden_dim, device=self.device)
                    c0 = torch.zeros(self.recurrent_layers, batch_size, self.recurrent_hidden_dim, device=self.device)
                    hidden_state = (h0, c0)
                rnn_out, new_hidden_state = self.recurrent(obs, hidden_state)
            else:  # GRU
                if hidden_state is None:
                    hidden_state = torch.zeros(self.recurrent_layers, batch_size, self.recurrent_hidden_dim, device=self.device)
                rnn_out, new_hidden_state = self.recurrent(obs, hidden_state)
            
            features_input = rnn_out[:, -1, :]  # Last timestep
            features = self.feature_extractor(features_input)
        else:
            features = self.feature_extractor(obs)
            new_hidden_state = None
        
        return features, new_hidden_state
    
    def forward(self, obs, hidden_state=None):
        """
        Forward pass.
        
        Returns:
            spoke_logits: (batch, n_spokes) - logits for spoke selection
            delta_mean: (batch, 1) - mean for delta
            delta_std: (batch, 1) - std for delta (broadcasted)
            value: (batch, 1) - state value
            new_hidden_state: updated hidden state (or None)
        """
        features, new_hidden_state = self._get_features(obs, hidden_state)
        
        # Spoke selection (discrete)
        spoke_logits = self.spoke_head(features)
        
        # Delta (continuous) - use tanh to bound mean to [-1, 1]
        delta_mean = torch.tanh(self.delta_mean_head(features))
        delta_std = torch.exp(self.delta_log_std).expand_as(delta_mean)
        
        # Value
        value = self.critic_head(features)
        
        return spoke_logits, delta_mean, delta_std, value, new_hidden_state
    
    def select_action(self, obs, hidden_state=None, deterministic=False):
        """
        Select action.
        
        Returns:
            action: numpy array [spoke_idx, delta]
            log_prob: combined log probability
            value: state value
            (new_hidden_state if recurrent)
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            spoke_logits, delta_mean, delta_std, value, new_hidden_state = self.forward(obs, hidden_state)
            
            # Spoke distribution (discrete)
            spoke_probs = F.softmax(spoke_logits, dim=-1)
            spoke_dist = Categorical(spoke_probs)
            
            # Delta distribution (continuous)
            delta_dist = Normal(delta_mean, delta_std)
            
            if deterministic:
                spoke_idx = spoke_logits.argmax(dim=-1)
                delta = delta_mean.squeeze(-1)
            else:
                spoke_idx = spoke_dist.sample()
                delta = delta_dist.sample().squeeze(-1)
            
            # Combined log probability
            spoke_log_prob = spoke_dist.log_prob(spoke_idx)
            delta_log_prob = delta_dist.log_prob(delta.unsqueeze(-1)).squeeze(-1)
            total_log_prob = spoke_log_prob + delta_log_prob
            
            # Create action array
            action = np.array([spoke_idx.item(), delta.item()], dtype=np.float32)
        
        if self.use_recurrent:
            return action, total_log_prob.cpu().numpy(), value.item(), new_hidden_state
        else:
            return action, total_log_prob.cpu().numpy(), value.item()
    
    def evaluate_actions(self, states, actions, hidden_states=None, masks=None):
        """
        Args:
            states: (batch, obs_dim) or (batch, seq_len, obs_dim)
            actions: (batch, 2) - [spoke_idx, delta]
            hidden_states: optional for recurrent
            masks: optional for recurrent
        
        Returns:
            values: (batch,)
            log_probs: (batch,)
            entropy: (batch,)
        """
        spoke_logits, delta_mean, delta_std, values, _ = self.forward(states, hidden_states)
        
        # Extract spoke indices and deltas from actions
        spoke_indices = actions[:, 0].long()
        deltas = actions[:, 1:2]  # Keep dim for broadcasting
        
        # Spoke distribution
        spoke_probs = F.softmax(spoke_logits, dim=-1)
        spoke_dist = Categorical(spoke_probs)
        spoke_log_prob = spoke_dist.log_prob(spoke_indices)
        spoke_entropy = spoke_dist.entropy()
        
        # Delta distribution
        delta_dist = Normal(delta_mean, delta_std)
        delta_log_prob = delta_dist.log_prob(deltas).squeeze(-1)
        delta_entropy = delta_dist.entropy().squeeze(-1)
        
        # Combined
        total_log_prob = spoke_log_prob + delta_log_prob
        total_entropy = spoke_entropy + delta_entropy
        
        return values.squeeze(-1), total_log_prob, total_entropy
    
    def reset_hidden_state(self, batch_size=1):
        """Reset hidden state for recurrent networks."""
        if not self.use_recurrent:
            return None
        
        if self.recurrent_type == 'lstm':
            h0 = torch.zeros(self.recurrent_layers, batch_size, self.recurrent_hidden_dim, device=self.device)
            c0 = torch.zeros(self.recurrent_layers, batch_size, self.recurrent_hidden_dim, device=self.device)
            self.hidden_state = (h0, c0)
        else:
            self.hidden_state = torch.zeros(self.recurrent_layers, batch_size, self.recurrent_hidden_dim, device=self.device)
        
        return self.hidden_state


class HybridPPOAgent:
    """
    PPO Agent using HybridActorCritic for mixed discrete-continuous action spaces.
    """
    
    def __init__(
        self,
        obs_dim,
        n_spokes,
        hidden_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        num_epochs=10,
        eps_clip=0.2,
        delta_std_init=0.3,
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
        self.device = device
        
        self.obs_dim = obs_dim
        self.n_spokes = n_spokes
        
        # Recurrent settings
        self.use_recurrent = use_recurrent
        self.recurrent_sequence_length = recurrent_sequence_length
        
        print(f"\n[HybridPPOAgent] Initializing {'RECURRENT' if use_recurrent else 'STANDARD'} Hybrid PPO agent")
        
        if use_recurrent:
            from collections import deque
            self.obs_history = deque(maxlen=recurrent_sequence_length)
            self.current_hidden_state = None
        
        # Create hybrid policy network
        self.policy = HybridActorCritic(
            obs_dim=obs_dim,
            n_spokes=n_spokes,
            hidden_dim=hidden_dim,
            delta_std_init=delta_std_init,
            device=device,
            use_recurrent=use_recurrent,
            recurrent_type=recurrent_type,
            recurrent_hidden_dim=recurrent_hidden_dim,
            recurrent_layers=recurrent_layers,
            recurrent_dropout=recurrent_dropout
        ).to(device)
        
        # Optimizer with different learning rates
        optimizer_params = [
            {'params': self.policy.feature_extractor.parameters()},
            {'params': self.policy.spoke_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.delta_mean_head.parameters(), 'lr': lr_actor},
            {'params': [self.policy.delta_log_std], 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic}
        ]
        if use_recurrent:
            optimizer_params.append({'params': self.policy.recurrent.parameters()})
        
        self.optimizer = torch.optim.Adam(optimizer_params)
        
        # Import RolloutBuffer from Models
        from Models import RolloutBuffer
        self.buffer = RolloutBuffer()
        
        self.mse_loss = nn.MSELoss()
    
    def reset_episode(self):
        """Reset recurrent states."""
        if self.use_recurrent:
            self.obs_history.clear()
            self.current_hidden_state = self.policy.reset_hidden_state(batch_size=1)
    
    def select_action(self, obs, deterministic=False):
        """Select action (handles recurrent vs standard)."""
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs.to(self.device)
        
        if self.use_recurrent:
            self.obs_history.append(obs_tensor)
            while len(self.obs_history) < self.recurrent_sequence_length:
                self.obs_history.insert(0, torch.zeros_like(obs_tensor))
            obs_seq = torch.stack(list(self.obs_history)).unsqueeze(0)
            
            result = self.policy.select_action(obs_seq, self.current_hidden_state, deterministic)
            if len(result) == 4:
                action, log_prob, value, self.current_hidden_state = result
            else:
                action, log_prob, value = result
        else:
            action, log_prob, value = self.policy.select_action(obs_tensor, deterministic=deterministic)
        
        return action, log_prob, value
    
    def compute_returns(self):
        """Compute discounted rewards."""
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        return returns
    
    def update_policy(self):
        """PPO update."""
        rewards_to_go = self.compute_returns()
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32, device=self.device)
        old_logprobs = torch.tensor(np.array(self.buffer.logprobs), dtype=torch.float32, device=self.device)
        state_vals = torch.tensor(np.array(self.buffer.state_values), dtype=torch.float32, device=self.device)
        
        advantages = rewards_to_go - state_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
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
                    state_values, logprobs, dist_entropy = self.policy.evaluate_actions(
                        batch_states.unsqueeze(1), batch_actions
                    )
                else:
                    state_values, logprobs, dist_entropy = self.policy.evaluate_actions(
                        batch_states, batch_actions
                    )
                
                # Handle old_logprobs shape
                if batch_old_logprobs.dim() > 1:
                    batch_old_logprobs = batch_old_logprobs.squeeze(-1)
                
                ratios = torch.exp(logprobs - batch_old_logprobs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * self.mse_loss(state_values, batch_rewards_to_go)
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * dist_entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        self.buffer.clear()
    
    def save(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': {
                'obs_dim': self.obs_dim,
                'n_spokes': self.n_spokes,
                'use_recurrent': self.use_recurrent,
            }
        }
        torch.save(checkpoint, path)
        print(f"[HybridPPOAgent] Model saved to {path}")
    
    def load(self, path, load_optimizer=True):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        if load_optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"[HybridPPOAgent] Model loaded from {path}")


class PPO:
    def __init__(self, env):
        super(PPO, self).__init__()

        self.env = env

        # extract environment information
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self._set_hyperparameters()

        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        # create actor-critic optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        # initialize covariance matrix for continuous action space
        self.action_cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.act_cov = torch.diag(self.action_cov_var)

        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lengths': [],
            'batch_rewards': [],
            'actor_losses': []
        }


    def learn(self, total_timesteps, timesteps_per_batch, max_eps_len, num_updates_per_itr, clip_thresh=0.2, save_every=1000, gamma=0.9):
        t_so_far = 0    # timesteps simulated so far
        i_so_far = 0

        while t_so_far < total_timesteps:
            # roll out multiple trajectories
            batch_obs, batch_actions, batch_logprobs, batch_reward_to_go, batch_eps_lens = self.collect_rollouts(
                timesteps_per_batch, 
                max_eps_len, 
                gamma
            )
            print("stage-1:", batch_obs.shape, batch_actions.shape, batch_logprobs.shape, batch_reward_to_go.shape)

            # calculate how many timesteps collected in this batch
            t_so_far += np.sum(batch_eps_lens)
            i_so_far += 1

            # logging timesteps and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # calculate value function V_{phi, k} using critic model
            V, _ = self.evaluate(batch_obs, batch_actions)

            # calculate advantage function A_k
            A_k = batch_reward_to_go - V.detach()

            # normalize advantage function
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(num_updates_per_itr):
                # calculate pi_theta(at | st)
                curr_V, curr_logprobs = self.evaluate(batch_obs, batch_actions)

                # calcuate ratios
                ratios = torch.exp(curr_logprobs - batch_logprobs)

                # calcuate surrogate losses
                surr1 = ratios * A_k

                # clips ratio to make sure we are not stepping too far in any direction during gradient ascent
                surr2 = torch.clamp(ratios, 1 - clip_thresh, 1 + clip_thresh) * A_k

                # calculate actor and critic losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(curr_V, batch_reward_to_go)

                # calculate gradients and backpropagate for actor network
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                # calculate gradients and backpropagate for critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                self.logger['actor_losses'].append(actor_loss.detach())
            
            # print a summary of the training so far
            self._log_summary(total_timesteps)

            if i_so_far % save_every == 0:
                torch.save(self.actor.state_dict(), './checkpoints/ppo_actor.pth')
                torch.save(self.critic.state_dict(), './checkpoints/ppo_critic.pth')


    def evaluate(self, batch_obs, batch_actions):
        value = self.critic(batch_obs).squeeze()
        # print(value.shape)

        # calculate the log probabilities of batch actions using most recent actor network
        mean = self.actor(batch_obs)
        # print("Stage-2", mean.shape, self.action_cov_mat.shape, batch_obs.shape, batch_actions.shape)
        dist = MultivariateNormal(mean, self.act_cov)
        # print("This would be printed", dist)
        logprob = dist.log_prob(batch_actions)
        # print("This would not be printed", dist)
        return value, logprob


    def collect_rollouts(self, max_timesteps, max_eps_len, gamma):
        observations = []
        actions = []
        logprobs = []
        rewards = []
        eps_lens = []

        t = 0
        while t < max_timesteps:
            # reset environment and get initial observation
            obs, _ = self.env.reset()
            done = False
            # print("Stage-2 after reset:", obs)

            eps_rewards = []
            for step in range(max_eps_len):
                action, logprob = self.select_action(obs)
                next_obs, reward, done, _, _ = self.env.step(action)
                t += 1

                # collect observation, action, log probabilities and reward
                observations.append(obs)
                actions.append(action)
                logprobs.append(logprob)
                eps_rewards.append(reward)

                obs = next_obs
                if done:
                    break
            
            # collect episode length and rewards
            rewards.append(eps_rewards)
            eps_lens.append(step+1)

        # reshape numpy data as tensors
        observations = torch.from_numpy(np.array(observations, dtype=np.float32))    # [max_timesteps, ns]
        actions = torch.from_numpy(np.array(actions, dtype=np.float32))    # [max_timesteps, na]
        actions = actions.unsqueeze(1)
        logprobs = torch.from_numpy(np.array(logprobs, dtype=np.float32))    # [max_timesteps]
        rewards_to_go = self.compute_reward_to_go(rewards, gamma)
        # print("Stage-0:", np.array(batch_rewards).shape, batch_reward_to_go.shape)
        # batch_episode_lengths = torch.tensor(batch_episode_lengths, dtype=torch.float32)

        # log the episodic rewards and lengths
        self.logger['batch_rewards'] = rewards
        self.logger['batch_lengths'] = eps_lens
        return observations, actions, logprobs, rewards_to_go, eps_lens


    def compute_reward_to_go(self, rewards, gamma):
        """
        Compute the discounted reward-to-go for each timestep in each episode
        Args:
            rewards: list of lists, where each inner list contains rewards for an episode
            gamma: discount  for future rewards
        Returns:
            rewards_to_go: list of reward-to-go for each timestep in each episode
        """
        rewards_to_go = []

        # iterate through each episodic rewards
        for eps_rewards in rewards:
            eps_rewards_to_go = []
            reward_sum = 0

            for r in reversed(eps_rewards):
                reward_sum = r + gamma * reward_sum    # discounted reward
                eps_rewards_to_go.append(reward_sum)

            eps_rewards_to_go = eps_rewards_to_go[::-1]
            rewards_to_go.append(eps_rewards_to_go)

        # convert reward-to-go into tensor
        rewards_to_go = np.array(rewards_to_go, dtype=np.float32)
        rewards_to_go = torch.flatten(torch.from_numpy(rewards_to_go))

        return rewards_to_go


    def estimate_action(self, obs):
        print("Stage-3:", obs)
        # query the actor network for mean of the distribution
        mean = self.actor(obs)

        # create multivariate normal distribution
        dist = MultivariateNormal(mean, self.act_cov)

        # sample an action from the distribution and compute its logprob
        action = dist.sample()
        logprob = dist.log_prob(action)

        return action.detach().numpy(), logprob.detach()


    def _set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Successfully set seed everywhere: {seed}")


    def _log_summary(self, total_timesteps):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = round((self.logger['delta_t'] - delta_t) / 1e9, 4)

        avg_episode_lens = np.mean(self.logger['batch_lengths'])
        avg_episode_rewards = round(np.mean([np.sum(ep_rewards) for ep_rewards in self.logger['batch_rewards']]), 4)
        avg_actor_loss = round(np.mean([losses.mean() for losses in self.logger['actor_losses']]), 4)

        print(f"{self.logger['t_so_far']}/{total_timesteps} | Avg Loss: {avg_actor_loss} | Avg Ep Len: {avg_episode_lens} | Avg Ep Reward: {avg_episode_rewards} | Itr {self.logger['i_so_far']} took {delta_t} s")