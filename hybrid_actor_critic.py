"""
Hybrid Actor-Critic for discrete spoke selection + continuous delta adjustment.

This network properly handles the mixed action space:
- Spoke selection: Categorical distribution (discrete)
- Delta adjustment: Normal distribution (continuous)

This avoids the gradient issues of rounding a continuous spoke index.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np


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
