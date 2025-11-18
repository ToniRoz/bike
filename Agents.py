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

from Models import DQN

"""
Todo:
    add lstm option to rainbow and ppo 
"""



class RainbowAgent:
    """Rainbow DQN agent with dynamic state and action space inference"""
    
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
        
        # ========== CREATE NETWORKS ==========
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
    
    def act(self, state):
        """Select action using the online network (for training with noise)"""
        with torch.no_grad():
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
        idxs, states, actions, returns, next_states, nonterminals, weights = memory.sample(
            self.config.batch_size
        )
        
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
    
    def update_target_net(self):
        """Update target network with online network weights"""
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    def train(self):
        """Set network to training mode"""
        self.online_net.train()
    
    def eval(self):
        """Set network to evaluation mode"""
        self.online_net.eval()
    
    def evaluate_q(self, state):
        """Evaluate Q-value for a state (used in evaluation)"""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            
            if state.dim() == len(self.state_shape):
                state = state.unsqueeze(0)
            
            state = state.reshape(1, -1)
            q_values, _ = self.online_net(state, self.num_quantiles_eval)
            q_values = q_values.mean(dim=2)  # Average over quantiles
            
            return q_values.max().item()
    
    def save(self, path, name='model.pth'):
        """Save model checkpoint"""
        import os
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, name)
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Model loaded from {path}")




#### PPO ####

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    def store_transition(self, state, action, logprob, reward, done, state_value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.dones.clear()


class ActorCritic(nn.Module):
    def __init__(
            self, 
            obs_dim, 
            action_dim, 
            hidden_dim, 
            continuous_action_space=False, 
            action_std_init=0.0, 
            device='cpu'
        ):
        super(ActorCritic, self).__init__()
        self.continuous_action_space = continuous_action_space
        self.device = device

        # create shared feature extractor for both actor and critic
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.Tanh()
        ).to(device)

        if continuous_action_space:
            self.action_var = nn.Parameter(torch.full(size=(action_dim,), fill_value=action_std_init * action_std_init)).to(device)
            self.actor_head = nn.Linear(hidden_dim, action_dim, dtype=torch.float32).to(device)
        else:
            self.actor_head = nn.Sequential(
                nn.Linear(hidden_dim, action_dim, dtype=torch.float32),
                nn.Softmax(dim=-1)
            ).to(device)

        self.critic_head = nn.Linear(hidden_dim, 1).to(device)


    def forward(self, obs):
        features = self.feature_extractor(obs)
        actor_out = self.actor_head(features)
        critic_out = self.critic_head(features)
        return actor_out, critic_out


    def select_action(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)    # add batch dimension if missing

        # to prevent unnecessary gradient computation
        with torch.no_grad():
            action_out, value = self.forward(obs)
            # print('stage-0:', action_out.shape, value, obs.shape)

            if self.continuous_action_space:
                action_cov = torch.diag(self.action_var)    # (na, na)
                # print('stage-1:', action_out.shape, action_cov.shape)
                dist = MultivariateNormal(action_out, action_cov)
            else:
                # print(action_out.shape)
                dist = Categorical(action_out)

            action = dist.sample()
            action_logprob = dist.log_prob(action)

            if self.continuous_action_space:
                if action.dim() == 2 and action.shape[0] == 1:
                    action = action.squeeze(0).cpu().numpy()
            else:
                # action = torch.clamp(action, -1.0, 1.0)
                action = action.item()

        return action, action_logprob.cpu().numpy(), value.item()


    def evaluate_actions(self, states, actions):
        action_out, values = self.forward(states)

        if self.continuous_action_space:
            action_cov = torch.diag(self.action_var)
            dist = MultivariateNormal(action_out, action_cov)
            action_logprobs = dist.log_prob(actions)
        else:
            dist = Categorical(action_out)
            action_logprobs = dist.log_prob(actions.squeeze(-1).long())
        dist_entropy = dist.entropy()

        return values.squeeze(), action_logprobs, dist_entropy


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
            device='cpu'
        ):
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_std_init = action_std_init
        self.continuous_action_space = continuous_action_space
        self.device = device

        self.policy = ActorCritic(
            obs_dim, 
            action_dim, 
            hidden_dim, 
            continuous_action_space=continuous_action_space,
            action_std_init=action_std_init,
            device=device,
        )

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.feature_extractor.parameters()},
            {'params': self.policy.actor_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic}
        ])

        self.buffer = RolloutBuffer()
        self.mse_loss = nn.MSELoss()  # Initialize MSE loss


    def compute_returns(self):
        returns = []
        discounted_reward = 0

        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = np.array(returns, dtype=np.float32)
        returns = torch.flatten(torch.from_numpy(returns).float()).to(self.device)
        return returns


    def update_policy(self):
        # print(len(self.buffer.rewards))
        rewards_to_go = self.compute_returns()
        # print(len(rewards_to_go))

        states = torch.from_numpy(np.array(self.buffer.states)).float().to(self.device)
        actions = torch.from_numpy(np.array(self.buffer.actions)).float().to(self.device)
        old_logprobs = torch.from_numpy(np.array(self.buffer.logprobs)).float().to(self.device)
        state_vals = torch.from_numpy(np.array(self.buffer.state_values)).float().to(self.device)

        # print('stage-0:', rewards_to_go.shape, state_vals.shape)
        # print('stage-1:', rewards_to_go.device, state_vals.device)
        advantages = rewards_to_go - state_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # print(states.shape, actions.shape, old_logprobs.shape, state_vals.shape, advantages.shape, rewards_to_go.shape)

        for _ in range(self.num_epochs):
            # generate random indices for minibatch
            indices = np.random.permutation(len(self.buffer.states))

            for start_idx in range(0, len(states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_rewards_to_go = rewards_to_go[batch_indices]
                
                # evaluate old actions and values
                state_values, logprobs, dist_entropy = self.policy.evaluate_actions(batch_states, batch_actions)
                # print(logprobs.shape, batch_old_logprobs.shape)

                # Finding the ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(logprobs - batch_old_logprobs.squeeze(-1))

                # Finding Surrogate Loss
                # print(ratios.shape, batch_advantages.shape)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages

                # final loss of clipped objective PPO
                actor_loss = -torch.min(surr1, surr2).mean()
                # print(state_values.dtype, batch_rewards_to_go.dtype)
                critic_loss = 0.5 * self.mse_loss(state_values.squeeze(), batch_rewards_to_go)
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * dist_entropy.mean()
                # print("Final loss:", actor_loss, critic_loss, dist_entropy, loss)

                # calculate gradients and backpropagate for actor network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        self.buffer.clear()