# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical


from Models import DQN


class RainbowAgent:
  def __init__(self, args, env):
      self.env = env
      self.action_space = 72
      self.device = args.device
      
      # IQN parameters (replacing C51 parameters)
      self.N = args.N_quantiles  # Number of quantiles for training
      self.K = args.K_quantiles  # Number of quantiles for evaluation
      self.kappa = args.kappa    # Huber loss threshold
      self.embedding_dim = args.embedding_dim
      
      # Training parameters
      self.batch_size = args.batch_size
      self.n = args.multi_step
      self.discount = args.discount

      self.online_net = DQN(args, self.action_space).to(device=args.device)
      
      if args.model_path:  # Load pretrained model if provided
          self.online_net.load_state_dict(torch.load(args.model_path, map_location='cpu'))
      
      self.online_net.train()

      self.target_net = DQN(args, self.action_space).to(device=args.device)
      self.update_target_net()
      self.target_net.train()
      for param in self.target_net.parameters():
          param.requires_grad = False

      self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  def act(self, state):
    if not torch.is_tensor(state) and state is not None:
        state = torch.tensor(state, dtype=torch.float32)
        device = next(self.online_net.parameters()).device
        state = state.to(device)
        with torch.no_grad():
            # Get quantile values: (1, action_space, K)
            quantile_values, _ = self.online_net(state.unsqueeze(0), self.K)
            # Take mean over quantiles to get Q-values: (1, action_space)
            q_values = quantile_values.mean(dim=2)
            # Return action with highest Q-value
            return q_values.argmax(1).item()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

    # Get current state quantile values with online network
    # Shape: (batch_size, action_space, N)
    current_quantiles, current_taus = self.online_net(states, self.N)
    
    # Select quantiles for taken actions
    # Shape: (batch_size, N)
    current_quantiles_a = current_quantiles[range(self.batch_size), actions, :]

    with torch.no_grad():
        # Double-Q learning: use online network to select actions
        next_quantiles_online, _ = self.online_net(next_states, self.N)
        # Get Q-values by averaging over quantiles: (batch_size, action_space)
        next_q_values = next_quantiles_online.mean(dim=2)
        # Select best actions
        next_actions = next_q_values.argmax(1)
        
        # Use target network to evaluate the selected actions
        self.target_net.reset_noise()
        next_quantiles_target, _ = self.target_net(next_states, self.N)
        # Select quantiles for best actions: (batch_size, N)
        next_quantiles_a = next_quantiles_target[range(self.batch_size), next_actions, :]
        
        # Compute target quantiles with Bellman equation
        target_quantiles = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * next_quantiles_a

    # Compute pairwise TD errors
    # current_quantiles_a: (batch_size, N) -> (batch_size, N, 1)
    # target_quantiles: (batch_size, N) -> (batch_size, 1, N)
    # td_errors: (batch_size, N, N)
    td_errors = current_quantiles_a.unsqueeze(2) - target_quantiles.unsqueeze(1)
    
    # Compute element-wise Huber loss
    abs_errors = torch.abs(td_errors)
    huber_loss = torch.where(
        abs_errors <= self.kappa,
        0.5 * td_errors ** 2,
        self.kappa * (abs_errors - 0.5 * self.kappa)
    )
    
    # Compute quantile weights: |τ - 1(td_error < 0)|
    # current_taus shape: (batch, N) -> (batch, N, 1) for broadcasting
    taus_expanded = current_taus.unsqueeze(2)
    quantile_weight = torch.abs(taus_expanded - (td_errors < 0).float())
    
    # Apply quantile weighting to Huber loss
    # Shape: (batch, N, N)
    quantile_huber = quantile_weight * huber_loss
    
    # Sum over target quantiles (dim=2), mean over current quantiles (dim=1)
    # Shape: (batch,)
    loss_per_sample = quantile_huber.sum(dim=2).mean(dim=1) / self.N
    
    # Apply importance sampling weights and compute final loss
    weighted_loss = (weights * loss_per_sample).mean()
    
    # Backpropagation
    self.online_net.zero_grad()
    weighted_loss.backward()
    clip_grad_norm_(self.online_net.parameters(), 10)
    self.optimiser.step()

    # Update priorities with loss values
    mem.update_priorities(idxs, loss_per_sample.detach().cpu().numpy())

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    # Convert numpy state to tensor if needed 
    if not torch.is_tensor(state):
        state = torch.tensor(state, dtype=torch.float32)
    
    device = next(self.online_net.parameters()).device
    state = state.to(device)
    
    with torch.no_grad():
        # Get quantile values: (1, action_space, K)
        quantile_values, _ = self.online_net(state.unsqueeze(0), self.K)
        # Take mean over quantiles and max over actions
        return quantile_values.mean(dim=2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()



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

        # print("Observation dim:", obs.dim())
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