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

"""
Todo:
 read and remove useless ai comments
"""



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


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.state_values = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.state_values[:]

    def store_transition(self, state, action, logprob, reward, done, state_value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.state_values.append(state_value)


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