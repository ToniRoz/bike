#!/usr/bin/env python3
"""
Test script for Recurrent Rainbow DQN implementation
Tests both standard and recurrent modes
"""

import torch
import numpy as np
from collections import namedtuple

# Mock config class
class MockConfig:
    def __init__(self, use_recurrent=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history_length = 1
        self.hidden_size = 128
        self.noisy_std = 0.1
        self.embedding_dim = 32
        self.num_quantiles = 8
        self.num_quantiles_eval = 8
        self.kappa = 1.0
        self.learning_rate = 0.001
        self.adam_eps = 1e-4
        self.discount = 0.99
        self.multi_step = 3
        self.priority_weight = 0.4
        self.priority_exponent = 0.5
        self.batch_size = 4
        
        # Recurrent settings
        self.use_recurrent = use_recurrent
        self.recurrent_sequence_length = 4
        self.recurrent_hidden_size = 64
        self.recurrent_type = "gru"
        self.recurrent_layers = 1
        self.recurrent_dropout = 0.0


def test_models():
    """Test both DQN and RecurrentDQN forward passes"""
    print("="*60)
    print("Testing Models...")
    print("="*60)
    
    from Models import DQN, RecurrentDQN
    
    action_space = 5
    state_dim = 10
    batch_size = 4
    
    # Test standard DQN
    print("\n1. Testing Standard DQN:")
    config = MockConfig(use_recurrent=False)
    dqn = DQN(config, action_space, state_dim).to(config.device)
    
    # Input: (batch, history * state_dim)
    input_dim = config.history_length * state_dim
    test_input = torch.randn(batch_size, input_dim).to(config.device)
    
    q_values, taus = dqn(test_input, num_quantiles=config.num_quantiles)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Q-values shape: {q_values.shape}")
    print(f"   Taus shape: {taus.shape}")
    print(f"   ✓ Standard DQN test passed!")
    
    # Test Recurrent DQN
    print("\n2. Testing Recurrent DQN:")
    config_rec = MockConfig(use_recurrent=True)
    rec_dqn = RecurrentDQN(config_rec, action_space, state_dim).to(config_rec.device)
    
    # Input: (batch, seq_len, state_dim + action_space)
    seq_len = config_rec.recurrent_sequence_length
    input_dim_rec = state_dim + action_space
    test_input_rec = torch.randn(batch_size, seq_len, input_dim_rec).to(config_rec.device)
    
    q_values_rec, taus_rec = rec_dqn(test_input_rec, num_quantiles=config_rec.num_quantiles)
    print(f"   Input shape: {test_input_rec.shape}")
    print(f"   Q-values shape: {q_values_rec.shape}")
    print(f"   Taus shape: {taus_rec.shape}")
    print(f"   ✓ Recurrent DQN test passed!")
    
    # Test with mask
    print("\n3. Testing Recurrent DQN with mask:")
    mask = torch.ones(batch_size, seq_len).to(config_rec.device)
    mask[:, :2] = 0  # Mask first 2 timesteps
    
    q_values_masked, _ = rec_dqn(test_input_rec, num_quantiles=config_rec.num_quantiles, mask=mask)
    print(f"   Q-values with mask shape: {q_values_masked.shape}")
    print(f"   ✓ Masking test passed!")


def test_memory():
    """Test Memory sampling for both standard and recurrent modes"""
    print("\n" + "="*60)
    print("Testing Memory...")
    print("="*60)
    
    from Memory import ReplayMemory
    
    state_shape = (10,)
    capacity = 100
    
    # Test standard memory
    print("\n1. Testing Standard Memory:")
    config = MockConfig(use_recurrent=False)
    memory = ReplayMemory(config, capacity, state_shape)
    
    # Add some transitions
    for i in range(20):
        state = np.random.randn(*state_shape).astype(np.float32)
        action = np.random.randint(0, 5)
        reward = np.random.randn()
        terminal = (i % 10 == 9)  # Terminal every 10 steps
        memory.append(state, action, reward, terminal)
    
    # Sample a batch
    batch = memory.sample(batch_size=4)
    print(f"   Number of returned values: {len(batch)}")
    print(f"   States shape: {batch[1].shape}")
    print(f"   Actions shape: {batch[2].shape}")
    print(f"   ✓ Standard memory sampling test passed!")
    
    # Test recurrent memory
    print("\n2. Testing Recurrent Memory:")
    config_rec = MockConfig(use_recurrent=True)
    memory_rec = ReplayMemory(config_rec, capacity, state_shape)
    
    # Add some transitions
    for i in range(50):  # Need more transitions for sequence sampling
        state = np.random.randn(*state_shape).astype(np.float32)
        action = np.random.randint(0, 5)
        reward = np.random.randn()
        terminal = (i % 15 == 14)  # Terminal every 15 steps
        memory_rec.append(state, action, reward, terminal)
    
    # Sample a batch
    batch_rec = memory_rec.sample(batch_size=4)
    print(f"   Number of returned values: {len(batch_rec)}")
    print(f"   State sequences shape: {batch_rec[1].shape}")
    print(f"   Action sequences shape: {batch_rec[2].shape}")
    print(f"   Masks shape: {batch_rec[3].shape}")
    print(f"   ✓ Recurrent memory sampling test passed!")


def test_agent():
    """Test Agent initialization and action selection"""
    print("\n" + "="*60)
    print("Testing Agent...")
    print("="*60)
    
    from Agents import RainbowAgent
    import gymnasium as gym
    
    # Create a simple environment
    env = gym.make('CartPole-v1')
    
    # Test standard agent
    print("\n1. Testing Standard Agent:")
    config = MockConfig(use_recurrent=False)
    agent = RainbowAgent(config, env)
    
    state, _ = env.reset()
    action = agent.act(state)
    print(f"   State shape: {state.shape}")
    print(f"   Selected action: {action}")
    print(f"   ✓ Standard agent test passed!")
    
    # Test recurrent agent
    print("\n2. Testing Recurrent Agent:")
    config_rec = MockConfig(use_recurrent=True)
    agent_rec = RainbowAgent(config_rec, env)
    
    state, _ = env.reset()
    agent_rec.reset_episode()
    
    # Take a few steps to build history
    for i in range(5):
        action = agent_rec.act(state)
        state, reward, done, truncated, _ = env.step(action)
        if done or truncated:
            state, _ = env.reset()
            agent_rec.reset_episode()
    
    print(f"   State shape: {state.shape}")
    print(f"   Action history length: {len(agent_rec.action_history)}")
    print(f"   State history length: {len(agent_rec.state_history)}")
    print(f"   ✓ Recurrent agent test passed!")
    
    env.close()


def test_sequence_construction():
    """Test input sequence construction for recurrent agent"""
    print("\n" + "="*60)
    print("Testing Sequence Construction...")
    print("="*60)
    
    from Agents import RainbowAgent
    import gymnasium as gym
    
    env = gym.make('CartPole-v1')
    config = MockConfig(use_recurrent=True)
    agent = RainbowAgent(config, env)
    
    state, _ = env.reset()
    agent.reset_episode()
    
    print(f"\n1. Initial state (no history):")
    input_seq = agent._construct_input_sequence(state)
    print(f"   Input sequence shape: {input_seq.shape}")
    print(f"   Expected: (1, {config.recurrent_sequence_length}, {agent.state_dim + agent.action_space})")
    
    # Take a few actions to build history
    for i in range(3):
        action = agent.act(state)
        state, _, done, truncated, _ = env.step(action)
        if done or truncated:
            break
    
    print(f"\n2. After {i+1} actions:")
    print(f"   Action history: {list(agent.action_history)}")
    print(f"   State history length: {len(agent.state_history)}")
    
    input_seq = agent._construct_input_sequence(state)
    print(f"   Input sequence shape: {input_seq.shape}")
    print(f"   ✓ Sequence construction test passed!")
    
    env.close()


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print(" RECURRENT RAINBOW DQN - TEST SUITE")
    print("="*60)
    
    try:
        test_models()
        test_memory()
        test_agent()
        test_sequence_construction()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)