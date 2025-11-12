"""
Main training script for RL agents
"""
import os
import time
import random
import argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

# Dummy writer for environments without tensorboard
class DummyWriter:
    def add_scalar(self, *args, **kwargs):
        pass
    def add_scalars(self, *args, **kwargs):
        pass
    def close(self):
        pass

writer = DummyWriter()  # Default writer

from config import RainbowConfig, PPOConfig, TrainingConfig, load_config
from trainers import RainbowTrainer, PPOTrainer
import sys
#sys.path.insert(0, '/content/project')
#from Environment.wheel_env import WheelEnv
from Environment.wheel_env import WheelEnv

def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    # JAX uses PRNG keys, handled per-trainer


def get_device(prefer_cuda: bool = True):
    """Get computing device for JAX"""
    devices = jax.devices()
    backend = jax.default_backend()
    
    print(f"JAX backend: {backend}")
    print(f"JAX devices: {devices}")
    
    if backend == 'gpu':
        return 'gpu'
    elif backend == 'tpu':
        return 'tpu'
    else:
        return 'cpu'


def setup_logging(config: TrainingConfig, algo_config):
    """Setup tensorboard logging"""

    if config.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            exp_name = config.experiment_name or f"{config.algorithm}_{time.strftime('%Y%m%d-%H%M%S')}"
            log_dir = os.path.join(config.log_dir, exp_name)
            
            # Create log directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)
            
            writer = SummaryWriter(log_dir=log_dir)
            print(f"Logging to: {log_dir}")
            print(f"TensorBoard command: tensorboard --logdir {config.log_dir}")
            return writer
        except ImportError:
            print("Warning: tensorboard not available, using DummyWriter")
            return None
    return None


def train_rainbow(config: RainbowConfig, training_config: TrainingConfig):
    """Train Rainbow DQN agent"""
    print("\n" + "="*50)
    print("Training Rainbow DQN")
    print("="*50)
    
    # Rainbow still uses PyTorch
    import torch
    
    # Set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Setup device (PyTorch)
    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    config.device = str(device)
    print(f"Using device: {device}")
    
    # Setup logging
    writer = setup_logging(training_config, config)
    
    # Create environment
    env = WheelEnv(reward_func="spoke",action_space_selection="discrete")
    
    # Create trainer
    trainer = RainbowTrainer(config, env, writer)
    
    # Train
    if training_config.mode == 'train':
        trainer.train()
    else:
        trainer.evaluate()
    
    # Cleanup
    if writer:
        writer.close()
    env.close()


def train_ppo(config: PPOConfig, training_config: TrainingConfig):
    """Train PPO agent"""
    print("\n" + "="*50)
    print("Training PPO")
    print("="*50)
    
    # PPO still uses PyTorch
    import torch
    
    # Set seed
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Setup device (PyTorch)
    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    config.device = str(device)
    print(f"Using device: {device}")
    
    # Setup logging
    writer = setup_logging(training_config, config)
    
    # Create environment
    env = WheelEnv(reward_func="spoke",action_space_selection="continous")
    
    # Create trainer
    trainer = PPOTrainer(config, env, writer)
    
    # Train or evaluate
    if training_config.mode == 'train':
        trainer.train()
    else:
        trainer.evaluate()
    
    # Cleanup
    if writer:
        writer.close()
    env.close()





def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train RL agents')
    parser.add_argument('--algorithm', type=str, default='rainbow', 
                       choices=['rainbow', 'ppo', 'tdmpc2'],
                       help='Algorithm to use')
    parser.add_argument('--config', type=str, default='/home/fhg/Desktop/bike/configs/rainbow_default.yaml',
                       help='Path to config file (optional)')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval'],
                       help='Training or evaluation mode')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for this experiment')
    parser.add_argument('--use-optuna', action='store_true',
                       help='Use Optuna for hyperparameter optimization')
    
    args = parser.parse_args()
    
    # Load algorithm config
    algo_config = load_config(args.algorithm, args.config)
    
    # Create training config
    training_config = TrainingConfig(
        algorithm=args.algorithm,
        mode=args.mode,
        experiment_name=args.experiment_name,
        use_optuna=args.use_optuna
    )
    
    # Print configuration
    print("\nTraining Configuration:")
    print("-" * 50)
    for key, value in training_config.to_dict().items():
        print(f"{key:25s}: {value}")
    
    print("\nAlgorithm Configuration:")
    print("-" * 50)
    for key, value in algo_config.to_dict().items():
        print(f"{key:25s}: {value}")
    print("-" * 50 + "\n")
    
    # Train
    if args.algorithm == 'rainbow':
        train_rainbow(algo_config, training_config)
    elif args.algorithm == 'ppo':
        train_ppo(algo_config, training_config)



if __name__ == "__main__":
    main()