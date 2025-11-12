"""
Configuration management for RL agents
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
import yaml
import json
from pathlib import Path


@dataclass
class RainbowConfig:
    """Configuration for Rainbow DQN agent"""
    # Experiment
    id: str = 'default'
    seed: int = 123
    
    # Training
    T_max: int = int(5e5)
    max_episode_length: int = 1000
    learn_start: int = int(50e3)
    batch_size: int = 64
    
    # Network Architecture
    history_length: int = 1
    architecture: Literal['canonical', 'data-efficient'] = 'canonical'
    hidden_size: int = 800
    noisy_std: float = 0.1
    
    # Rainbow Components
    atoms: int = 51
    V_min: float = -10.0
    V_max: float = 10.0
    
    # Memory
    memory_capacity: int = int(1e6)
    replay_frequency: int = 4
    priority_exponent: float = 0.5
    priority_weight: float = 0.4
    
    # Multi-step and Discount
    multi_step: int = 3
    discount: float = 0.99
    
    # Optimization
    learning_rate: float = 0.0000625
    adam_eps: float = 1.5e-4
    norm_clip: float = 10.0
    target_update: int = 6000
    
    # Evaluation
    evaluation_interval: int = 100000
    evaluation_episodes: int = 10
    evaluation_size: int = 500
    
    # Checkpointing
    checkpoint_interval: int = 100000
    
    # Misc
    reward_clip: int = 0
    render: bool = False
    device: str = 'cuda'
    enable_cudnn: bool = False
    
    # Paths
    model_path: Optional[str] = None
    memory_path: Optional[str] = None
    results_dir: str = 'results'
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class PPOConfig:
    """Configuration for PPO agent"""
    # Experiment
    env_name: str = "WheelEnv"
    random_seed: int = 42
    exp_name: str = ""
    
    # Training
    num_train_steps: int = 500000
    max_eps_steps: int = 1500
    update_interval: int = 1000
    
    # PPO Hyperparameters
    num_epochs: int = 100
    batch_size: int = 64
    gamma: float = 0.99
    eps_clip: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Network Architecture
    hidden_dim: int = 64
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    activation: str = 'tanh'
    
    # Action Space
    continuous_action_space: bool = True
    action_std_init: float = 0.1
    
    # Logging and Checkpointing
    log_interval: int = 10000
    save_interval: int = 100000
    log_dir: str = "logs"
    ckpt_dir: str = "checkpoints"
    
    # Evaluation
    num_eval_eps: int = 10
    eval_ckpt_name: str = ""
    render_mode: bool = False
    
    # Device
    device: str = 'cpu'
    
    # Misc
    verbose: bool = False
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)




@dataclass
class TrainingConfig:
    """General training configuration"""
    algorithm: Literal['rainbow', 'ppo', 'tdmpc2'] = 'ppo'
    mode: Literal['train', 'eval'] = 'train'
    
    # Experiment tracking
    use_tensorboard: bool = True
    log_dir: str = 'runs'
    experiment_name: Optional[str] = None
    
    # Optuna (for hyperparameter optimization)
    use_optuna: bool = False
    optuna_study_name: str = 'rl_optimization'
    optuna_n_trials: int = 100
    optuna_timeout: Optional[int] = None
    
    def to_dict(self):
        return asdict(self)


def load_config(algorithm: str, config_path: Optional[str] = None):
    """
    Load configuration for specified algorithm
    
    Args:
        algorithm: 'rainbow', 'ppo', or 'tdmpc2'
        config_path: Path to custom config file. If None, uses default.
    
    Returns:
        Config object (RainbowConfig, PPOConfig, or TDMPC2Config)
    """
    algorithm = algorithm.lower()
    
    if algorithm == 'rainbow':
        if config_path is None:
            return RainbowConfig()
        return RainbowConfig.from_yaml(config_path)
    elif algorithm == 'ppo':
        if config_path is None:
            return PPOConfig()
        return PPOConfig.from_yaml(config_path)
    elif algorithm == 'tdmpc2':
        if config_path is None:
            return TDMPC2Config()
        return TDMPC2Config.from_yaml(config_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def create_default_configs():
    """Create default configuration files"""
    Path('configs').mkdir(exist_ok=True)
    
    rainbow_config = RainbowConfig()
    rainbow_config.save_yaml('configs/rainbow_default.yaml')
    
    ppo_config = PPOConfig()
    ppo_config.save_yaml('configs/ppo_default.yaml')
    
    tdmpc2_config = TDMPC2Config()
    tdmpc2_config.save_yaml('configs/tdmpc2_default.yaml')
    
    print("Default configuration files created in 'configs/' directory")


if __name__ == "__main__":
    create_default_configs()