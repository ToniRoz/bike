"""
train_ppo_vectorized_temporal.py

Modified version of train_ppo_vectorized.py with frame stacking support.
Copy this to your project and use it for frame stacking experiments.

Usage:
    # Baseline
    python train_ppo_vectorized_temporal.py env.state_space_selection=rimpoints
    
    # Frame stacking
    python train_ppo_vectorized_temporal.py env.state_space_selection=rimpoints +frame_stack_size=4
    
    # LSTM/GRU (already supported)
    python train_ppo_vectorized_temporal.py env.state_space_selection=rimpoints use_recurrent=true recurrent_type=lstm
"""

import os
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from Environment.wheel_env import WheelEnv
from trainers import PPOTrainer
from config import TrainingConfig

# Import vectorized components
from wheel_env_vectorized import SubprocVecEnv, DummyVecEnv
from ppo_trainer_vectorized import VectorizedPPOTrainer

# Import frame stacking
try:
    from frame_stacking import VecFrameStackWrapper
    FRAME_STACK_AVAILABLE = True
except ImportError:
    FRAME_STACK_AVAILABLE = False
    print("[Warning] frame_stacking.py not found - frame stacking disabled")


def setup_logging(output_dir: str, use_tensorboard: bool = True):
    """Setup TensorBoard logging inside Hydra run dir"""
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(output_dir, "tensorboard")
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            print(f"Logging to: {log_dir}")
            print(f"TensorBoard command: tensorboard --logdir {output_dir}")
            return writer
        except ImportError:
            print("Warning: tensorboard not available, using DummyWriter")
            return None
    return None


def train(cfg: DictConfig, output_dir: str = None):
    """Train PPO agent using Hydra with vectorized environments"""
    print("\n" + "=" * 50)
    print("Training PPO (Vectorized) - Temporal Support")
    print("=" * 50)

    # Print current configuration
    print(OmegaConf.to_yaml(cfg))

    ##############################
    # 1. Set seed and device
    ##############################
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    cfg.device = str(device)
    print(f"Using device: {device}")

    ##############################
    # 2. Setup logging
    ##############################
    if output_dir is None:
        try:
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        except:
            output_dir = "outputs/default_run"
            os.makedirs(output_dir, exist_ok=True)
    
    writer = setup_logging(str(output_dir))

    ##############################
    # 3. Create VECTORIZED environment
    ##############################
    n_envs = getattr(cfg, 'n_envs', 8)
    use_subproc = getattr(cfg, 'use_subproc', True)
    
    print(f"\nCreating {n_envs} parallel environments...")
    print(f"Using {'SubprocVecEnv' if use_subproc else 'DummyVecEnv'}")
    
    # Extract environment kwargs from config
    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    env_kwargs.pop('_target_', None)
    
    print(f"Environment kwargs: {env_kwargs}")
    
    # Create environment factory function
    def make_env(seed):
        def _init():
            env = WheelEnv(**env_kwargs)
            return env
        return _init
    
    # Create vectorized environment
    env_fns = [make_env(cfg.random_seed + i) for i in range(n_envs)]
    
    if use_subproc:
        vec_env = SubprocVecEnv(env_fns, start_method='spawn')
    else:
        vec_env = DummyVecEnv(env_fns)
    
    ##############################
    # 3b. Apply Frame Stacking (NEW)
    ##############################
    frame_stack_size = getattr(cfg, 'frame_stack_size', 0)
    if frame_stack_size > 1:
        if FRAME_STACK_AVAILABLE:
            include_actions = getattr(cfg, 'frame_stack_include_actions', True)
            print(f"\n[Frame Stacking] Applying wrapper:")
            print(f"    Stack size: {frame_stack_size}")
            print(f"    Include actions: {include_actions}")
            vec_env = VecFrameStackWrapper(vec_env, stack_size=frame_stack_size, include_actions=include_actions)
        else:
            print("[ERROR] Frame stacking requested but frame_stacking.py not found!")
            raise ImportError("frame_stacking.py required for frame stacking")
    
    print(f"\nVectorized environment created:")
    print(f"  - Observation space: {vec_env.observation_space}")
    print(f"  - Action space: {vec_env.action_space}")
    print(f"  - Num envs: {n_envs}")

    ##############################
    # 4. Create VECTORIZED trainer
    ##############################
    trainer = VectorizedPPOTrainer(
        config=cfg,
        vec_env=vec_env,
        writer=writer,
        output_dir=output_dir
    )

    ##############################
    # 5. Train or evaluate
    ##############################
    mode = getattr(cfg, "mode", "train")
    if mode == "train":
        trainer.train()
    else:
        print("Note: Evaluation uses single environment")
        single_env = WheelEnv(**env_kwargs)
        single_trainer = PPOTrainer(cfg, single_env, writer, output_dir=output_dir)
        single_trainer.evaluate()
        single_env.close()

    ##############################
    # 6. Cleanup
    ##############################
    if writer:
        writer.close()
    vec_env.close()


if __name__ == "__main__":
    @hydra.main(config_path="configs", config_name="ppo_default", version_base=None)
    def main(cfg: DictConfig):
        train(cfg)
    
    main()