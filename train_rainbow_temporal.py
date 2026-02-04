

import os
import random
import time
import numpy as np
import torch

num_runs = int(os.getenv("NUM_PARALLEL_RUNS", "1"))
fraction = 0.9 / num_runs
torch.cuda.set_per_process_memory_fraction(fraction)
print(f"[GPU] Using {fraction:.2f} of total VRAM per process")

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from trainers import RainbowTrainer  
from config import TrainingConfig  

# Import frame stacking
try:
    from frame_stacking import FrameStackWrapper
    FRAME_STACK_AVAILABLE = True
except ImportError:
    FRAME_STACK_AVAILABLE = False
    print("[Warning] frame_stacking.py not found - frame stacking disabled")


def setup_logging(output_dir: str, use_tensorboard: bool = True):
    """Setup tensorboard logging with Hydra output dir"""
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
    """Train Rainbow DQN using Hydra"""
    print("\n" + "=" * 50)
    print("Training Rainbow DQN - Temporal Support")
    print("=" * 50)

    # Print config summary
    print(OmegaConf.to_yaml(cfg))

    ##############################
    # 1. Set seeds and device
    ##############################
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = cfg.enable_cudnn

    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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
    # 3. Create environment
    ##############################
    env = instantiate(cfg.env)
    
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
            env = FrameStackWrapper(env, stack_size=frame_stack_size, include_actions=include_actions)
            # Override history_length since stacking is handled by wrapper
            cfg.history_length = 1
        else:
            print("[ERROR] Frame stacking requested but frame_stacking.py not found!")
            raise ImportError("frame_stacking.py required for frame stacking")
    
    print(f"\nEnvironment created:")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    
    ##############################
    # 4. Create trainer
    ##############################
    trainer = RainbowTrainer(cfg, env, writer, output_dir=output_dir)
    
    ##############################
    # 5. Train or evaluate
    ##############################
    mode = getattr(cfg, "mode", "train")
    if mode == "train":
        trainer.train()
    else:
        trainer.evaluate()

    ##############################
    # 6. Cleanup
    ##############################
    if writer:
        writer.close()
    env.close()


if __name__ == "__main__":
    @hydra.main(config_path="configs", config_name="rainbow_default", version_base=None)
    def main(cfg: DictConfig):
        train(cfg)
    
    main()