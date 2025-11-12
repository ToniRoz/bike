import os
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from Environment.wheel_env import WheelEnv
from trainers import PPOTrainer        # adjust import path as needed
from config import TrainingConfig  # adjust path if needed


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


@hydra.main(config_path="configs", config_name="ppo_default", version_base=None)
def train(cfg: DictConfig):
    """Train PPO agent using Hydra"""
    print("\n" + "=" * 50)
    print("Training PPO (Hydra)")
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
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    writer = setup_logging(output_dir)

    ##############################
    # 3. Create environment
    ##############################
    #env = WheelEnv(reward_func="spoke", action_space_selection="discrete")
    env = instantiate(cfg.env)

    ##############################
    # 4. Create trainer
    ##############################
    trainer = PPOTrainer(cfg, env, writer)

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
    train()
