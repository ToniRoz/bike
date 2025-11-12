import os
from pathlib import Path
from hydra import initialize, compose
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from train_ppo import train  # the plain training function, not @hydra.main

def run_experiment(reward, hidden_dim, outdir):
    # Initialize Hydra
    with initialize(config_path="configs", version_base=None):
        cfg = compose(
            config_name="ppo_default",  # your PPO config
            overrides=[
                f"hidden_dim={hidden_dim}",     # override hidden_dim directly
                f"env.reward_func={reward}"    # override reward_func in env config
            ],
            return_hydra_config=False,
        )

    # Setup experiment folder
    exp_name = f"ppo_reward_{reward}_dim_{hidden_dim}"
    output_dir = Path(outdir) / "exp_folder" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    tb_dir = output_dir / "tensorboard"
    tb_dir.mkdir(exist_ok=True)

    # Call train
    train(cfg, output_dir)


if __name__ == "__main__":
    reward_funcs = ["raw", "spoke", "normalized", "percentage"]
    hidden_dims = [64, 128, 256, 500]

    for reward in reward_funcs:
        for hidden_dim in hidden_dims:
            outdir = "outputs"
            os.makedirs(outdir, exist_ok=True)
            run_experiment(reward, hidden_dim, outdir)
