import os
from pathlib import Path
from hydra import initialize, compose
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from train_rainbow import train  

def run_experiment(reward, hidden_dim, outdir):
    # Initialize Hydra
    with initialize(config_path="configs", version_base=None):
        cfg = compose(
            config_name="rainbow_default",  
            overrides=[
                f"hidden_size={hidden_dim}",     # override hidden_dim directly
                f"env.reward_func={reward}"    # override reward_func in env config
            ],
            return_hydra_config=False,
        )

    # Setup experiment folder
    exp_name = f"rainbow_reward_{reward}_dim_{hidden_dim}"
    output_dir = Path(outdir) / "exp_folder_more_state" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")

    # Call train with output_dir parameter
    train(cfg, output_dir=str(output_dir))


if __name__ == "__main__":
    reward_funcs = ["raw", "spoke", "normalized", "percentage"]
    hidden_dims = [1000, 500]

    for reward in reward_funcs:
        for hidden_dim in hidden_dims:
            outdir = "outputs"
            os.makedirs(outdir, exist_ok=True)
            run_experiment(reward, hidden_dim, outdir)
            print(f"\n✓ Completed: reward={reward}, hidden_dim={hidden_dim}\n")