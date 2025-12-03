import os
from pathlib import Path
from hydra import initialize, compose
from omegaconf import OmegaConf
from train_ppo import train

def run_experiment(reward, hidden_dim, state_space, outdir):
    # Initialize Hydra
    with initialize(config_path="configs", version_base=None):
        cfg = compose(
            config_name="ppo_default",
            overrides=[
                f"hidden_dim={hidden_dim}",
                f"env.reward_func={reward}",                    # Remove + prefix
                f"env.state_space_selection={state_space}",    # Remove + prefix
                f"exp_name=ppo_{reward}_{hidden_dim}_{state_space}",
                "device=cuda"  # Force cuda
            ],
            return_hydra_config=False,
        )

    # Setup experiment folder
    exp_name = f"ppo_reward_{reward}_dim_{hidden_dim}_{state_space}"
    output_dir = Path(outdir) / "exp_folder" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override log/checkpoint directories
    cfg.log_dir = str(output_dir / "logs")
    cfg.ckpt_dir = str(output_dir / "checkpoints")

    # Call train
    train(cfg, output_dir=str(output_dir))


if __name__ == "__main__":
    reward_funcs = ["percentage", "raw", "spokes"]
    state_spaces = ["rimandspokes", "rimpoints", "spoketensions"]
    hidden_dims = [250, 500]

    for reward in reward_funcs:
        for hidden_dim in hidden_dims:
            for state_space in state_spaces:
                outdir = "outputs"
                os.makedirs(outdir, exist_ok=True)
                try:
                    run_experiment(reward, hidden_dim, state_space, outdir)
                    print(f"\n✓ Completed: reward={reward}, hidden_dim={hidden_dim}, state={state_space}\n")
                except Exception as e:
                    print(f"\n✗ Failed: reward={reward}, hidden_dim={hidden_dim}, state={state_space}")
                    print(f"Error: {e}\n")