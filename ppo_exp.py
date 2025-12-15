import os
from pathlib import Path
from hydra import initialize, compose
from train_ppo import train


def run_experiment(reward, use_rnn, state_space, action_space, outdir):

    # Determine if the action space is continuous
    continuous_flag = str(action_space in ["continous", "all_spokes"]).lower()

    # Initialize Hydra
    with initialize(config_path="configs", version_base=None):
        cfg = compose(
            config_name="ppo_default",
            overrides=[
                f"use_recurrent={use_rnn}",
                f"env.reward_func={reward}",
                f"env.action_space_selection={action_space}",
                f"env.state_space_selection={state_space}",
                f"continuous_action_space={continuous_flag}",
                f"exp_name=ppo_{reward}_rnn_{use_rnn}_{state_space}_{action_space}",
                "device=cuda"
            ],
            return_hydra_config=True,
        )

    # Setup experiment folder
    exp_name = f"ppo_reward_{reward}_rnn_{use_rnn}_{state_space}_{action_space}"
    output_dir = Path(outdir) / "exp_folder" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override log/checkpoint directories
    cfg.log_dir = str(output_dir / "logs")
    cfg.ckpt_dir = str(output_dir / "checkpoints")

    # Call train
    train(cfg, output_dir=str(output_dir))


if __name__ == "__main__":
    reward_funcs = ["normalized", "raw", "spokes"]
    state_spaces = ["rimandspokes", "rimpoints", "spoketensions", "fourier", "fourier_and_spokes"]
    use_rnn_vals = [ "false"]
    action_spaces = ["discrete", "continous", "all_spokes"]

    outdir = "ppo_exp"

    for reward in reward_funcs:
        for rnn_flag in use_rnn_vals:
            for state_space in state_spaces:
                for act in action_spaces:

                    exp_name = f"ppo_reward_{reward}_rnn_{rnn_flag}_{state_space}_{act}"
                    exp_dir = Path(outdir) / "exp_folder" / exp_name

                    if exp_dir.exists():
                        continue

                    run_experiment(
                        reward,
                        rnn_flag,
                        state_space,
                        act,
                        outdir
                    )
