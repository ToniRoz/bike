import os
from hydra import initialize_config_dir, compose
from pathlib import Path
from train_ppo_vectorized import train

CONFIG_DIR = Path.cwd() / "configs"

# GPU memory allocation (SAME LOGIC AS RAINBOW)
import torch
num_runs = int(os.getenv("NUM_PARALLEL_RUNS", "1"))
torch.cuda.set_per_process_memory_fraction(0.9 / num_runs)

def main():
    import sys

    overrides = sys.argv[1:]
    output_dir = None

    for o in overrides:
        if o.startswith("output_dir="):
            output_dir = o.split("=", 1)[1]

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.absolute()),
        version_base=None
    ):
        cfg = compose(
            config_name="ppo_default",
            overrides=[o for o in overrides if not o.startswith("output_dir=")],
            return_hydra_config=True,
        )

    train(cfg, output_dir=output_dir)

if __name__ == "__main__":
    main()
