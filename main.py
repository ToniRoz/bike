import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Import your trainers
from train_rainbow import train as train_rainbow
from train_ppo import train as train_ppo


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("🔥 Hydra main() was called")

    print("\n" + "=" * 60)
    print(f"🚀 Training {cfg.alg.env_name if 'env_name' in cfg.alg else 'Agent'}")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # Instantiate environment
    env = instantiate(cfg.env)

    # Choose algorithm dynamically
    alg_name = cfg.alg.get("exp_name", "").lower()

    if "rainbow" in alg_name:
        train_rainbow(cfg.alg, env)
    elif "ppo" in alg_name:
        train_ppo(cfg.alg, env)
    else:
        raise ValueError(f"Unknown algorithm type: {alg_name}")

if __name__ == "__main__":
    main()
