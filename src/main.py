# my_script.py
from omegaconf import DictConfig
import hydra


@hydra.main(config_path="../conf", config_name="model")
def my_app(cfg: DictConfig) -> None:
    print(cfg.model.random_state)


if __name__ == "__main__":
    my_app()
