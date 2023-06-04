from pathlib import Path
from omegaconf import OmegaConf


class PathProcessor:

    def __init__(self, mode, target_file):
        self.mode = mode
        self.target_file = target_file

        match self.mode:
            case "colab":
                self.base = "/content/drive/MyDrive/fs/thesis/fraud-detection"
            case "local":
                self.base = ".."

        self.base += "/conf/"

    def get_path_prefix(self) -> str:
        return OmegaConf.load(self.base + "path.yaml")["prefix"][self.mode]

    def get_full_path(self) -> Path:
        return Path(self.get_path_prefix() + self.target_file)

    def get_param(self, which: str):
        return OmegaConf.load(self.base + "params.yaml")[which]

    def get_random(self):
        return OmegaConf.load(self.base + "random.yaml")

