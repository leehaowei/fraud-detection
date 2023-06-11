from pathlib import Path
from omegaconf import OmegaConf


class PathProcessor:

    def __init__(self, mode):
        self.mode = mode

        match self.mode:
            case "colab":
                self.base = "/content/fraud-detection"
            case "local":
                self.base = ".."

        self.base += "/conf/"

    def get_path_prefix(self) -> str:
        return OmegaConf.load(self.base + "path.yaml")["prefix"][self.mode]

    def get_data_path(self, target_file: str) -> Path:
        bucket = "https://fs-fraud-dectection.s3.eu-central-1.amazonaws.com/data/"
        return Path(bucket + target_file)

    def get_param(self, which: str):
        return OmegaConf.load(self.base + "params.yaml")[which]

    def get_random(self):
        return OmegaConf.load(self.base + "random.yaml")

    def get_mapping(self, which: str):
        dtype_mapping_str = OmegaConf.load(self.base + "params.yaml")[which]
        dtype_mapping = {k: eval(v) for k, v in dtype_mapping_str.items()}
        return dtype_mapping

