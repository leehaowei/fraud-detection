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

    def get_data_path(self, target_file: str) -> str:
        bucket = "https://fs-fraud-dectection.s3.eu-central-1.amazonaws.com/data/"
        return bucket + target_file

    def get_feature_params(self, which: str):
        return OmegaConf.load(self.base + "feature_params.yaml")[which]

    def get_filter_params(self):
        return OmegaConf.load(self.base + "filter_params.yaml")

    def get_mapping(self, which: str):
        dtype_mapping_str = OmegaConf.load(self.base + "feature_params.yaml")[which]
        dtype_mapping = {k: eval(v) for k, v in dtype_mapping_str.items()}
        return dtype_mapping

