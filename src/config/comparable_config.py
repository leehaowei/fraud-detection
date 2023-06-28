import os


class ComparableConfig:
    def __init__(self):
        self.n_records = int(os.getenv("N_RECORDS"))
        self.column_compared = os.getenv("TARGET_PERCENTILE_COLUMN")
        self.lower_bound = float(os.getenv("PERCENTILE_LOWER_BOUND"))
        self.upper_bound = float(os.getenv("PERCENTILE_UPPER_BOUND"))
        self.mode = os.getenv("MODE")
        self.log_file_path = os.getenv("LOG_FILE_PATH")
        self.file_path = ComparableConfig.get_file_path()

    @staticmethod
    def get_file_path():
        file_path_dict = {
            "wrds": "wrds.parquet",
            "gvkeys": "gvkeys.yaml",
        }
        return file_path_dict
