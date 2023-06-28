import chardet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import requests


def get_violation_years(gvkey: str, gvkeys_icw: dict) -> tuple:
    return gvkeys_icw[gvkey] if gvkey in gvkeys_icw else (0, 0)


def check_if_icw(gvkey: str, gvkeys_icw) -> bool:
    return True if gvkey in gvkeys_icw else False


def remove_nan(df, col):
    return df.dropna(subset=[col])


def remove_by_year(df, year: int):
    df = df.copy()
    df["datadate"] = pd.to_datetime(df["datadate"])
    date_filter = df["datadate"].dt.year <= year
    return df[date_filter]


def add_min_max_years(df):
    df = df.copy()

    # Calculate the earliest and latest years for each 'gvkey' group
    min_year = df.groupby("gvkey")[["fyear", "bv_year", "ev_year"]].min().min(axis=1)
    max_year = df.groupby("gvkey")[["fyear", "bv_year", "ev_year"]].max().max(axis=1)

    # Convert these to DataFrames so they can be joined with the original DataFrame
    min_year = min_year.to_frame(name="earliest_year")
    max_year = max_year.to_frame(name="latest_year")

    # Join these DataFrames with the original DataFrame
    df = df.join(min_year, on="gvkey")
    df = df.join(max_year, on="gvkey")

    return df


def rename_motive(df):
    df = df.copy()
    motive_filter = df["motive"] == "Weak internal control"

    df.loc[motive_filter, "motive"] = "1"
    df.loc[~motive_filter, "motive"] = "0"

    return df


def create_dict(df):
    data_dict = {}

    categories = {"1": "internal_control_weakness", "0": "other_motive"}

    for value, category in categories.items():
        filtered_df = df[df["motive"] == value]
        inner_dict = filtered_df.set_index("gvkey")[
            ["earliest_year", "latest_year"]
        ].to_dict("split")
        inner_dict = {
            k: list(v) for k, v in zip(inner_dict["index"], inner_dict["data"])
        }
        data_dict[category] = inner_dict

    return data_dict


def load_yaml(file):
    with open(file, "r") as f:
        data_dict = yaml.safe_load(f)
    return data_dict


def load_yaml_from_public_s3(url):
    """Load a YAML file from a public S3 bucket."""
    response = requests.get(url)

    try:
        data = yaml.safe_load(response.text)
        return data
    except yaml.YAMLError as error:
        print(f"Error parsing YAML file: {error}")
        return None


def get_bounding(target_value: float, bound: float):
    return target_value * bound


def save_yaml(target, file):
    with open(file, "w") as file:
        yaml.dump(target, file)


def get_keys(data_dict, category):
    return list(data_dict[category].keys())


def get_encoding(file):
    rawdata = open(file, "rb").read()
    result = chardet.detect(rawdata)
    return result["encoding"]


def create_non_fraud_dict(gvkey_list):
    data_dict = {"non_fraud": {gvkey: [] for gvkey in gvkey_list}}
    return data_dict


def get_nunique(df: pd.DataFrame, feature: str):
    return df[feature].nunique()


def get_quantile(df: pd.DataFrame, feature: str, pct: float):
    return df[feature].quantile(pct)


def plot_distribution(df, column, max_value=None):
    """
    Plot the distribution of a specific column in a DataFrame up to a maximum value.

    Parameters:
    df : pandas DataFrame
    column : str, name of the column to plot
    max_value : float or int, optional, maximum value to consider in the plot
    """
    # Filter the DataFrame based on max_value if provided
    if max_value is not None:
        df = df[df[column] <= max_value]

    # Create a Figure and Axes
    fig, ax = plt.subplots()

    # Plot the distribution using seaborn
    sns.histplot(data=df, x=column, kde=True, ax=ax)

    # Set title and labels
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    # Show the plot
    plt.show()
