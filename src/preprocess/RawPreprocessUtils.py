import chardet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import requests


# Setting motive and filtering data
def process_icw_keys(df, gvkeys_icw):
    # Loop over each ICW key to add a 'motive' column and drop certain records
    for gvkey, year_list in gvkeys_icw.items():
        # Various filters for key and year range
        gvkey_filter = df["gvkey"] == gvkey
        year_filter = df["year"].between(year_list[0], year_list[1], inclusive=True)
        filters = gvkey_filter & year_filter

        df.loc[
            filters, "motive"
        ] = 1  # Set 'motive' to 1 for records matching the filters

        filter_to_drop = (
            df["year"] > year_list[1]
        ) & gvkey_filter  # Create a filter for records to drop

        filtered_df = df[~filter_to_drop]  # Apply the filter

        df = filtered_df.copy()
    return df


# Finding comparable companies
def find_comparable_companies(df, gvkeys_icw, gvkeys_non_fraud_temp):
    # Loop over each fraud gvkey to find a comparable company
    comparable_gvkey_dict = {}
    for fraud_gvkeys in list(gvkeys_icw.keys()):
        # Various setup and filters
        temp_dict = {}
        non_fraud_filter = df["gvkey"].isin(list(gvkeys_non_fraud_temp.keys()))
        non_fraud_df_temp = df.loc[non_fraud_filter, :]
        fraud_df_temp = df[df["gvkey"] == fraud_gvkeys].iloc[-1]
        last_fraud_year = int(fraud_df_temp["year"])
        temp_dict["comparable_year"] = last_fraud_year
        at_lower_bound = fraud_df_temp["at_lower_bound"]
        at_upper_bound = fraud_df_temp["at_upper_bound"]
        year_filter = non_fraud_df_temp["year"] == last_fraud_year
        at_filter = non_fraud_df_temp["at"].between(
            at_lower_bound, at_upper_bound, inclusive=True
        )
        filters = year_filter & at_filter

        # Select the gvkey of the comparable company
        non_fraud_gvkey_pass = non_fraud_df_temp[filters]["gvkey"]
        selected = non_fraud_gvkey_pass.iloc[0]

        # Update the temporary dictionary and remove the selected key from the temporary non-fraud keys
        temp_dict["comparable_company"] = selected
        comparable_gvkey_dict[fraud_gvkeys] = temp_dict
        gvkeys_non_fraud_temp.pop(selected)
    return comparable_gvkey_dict


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


def add_percentile_columns(
    df: pd.DataFrame, target_column: str, lower_bound: float, upper_bound: float
) -> pd.DataFrame:
    df[f"{target_column}_lower_bound"] = df[target_column] * lower_bound
    df[f"{target_column}_upper_bound"] = df[target_column] * upper_bound
    return df


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
