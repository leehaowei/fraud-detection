import numpy as np
import pandas as pd


def fillna_with_median(df, group_col, target_cols):
    """
    Fill NaN values in the specified columns of a DataFrame with the median of each group.

    Parameters:
    - df: The DataFrame.
    - group_col: The column to group by.
    - target_cols: A list of columns to fill NaN values in.

    Returns:
    - A DataFrame with NaN values filled.
    """
    for col in target_cols:
        medians = df.groupby(group_col)[col].transform("median")
        df.loc[df[col].isna(), col] = medians
    return df


def zero_to_nan(df, columns):
    """
    This function converts zeros in the specified column of the dataframe to NaN.

    Parameters:
        df (pandas.DataFrame): input DataFrame
        column (str): name of the column

    Returns:
        pandas.DataFrame: DataFrame with zeros in the specified column replaced with NaN
    """
    df = df.copy()  # make a copy to avoid modifying the original DataFrame
    for column in columns:
        df.loc[df[column] == 0, column] = np.nan
    return df


def shift_dataframe(df, n, targets: list, use_pct=False, decimals=4):
    df = df.sort_values(["gvkey", "year"])  # Ensure data is sorted
    new_columns = {}  # Initialize an empty dictionary for new columns

    # First pass: Create all shifted columns
    for target in targets:
        for i in range(1, n + 1):
            shifted_column_name = f"{target}_t-{i}"
            new_columns[shifted_column_name] = df.groupby("gvkey")[target].shift(i)

    # Second pass: Calculate percentage changes, if requested
    if use_pct:
        for target in targets:
            for i in range(1, n + 1):
                pct_change_column_name = f"{target}_t-{i}_pct_change"
                if (
                    i == 1
                ):  # for t vs t-1, we directly use target column to compute pct_change
                    new_columns[pct_change_column_name] = df.groupby("gvkey")[
                        target
                    ].pct_change()
                else:  # for t vs t-i (i > 1), we use the column of t-(i-1) to compute pct_change
                    new_columns[pct_change_column_name] = new_columns[
                        f"{target}_t-{i-1}"
                    ].pct_change()
                if decimals is not None:
                    new_columns[pct_change_column_name] = new_columns[
                        pct_change_column_name
                    ].round(decimals)

    new_df = pd.DataFrame(new_columns)  # Create a new DataFrame from the dictionary
    df = pd.concat(
        [df, new_df], axis=1
    )  # Concatenate the original DataFrame and the new DataFrame
    return df


def cleanup_dataframe(df, n, targets):
    if isinstance(
        targets, str
    ):  # if only one target is provided as a string, make it into a list
        targets = [targets]
    columns_to_drop = []
    for target in targets:
        columns_to_drop.append(target)  # the original target column
        for i in range(1, n + 1):
            columns_to_drop.append(f"{target}_t-{i}")  # the shifted columns
    df_clean = df.drop(columns=columns_to_drop)
    return df_clean


def clean_dict(data_dict, valid_keys):
    """
    Filters the input dictionary based on valid keys.

    Parameters:
    - data_dict (dict): The original dictionary.
    - valid_keys (list): List of valid keys.

    Returns:
    - dict: A dictionary filtered based on valid keys.
    """
    data_dict_copy = {k: v for k, v in data_dict.items() if (k in valid_keys) and (v["comparable_company"] in valid_keys)}
    return data_dict_copy


def get_years_to_drop(df, data_dict):
    """
    Add 'years_to_drop' to the info dict for each key in the data_dict.

    Parameters:
    - df (pd.DataFrame): Dataframe containing the 'gvkey' and 'year' columns.
    - data_dict (dict): Dictionary containing info for each 'gvkey'.

    Returns:
    - dict: An updated dictionary with 'years_to_drop' added to each key's info.
    """
    for key in data_dict:
        gvkey_filter = df["gvkey"] == key
        temp_df = df[gvkey_filter]
        years = temp_df["year"].tolist()
        data_dict[key]["years_to_drop"] = years
    return data_dict


def keep_only_comparable_year(df, data_dict):
    """
    Creates a concatenated DataFrame by appending rows from the input DataFrame
    that satisfy certain conditions based on data_dict.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - data_dict (dict): Dictionary containing 'gvkey' and corresponding info.

    Returns:
    - pd.DataFrame: A concatenated DataFrame.
    """
    empty_df = pd.DataFrame(columns=df.columns)
    concatenated_df = empty_df.copy()

    for key, info in data_dict.items():
        filter_1 = (df["gvkey"] == key) | (df["gvkey"] == info["comparable_company"])

        if info["years_to_drop"] != []:
            filter_2 = ~df["year"].isin(info["years_to_drop"])
            filters = filter_1 & filter_2
            df_to_add = df[filters]
        else:
            df_to_add = df[filter_1]

        concatenated_df = pd.concat([concatenated_df, df_to_add])

    return concatenated_df
