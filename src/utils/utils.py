import numpy as np
import pandas as pd


def inspector(df: pd.DataFrame, col, target):
    filter_ = df[col] == target
    return df[filter_]


def generate_records(
    df: pd.DataFrame, n: int, target_col: str, start_col: str, end_col: str
):
    rows = []
    for _, row in df.iterrows():
        start_year = row[start_col] - n
        end_year = row[end_col]
        for year in range(start_year, end_year + 1):
            new_row = row.copy()
            new_row[target_col] = year
            rows.append(new_row)
    new_df = pd.concat(rows, axis=1).T
    return new_df.reset_index(drop=True)


def fill_nan_str(df, col: str, to_fill: str):
    # Copy the dataframe to avoid modifying the original one
    df_filled = df.copy()

    # Replace NaN values
    df_filled[col] = df_filled[col].fillna(to_fill)

    return df_filled


def validate_comparable(df, comparable_dict: dict):
    for target_gvkey in comparable_dict:
        filter_1 = df["gvkey"] == target_gvkey
        filter_2 = df["gvkey"] == comparable_dict[target_gvkey]["comparable"]
        arr = df[filter_1]["year"].values == df[filter_2]["year"].values
        print(target_gvkey)
        print(np.any(arr == False))


def keep_only_bv_records(df, data_dict):
    concatenated_df = pd.DataFrame(columns=df.columns)

    for key, info in data_dict.items():
        filter_1 = (df["gvkey"] == key) | (df["gvkey"] == info["comparable"])

        current_temp_df = df[filter_1]
        bv_year = current_temp_df["year"].unique()[0]
        # print(f"bv_year: {bv_year}")
        filter_2 = df["year"] <= bv_year

        filters = filter_1 & filter_2

        df_to_add = df[filters]

        concatenated_df = pd.concat([concatenated_df, df_to_add])

    return concatenated_df
