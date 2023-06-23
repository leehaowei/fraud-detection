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
