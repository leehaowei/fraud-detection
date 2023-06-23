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
        medians = df.groupby(group_col)[col].transform('median')
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
    df = df.copy() # make a copy to avoid modifying the original DataFrame
    for column in columns:
        df.loc[df[column] == 0, column] = np.nan
    return df


def shift_dataframe(df, n, targets: list, use_pct=False, decimals=4):
    df = df.sort_values(['gvkey', 'year'])  # Ensure data is sorted
    new_columns = {}  # Initialize an empty dictionary for new columns

    # First pass: Create all shifted columns
    for target in targets:
        for i in range(1, n+1):
            shifted_column_name = f'{target}_t-{i}'
            new_columns[shifted_column_name] = df.groupby('gvkey')[target].shift(i)

    # Second pass: Calculate percentage changes, if requested
    if use_pct:
        for target in targets:
            for i in range(1, n+1):
                pct_change_column_name = f'{target}_t-{i}_pct_change'
                if i == 1:  # for t vs t-1, we directly use target column to compute pct_change
                    new_columns[pct_change_column_name] = df.groupby('gvkey')[target].pct_change()
                else:  # for t vs t-i (i > 1), we use the column of t-(i-1) to compute pct_change
                    new_columns[pct_change_column_name] = new_columns[f'{target}_t-{i-1}'].pct_change()
                if decimals is not None:
                    new_columns[pct_change_column_name] = new_columns[pct_change_column_name].round(decimals)

    new_df = pd.DataFrame(new_columns)  # Create a new DataFrame from the dictionary
    df = pd.concat([df, new_df], axis=1)  # Concatenate the original DataFrame and the new DataFrame
    return df


def cleanup_dataframe(df, n, targets):
    if isinstance(targets, str):  # if only one target is provided as a string, make it into a list
        targets = [targets]
    columns_to_drop = []
    for target in targets:
        columns_to_drop.append(target)  # the original target column
        for i in range(1, n+1):
            columns_to_drop.append(f'{target}_t-{i}')  # the shifted columns
    df_clean = df.drop(columns=columns_to_drop)
    return df_clean