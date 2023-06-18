import pandas as pd


def inspector(df: pd.DataFrame, col, target):
    filter_ = df[col] == target
    return df[filter_]
