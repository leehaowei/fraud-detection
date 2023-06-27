import numpy as np
import pandas as pd


class TimeSeriesWindowShifter:
    def __init__(self, df: pd.DataFrame, target_columns: list, target_features: list):
        self.df = df[target_columns]
        self.target_columns = target_columns
        self.target_features = target_features

    def fillna_with_median(self, group_col: str):
        """
        Fill NaN values in the specified columns of a DataFrame with the median of each group.

        Parameters:
        - group_col: The column to group by.

        Returns:
        - self for method chaining
        """
        for col in self.target_features:
            medians = self.df.groupby(group_col)[col].transform("median")
            self.df.loc[self.df[col].isna(), col] = medians
        return self

    def zero_to_nan(self):
        """
        This function converts zeros in the target features of the dataframe to NaN.

        Returns:
        - self for method chaining
        """
        for column in self.target_features:
            self.df.loc[self.df[column] == 0, column] = np.nan
        return self

    def dropna(self, axis: int, how: str):
        """
        Drop rows/columns with NaN values.

        Parameters:
        - axis: {0 or ‘index’, 1 or ‘columns’}, default 0. 0 for rows and 1 for columns
        - how: {'any', 'all'}, default 'any'. Determine if row or column is removed from DataFrame, when we have at least one NA or all NA.

        Returns:
        - self for method chaining
        """
        self.df = self.df.dropna(axis=axis, how=how)

        self.target_features = self.df.drop(["motive", "gvkey", "year"], axis=1).columns

        return self

    def shift_dataframe(self, n: int, use_pct: bool = False, decimals: int = 4):
        """
        Shift data in dataframe by n rows and use percentage change

        Parameters:
        - n: number of rows to shift
        - use_pct: whether to use percentage change or not
        - decimals: number of decimal places to round to (if None, no rounding is applied)

        Returns:
        - self for method chaining
        """
        self.df = self.df.sort_values(["gvkey", "year"])  # Ensure data is sorted
        new_columns = {}  # Initialize an empty dictionary for new columns

        # First pass: Create all shifted columns
        for target in self.target_features:
            for i in range(1, n + 1):
                shifted_column_name = f"{target}_t-{i}"
                new_columns[shifted_column_name] = self.df.groupby("gvkey")[
                    target
                ].shift(i)

        # Second pass: Calculate percentage changes, if requested
        if use_pct:
            for target in self.target_features:
                for i in range(1, n + 1):
                    pct_change_column_name = f"{target}_t-{i}_pct_change"
                    if (
                        i == 1
                    ):  # for t vs t-1, we directly use target column to compute pct_change
                        new_columns[pct_change_column_name] = self.df.groupby("gvkey")[
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
        self.df = pd.concat(
            [self.df, new_df], axis=1
        )  # Concatenate the original DataFrame and the new DataFrame
        return self

    def get_processed_df(self):
        return self.df
