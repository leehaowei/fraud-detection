import chardet
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


def get_encoding(file):
    rawdata = open(file, 'rb').read()
    result = chardet.detect(rawdata)
    return result['encoding']


def get_nunique(df: DataFrame, feature: str):
    return df[feature].nunique()


def get_quantile(df: DataFrame, feature: str, pct: float):
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
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

    # Show the plot
    plt.show()
