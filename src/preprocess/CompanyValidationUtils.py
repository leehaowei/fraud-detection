import pandas as pd


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


def select_gvkeys_and_create_df(df, comparable_gvkey_dict, gvkeys_icw):
    """
    Select relevant gvkeys and create dataframe

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing all the data.
    comparable_gvkey_dict : dict
        A dictionary where the key is a gvkey and the value is a dict containing
        the comparable company and year for that gvkey.
    gvkeys_icw : list
        A list containing gvkeys related to internal control weaknesses

    Returns
    -------
    DataFrame
        A DataFrame that contains data for the selected gvkeys.
    """
    # Generate list of selected gvkeys
    comparable_non_fraud_list = [
        v["comparable_company"] for k, v in comparable_gvkey_dict.items()
    ]
    all_selected_gvkeys = comparable_non_fraud_list + list(gvkeys_icw.keys())

    # Filter dataframe for selected gvkeys
    filter_ = df["gvkey"].isin(all_selected_gvkeys)
    df_selected = df[filter_]

    return df_selected


def concatenate_selected_df(df_selected, comparable_gvkey_dict):
    """
    Concatenate selected dataframes based on gvkeys.

    Parameters
    ----------
    df_selected : DataFrame
        The DataFrame containing data for selected gvkeys.
    comparable_gvkey_dict : dict
        A dictionary where the key is a gvkey and the value is a dict containing
        the comparable company and year for that gvkey.

    Returns
    -------
    DataFrame
        A DataFrame that contains the concatenated data.
    """
    # Initialize an empty dataframe
    empty_df = pd.DataFrame(columns=df_selected.columns)
    concatenated_df = empty_df.copy()

    # Iterate over each gvkey
    for fraud_gvkey, info in comparable_gvkey_dict.items():
        comparable_gvkey = info["comparable_company"]
        comparable_year = info["comparable_year"]
        fraud_df_temp = df_selected[df_selected["gvkey"] == fraud_gvkey]

        # Create filters
        filter_1 = df_selected["gvkey"] == comparable_gvkey
        filter_2 = df_selected["year"] <= comparable_year
        filters = filter_1 & filter_2

        # Create non-fraud dataframe
        non_fraud_df_temp = df_selected[filters]

        # Concatenate dataframes
        concatenated_df = pd.concat([concatenated_df, fraud_df_temp])
        concatenated_df = pd.concat([concatenated_df, non_fraud_df_temp])

    return concatenated_df


def filter_data_sufficient_amount(df_selected):
    """
    Filter data based on the number of records per gvkey.

    Parameters
    ----------
    df_selected : DataFrame
        The DataFrame containing selected data.

    Returns
    -------
    DataFrame
        A DataFrame that contains data with sufficient amount.
    """
    # Create filters
    filter_sufficient_amount = df_selected.groupby("gvkey").size() >= 10
    filter_selected_sufficient = df_selected["gvkey"].isin(
        list(filter_sufficient_amount[filter_sufficient_amount].index)
    )

    # Filter data
    df_sufficient = df_selected[filter_selected_sufficient]
    return df_sufficient


def select_and_clean_data(df_sufficient):
    """
    Select and clean data by selecting last 10 years of data for each gvkey
    and dropping unnecessary columns.

    Parameters
    ----------
    df_sufficient : DataFrame
        The DataFrame containing data with sufficient amount.

    Returns
    -------
    DataFrame
        A DataFrame that contains selected and cleaned data.
    """
    df1 = df_sufficient.copy()
    df1.loc[:, "rank"] = df1.groupby("gvkey").cumcount(ascending=False)
    selected_df_10_years = df1[df1["rank"] < 10]
    selected_df_10_years = selected_df_10_years.drop(columns=["rank"])

    # Reset index and clean dataframe
    selected_df_10_years.reset_index(inplace=True, drop=True)
    selected_df_10_years["datadate"] = selected_df_10_years.pop("year")
    selected_df_10_years = selected_df_10_years.drop(
        ["at_lower_bound", "at_upper_bound"], axis=1
    )
    selected_df_10_years.rename(columns={"datadate": "year"}, inplace=True)

    return selected_df_10_years


def info(df: pd.DataFrame):
    print(df.shape)
    return df.head()
