import pandas as pd


def select_gvkeys_and_create_df(df, comparable_gvkey_dict, gvkeys_icw):
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
    # Create filters
    filter_sufficient_amount = df_selected.groupby("gvkey").size() > 10
    filter_selected_sufficient = df_selected["gvkey"].isin(
        list(filter_sufficient_amount[filter_sufficient_amount].index)
    )

    # Filter data
    df_sufficient = df_selected[filter_selected_sufficient]
    return df_sufficient


def select_and_clean_data(df_sufficient):
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
