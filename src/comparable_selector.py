# import self-defined packages
from path.PathProcessor import PathProcessor
from preprocess.raw_preprocess_utils import *
from preprocess.company_validation_utils import *
from utils.utils import *

# Constants

WRDS_FILE_NAME = "wrds.parquet"
GVKEYS_FILE_NAME = "gvkeys.yaml"
TARGET_FILE_NAME = "data_10_years.parquet"


class ComparableSelector:
    def __init__(self, n_records, column_compared, lower_bound, upper_bound, mode):
        self.n_records = n_records
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.column_compared = column_compared
        self.mode = mode

    def process(self):
        # Load and parse the data files
        df, gvkeys_dict = self.load_and_parse_data()
        # Additional processing methods here...
        gvkeys_icw = gvkeys_dict["internal_control_weakness"]
        gvkeys_non_fraud = gvkeys_dict["non_fraud"]

        # Filter out unwanted records and add ICW-related information
        df_icw_and_non_fraud = self.remove_other_motive_add_icw_info(df, gvkeys_dict)

        # Process the data
        df_labeled = self.assign_label_and_remove_post_violation(df_icw_and_non_fraud)

        # Add percentile columns and process non-fraud data
        df_labeled = self.add_percentile_columns(
            df_labeled,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )

        # First Filter for gvkey more than N records
        df_filtered = self.ensure_enough_records(df_labeled)

        # reassign gvkeys_icw: only includes gvkeys_icw with more than N records
        gvkeys_icw_filtered = self.filter_gvkeys(
            df=df_filtered, original_gvkeys_icw=gvkeys_icw, target_motive=1
        )
        gvkeys_non_fraud_filtered = self.filter_gvkeys(
            df=df_filtered, original_gvkeys_icw=gvkeys_non_fraud, target_motive=0
        )

        # Filter the DataFrame for non-fraud records
        fraud_df, non_fraud_df = self.split_fraud_and_non(
            df=df_filtered, fraud_check_col="is_icw"
        )

        # Find comparable companies for each fraud gvkey
        comparable_gvkey_dict = self.find_comparable_companies(
            fraud_df=fraud_df,
            non_fraud_df=non_fraud_df,
            gvkeys_icw=gvkeys_icw_filtered,
            gvkeys_non_fraud=gvkeys_non_fraud_filtered,
        )

        # Write the comparable gvkeys dictionary to a YAML file
        save_yaml(target=comparable_gvkey_dict, file="comparable_gvkeys.yaml")

        # Select relevant gvkeys and create dataframe
        df_selected = self.select_gvkeys_and_create_df(
            df_filtered, comparable_gvkey_dict, gvkeys_icw_filtered
        )

        df_selected_clean = self.remove_comparable_after_bv_year(
            df_selected, comparable_gvkey_dict
        )

        df_sufficient = self.filter_data_sufficient_amount(df_selected_clean)

        selected_df_10_years = self.select_and_clean_data(df_sufficient)

        # Save results
        selected_df_10_years.to_parquet("data_10_years.parquet")

        print(selected_df_10_years.shape)
        print(len(comparable_gvkey_dict))

    def load_and_parse_data(self):
        # Setup and file paths
        # Create instances of PathProcessor for WRDS, fraud, and gvkeys data
        processor = PathProcessor(mode=self.mode)

        # Get the data path for each file
        wrds_file_path = processor.get_data_path(target_file=WRDS_FILE_NAME)
        gvkeys_file_path = processor.get_data_path(target_file=GVKEYS_FILE_NAME)

        # Load data
        # Read the Parquet files into DataFrame
        df_wrds = pd.read_parquet(wrds_file_path)

        # Load the YAML file into a dictionary
        gvkeys_dict = load_yaml_from_public_s3(gvkeys_file_path)

        return df_wrds, gvkeys_dict

    def ensure_enough_records(self, df):
        counts = df["gvkey"].value_counts()
        mask = df["gvkey"].isin(counts[counts >= self.n_records].index)
        return df[mask]

    # Rest of your functions here...
    def remove_other_motive_add_icw_info(self, df_wrds, gvkeys_dict):
        # Extract key lists from the dictionary
        gvkeys_icw = gvkeys_dict["internal_control_weakness"]
        gvkeys_other_motive = gvkeys_dict["other_motive"]

        # Filter out other motive keys and update the data
        df_wrds = df_wrds.loc[
            ~df_wrds["gvkey"].isin(gvkeys_other_motive.keys()), :
        ].copy()

        # Add new columns based on icw
        df_wrds.loc[:, "is_icw"] = (
            df_wrds["gvkey"]
            .apply(lambda x: check_if_icw(gvkey=x, gvkeys_icw=gvkeys_icw))
            .astype(int)
        )

        # Avoid SettingWithCopyWarning by creating temporary DataFrame and splitting it into two columns
        temp_df = (
            df_wrds["gvkey"]
            .apply(lambda x: get_violation_years(gvkey=x, gvkeys_icw=gvkeys_icw))
            .apply(pd.Series)
        )

        df_wrds.loc[:, "bv_year"] = temp_df[0]
        df_wrds.loc[:, "ev_year"] = temp_df[1]

        return df_wrds

    def assign_label_and_remove_post_violation(self, df_wrds):
        # Assign label
        df_wrds = self.assign_label(df_wrds, label=1)

        # Remove records
        df_wrds = self.remove_records_after_bv_years(df_wrds)

        return df_wrds

    def filter_data_sufficient_amount(self, df_selected):
        """
        Filter data based on the number of records per gvkey.

        Parameters
        ----------
        df_selected : DataFrame
            The DataFrame containing selected data.
        n : int, optional
            The minimum number of records required per gvkey for the data to be considered sufficient.
            Default is 10.

        Returns
        -------
        DataFrame
            A DataFrame that contains data with sufficient amount.
        """
        # Create filters
        filter_sufficient_amount = df_selected.groupby("gvkey").size() >= self.n_records
        filter_selected_sufficient = df_selected["gvkey"].isin(
            list(filter_sufficient_amount[filter_sufficient_amount].index)
        )

        # Filter data
        df_sufficient = df_selected[filter_selected_sufficient]
        return df_sufficient

    def select_and_clean_data(self, df):
        """
        Select and clean data by selecting last 10 years of data for each gvkey
        and dropping unnecessary columns.

        Parameters
        ----------
        df : DataFrame
            The DataFrame containing data with sufficient amount.
        n : int, optional
            The minimum number of records required per gvkey for the data to be considered sufficient.
            Default is 10.

        Returns
        -------
        DataFrame
            A DataFrame that contains selected and cleaned data.
        """
        df_copy = df.copy()
        df_copy.loc[:, "rank"] = df_copy.groupby("gvkey").cumcount(ascending=False)
        selected_df_n_years = df_copy[df_copy["rank"] < self.n_records]
        selected_df_n_years = selected_df_n_years.drop(columns=["rank"])

        # Reset index and clean dataframe
        selected_df_n_years.reset_index(inplace=True, drop=True)
        selected_df_n_years["datadate"] = selected_df_n_years.pop("year")
        selected_df_n_years = selected_df_n_years.drop(
            ["at_lower_bound", "at_upper_bound"], axis=1
        )
        selected_df_n_years.rename(columns={"datadate": "year"}, inplace=True)

        return selected_df_n_years

    def filter_gvkeys(self, df, original_gvkeys_icw, target_motive: int):
        gvkeys_filtered: list = list(
            df[df["motive"] == target_motive]["gvkey"].unique()
        )

        return {k: v for k, v in original_gvkeys_icw.items() if k in gvkeys_filtered}

    def split_fraud_and_non(self, df, fraud_check_col: str = "is_icw"):
        fraud_filter = df[fraud_check_col] == 1
        non_fraud_filter = df[fraud_check_col] == 0

        return df.loc[fraud_filter, :], df.loc[non_fraud_filter, :]

    def find_comparable_companies(
        self,
        fraud_df,
        non_fraud_df,
        gvkeys_icw,
        gvkeys_non_fraud,
    ):
        gvkeys_non_fraud_copy = gvkeys_non_fraud.copy()

        # Loop over each fraud gvkey to find a comparable company
        comparable_gvkey_dict = {}

        for fraud_gvkeys in list(gvkeys_icw.keys()):
            # Various setup and filters
            inner_dict = {}

            current_fraud_gvkeys_df = fraud_df[fraud_df["gvkey"] == fraud_gvkeys]

            current_fraud_df_last = current_fraud_gvkeys_df[
                current_fraud_gvkeys_df["year"] == current_fraud_gvkeys_df["ev_year"]
            ]
            comparable_year: int = current_fraud_df_last["ev_year"].item()
            at_lower_bound: float = current_fraud_df_last[
                f"{self.column_compared}_lower_bound"
            ].item()
            at_upper_bound: float = current_fraud_df_last[
                f"{self.column_compared}_upper_bound"
            ].item()

            non_fraud_df_temp_filter = non_fraud_df["gvkey"].isin(
                list(gvkeys_non_fraud_copy.keys())
            )

            non_fraud_df_temp = non_fraud_df.loc[non_fraud_df_temp_filter, :]

            year_filter = non_fraud_df_temp["year"] == comparable_year

            at_filter = non_fraud_df_temp[self.column_compared].between(
                at_lower_bound, at_upper_bound, inclusive="both"
            )

            # add filter for industry
            industry = current_fraud_gvkeys_df["naics"].unique()[0]
            industry_filter = non_fraud_df_temp["naics"] == industry

            try:
                filters = year_filter & at_filter & industry_filter
                # Select the gvkey of the comparable company
                non_fraud_gvkey_pass = non_fraud_df_temp[filters]["gvkey"]

                selected = non_fraud_gvkey_pass.iloc[0]

            except:
                print("comparable company not found in the same industry")
                filters = year_filter & at_filter
                # Select the gvkey of the comparable company
                non_fraud_gvkey_pass = non_fraud_df_temp[filters]["gvkey"]
                selected = non_fraud_gvkey_pass.iloc[0]

            # Update the temporary dictionary and remove the selected key from the temporary non-fraud keys
            inner_dict["comparable_year"] = comparable_year
            inner_dict["comparable_company"] = selected
            comparable_gvkey_dict[fraud_gvkeys] = inner_dict
            gvkeys_non_fraud_copy.pop(selected)

        return comparable_gvkey_dict

    def select_gvkeys_and_create_df(self, df, comparable_gvkey_dict, gvkeys_icw):
        """
        Select relevant gvkeys and create dataframe

        Parameters
        ----------
        df : DataFrame
            The DataFrame containing all the data.
        comparable_gvkey_dict : dict
            A dictionary where the key is a gvkey and the value is a dict containing
            the comparable company and year for that gvkey.
        gvkeys_icw : dict
            A dict containing gvkeys related to internal control weaknesses

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

    def remove_comparable_after_bv_year(self, df_selected, comparable_gvkey_dict):
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

    def add_percentile_columns(
        self,
        df: pd.DataFrame,
        lower_bound: float,
        upper_bound: float,
        lower_bound_suffix: str = "lower_bound",
        upper_bound_suffix: str = "upper_bound",
    ) -> pd.DataFrame:
        df[f"{self.column_compared}_{lower_bound_suffix}"] = df[
            self.column_compared
        ].apply(lambda x: get_bounding(target_value=x, bound=lower_bound))
        df[f"{self.column_compared}_{upper_bound_suffix}"] = df[
            self.column_compared
        ].apply(lambda x: get_bounding(target_value=x, bound=upper_bound))
        return df

    def assign_label(self, df, label: int = 1):
        df = df.copy()

        df["motive"] = 0

        gvkey_filter = df["is_icw"] == 1
        year_filter = df["year"].between(df["bv_year"], df["ev_year"], inclusive="both")
        filters = gvkey_filter & year_filter
        df.loc[filters, "motive"] = label

        return df

    def remove_records_after_bv_years(self, df):
        df = df.copy()

        gvkey_filter = df["is_icw"] == 1

        filter_to_remove = (
            df["year"] > df["ev_year"]
        ) & gvkey_filter  # Create a filter for records to drop

        filtered_df = df[~filter_to_remove]  # Apply the filter

        return filtered_df

    def info(self, df: pd.DataFrame):
        print(df.shape)
        return df.head()
