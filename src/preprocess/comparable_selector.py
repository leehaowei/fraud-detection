import json
import logging

# import self-defined packages
from path.PathProcessor import PathProcessor
from preprocess.raw_preprocess_utils import *
from config.comparable_config import ComparableConfig
from utils.utils import *


class ComparableSelector:
    def __init__(self, comparable_config: ComparableConfig):
        self.n_records = comparable_config.n_records
        self.lower_bound = comparable_config.lower_bound
        self.upper_bound = comparable_config.upper_bound
        self.column_compared = comparable_config.column_compared
        self.mode = comparable_config.mode
        self.log_file_path = comparable_config.log_file_path

        self.wrds_file_path = comparable_config.file_path["wrds"]
        self.gvkeys_file_path = comparable_config.file_path["gvkeys"]

        # Setup logger
        self.set_logger()

    def set_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        handler = logging.FileHandler(self.log_file_path)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def process(self):
        # Load and parse the data files
        df, gvkeys_dict = self.load_and_parse_data()
        # Additional processing methods here...
        gvkeys_icw = gvkeys_dict["internal_control_weakness"]
        gvkeys_non_fraud = gvkeys_dict["non_fraud"]

        df_filtered = (
            df.pipe(
                ComparableSelector.remove_other_motive_add_icw_info,
                gvkeys_dict,  # Filter out unwanted records and add ICW-related information
            )
            .pipe(
                ComparableSelector.assign_label_and_remove_post_violation  # Process the data
            )
            .pipe(
                self.add_percentile_columns,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,  # Add percentile columns and process non-fraud data
            )
            .pipe(
                self.ensure_enough_records  # First Filter for gvkey more than N records
            )
        )

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

        # Filter the df with comparable gvkeys
        df_comparable = self.select_gvkeys_and_create_df(
            df_filtered, comparable_gvkey_dict, gvkeys_icw_filtered
        )

        df_comparable_lean = self.remove_comparable_after_bv_year(
            df_comparable, comparable_gvkey_dict
        )

        df_sufficient = self.filter_data_sufficient_amount(df_comparable_lean)

        df_comparable_n_years = self.select_and_clean_data(df_sufficient)

        # Save results
        df_comparable_n_years.to_parquet(f"data_{self.n_records}_years.parquet")
        self.df = df_comparable_n_years

        print(df_comparable_n_years.shape)
        print(len(comparable_gvkey_dict))

    def load_and_parse_data(self):
        # Setup and file paths
        # Create instances of PathProcessor for WRDS, fraud, and gvkeys data
        processor = PathProcessor(mode=self.mode)

        # Get the data path for each file
        wrds_file_path = processor.get_data_path(target_file=self.wrds_file_path)
        gvkeys_file_path = processor.get_data_path(target_file=self.gvkeys_file_path)

        # Load data
        # Read the Parquet files into DataFrame
        df_wrds = pd.read_parquet(wrds_file_path)

        # Load the YAML file into a dictionary
        gvkeys_dict = load_yaml_from_public_s3(gvkeys_file_path)

        return df_wrds, gvkeys_dict

    @staticmethod
    def remove_other_motive_add_icw_info(df_wrds, gvkeys_dict):
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

    @staticmethod
    def assign_label_and_remove_post_violation(df):
        # Assign label
        df = ComparableSelector.assign_label(df, label=1)

        # Remove records
        df = ComparableSelector.remove_records_after_ev_years(df)

        return df

    @staticmethod
    def assign_label(df, label: int = 1):
        df = df.copy()

        df["motive"] = 0

        gvkey_filter = df["is_icw"] == 1
        year_filter = df["year"].between(df["bv_year"], df["ev_year"], inclusive="both")
        filters = gvkey_filter & year_filter
        df.loc[filters, "motive"] = label

        return df

    @staticmethod
    def remove_records_after_ev_years(df):
        df = df.copy()

        gvkey_filter = df["is_icw"] == 1

        filter_to_remove = (
            df["year"] > df["ev_year"]
        ) & gvkey_filter  # Create a filter for records to drop

        filtered_df = df[~filter_to_remove]  # Apply the filter

        return filtered_df

    def add_percentile_columns(
        self,
        df: pd.DataFrame,
        lower_bound: float,
        upper_bound: float,
        lower_bound_suffix: str = "lower_bound",
        upper_bound_suffix: str = "upper_bound",
    ) -> pd.DataFrame:
        df = (
            df.copy()
        )  # Creates a copy of the DataFrame to avoid SettingWithCopyWarning
        df[f"{self.column_compared}_{lower_bound_suffix}"] = df[
            self.column_compared
        ].apply(lambda x: get_bounding(target_value=x, bound=lower_bound))
        df[f"{self.column_compared}_{upper_bound_suffix}"] = df[
            self.column_compared
        ].apply(lambda x: get_bounding(target_value=x, bound=upper_bound))

        return df

    def ensure_enough_records(self, df):
        counts = df["gvkey"].value_counts()
        mask = df["gvkey"].isin(counts[counts >= self.n_records].index)
        return df[mask]

    @staticmethod
    def filter_gvkeys(df, original_gvkeys_icw, target_motive: int):
        gvkeys_filtered: list = list(
            df[df["motive"] == target_motive]["gvkey"].unique()
        )

        return {k: v for k, v in original_gvkeys_icw.items() if k in gvkeys_filtered}

    @staticmethod
    def split_fraud_and_non(df, fraud_check_col: str = "is_icw"):
        fraud_filter = df[fraud_check_col] == 1
        non_fraud_filter = df[fraud_check_col] == 0

        return df.loc[fraud_filter, :], df.loc[non_fraud_filter, :]

    def apply_filters_recursively(
        self,
        non_fraud_df_temp,
        filters,
        inner_dict,
        inner_dict_json,
        industry_dict,
        industry,
    ):
        if not filters:
            raise IndexError("No more filters to apply!")

        try:
            combined_filter = filters[0]
            for filter in filters[1:]:
                combined_filter = combined_filter & filter

            non_fraud_pass = non_fraud_df_temp[combined_filter]
            non_fraud_pass_first = non_fraud_pass.iloc[0, :]
            non_fraud_pass_first_gvkey = non_fraud_pass_first["gvkey"]
            non_fraud_pass_first_naics = non_fraud_pass_first["naics"]

            if len(filters) == 3:
                industry_dict["same"] = True
                inner_dict["industry_filter"] = True
                inner_dict_json["industry_filter"] = True
                industry_dict["naics_compared"] = industry
            else:
                industry_dict["same"] = False
                inner_dict["industry_filter"] = False
                inner_dict_json["industry_filter"] = False
                industry_dict["naics_compared"] = non_fraud_pass_first_naics

            inner_dict["industry"] = industry_dict
            inner_dict["fin_filter"] = True
            inner_dict_json["fin_filter"] = True

            if len(filters) == 1:
                inner_dict["use_financial_filter"] = False
                inner_dict_json["fin_filter"] = False
            inner_dict_json["num_comparable"] = non_fraud_pass["gvkey"].nunique()

            return non_fraud_pass_first_gvkey
        except IndexError:
            return self.apply_filters_recursively(
                non_fraud_df_temp,
                filters[:-1],
                inner_dict,
                inner_dict_json,
                industry_dict,
                industry,
            )

    def find_comparable_companies(
        self,
        fraud_df,
        non_fraud_df,
        gvkeys_icw,
        gvkeys_non_fraud,
        which_year: str = "bv_year"
    ):
        gvkeys_non_fraud_copy = gvkeys_non_fraud.copy()

        # Loop over each fraud gvkey to find a comparable company
        comparable_gvkey_dict = {}
        comparable_gvkey_dict_json = {}

        # log json
        info_json = {
            "metadata": {
                "factor": self.column_compared,
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
            },
            "data": comparable_gvkey_dict_json,
        }

        for fraud_gvkeys in list(gvkeys_icw.keys()):
            # Various setup and filters
            inner_dict = {}
            inner_dict_json = {}

            current_fraud_gvkeys_df = fraud_df[fraud_df["gvkey"] == fraud_gvkeys]

            current_fraud_df_last = current_fraud_gvkeys_df[
                current_fraud_gvkeys_df["year"] == current_fraud_gvkeys_df[which_year]
            ]
            comparable_year: int = current_fraud_df_last[which_year].item()
            lower_bound: float = current_fraud_df_last[
                f"{self.column_compared}_lower_bound"
            ].item()
            upper_bound: float = current_fraud_df_last[
                f"{self.column_compared}_upper_bound"
            ].item()

            non_fraud_df_temp_filter = non_fraud_df["gvkey"].isin(
                list(gvkeys_non_fraud_copy.keys())
            )

            non_fraud_df_temp = non_fraud_df.loc[non_fraud_df_temp_filter, :]

            year_filter = non_fraud_df_temp["year"] == comparable_year

            financial_filter = non_fraud_df_temp[self.column_compared].between(
                lower_bound, upper_bound, inclusive="both"
            )

            # add filter for industry
            industry = current_fraud_gvkeys_df["naics"].unique()[0]
            industry_filter = non_fraud_df_temp["naics"] == industry

            industry_dict = {"naics_ori": industry}

            filters = [year_filter, financial_filter, industry_filter]

            selected = None  # Set a default value for selected
            try:
                selected = self.apply_filters_recursively(
                    non_fraud_df_temp,
                    filters,
                    inner_dict,
                    inner_dict_json,
                    industry_dict,
                    industry,
                )
            except IndexError:
                print(f"No comparable company found for gvkey {fraud_gvkeys}")

            # Update the temporary dictionary and remove the selected key from the temporary non-fraud keys
            inner_dict["year"] = comparable_year
            inner_dict["comparable"] = selected

            comparable_gvkey_dict[fraud_gvkeys] = inner_dict
            comparable_gvkey_dict_json[fraud_gvkeys] = inner_dict_json

            gvkeys_non_fraud_copy.pop(selected)

        # store it into a json file:
        with open(
            f"out/compared/{self.column_compared}_{int(self.lower_bound * 10)}_{int(self.upper_bound * 10)}.json",
            "w",
        ) as f:
            json.dump(info_json, f)

        return comparable_gvkey_dict

    @staticmethod
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
        gvkeys_icw : dict
            A dict containing gvkeys related to internal control weaknesses

        Returns
        -------
        DataFrame
            A DataFrame that contains data for the selected gvkeys.
        """
        # Generate list of selected gvkeys
        comparable_non_fraud_list = [
            v["comparable"] for k, v in comparable_gvkey_dict.items()
        ]
        all_selected_gvkeys = comparable_non_fraud_list + list(gvkeys_icw.keys())

        # Filter dataframe for selected gvkeys
        filter_ = df["gvkey"].isin(all_selected_gvkeys)
        df_selected = df[filter_]

        return df_selected

    @staticmethod
    def remove_comparable_after_bv_year(df_selected, comparable_gvkey_dict):
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
            comparable_gvkey = info["comparable"]
            comparable_year = info["year"]
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
            [
                f"{self.column_compared}_lower_bound",
                f"{self.column_compared}_upper_bound",
            ],
            axis=1,
        )
        selected_df_n_years.rename(columns={"datadate": "year"}, inplace=True)

        return selected_df_n_years

    @staticmethod
    def info(df: pd.DataFrame):
        print(df.shape)
        return df.head()
