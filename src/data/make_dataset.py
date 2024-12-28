import pandas as pd
from src.utils.data_preprocess_log_config import logger


class CleanData:
    """
    Class to clean news data
    """
    def __init__(self, source_file_path: str, save_path: str,) -> None:
        logger.info("initializing news data cleaner")
        self.source_file_path = source_file_path
        self.save_path = save_path

    def clean(self, output_file_name: str = "cleaned_scraped_data") -> None:
        """
        Cleans the scraped news data by selecting important columns, sorting by datetime, 
        dropping rows with zero values, and filling missing news text, then saves the cleaned data to a CSV file.

        Args:
            output_file_name (str, optional): The name of the output CSV file to save the cleaned data. Default is "cleaned_scraped_data".

        Returns:
            None: The function saves the cleaned data to a CSV file at the specified path.

        Behavior:
            - Loads the raw data using `__load_data`.
            - Selects important columns: "datetime", "impact", and "link_text".
            - Sorts the data by datetime.
            - Removes rows where news data is missing or invalid.
            - Fills any missing news text.
            - Saves the cleaned data to a CSV file with the specified output file name.

        Logs:
            - Logs the cleaning process and file saving.
        """
        df = self.__load_data()
        df = self.__select_important_cols(
            df,
            imp_cols= ["datetime", "impact", "link_text"]
        )
        df = self.__sort_by_datetime(df)
        df = self.__drop_zero_value_news(df)
        df = self.__fill_empty_news_text(df)
        df.to_csv(self.save_path + f"/{output_file_name}.csv", index=False)

    def __load_data(self) -> pd.DataFrame:
        """
        Loads the data from a CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the data loaded from the CSV file.

        Behavior:
            - Reads the CSV file located at `self.source_file_path` and returns it as a pandas DataFrame.
        
        Example:
            If the source file contains news data, it will be loaded into a DataFrame for further processing.
        """
        return pd.read_csv(self.source_file_path)
    
    @staticmethod
    def __select_important_cols(
            df: pd.DataFrame,
            imp_cols: list,
        ) -> pd.DataFrame:
        """
        Selects specified important columns from the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame from which columns will be selected.
            imp_cols (list): A list of column names to be selected from the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing only the selected important columns.

        Behavior:
            - Filters the input DataFrame to include only the specified columns from `imp_cols`.
        
        Example:
            If `imp_cols = ["datetime", "impact", "link_text"]`, the function will return a DataFrame with only those columns.
        """
        return df[imp_cols]
    
    @staticmethod
    def __sort_by_datetime(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts the DataFrame by the 'datetime' column in ascending order and resets the index.

        Args:
            df (pd.DataFrame): The DataFrame to be sorted by datetime.

        Returns:
            pd.DataFrame: The DataFrame sorted by 'datetime' with the index reset.

        Behavior:
            - Sorts the DataFrame by the 'datetime' column in ascending order.
            - Resets the DataFrame's index after sorting to ensure a continuous index.

        Example:
            The function will reorder the rows based on the 'datetime' column, making it easier to analyze time-based data.
        """
        df.sort_values(by="datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    @staticmethod
    def __drop_zero_value_news(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows from the DataFrame where the 'impact' column is "Non-Economic" 
        and the 'link_text' column contains a missing value (NaN).

        Args:
            df (pd.DataFrame): The DataFrame from which rows will be dropped.

        Returns:
            pd.DataFrame: The DataFrame with rows containing zero-value news removed.

        Behavior:
            - Iterates over the DataFrame and removes rows where 'impact' is "Non-Economic" 
            and 'link_text' is missing or NaN.
            - Resets the DataFrame's index after dropping rows.

        Example:
            If a news row has no significant impact ("Non-Economic") and no associated link text, 
            it will be dropped from the DataFrame.
        """
        for index, row in df.iterrows():
            if row["impact"] == "Non-Economic" and str(row["link_text"]) == "nan":
                df.drop(index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    @staticmethod
    def __fill_empty_news_text(
            df: pd.DataFrame,
            column: str= "link_text",
            n: int = 20
        ) -> pd.DataFrame:
        """
        Fills missing or short news text in the specified column by replacing it with 
        the previous row's value if the text length is shorter than the specified threshold.

        Args:
            df (pd.DataFrame): The DataFrame in which the missing or short text will be filled.
            column (str, optional): The column to check for short or missing text. Defaults to "link_text".
            n (int, optional): The threshold length for the text. Rows with text shorter than this will be filled. Defaults to 20.

        Returns:
            pd.DataFrame: The DataFrame with empty or short text in the specified column filled.

        Behavior:
            - Iterates through the DataFrame and checks the length of the text in the specified column.
            - If the text length is shorter than the threshold (`n`), it replaces the text with the value from the previous row.
        
        Example:
            If a news article has a link text shorter than 20 characters, it will be replaced with the previous article's link text.
        """
        for index, row in df.iterrows():
            if len(str(row["link_text"])) < n:
                df.loc[index, column] = df[column].iloc[index - 1]
        return df
    

class MergeData:
    """
    Class for merging news data with price data.
    """
    def __init__(
            self,
            price_source_path: str,
            news_source_path: str,
            save_path: str
        ) -> None:
        self.price_source_path = price_source_path
        self.news_source_path = news_source_path
        self.save_path = save_path
        self.__price_df = None
        self.__news_df = None
        self.__impact_priority = {
            'Non Economic': 0,
            'Low Impact Expected': 1,
            'Medium Impact Expected': 2,
            'High Impact Expected': 3
        }

    def merge(self, output_file_name: str = "merged_scrapped_n_price") -> None:
        """
        Merges news data with price data, resamples the news data, and removes duplicates 
        before saving the merged result to a CSV file.

        Args:
            output_file_name (str, optional): The name of the output CSV file where 
                                            the merged data will be saved. Defaults to "merged_scrapped_n_price".

        Returns:
            None: The merged data is saved to a CSV file at the specified location.

        Behavior:
            - Loads the news and price data.
            - Resamples the news data to align with the price data.
            - Removes any duplicate timestamps from the news data.
            - Fills in missing timestamps in the news data based on the price data.
            - Merges the news and price data into a single DataFrame.
            - Saves the merged data to a CSV file.
        """
        self.__load_data()
        news_df = self.__resample_news_data(self.__news_df)
        news_df = self.__remove_duplicate_news_timestamps(news_df)
        news_df = self.__fill_empty_news_timestamps_in_price(news_df, self.__price_df)
        merged_data = self.__merge_news_n_price(news_df, self.__price_df)
        merged_data.to_csv(self.save_path + f"/{output_file_name}.csv", index=False)

    def __load_data(self) -> None:
        """
        Loads price and news data from the specified source paths into DataFrames.

        Args:
            None

        Returns:
            None: The method loads the data into the instance variables `__price_df` and `__news_df`.

        Behavior:
            - Reads the CSV file at the path specified by `price_source_path` into `__price_df`.
            - Reads the CSV file at the path specified by `news_source_path` into `__news_df`.
        """
        self.__price_df = pd.read_csv(self.price_source_path)
        self.__news_df = pd.read_csv(self.news_source_path)
    
    def __resample_news_data(self, news_df: pd.DataFrame, interval: int = 15) -> pd.DataFrame:
        """
        Resamples news data into specified time intervals and assigns an impact priority to each news item.

        Args:
            news_df (pd.DataFrame): The DataFrame containing the news data to be resampled.
            interval (int, optional): The time interval (in minutes) for resampling the news data. Defaults to 15 minutes.

        Returns:
            pd.DataFrame: The resampled news DataFrame with a new `bucket` column representing the resampled time intervals.

        Behavior:
            - Adds a new column `impact_priority` based on the "impact" column, mapped from the `__impact_priority` dictionary.
            - Converts the "datetime" column to a pandas datetime format.
            - Creates a new column `bucket` representing the resampled time intervals, rounded up to the specified `interval` (in minutes).
        
        Example:
            If the news data is recorded at irregular intervals, this function resamples the data into 15-minute buckets.
        """
        news_df["impact_priority"] = news_df["impact"].map(
            self.__impact_priority
        )

        news_df["datetime"] = pd.to_datetime(news_df["datetime"])
        news_df["bucket"] = news_df["datetime"].dt.ceil(f"{interval}min")

        return news_df
    
    @staticmethod
    def __remove_duplicate_news_timestamps(news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate news entries for the same timestamp by keeping the one with the highest priority.

        Args:
            news_df (pd.DataFrame): The DataFrame containing the news data to be processed.

        Returns:
            pd.DataFrame: The DataFrame with duplicate news entries removed, keeping the one with the highest impact priority.

        Behavior:
            - Sorts the news data by the "impact_priority" column in descending order.
            - Groups the news data by the `bucket` column (representing time intervals).
            - For each group (i.e., each time bucket), keeps the first entry based on the highest impact priority and 
            combines the `link_text` from all rows in that group (removes duplicates by converting to a set).
            - Resets the index of the resulting DataFrame.

        Example:
            This function ensures that for each time bucket, only the most impactful news item is retained and that any duplicate 
            links within the same time bucket are combined into a single string.
        """
        news_no_duplicate_timestamps = (
            news_df.sort_values(by="impact_priority", ascending=False)
            .groupby("bucket")
            .agg({
                "impact": "first",
                "link_text": lambda x: " ".join(set(x))
            })
            .reset_index()
        )
        return news_no_duplicate_timestamps
    

    @staticmethod
    def __fill_empty_news_timestamps_in_price(
            news_df: pd.DataFrame,
            price_df: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Fills missing news data timestamps by forward filling based on price data timestamps.

        Args:
            news_df (pd.DataFrame): The DataFrame containing the news data to be filled.
            price_df (pd.DataFrame): The DataFrame containing price data with timestamps that will be used to fill missing news data.

        Returns:
            pd.DataFrame: The news DataFrame with missing timestamps filled by forward filling based on price data timestamps.

        Behavior:
            - Sets the "bucket" column in the `news_df` DataFrame as the index.
            - Reindexes the `news_df` DataFrame based on the "time" column from the `price_df` to align the timestamps.
            - Forward fills any missing values in the news data using the available values in the previous rows.
            - Resets the index of the filled DataFrame.
        """
        news_df.set_index("bucket", inplace=True)
        filled_news_df = news_df.reindex(price_df["time"]).ffill().reset_index()

        return filled_news_df
    
    @staticmethod
    def __merge_news_n_price(
            news_df: pd.DataFrame,
            price_df: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Merges news data with price data based on timestamps.

        Args:
            news_df (pd.DataFrame): The DataFrame containing the news data.
            price_df (pd.DataFrame): The DataFrame containing the price data.

        Returns:
            pd.DataFrame: The merged DataFrame containing both news and price data.

        Behavior:
            - Merges the `news_df` and `price_df` DataFrames based on the "time" column.
            - Performs a left join, where all entries from `price_df` are retained, and the corresponding news data is added where available.
            - Drops any rows with missing values (i.e., where no corresponding news data is found for a price timestamp).
            - Resets the index of the resulting DataFrame.

        Example:
            This function combines news and price data for each timestamp, ensuring that each price record is associated 
            with the relevant news data, while removing rows that lack corresponding news information.
        """
        merged_data = pd.merge(
            price_df,
            news_df,
            left_on="time",
            right_on="time",
            how="left"
        )
        merged_data.dropna(axis=0, inplace=True)
        merged_data.reset_index(drop=True, inplace=True)
        return merged_data
    
if __name__ == "__main__":
    news_data_cleaner = CleanData(
        source_file_path="./data/interim/combined_scrapped_data.csv",
        save_path="./data/interim"
    )
    news_data_cleaner.clean()

    data_merger = MergeData(
        price_source_path="./data/raw/price_data.csv",
        news_source_path="./data/interim/cleaned_scraped_data.csv",
        save_path="./data/interim"
    )
    data_merger.merge()
