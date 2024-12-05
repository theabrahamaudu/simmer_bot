import pandas as pd
from src.utils.data_preprocess_log_config import logger


class CleanData:
    def __init__(self, source_file_path: str, save_path: str,) -> None:
        logger.info("initializing news data cleaner")
        self.source_file_path = source_file_path
        self.save_path = save_path

    def clean(self, output_file_name: str = "cleaned_scraped_data") -> None:
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
        return pd.read_csv(self.source_file_path)
    
    @staticmethod
    def __select_important_cols(
            df: pd.DataFrame,
            imp_cols: list,
        ) -> pd.DataFrame:
        return df[imp_cols]
    
    @staticmethod
    def __sort_by_datetime(df: pd.DataFrame) -> pd.DataFrame:
        df.sort_values(by="datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    @staticmethod
    def __drop_zero_value_news(df: pd.DataFrame) -> pd.DataFrame:
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
        Replace text in a column (default: `link_text`) with the previous row's text if the string is shorter than n characters.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column (str): The name of the column to process.
            n (int): The minimum number of characters a string should have to remain unchanged.

        Returns:
            pd.DataFrame: A DataFrame with the modified column.
        """
        for index, row in df.iterrows():
            if len(str(row["link_text"])) < n:
                df.loc[index, column] = df[column].iloc[index - 1]
        return df
    

class MergeData:
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
        self.__load_data()
        news_df = self.__resample_news_data(self.__news_df)
        news_df = self.__remove_duplicate_news_timestamps(news_df)
        news_df = self.__fill_empty_news_timestamps_in_price(news_df, self.__price_df)
        merged_data = self.__merge_news_n_price(news_df, self.__price_df)
        merged_data.to_csv(self.save_path + f"/{output_file_name}.csv", index=False)

    def __load_data(self) -> None:
        self.__price_df = pd.read_csv(self.price_source_path)
        self.__news_df = pd.read_csv(self.news_source_path)
    
    def __resample_news_data(self, news_df: pd.DataFrame, interval: int = 15) -> pd.DataFrame:
        news_df["impact_priority"] = news_df["impact"].map(
            self.__impact_priority
        )

        news_df["datetime"] = pd.to_datetime(news_df["datetime"])
        news_df["bucket"] = news_df["datetime"].dt.ceil(f"{interval}min")

        return news_df
    
    @staticmethod
    def __remove_duplicate_news_timestamps(news_df: pd.DataFrame) -> pd.DataFrame:
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
        news_df.set_index("bucket", inplace=True)
        filled_news_df = news_df.reindex(price_df["time"]).ffill().reset_index()

        return filled_news_df
    
    @staticmethod
    def __merge_news_n_price(
            news_df: pd.DataFrame,
            price_df: pd.DataFrame
        ) -> pd.DataFrame:
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
