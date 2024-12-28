from tqdm import tqdm
import pandas as pd
from src.features.preprocess_utils import (
    LLMSentimentAnalysis,
    TAIndicators,
    NumericalPreprocess,
    SplitData
)
from src.utils.data_preprocess_log_config import logger

class TrainPreprocessPipeline:
    def __init__(self, data_path: str = "./data/interim/parsed_scraped_data_clipped.csv"):
        logger.info("Initializing TrainPreprocessPipeline")
        self.__source_file_path = data_path
        self.llm_sentiment_analysis = LLMSentimentAnalysis()
        self.ta_indicators = TAIndicators()
        self.numerical_preprocess = NumericalPreprocess()
        self.split_data = SplitData()
    
    def __load_data(self) -> pd.DataFrame:
        """
        Loads data from the specified file path.

        Args:
            None

        Returns:
            pd.DataFrame: The loaded data from the CSV file.

        Raises:
            FileNotFoundError: If the file specified in the source file path does not exist.
        """
        logger.info("Loading data from %s", self.__source_file_path)
        try:
            return pd.read_csv(self.__source_file_path)
        except FileNotFoundError as e:
            logger.error("Error loading data from %s:\n %s", self.__source_file_path, e)
    
    def run(
            self,
            lookback: int = 6,
            target_column: str = "target",
            save_path: str = "./data/processed/",
            with_llm_sentiment: bool = False
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Runs the preprocessing pipeline to process raw data, apply sentiment analysis (optional), 
        add technical analysis indicators, and prepare time series data for training, testing, and validation.

        Args:
            lookback (int): The number of previous time steps to use when preparing time series data. Defaults to 6.
            target_column (str): The name of the target column in the dataset. Defaults to "target".
            with_llm_sentiment (bool): Whether to include sentiment analysis from an LLM. Defaults to False.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                A tuple containing the preprocessed DataFrames for training, testing, and validation data.

        Raises:
            Any errors encountered during data loading, preprocessing, or time series preparation will be logged.
        """
        data = self.__load_data()
        if with_llm_sentiment:
            data = self.llm_sentiment_analysis.parse_dataframe(data)
        data = self.ta_indicators.add_indicators(data)
        train, test, validate = self.split_data.raw_split(data)

        # Numerical preprocessing
        train_preprocessed = self.numerical_preprocess.run(
            data=train,
            file_name="train_preprocessed"
        )

        test_preprocessed = self.numerical_preprocess.run(
            data=test,
            train=False,
            file_name="test_preprocessed"
        )
        validate_preprocessed = self.numerical_preprocess.run(
            data=validate,
            train=False,
            file_name="validate_preprocessed"
        )

        # Prepare time series data
        # train
        train_preprocessed = self.__prepare_time_series_data(
            train_preprocessed, lookback, target_column
        )
        train_preprocessed.to_csv(
            save_path + "series_train_preprocessed.csv", index=False
        )
        
        # test
        test_preprocessed = self.__prepare_time_series_data(
            test_preprocessed, lookback, target_column
        )
        test_preprocessed.to_csv(
            save_path + "series_test_preprocessed.csv", index=False
        )
        
        # validation
        validate_preprocessed = self.__prepare_time_series_data(
            validate_preprocessed, lookback, target_column
        )
        validate_preprocessed.to_csv(
            save_path + "series_validate_preprocessed.csv", index=False
        )
        logger.info("TrainPreprocessPipeline completed")
        return train_preprocessed, test_preprocessed, validate_preprocessed
    
    
    @staticmethod
    def __prepare_time_series_data(df: pd.DataFrame, n: int, target_column: str) -> pd.DataFrame:
        """
        Prepares time series data using the last n timestamps and the current timestamp
        to predict the target column.

        Args:
        - df (pd.DataFrame): The input data containing timestamps and feature columns.
        - n (int): The number of previous timestamps to use as features.
        - target_column (str): The name of the target column.

        Returns:
        - pd.DataFrame: A DataFrame where each row contains n+1 timestamps of features and the target value.
        """
        feature_columns = [col for col in df.columns if col != target_column and col != 'time']
        data = []
        columns = []

        # Generate column names for the reorganized DataFrame
        for i in range(n, 0, -1):
            columns += [f"{col}_t-{i}" for col in feature_columns]
        columns += [f"{col}_t" for col in feature_columns]  # Add current timestep features
        columns += [target_column]

        # Populate the rows with shifted data
        for i in tqdm(range(n, len(df)),
                desc="Preparing time series data",
                total=len(df),
                unit=" rows"
                ):
            row_features = df.iloc[i-n:i][feature_columns].values.flatten()  # Last n timesteps
            current_features = df.iloc[i][feature_columns].values  # Current timestep
            row_target = df.iloc[i][target_column]  # Target value
            data.append(list(row_features) + list(current_features) + [row_target])
        
        return pd.DataFrame(data, columns=columns)

class InferencePreprocessPipeline:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.llm_sentiment_analysis = LLMSentimentAnalysis(
            data=self.data,
            save_path=None,
            mock=True)
        self.ta_indicators = TAIndicators()
        self.numerical_preprocess = NumericalPreprocess(
            inference_mode=True
        )
    
    def run(self,
            lookback: int = 6,
            target_column: str = "target"
        ) -> pd.DataFrame:
        """
        Runs the inference preprocessing pipeline to process the input data, 
        apply sentiment analysis, add technical analysis indicators, and prepare 
        the data for time series prediction.

        Args:
            lookback (int): The number of previous time steps to use when preparing time series data. Defaults to 6.
            target_column (str): The name of the target column in the dataset. Defaults to "target".

        Returns:
            pd.DataFrame: The preprocessed data ready for inference.

        Raises:
            Any errors encountered during preprocessing steps will be logged.
        """
        data = self.llm_sentiment_analysis.parse_dataframe()
        data = self.ta_indicators.add_indicators(data)
        data = self.numerical_preprocess.run(
            data,
            train=False
        )
        data = self.__prepare_time_series_data(data, lookback, target_column)
        logger.info("InferencePreprocessPipeline completed")
        
        return data

    @staticmethod
    def __prepare_time_series_data(df: pd.DataFrame, n: int, target_column: str) -> pd.DataFrame:
        """
        Prepares time series data using the last n timestamps and the current timestamp
        to predict the target column.

        Args:
        - df (pd.DataFrame): The input data containing timestamps and feature columns.
        - n (int): The number of previous timestamps to use as features.
        - target_column (str): The name of the target column.

        Returns:
        - pd.DataFrame: A DataFrame where each row contains n+1 timestamps of features and the target value.
        """
        feature_columns = [col for col in df.columns if col != target_column and col != 'time']
        data = []
        columns = []

        # Generate column names for the reorganized DataFrame
        for i in range(n, 0, -1):
            columns += [f"{col}_t-{i}" for col in feature_columns]
        columns += [f"{col}_t" for col in feature_columns]  # Add current timestep features

        # Populate the rows with shifted data
        for i in tqdm(range(n, len(df)),
                desc="Preparing time series data",
                total=len(df),
                unit=" rows"
                ):
            row_features = df.iloc[i-n:i][feature_columns].values.flatten()  # Last n timesteps
            current_features = df.iloc[i][feature_columns].values  # Current timestep
            data.append(list(row_features) + list(current_features))
        
        return pd.DataFrame(data, columns=columns) 
    
if __name__ == "__main__":
    pipeline = TrainPreprocessPipeline()
    train_data, test_data, validate_data = pipeline.run()