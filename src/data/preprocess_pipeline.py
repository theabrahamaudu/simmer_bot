import pandas as pd
from src.data.preprocess_utils import (
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
        logger.info("Loading data from %s", self.__source_file_path)
        try:
            return pd.read_csv(self.__source_file_path)
        except FileNotFoundError as e:
            logger.error("Error loading data from %s:\n %s", self.__source_file_path, e)
    
    def run(self):
        data = self.__load_data()
        
        data = self.ta_indicators.add_indicators(data)
        train, test, validate = self.split_data.raw_split(data)
        train_preprocessed = self.numerical_preprocess.run(
            train,
            file_name="train_preprocessed"
        )
        
        test_preprocessed = self.numerical_preprocess.run(
            test,
            train=False,
            file_name="test_preprocessed"
        )
        validate_preprocessed = self.numerical_preprocess.run(
            validate,
            train=False,
            file_name="validate_preprocessed"
        )
        
        return train_preprocessed, test_preprocessed, validate_preprocessed
    
    def run_with_llm_sentiment(self):
        data = self.__load_data()
        data = self.llm_sentiment_analysis.parse_dataframe()
        data = self.ta_indicators.add_indicators(data)
        train, test, validate = self.split_data.raw_split(data)
        train_preprocessed = self.numerical_preprocess.run(
            train,
            file_name="train_preprocessed"
        )
        
        test_preprocessed = self.numerical_preprocess.run(
            test,
            train=False,
            file_name="test_preprocessed"
        )
        validate_preprocessed = self.numerical_preprocess.run(
            validate,
            train=False,
            file_name="validate_preprocessed"
        )
        
        return train_preprocessed, test_preprocessed, validate_preprocessed

class InferencePreprocessPipeline:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.llm_sentiment_analysis = LLMSentimentAnalysis()
        self.ta_indicators = TAIndicators()
        self.numerical_preprocess = NumericalPreprocess()
    
    def run(self):
        data = self.llm_sentiment_analysis.parse_dataframe(self.data)
        data = self.ta_indicators.add_indicators(data)
        data = self.numerical_preprocess.run(
            data,
            train=False
        )
        
        return 
    
if __name__ == "__main__":
    pipeline = TrainPreprocessPipeline()
    train, test, validate = pipeline.run()