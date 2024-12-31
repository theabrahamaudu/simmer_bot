import joblib
import yaml
from tqdm import tqdm
import pandas as pd
from src.features.preprocess_pipeline import InferencePreprocessPipeline
from src.models.predict_model import PredictModel

def get_predictions(
        preds_path: str = "./data/processed/predictions.pkl",
        backtest_data_path: str = "./data/processed/X_backtest.csv",
        raw_data_path: str = "./data/interim/merged_scrapped_n_price.csv",
    ) -> list:
    """
    Retrieves or generates predictions for the backtest dataset.

    Args:
        preds_path (str): Path to the file containing precomputed predictions. Defaults to 
            "./data/processed/predictions.pkl".
        backtest_data_path (str): Path to the backtest dataset file. Defaults to 
            "./data/processed/X_backtest.csv".
        raw_data_path (str): Path to the raw dataset file. Defaults to 
            "./data/interim/merged_scrapped_n_price.csv".

    Returns:
        list: A list of predictions.

    Raises:
        FileNotFoundError: If required data files are missing.

    Notes:
        - If predictions are not already saved in the file specified by `preds_path`, the function
          generates them by preprocessing the raw data and using the model's prediction pipeline.
        - Predictions are saved to the `preds_path` after generation for future use.

    Example:
        predictions = get_predictions()
    """
    try:
        print("Loading predictions...")
        predictions: list = joblib.load(preds_path)
    except FileNotFoundError as e:
        print("Error loading predictions...")
        raise FileNotFoundError(f"File not found: {e.filename}") from e
    
    if predictions:
        return predictions
    
    else:
        try:
            print("Generating predictions from raw data...")
            with open("./config/config.yaml", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            backtest_data = pd.read_csv(backtest_data_path)
            raw_data = pd.read_csv(raw_data_path)
            raw_data = raw_data.iloc[-(len(backtest_data)+205):,:]
            raw_data.reset_index(drop=True, inplace=True)
            preprocessed_data = InferencePreprocessPipeline(
                raw_data,
                mock=bool(config["mock_sentiment_backtest"])
            ).run()
            model = PredictModel()
            predictions = []
            for index, _ in tqdm(
                preprocessed_data.iterrows(),
                total=len(backtest_data),
                desc="Generating predictions",
                unit=" rows"
                ):
                predictions.append(
                    model.stack_predict(
                        pd.DataFrame(preprocessed_data.loc[[index],:])
                    )
                )
            joblib.dump(predictions, preds_path)
            return predictions
        except FileNotFoundError as e:
            print("Error generating predictions...")
            raise FileNotFoundError(f"File not found: {e.filename}") from e

PREDICTIONS = get_predictions()
