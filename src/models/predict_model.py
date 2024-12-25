import pandas as pd
import numpy as np
import tensorflow as tf
# from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import joblib


class PredictModel:
    def __init__(
            self,
            meta_model_path: str = "./models/meta_model.pkl",
            xgb_model_path: str = "./models/xgb_model_top_20.ubj",
            lgb_model_path: str = "./models/lgb_model_top_20.txt",
            gru_model_path: str = "./models/gru_model.h5",
            scaler_path: str = "./artefacts/scaler.pkl"
        ) -> None:
        self.meta_model_path = meta_model_path
        self.xgb_model_path = xgb_model_path
        self.lgb_model_path = lgb_model_path
        self.gru_model_path = gru_model_path
        self.scaler_path = scaler_path

        self.meta_model = self.__load_model("meta")
        self.xgb_model = self.__load_model("xgb")
        self.lgb_model = self.__load_model("lgb")
        self.gru_model = self.__load_model("gru")

        self.scaler = self.__load_scaler(self.scaler_path)

    def stack_predict(self, data: pd.DataFrame) -> float:
        """
        Predicts the target column using the stacked model.
        """
        data = self.__layer_output(data)
        output = self.meta_model.predict(data)
        unscaled_output = self.__invTransform(output)
        return unscaled_output
    
    def __load_model(self, model_name: str):
        """
        Loads the specified model.
        """
        match model_name:
            case "xgb":
                model = XGBRegressor()
                model.load_model(self.xgb_model_path)
            case "lgb":
                model = lgb.Booster(model_file=self.lgb_model_path)
                return model
            case "gru":
                model = tf.keras.models.load_model(self.gru_model_path)
            case "meta":
                model = joblib.load(self.meta_model_path)
            case _:
                raise ValueError(f"Invalid model name: {model_name}")
            
        return model
    
    def __load_scaler(self, scaler_path: str):
        """
        Loads the specified scaler.
        """
        return joblib.load(scaler_path)
    
    def __layer_output(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Returns dataframe with predictions from base models layer.
        """
        xgb_preds = self.xgb_model.predict(data)
        lgb_preds = self.lgb_model.predict(data)
        gru_preds = self.gru_model.predict(
            data.to_numpy().reshape(-1, 1, (data.shape[1]))
        ).flatten()

        return pd.DataFrame({
            "GRU": gru_preds,
            "XGB": xgb_preds,
            "LGB": lgb_preds
        })
    
    def __invTransform(self, data: float) -> float:
        data = np.asarray([data])
        dummy = np.zeros((len(data), 136))
        dummy[:,-1] = data
        dummy = pd.DataFrame(self.scaler.inverse_transform(dummy))
        return dummy.iloc[:,-1].values[0]
        