from datetime import datetime
from time import perf_counter
import json
from math import sqrt
import pandas as pd
from pandas import Index
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, Booster
import lightgbm as lgb
from src.utils.model_train_log_config import logger

# Tensorflow libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as L

class SelectFeatures():
    def __init__(
            self,
            top_k: int = 20,
            train_data_path: str = "./data/processed/series_train_preprocessed.csv",
            test_data_path: str = "./data/processed/series_test_preprocessed.csv",  
            val_data_path: str = "./data/processed/series_validate_preprocessed.csv",
            save_path: str = "./artefacts/",
            report_path: str = "./reports/",         
        ) -> None:
        self.top_k = top_k

        self.__save_path = save_path
        self.__report_path = report_path

        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.val_data_path = val_data_path

        self.train_data: pd.DataFrame = None
        self.test_data: pd.DataFrame = None
        self.val_data: pd.DataFrame = None

        self.__load_data()
        self.X_train, self.y_train = self.__get_X_y(self.train_data)
        self.X_test, self.y_test = self.__get_X_y(self.test_data)
        self.X_val, self.y_val = self.__get_X_y(self.val_data)

        self.rf_rmse: float = None
        self.xgb_rmse: float = None
        self.lgb_rmse: float = None

        self.rf_lstm_rmse: float = None
        self.xgb_lstm_rmse: float = None
        self.lgb_lstm_rmse: float = None

        self.rf_time: float = None
        self.xgb_time: float = None
        self.lgb_time: float = None
        self.rf_lstm_time: float = None
        self.xgb_lstm_time: float = None
        self.lgb_lstm_time: float = None

        self.top_features: list = None

    def run(self) -> list:
        logger.info(
            "Running feature selection pipeline to select top %s features",
            self.top_k
        )
        # Train base models
        logger.info("Training base models")
        start = perf_counter()
        _, self.rf_rmse, rf_feature_importances, rf_feature_names = self.__RF_model()
        self.rf_time = perf_counter() - start

        start = perf_counter()
        _, self.xgb_rmse, xgb_feature_importances, xgb_feature_names = self.__XGB_model()
        self.xgb_time = perf_counter() - start

        start = perf_counter
        _, self.lgb_rmse, lgb_feature_importances, lgb_feature_names = self.__LGB_model()
        self.lgb_time = perf_counter() - start

        # Get LSTM rmse for top k features from each model
        logger.info("Getting LSTM rmse for top %s features", self.top_k)

        start = perf_counter()
        self.rf_lstm_rmse = self.__LSTM_model_rmse(rf_feature_names[:self.top_k])
        self.rf_lstm_time = perf_counter() - start
        logger.info("Random Forest LSTM RMSE: %.4f", self.rf_lstm_rmse)

        start = perf_counter()
        self.xgb_lstm_rmse = self.__LSTM_model_rmse(xgb_feature_names[:self.top_k])
        self.xgb_lstm_time = perf_counter() - start
        logger.info("XGBoost LSTM RMSE: %.4f", self.xgb_lstm_rmse)

        start = perf_counter()
        self.lgb_lstm_rmse = self.__LSTM_model_rmse(lgb_feature_names[:self.top_k])
        self.lgb_lstm_time = perf_counter() - start
        logger.info("LightGBM LSTM RMSE: %.4f", self.lgb_lstm_rmse)

        logger.info("Selecting top %s features", self.top_k)
        top_features = self.__weighted_top_features(
            rf_feature_importances,
            xgb_feature_importances,
            lgb_feature_importances,
            rf_feature_names,
            xgb_feature_names,
            lgb_feature_names,
            self.rf_lstm_rmse,
            self.xgb_lstm_rmse,
            self.lgb_lstm_rmse
        )

        self.__save_top_features(top_features)
        self.generate_report()
        return top_features

    def __save_top_features(self, top_features: list) -> None:
        try:  
            joblib.dump(
                top_features,
                self.__save_path + f"top_{self.top_k}_features_w_pipeline.pkl"
            )
            logger.info("Top %s features saved to %s", self.top_k, self.__save_path)
        except Exception as e:
            logger.error("Error saving top features: %s", e)


    def __weighted_top_features(
        self,
        rf_feature_importances: np.ndarray,
        xgb_feature_importances: np.ndarray,
        lgb_feature_importances: np.ndarray,
        rf_sorted_features: Index,
        xgb_sorted_features: Index,
        lgb_sorted_features: Index,
        rmse_rf_lstm: float,
        rmse_xgb_lstm: float,
        rmse_lgb_lstm: float
    ) -> list:
        
        # get set of top k features common to all models
        all_k = []
        for list in rf_sorted_features[:self.top_k], xgb_sorted_features[:self.top_k], lgb_sorted_features[:self.top_k]:
            all_k.extend(list)
        top_features_k = set(all_k)
        
        # Scale feature importances
        scaled_rf_feature_importances = MinMaxScaler().fit_transform(
            rf_feature_importances.reshape(-1, 1)
        )
        scaled_xgb_feature_importances = MinMaxScaler().fit_transform(
            xgb_feature_importances.reshape(-1, 1)
        )    
        scaled_lgb_feature_importances = MinMaxScaler().fit_transform(
            lgb_feature_importances.reshape(-1, 1)
        )

        # Calculate weighted average of feature importances
        scores = []
        for feature in top_features_k:
            feature_score = sum([
                (scaled_rf_feature_importances[
                    rf_sorted_features.to_list().index(feature)
                    ][0])/rmse_rf_lstm,
                (scaled_xgb_feature_importances[
                    xgb_sorted_features.to_list().index(feature)
                    ][0])/rmse_xgb_lstm,
                (scaled_lgb_feature_importances[
                    lgb_sorted_features.to_list().index(feature)
                    ][0])/rmse_lgb_lstm
            ])
            scores.append((feature, feature_score))

        # Sort features by weighted average
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        return [feature for feature, _ in sorted_scores][:self.top_k]

    def __load_data(self) -> None:
        try:
            self.train_data = pd.read_csv(self.train_data_path)
            self.test_data = pd.read_csv(self.test_data_path)
            self.val_data = pd.read_csv(self.val_data_path)
            logger.info("Train, test, and val data loaded successfully")
        except FileNotFoundError as e:
            logger.error("Error loading train, test, and val data: %s", e)

    def __get_X_y(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = data.drop("target", axis=1)
        y = data["target"]
        return X, y
    
    def __RF_model(self) -> tuple[
        RandomForestRegressor,
        float,
        np.ndarray,
        Index
    ]:
        logger.info("Training Random Forest model")
        model = RandomForestRegressor(
            n_estimators=100,         # Number of trees in the forest
            max_depth=15,             # Maximum depth of each tree
            min_samples_split=10,     # Minimum samples required to split an internal node
            min_samples_leaf=4,       # Minimum samples required to be at a leaf node
            max_features='sqrt',      # Number of features to consider for best split
            bootstrap=True,           # Whether to use bootstrap samples
            n_jobs=-1,                # Use all available CPU cores
            random_state=42           # For reproducibility
        )
        model.fit(self.X_train, self.y_train)

        logger.info("Evaluating Random Forest model")
        y_pred = model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)
        logger.info("Random Forest RMSE: %.4f", rmse)

        feature_importances = model.feature_importances_
        feature_names = self.X_train.columns

        # Sort features by importance
        sorted_idx = np.argsort(feature_importances)[::-1]
        feature_importances = feature_importances[sorted_idx]
        feature_names = feature_names[sorted_idx]

        return model, rmse, feature_importances, feature_names
    
    def __XGB_model(self) -> tuple[
        XGBRegressor,
        float,
        np.ndarray,
        Index
    ]:
        logger.info("Training XGBoost model")
        model = XGBRegressor(
            n_estimators=500,           # Number of trees
            learning_rate=0.1,          # Step size for weight updates
            max_depth=6,                # Depth of each tree
            min_child_weight=10,        # Minimum sum of weights in child nodes
            subsample=0.8,              # Subsample ratio for training data
            colsample_bytree=0.8,       # Subsample ratio for features
            gamma=0,                    # Minimum loss reduction for further partitioning
            reg_alpha=0.1,              # L1 regularization (prevents overfitting)
            reg_lambda=1.0,             # L2 regularization (prevents overfitting)
            random_state=42,            # Ensures reproducibility
            tree_method='hist',         # Faster tree construction for large datasets
            n_jobs=-1,                  # Use all available CPU threads
            early_stopping_rounds=10
        )
        model.fit(self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=0
        )

        logger.info("Evaluating XGBoost model")
        y_pred = model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)
        logger.info("XGBoost RMSE: %.4f", rmse)

        feature_importances = model.feature_importances_
        feature_names = self.X_train.columns

        # Sort features by importance
        sorted_idx = np.argsort(feature_importances)[::-1]
        feature_importances = feature_importances[sorted_idx]
        feature_names = feature_names[sorted_idx]

        return model, rmse, feature_importances, feature_names
    
    def __LGB_model(self) -> tuple[
        lgb.Booster,
        float,
        np.ndarray,
        Index
    ]:
        logger.info("Training LightGBM model")
        params = {
            "boosting_type": "gbdt",           # Gradient Boosting Decision Tree
            "objective": "regression",         # Regression task
            "metric": "rmse",                  # Root Mean Squared Error for evaluation
            "num_leaves": 31,                  # Maximum number of leaves in one tree
            "learning_rate": 0.05,             # Lower learning rate with more boosting rounds
            "feature_fraction": 0.8,           # Percentage of features used per tree
            "bagging_fraction": 0.8,           # Percentage of data used per tree
            "bagging_freq": 5,                 # Perform bagging every 5 iterations
            "max_depth": -1,                   # No maximum depth restriction
            "lambda_l1": 0.1,                  # L1 regularization
            "lambda_l2": 0.2,                  # L2 regularization
            "verbosity": -1,                   # Suppress output
            "n_jobs": -1,                      # Use all CPU cores
            "seed": 42,                        # Seed for reproducibility
            "early_stopping_rounds": 10,       # Stop if no improvement in 10 rounds
            "verbose_eval": 10                 # Print evaluation results every 10 rounds
        }

        train_dataset = lgb.Dataset(self.X_train, label=self.y_train)
        val_dataset = lgb.Dataset(self.X_val, label=self.y_val, reference=train_dataset)

        model = lgb.train(
            params,
            train_dataset,
            valid_sets=[train_dataset, val_dataset],
            num_boost_round=1000
        )

        logger.info("Evaluating LightGBM model")
        y_pred = model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)
        logger.info("LightGBM RMSE: %.4f", rmse)

        feature_importances = model.feature_importance(importance_type="gain")
        feature_names = self.X_train.columns

        # Sort features by importance
        sorted_idx = np.argsort(feature_importances)[::-1]
        feature_importances = feature_importances[sorted_idx]
        feature_names = feature_names[sorted_idx]

        return model, rmse, feature_importances, feature_names
    
    def __LSTM_model_rmse(self, top_features: list) -> float:
        timesteps = 1
        features = self.X_train[top_features].shape[2]
        lr = 0.0001
        epochs = 100
        batch_size = 128
        patience = int(sqrt(epochs))

        logger.info("Training LSTM model")
        # Define the model
        model = Sequential()
        model.add(L.Bidirectional(L.LSTM(timesteps, activation='tanh', input_shape=(timesteps, features), return_sequences=True)))
        model.add(L.LSTM(256, activation='tanh', return_sequences=True))
        model.add(L.LSTM(128, activation='tanh', return_sequences=True))
        model.add(L.LSTM(64, activation='tanh', return_sequences=False))
        model.add(L.BatchNormalization())
        model.add(L.RepeatVector(timesteps))
        model.add(L.LSTM(timesteps, activation='tanh', return_sequences=True))
        model.add(L.BatchNormalization())
        model.add(L.LSTM(64, activation='tanh', return_sequences=True))
        model.add(L.LSTM(128, activation='tanh', return_sequences=True))
        model.add(L.LSTM(256, activation='tanh', return_sequences=True))
        model.add(L.Bidirectional(L.LSTM(128, activation='tanh', return_sequences=False)))
        model.add(L.Dropout(0.2))
        model.add(L.Dense(1))

        # Compile the model
        adam = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer=adam, loss='mse', metrics=['mse'])  
        model.build(input_shape=(None, timesteps, features))
        
        # Prepare the data
        X_train = self.X_train[top_features].values.reshape(-1, timesteps, features)
        X_val = self.X_val.values.reshape(-1, timesteps, features)
        X_test = self.X_test.values.reshape(-1, timesteps, features)
        
        # Train the model
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            restore_best_weights=True,
        )

        model.fit(
            X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(
                X_val,
                self.y_val,
            ),
            callbacks=[early_stopping],
        )

        logger.info("Evaluating LSTM model")
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)

        return rmse
    
    def generate_report(self) -> None:
        report = {
            "top_k": self.top_k,
            "top_k_features": self.top_features,
            "count_init_features": len(self.X_train.columns),
            "base_rf_rmse": self.rf_rmse,
            "base_xgb_rmse": self.xgb_rmse,
            "base_lgb_rmse": self.lgb_rmse,
            "rf_features_LSTM_rmse": self.rf_lstm_rmse,
            "xgb_features_LSTM_rmse": self.xgb_lstm_rmse,
            "lgb_features_LSTM_rmse": self.lgb_lstm_rmse,
            "rf_train_time": self.rf_time,
            "xgb_train_time": self.xgb_time,
            "lgb_train_time": self.lgb_time,
            "rf_LSTM_train_time": self.rf_lstm_time,
            "xgb_LSTM_train_time": self.xgb_lstm_time,
            "lgb_LSTM_train_time": self.lgb_lstm_time,

        }
        filename = self.__report_path+\
            "FS_report_{}_{}.json".format(
                self.top_k,
                datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            )
        with open(filename, "w") as f:
            json.dump(report, f, indent=4)

        logger.info("Feature Selection Report saved to %s", filename)

class TrainStackedModel(SelectFeatures):
    def __init__(
            self,
            save_path: str = "./models/",
            top_k_features: list = None,
            top_features_path: str = "./artefacts/top_20_features_w_pipeline.pkl" 
        ) -> None:
        super().__init__(save_path=save_path)
        self.top_k_features = top_k_features
        self.top_features_path = top_features_path
        self.__weighted_top_features()

        self.X_train, self.y_train = self.__get_X_y(self.train_data)
        self.X_test, self.y_test = self.__get_X_y(self.test_data)
        self.X_val, self.y_val = self.__get_X_y(self.val_data)

        self.trained_xgb_model: Booster = None
        self.trained_lgb_model: lgb.Booster = None
        self.trained_gru_model: tf.keras.Model = None
        self.trained_meta_model: MLPRegressor = None

        self.xgb_rmse: float = None
        self.lgb_rmse: float = None
        self.gru_rmse: float = None
        self.meta_rmse: float = None

        self.xgb_train_time: float = None
        self.lgb_train_time: float = None
        self.gru_train_time: float = None
        self.meta_train_time: float = None

    def train_stack(self) -> None:
        logger.info("Training Stacked Model")

        start = perf_counter()
        self.trained_xgb_model, self.xgb_rmse = self.__XGB_model()
        self.xgb_train_time = perf_counter() - start

        start = perf_counter()
        self.trained_lgb_model, self.lgb_rmse = self.__LGB_model()
        self.lgb_train_time = perf_counter() - start

        start = perf_counter()
        self.trained_gru_model, self.gru_rmse = self.__GRU_model()
        self.gru_train_time = perf_counter() - start

        X_train_stacked, y_train_stacked = self.__split_data(
            self.X_train,
            self.y_train
        )

        layer_output = self.__layer_output(X_train_stacked)

        start = perf_counter()
        self.trained_meta_model, self.meta_rmse = self.__meta_model(
            layer_output,
            y_train_stacked
        )
        self.meta_train_time = perf_counter() - start

    def __layer_output(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions from base models (GRU, XGBoost, LightGBM).

        Args:
            data (pd.DataFrame): Input data for prediction.

        Returns:
            pd.DataFrame: DataFrame with predictions from 'GRU', 'XGB', and 'LGB' models.

        Notes:
            - GRU model input is reshaped to (-1, 1, n_features).
        """
        logger.info("Generating predictions from base models")
        xgb_preds: np.ndarray = self.trained_xgb_model.predict(data)
        lgb_preds: np.ndarray = self.trained_lgb_model.predict(data)
        gru_preds: np.ndarray = self.trained_gru_model.predict(
            data.to_numpy().reshape(-1, 1, (data.shape[1])),
            verbose=0
        ).flatten()

        return pd.DataFrame({
            "GRU": gru_preds,
            "XGB": xgb_preds,
            "LGB": lgb_preds
        })


    def __XGB_model(self) -> tuple[
            XGBRegressor,
            float
        ]:
        logger.info("Training XGBoost model")
        model = XGBRegressor(
            n_estimators=60,           # Number of trees
            learning_rate=0.1,          # Step size for weight updates
            max_depth=20,                # Depth of each tree
            min_child_weight=10,        # Minimum sum of weights in child nodes
            subsample=0.8,              # Subsample ratio for training data
            colsample_bytree=0.8,       # Subsample ratio for features
            gamma=0,                    # Minimum loss reduction for further partitioning
            reg_alpha=0.1,              # L1 regularization (prevents overfitting)
            reg_lambda=1.0,             # L2 regularization (prevents overfitting)
            random_state=42,            # Ensures reproducibility
            tree_method='hist',         # Faster tree construction for large datasets
            n_jobs=-1,                  # Use all available CPU threads
            # early_stopping_rounds=10
        )
        model.fit(self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=0
        )

        logger.info("Evaluating XGBoost model")
        y_pred = model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)
        logger.info(f"XGBoost RMSE: {rmse}")

        model.save_model(f"{self.__save_path}xgb_model_w_pipeline.ubj")
        logger.info("XGBoost model saved to %s", self.__save_path)

        return model, rmse
    
    def __LGB_model(self) -> tuple[
        lgb.Booster,
        float
    ]:
        logger.info("Training LightGBM model")
        params = {
            "boosting_type": "gbdt",           # Gradient Boosting Decision Tree
            "objective": "regression",         # Regression task
            "metric": "rmse",                  # Root Mean Squared Error for evaluation
            "num_leaves": 31,                  # Maximum number of leaves in one tree
            "learning_rate": 0.05,             # Lower learning rate with more boosting rounds
            "feature_fraction": 0.8,           # Percentage of features used per tree
            "bagging_fraction": 0.8,           # Percentage of data used per tree
            "bagging_freq": 5,                 # Perform bagging every 5 iterations
            "max_depth": -1,                   # No maximum depth restriction
            "lambda_l1": 0.1,                  # L1 regularization
            "lambda_l2": 0.2,                  # L2 regularization
            "verbosity": -1,                   # Suppress output
            "n_jobs": -1,                      # Use all CPU cores
            "seed": 42,                        # Seed for reproducibility
            "early_stopping_rounds": 10,       # Stop if no improvement in 10 rounds
            "verbose_eval": 10                 # Print evaluation results every 10 rounds
        }

        train_dataset = lgb.Dataset(self.X_train, label=self.y_train)
        val_dataset = lgb.Dataset(self.X_val, label=self.y_val, reference=train_dataset)

        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=1000,
            valid_sets=[train_dataset, val_dataset]
        )

        logger.info("Evaluating LightGBM model")
        y_pred = model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)
        logger.info(f"LightGBM RMSE: {rmse}")

        model.save_model(
            f"{self.__save_path}lgb_model_w_pipeline.txt",
            num_iteration=model.best_iteration
        )
        logger.info("LightGBM model saved to %s", self.__save_path)

        return model, rmse


    def __GRU_model(self) -> tuple[
        tf.keras.Model,
        float
    ]:
        timesteps = 1
        features = self.X_train.shape[2]
        lr = 0.0001
        epochs = 100
        batch_size = 128
        patience = int(sqrt(epochs))

        logger.info("Training GRU model")
        # Define the model
        model = Sequential()
        model.add(L.GRU(units=256, return_sequences=True, input_shape=(timesteps, features), recurrent_dropout=0.2))
        model.add(L.Dropout(0.2))
        model.add(L.GRU(units=128, return_sequences=True, recurrent_dropout=0.2))
        model.add(L.Dropout(0.2))
        model.add(L.GRU(units=64, return_sequences=False))
        model.add(L.Dense(1, activation='linear'))

        # Compile the model
        adam = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer=adam, loss='mse', metrics=['mse'])  

        # Prepare the data
        X_train = self.X_train.values.reshape(-1, timesteps, features)
        X_val = self.X_val.values.reshape(-1, timesteps, features)
        X_test = self.X_test.values.reshape(-1, timesteps, features)

        # Train the model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=patience, 
                                                mode='min',
                                                restore_best_weights=True)
        model.fit(
            X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(
                X_val,
                self.y_val
            ),
            callbacks=[early_stopping]
        )

        logger.info("Evaluating GRU model")
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)
        logger.info(f"GRU RMSE: {rmse}")

        model.save(f"{self.__save_path}gru_model_w_pipeline.h5")
        logger.info("GRU model saved to %s", self.__save_path)

        return model, rmse

    def __meta_model(
            self,
            features_stacked: pd.DataFrame,
            target_stacked: pd.Series
        ) -> tuple[
        MLPRegressor,
        float
        ]:

        logger.info("Training meta model")
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=200,
            shuffle=True,
            random_state=42,
            tol=0.0001,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=True,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            n_iter_no_change=10,
            max_fun=15000
        )

        # Split the data
        X_train_stacking, X_test_stacking, y_train_stacking, y_test_stacking = train_test_split(
            features_stacked,
            target_stacked,
            test_size=0.2,
            random_state=42,
            shuffle=False
        )

        model.fit(X_train_stacking, y_train_stacking)

        logger.info("Evaluating meta model")
        y_pred = model.predict(X_test_stacking)
        rmse = root_mean_squared_error(y_test_stacking, y_pred)
        logger.info(f"Meta model RMSE: {rmse}")

        joblib.dump(model, f"{self.__save_path}meta_model_w_pipeline.pkl")
        logger.info("Meta model saved to %s", self.__save_path)

        return model, rmse

    def __weighted_top_features(self) -> None:
        logger.info("Loading top %s features", self.top_k)
        if self.top_k_features is None:
            self.top_k_features: list = joblib.load(self.top_features_path)

    def __get_X_y(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        logger.info("Splitting data into X and y for top %s features", self.top_k)
        data = data[self.top_k_features]
        X = data.drop("target", axis=1)
        y = data["target"]
        return X, y
    
    def __split_data(
            self,
            features: pd.DataFrame,
            target: pd.Series,
            backtest_size: float=0.3,
            save: bool=True,
            save_path: str="./data/processed/"
        ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        logger.info("Splitting stacked training data into train, test, and backtest sets")
        split_idx = int(features.shape[0] * (1 - backtest_size))
        X_train = features[:split_idx]
        y_train = target[:split_idx]
        X_backtest = features[split_idx:]
        y_backtest = target[split_idx:]

        if save:
            X_backtest.to_csv(save_path + "X_backtest.csv", index=False)
            y_backtest.to_csv(save_path + "y_backtest.csv", index=False)
            logger.info("Backtest data saved to %s", save_path)
        return X_train, y_train

        
    def generate_report(self) -> None:
        report = {
            "features": self.top_k_features,
            "xgb_rmse": self.xgb_rmse,
            "lgb_rmse": self.lgb_rmse,
            "gru_rmse": self.gru_rmse,
            "meta_rmse": self.meta_rmse,
            "xgb_train_time": self.xgb_train_time,
            "lgb_train_time": self.lgb_train_time,
            "gru_train_time": self.gru_train_time,
            "meta_train_time": self.meta_train_time
        }
        filename = self.__report_path+\
            "SM_training_report_{}_{}.json".format(
                self.top_k,
                datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            )
        with open(filename, "w") as f:
            json.dump(report, f, indent=4)

        logger.info("Stacked Model Training Report saved to %s", filename)