from typing import Any
from math import sqrt
from tqdm import tqdm
import pandas as pd
from pandas import Index
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import lightgbm as lgb


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
            save_path: str = "./artefacts/"         
        ) -> None:
        self.top_k = top_k
        self.__save_path = save_path

        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.val_data_path = val_data_path

        self.train_data = None
        self.test_data = None
        self.val_data = None

        self.__load_data()
        self.X_train, self.y_train = self.__get_X_y(self.train_data)
        self.X_test, self.y_test = self.__get_X_y(self.test_data)
        self.X_val, self.y_val = self.__get_X_y(self.val_data)

    def run(self) -> list:
        # Train base models
        _, rf_rmse, rf_feature_importances, rf_feature_names = self.__RF_model()
        _, xgb_rmse, xgb_feature_importances, xgb_feature_names = self.__XGB_model()
        _, lgb_rmse, lgb_feature_importances, lgb_feature_names = self.__LGB_model()

        # Get LSTM rmse for top k features from each model
        rf_lstm_rmse = self.__LSTM_model_rmse(rf_feature_names[:self.top_k])
        xgb_lstm_rmse = self.__LSTM_model_rmse(xgb_feature_names[:self.top_k])
        lgb_lstm_rmse = self.__LSTM_model_rmse(lgb_feature_names[:self.top_k])

        top_features = self.__weighted_top_features(
            rf_feature_importances,
            xgb_feature_importances,
            lgb_feature_importances,
            rf_feature_names,
            xgb_feature_names,
            lgb_feature_names,
            rf_lstm_rmse,
            xgb_lstm_rmse,
            lgb_lstm_rmse
        )

        self.__save_top_features(top_features)
        return top_features

    def __save_top_features(self, top_features: list) -> None:  
        joblib.dump(
            top_features,
            self.__save_path + f"top_{self.top_k}_features_w_pipeline.pkl"
        )


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
        self.train_data = pd.read_csv(self.train_data_path)
        self.test_data = pd.read_csv(self.test_data_path)
        self.val_data = pd.read_csv(self.val_data_path)

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

        y_pred = model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)

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

        y_pred = model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)

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

        y_pred = model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)

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

        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)

        return rmse
    
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


    def __XGB_model(self) -> tuple[
            XGBRegressor,
            float
        ]:
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

        y_pred = model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)

        return model, rmse
    
    def __LGB_model(self) -> tuple[
        lgb.Booster,
        float
    ]:
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

        y_pred = model.predict(self.X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)

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
        model.compile(optimizer=adam, loss='mse', metrics=['mse'])  # Mean squared error for regression, mean absolute error
        model.build(input_shape=(None, timesteps, features))

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

        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(self.y_test, y_pred)

        return model, rmse

    def __weighted_top_features(self) -> None:
        if self.top_k_features is None:
            self.top_k_features: list = joblib.load(self.top_features_path)

    def __get_X_y(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        data = data[self.top_k_features]
        X = data.drop("target", axis=1)
        y = data["target"]
        return X, y
        
