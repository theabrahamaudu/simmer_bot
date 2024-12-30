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
        """
        Executes the feature selection pipeline to select the top `k` features based on 
        Random Forest, XGBoost, and LightGBM feature importances and their performance 
        with an LSTM model.

        The pipeline involves:
            1. Training base models (Random Forest, XGBoost, LightGBM) and calculating their RMSE.
            2. Calculating LSTM model RMSE using the top `k` features from each base model.
            3. Aggregating feature importances and selecting the top features using a weighted approach.
            4. Saving the selected top features and generating a report.

        Returns:
            list: A list of the top `k` selected features.

        Attributes:
            top_k (int): The number of top features to select.
            rf_rmse (float): RMSE of the Random Forest model.
            xgb_rmse (float): RMSE of the XGBoost model.
            lgb_rmse (float): RMSE of the LightGBM model.
            rf_lstm_rmse (float): RMSE of the LSTM model using top `k` features from Random Forest.
            xgb_lstm_rmse (float): RMSE of the LSTM model using top `k` features from XGBoost.
            lgb_lstm_rmse (float): RMSE of the LSTM model using top `k` features from LightGBM.
            rf_time (float): Execution time for training the Random Forest model.
            xgb_time (float): Execution time for training the XGBoost model.
            lgb_time (float): Execution time for training the LightGBM model.
            rf_lstm_time (float): Execution time for calculating LSTM RMSE with Random Forest features.
            xgb_lstm_time (float): Execution time for calculating LSTM RMSE with XGBoost features.
            lgb_lstm_time (float): Execution time for calculating LSTM RMSE with LightGBM features.

        Example:
            selected_features = feature_selector.run()

        Raises:
            Any exceptions encountered during the feature selection process will be logged and raised.
        """
        logger.info(
            "Running feature selection pipeline to select top %s features",
            self.top_k
        )
        # Train base models
        logger.info("Training base models")
        try:
            start = perf_counter()
            _, self.rf_rmse, rf_feature_importances, rf_feature_names = self.__RF_model()
            self.rf_time = perf_counter() - start

            start = perf_counter()
            _, self.xgb_rmse, xgb_feature_importances, xgb_feature_names = self.__XGB_model()
            self.xgb_time = perf_counter() - start

            start = perf_counter()
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
            self.top_features = self.__weighted_top_features(
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

            self.__save_top_features(self.top_features)
            self.generate_report()
            logger.info("Top %s features selected: %s", self.top_k, self.top_features)
            return self.top_features
        except Exception as e:
            logger.exception("Error running feature selection pipeline: %s", e)

    def __save_top_features(self, top_features: list) -> None:
        """
        Saves the selected top features to a file.

        The top features are serialized using `joblib` and stored as a `.pkl` file 
        in the specified save path. Logs the operation's success or failure.

        Args:
            top_features (list): A list of the top features to save.

        Raises:
            Exception: If an error occurs while saving the file, it logs the error 
                    and re-raises the exception.

        Example:
            self.__save_top_features(["feature1", "feature2", "feature3"])
        """
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
        """
        Computes the weighted top features based on feature importances and model RMSE.

        This method identifies the top `k` features shared across Random Forest (RF),
        XGBoost (XGB), and LightGBM (LGB) models. It calculates a weighted average of 
        feature importances scaled using Min-Max Scaling, adjusted by the inverse of 
        each model's LSTM RMSE.

        Args:
            rf_feature_importances (np.ndarray): Feature importances from the RF model.
            xgb_feature_importances (np.ndarray): Feature importances from the XGB model.
            lgb_feature_importances (np.ndarray): Feature importances from the LGB model.
            rf_sorted_features (Index): Features sorted by importance in the RF model.
            xgb_sorted_features (Index): Features sorted by importance in the XGB model.
            lgb_sorted_features (Index): Features sorted by importance in the LGB model.
            rmse_rf_lstm (float): RMSE of the LSTM model trained on RF-selected features.
            rmse_xgb_lstm (float): RMSE of the LSTM model trained on XGB-selected features.
            rmse_lgb_lstm (float): RMSE of the LSTM model trained on LGB-selected features.

        Returns:
            list: A list of the top `k` features ranked by weighted average importance.

        Example:
            top_features = self.__weighted_top_features(
                rf_importances, xgb_importances, lgb_importances,
                rf_features, xgb_features, lgb_features,
                rmse_rf, rmse_xgb, rmse_lgb
            )
        """
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
        """
        Loads training, testing, and validation data from CSV files.

        This method attempts to load the train, test, and validation data from the 
        specified file paths. If any of the files are not found, an error message is 
        logged. Upon successful loading, a success message is logged.

        Attributes:
            train_data (pd.DataFrame): The loaded training dataset.
            test_data (pd.DataFrame): The loaded test dataset.
            val_data (pd.DataFrame): The loaded validation dataset.

        Raises:
            FileNotFoundError: If any of the CSV files are not found at the specified paths.

        Example:
            self.load_data()
        """
        try:
            self.train_data = pd.read_csv(self.train_data_path)
            self.test_data = pd.read_csv(self.test_data_path)
            self.val_data = pd.read_csv(self.val_data_path)
            logger.info("Train, test, and val data loaded successfully")
        except FileNotFoundError as e:
            logger.error("Error loading train, test, and val data: %s", e)

    def __get_X_y(self, data: pd.DataFrame, target: str="target") -> tuple[pd.DataFrame, pd.Series]:
        """
        Splits the input data into features (X) and target (y).

        Args:
            data (pd.DataFrame): The input DataFrame containing both features and target.
            target (str, optional): The name of the target column. Defaults to "target".

        Returns:
            tuple: A tuple containing:
                - X (pd.DataFrame): The features (all columns except the target column).
                - y (pd.Series): The target column.

        Example:
            X, y = self.__get_X_y(data)
        """
        X = data.drop(target, axis=1)
        y = data[target]
        return X, y
    
    def __RF_model(self) -> tuple[
        RandomForestRegressor,
        float,
        np.ndarray,
        Index
    ]:
        """
        Trains a Random Forest model and evaluates its performance.

        This method initializes and trains a Random Forest model using the training 
        dataset and evaluates its performance on the test set. It returns the trained model, 
        the Root Mean Squared Error (RMSE) of the model, and the feature importances with their 
        corresponding feature names, sorted by importance.

        Args:
            None

        Returns:
            tuple: A tuple containing:
                - model (RandomForestRegressor): The trained Random Forest model.
                - rmse (float): The Root Mean Squared Error (RMSE) of the model on the test data.
                - feature_importances (np.ndarray): Array of feature importances.
                - feature_names (Index): Sorted feature names corresponding to the feature importances.

        Example:
            model, rmse, feature_importances, feature_names = self.__RF_model()
        """
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
        """
        Trains an XGBoost model and evaluates its performance.

        This method initializes and trains an XGBoost model using the training 
        dataset and evaluates its performance on the test set. It returns the trained model, 
        the Root Mean Squared Error (RMSE) of the model, and the feature importances with their 
        corresponding feature names, sorted by importance.

        Args:
            None

        Returns:
            tuple: A tuple containing:
                - model (XGBRegressor): The trained XGBoost model.
                - rmse (float): The Root Mean Squared Error (RMSE) of the model on the test data.
                - feature_importances (np.ndarray): Array of feature importances.
                - feature_names (Index): Sorted feature names corresponding to the feature importances.

        Example:
            model, rmse, feature_importances, feature_names = self.__XGB_model()
        """
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
        """
        Trains a LightGBM model and evaluates its performance.

        This method initializes and trains a LightGBM model using the training dataset and evaluates 
        its performance on the test set. It returns the trained model, the Root Mean Squared Error (RMSE) 
        of the model, and the feature importances with their corresponding feature names, sorted by importance.

        Args:
            None

        Returns:
            tuple: A tuple containing:
                - model (lgb.Booster): The trained LightGBM model.
                - rmse (float): The Root Mean Squared Error (RMSE) of the model on the test data.
                - feature_importances (np.ndarray): Array of feature importances.
                - feature_names (Index): Sorted feature names corresponding to the feature importances.

        Example:
            model, rmse, feature_importances, feature_names = self.__LGB_model()
        """
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
        """
        Trains an LSTM model using the provided top features and calculates its RMSE.

        This method trains a Bidirectional LSTM model on the given top features from the training data, 
        evaluates its performance on the test data, and returns the Root Mean Squared Error (RMSE) of 
        the model's predictions. The model is trained with early stopping to prevent overfitting.

        Args:
            top_features (list): A list of the top features to be used for training the LSTM model.

        Returns:
            float: The Root Mean Squared Error (RMSE) of the LSTM model on the test data.

        Example:
            rmse = self.__LSTM_model_rmse(top_features)
        """
        timesteps = 1
        features = self.X_train[top_features].shape[1]
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
        X_val = self.X_val[top_features].values.reshape(-1, timesteps, features)
        X_test = self.X_test[top_features].values.reshape(-1, timesteps, features)
        
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
        """
        Generates and saves a feature selection report as a JSON file.

        This method creates a detailed report containing information about the feature selection 
        process, including RMSE scores for base models (Random Forest, XGBoost, LightGBM) and 
        their corresponding LSTM-based feature subsets, training times, and the top selected features. 
        The report is saved as a JSON file with a timestamped filename.

        The report includes the following data:
        - Top k features selected
        - Initial number of features used for training
        - RMSE values for base models and their LSTM-based feature subsets
        - Training times for base models and LSTM models

        The generated report is saved to the specified report path with a unique filename based 
        on the `top_k` value and the current timestamp.

        Args:
            None

        Returns:
            None

        Example:
            self.generate_report()
        """
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
            report_path: str = "./reports/",
            data_save_path: str="./data/processed/",
            top_k_features: list = None,
            top_features_path: str = "./artefacts/top_20_features_w_pipeline.pkl" 
        ) -> None:
        super().__init__()
        self.top_k_features = top_k_features
        self.top_features_path = top_features_path
        self.__save_path = save_path
        self.__report_path = report_path
        self.data_save_path = data_save_path
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
        """
        Trains a stacked model using multiple base models and a meta model.

        This method sequentially trains several base models (XGBoost, LightGBM, and GRU) and 
        then trains a meta model using the predictions from the base models as input. The method 
        also computes and logs the RMSE values for each base model and the meta model. The training 
        times for each of the base models and the meta model are also logged.

        The base models trained in this method include:
        - XGBoost
        - LightGBM
        - GRU (Gated Recurrent Units)

        The trained meta model uses the outputs of these base models as input features for final predictions.

        Args:
            None

        Returns:
            None

        Example:
            model.train_stack()
        """
        logger.info("Training Stacked Model")
        try:
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
            self.generate_report()
            logger.info("Stacked Model Training Complete")
        except Exception as e:
            logger.exception("Error training stacked model: %s", e)

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
        """
        Trains an XGBoost regression model and evaluates its performance.

        This method initializes and trains an XGBoost model with the specified hyperparameters.
        It also evaluates the model's performance using RMSE on the test dataset. The trained model
        is then saved to the specified file path.

        The XGBoost model is trained using the following hyperparameters:
        - `n_estimators=60`: Number of trees (estimators) in the model.
        - `learning_rate=0.1`: Step size for weight updates during training.
        - `max_depth=20`: Maximum depth of each tree.
        - `min_child_weight=10`: Minimum sum of weights in child nodes for further partitioning.
        - `subsample=0.8`: Subsample ratio of training data.
        - `colsample_bytree=0.8`: Subsample ratio of features for each tree.
        - `gamma=0`: Minimum loss reduction for further partitioning.
        - `reg_alpha=0.1`: L1 regularization parameter to prevent overfitting.
        - `reg_lambda=1.0`: L2 regularization parameter to prevent overfitting.
        - `random_state=42`: Seed for reproducibility.
        - `tree_method='hist'`: Tree construction method optimized for large datasets.

        Args:
            None

        Returns:
            tuple: A tuple containing:
                - `model` (XGBRegressor): The trained XGBoost model.
                - `rmse` (float): The RMSE of the model on the test set.

        Example:
            model, rmse = self.__XGB_model()
        """
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
        """
        Trains a LightGBM regression model and evaluates its performance.

        This method initializes and trains a LightGBM model using the specified hyperparameters.
        It evaluates the model's performance using RMSE on the test dataset and saves the trained model
        to a specified file path.

        The LightGBM model is trained with the following hyperparameters:
        - `boosting_type="gbdt"`: Gradient Boosting Decision Tree (GBDT) method.
        - `objective="regression"`: Regression task.
        - `metric="rmse"`: Root Mean Squared Error (RMSE) as the evaluation metric.
        - `num_leaves=31`: Maximum number of leaves in each tree.
        - `learning_rate=0.05`: Lower learning rate with more boosting rounds.
        - `feature_fraction=0.8`: Percentage of features used per tree.
        - `bagging_fraction=0.8`: Percentage of data used per tree.
        - `bagging_freq=5`: Perform bagging every 5 iterations.
        - `max_depth=-1`: No maximum depth restriction.
        - `lambda_l1=0.1`: L1 regularization to prevent overfitting.
        - `lambda_l2=0.2`: L2 regularization to prevent overfitting.
        - `verbosity=-1`: Suppress output.
        - `n_jobs=-1`: Use all available CPU cores.
        - `seed=42`: Seed for reproducibility.
        - `early_stopping_rounds=10`: Stop training if no improvement in 10 rounds.
        - `verbose_eval=10`: Print evaluation results every 10 rounds.

        Args:
            None

        Returns:
            tuple: A tuple containing:
                - `model` (lgb.Booster): The trained LightGBM model.
                - `rmse` (float): The RMSE of the model on the test set.

        Example:
            model, rmse = self.__LGB_model()
        """
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
        """
        Trains a GRU (Gated Recurrent Unit) model and evaluates its performance.

        This method initializes and trains a GRU model using the specified hyperparameters.
        It evaluates the model's performance using RMSE on the test dataset and saves the trained model
        to a specified file path.

        The GRU model is trained with the following architecture and hyperparameters:
        - 3 GRU layers with decreasing units: 256, 128, and 64.
        - Dropout layers with a rate of 0.2 between GRU layers to prevent overfitting.
        - A final Dense layer with a linear activation to output the prediction.
        - Adam optimizer with a learning rate of 0.0001.
        - Early stopping based on validation loss with a patience equal to the square root of the number of epochs.

        Args:
            None

        Returns:
            tuple: A tuple containing:
                - `model` (tf.keras.Model): The trained GRU model.
                - `rmse` (float): The RMSE of the model on the test set.

        Example:
            model, rmse = self.__GRU_model()
        """
        timesteps = 1
        features = self.X_train.shape[1]
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
        """
        Trains a meta model using a Multi-Layer Perceptron (MLP) Regressor and evaluates its performance.

        This method trains an MLP regressor model using stacked features (predictions from base models) and the 
        target variable. It evaluates the model's performance using RMSE on a held-out test dataset, then saves 
        the trained model to a file.

        The MLP Regressor model is trained with the following hyperparameters:
        - Two hidden layers: 100 and 50 neurons.
        - ReLU activation function for hidden layers.
        - Adam solver for optimization with learning rate initialization of 0.001.
        - Early stopping if no improvement is seen for 10 iterations.
        - Other regularization and optimization parameters such as momentum and batch size.

        Args:
            features_stacked (pd.DataFrame): The features from stacked models (predictions from base models).
            target_stacked (pd.Series): The true target values corresponding to the stacked features.

        Returns:
            tuple: A tuple containing:
                - `model` (MLPRegressor): The trained meta model.
                - `rmse` (float): The RMSE of the meta model on the test set.

        Example:
            model, rmse = self.__meta_model(features_stacked, target_stacked)
        """
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
        """
        Loads the top-k weighted features from a specified path if not provided during initialization.

        This method loads a pre-saved list of top-k features (based on their importance or weight) from a file. 
        The file path defined by `self.top_features_path`. 

        Args:
            None

        Returns:
            None

        Example:
            self.__weighted_top_features()
        """
        logger.info("Loading top %s features", self.top_k)
        if self.top_k_features is None:
            self.top_k_features: list = joblib.load(self.top_features_path)

    def __get_X_y(self, data: pd.DataFrame, target: str="target") -> tuple[pd.DataFrame, pd.Series]:
        """
        Splits the dataset into features (X) and target (y) using top-k features.

        Args:
            data (pd.DataFrame): Dataset containing features and the target column.
            target (str): Name of the target column. Default is "target".

        Returns:
            tuple[pd.DataFrame, pd.Series]: Feature matrix (X) and target vector (y).
        """
        logger.info("Splitting data into X and y for top %s features", self.top_k)
        X = data[self.top_k_features]
        y = data[target]
        return X, y
    
    def __split_data(
            self,
            features: pd.DataFrame,
            target: pd.Series,
            backtest_size: float=0.3,
            save: bool=True,
        ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Splits features and target data into stacked training and backtest sets.

        Args:
            features (pd.DataFrame): Feature matrix.
            target (pd.Series): Target vector.
            backtest_size (float): Proportion of data for backtesting. Default is 0.3.
            save (bool): If True, saves backtest sets to files. Default is True.

        Returns:
            tuple: Training features (X_train), training target (y_train),
                backtest features (X_backtest), backtest target (y_backtest).
        """
        logger.info("Splitting stacked training data into train, test, and backtest sets")
        split_idx = int(features.shape[0] * (1 - backtest_size))
        X_train = features[:split_idx]
        y_train = target[:split_idx]
        X_backtest = features[split_idx:]
        y_backtest = target[split_idx:]

        if save:
            X_backtest.to_csv(self.data_save_path + "X_backtest.csv", index=False)
            y_backtest.to_csv(self.data_save_path + "y_backtest.csv", index=False)
            logger.info("Backtest data saved to %s", self.data_save_path)
        return X_train, y_train

        
    def generate_report(self) -> None:
        """
        Generates and saves a training report for the stacked model.
        """
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