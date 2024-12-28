"""
Module to download price data
"""

import os
from datetime import datetime as dt
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from src.utils.data_fetch_log_config import logger

class PriceData:
    """
    Class to download price data from broker
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.__login = os.environ.get("LOGIN", None)
        self.__password = os.environ.get("PASSWORD", None)
        self.__server = os.environ.get("SERVER", None)
        self.__setup()

    def __setup(self):
        """
        Initializes and logs into the account on the specified server.

        This method sets up the connection by initializing and logging into the 
        trading account using the provided login credentials and server details.

        Raises:
            Exception: If an error occurs during initialization or login, it is logged.
        """
        try:
            logger.info(
                f"initializing and logging into account -> {self.__login} on server -> {self.__server}"
            )
            mt5.initialize()
            mt5.login(
                self.__login,
                self.__password,
                self.__server
            )
            logger.info("login successful")
        except Exception as e:
            logger.error(f"Error initializing and logging into account -> {self.__login}: {e}")

    def fetch(self, start_date: dt, end_date: dt, timeframe: int) -> np.ndarray:
        """Fetch all price data from start to end date range

        Args:
            start_date (dt): _description_
            end_date (dt): _description_
            timeframe (int): _description_

        Returns:
            np.ndarray: _description_
        """
        start_date = start_date.strftime("%Y-%m-%d %H:%M:%S.%f")
        start_date = dt.strptime(start_date, "%Y-%m-%d %H:%M:%S.%f")

        end_date = end_date.strftime("%Y-%m-%d %H:%M:%S.%f")
        end_date = dt.strptime(end_date, "%Y-%m-%d %H:%M:%S.%f")

        try:
            logger.info(f"fetching {self.symbol} M{timeframe} price data from {start_date} to {end_date}")
            price_data = mt5.copy_rates_range(
                self.symbol,
                timeframe,
                start_date,
                end_date,
            )
            mt5.shutdown()
            logger.info(f"price fetch success")
        except Exception as e:
            mt5.shutdown()
            logger.error(f"error fetching price data: {e}")
            print("error fetching price data:\n", e)

        if isinstance(price_data, np.ndarray):
            return price_data
        else:
            print("error retrieving data from MT5 server")
    
    def fetch_n(self, num_bars: int, timeframe: int, start_bar: int = 0) -> np.ndarray:
        """Fetch `num_bars` number of price data backwards from current timestamp

        Args:
            num_bars (int): _description_
            timeframe (int): _description_
            start_bar (int, optional): _description_. Defaults to 0.

        Returns:
            np.ndarray: _description_
        """
        try:
            bars = mt5.copy_rates_from_pos(
                self.symbol,
                timeframe,
                start_bar,
                num_bars
            )
            mt5.shutdown()
        except Exception as e:
            mt5.shutdown()
            print(f"error fetching {num_bars} rows of price data\n", e)

        if isinstance(bars, np.ndarray):
            return bars
        else:
            print("error fetching price data from MT5 server")

    def save(self, price_array: np.ndarray, path: str):
        """
        Saves the given price data array to a CSV file after converting the timestamp to a readable datetime format.

        Args:
            price_array (np.ndarray): A NumPy array containing the price data, where one of the columns is a timestamp in seconds.
            path (str): The file path where the CSV file will be saved.

        Returns:
            None: The function saves the data to a CSV file at the specified path.

        Behavior:
            - Creates any necessary directories for the file path if they don't already exist.
            - Converts the 'time' column from Unix timestamp (seconds) to a readable datetime format.
            - Saves the data to a CSV file without the index.
        
        Logs:
            - Logs an info message indicating successful saving of price data.
            - Logs an error message if an exception occurs while saving the data.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            price_df = pd.DataFrame(price_array)
            price_df['time']=pd.to_datetime(price_df['time'], unit='s')
            price_df.to_csv(path, index=False)
            logger.info(f"price data saved to {path}")
        except Exception as e:
            logger.error(f"error saving price data to {path}:\n {e}")

if __name__ == "__main__":
    price_data = PriceData("EURUSD")
    fetched_data = price_data.fetch(dt(2014, 1, 1), dt(2024, 11, 18), mt5.TIMEFRAME_M15)
    price_data.save(price_array=fetched_data, path="./data/raw/price_data.csv")
