"""
MOdule to download price and news data
"""

import os
from datetime import datetime as dt
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from src.utils.data_log_config import logger

class PriceData:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.__login = os.environ.get("LOGIN", None)
        self.__password = os.environ.get("PASSWORD", None)
        self.__server = os.environ.get("SERVER", None)
        self.__setup()

    def __setup(self):
        mt5.initialize()
        mt5.login(
            self.__login,
            self.__password,
            self.__server
        )

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

        logger.info(f"fetching {self.symbol} M{timeframe} price data from {start_date} to {end_date}")
        try:
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
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            price_df = pd.DataFrame(price_array)
            price_df['time']=pd.to_datetime(price_df['time'], unit='s')
            price_df.to_csv(path, index=False)
            logger.info(f"price data saved to {path}")
        except Exception as e:
            print(f"error saving price data to {path}", e)
            logger.error(f"error saving price data to {path}:\n {e}")

if __name__ == "__main__":
    price_data = PriceData("EURUSD")
    fetched_data = price_data.fetch(dt(2024, 1, 1), dt(2024, 11, 14), mt5.TIMEFRAME_M15)
    price_data.save(price_array=fetched_data, path="./data/raw/price_data.csv")
