"""
MOdule to download price and news data
"""

import os
from datetime import datetime as dt
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

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
        
        price_data = mt5.copy_rates_range(
            self.symbol,
            timeframe,
            start_date,
            end_date,
        )

        return price_data
    
    def fetch_n(self, num_bars: int, timeframe: int, start_bar: int = 0) -> np.ndarray:
        """Fetch `num_bars` number of price data backwards from current timestamp

        Args:
            num_bars (int): _description_
            timeframe (int): _description_
            start_bar (int, optional): _description_. Defaults to 0.

        Returns:
            np.ndarray: _description_
        """
        bars = mt5.copy_rates_from_pos(
            self.symbol,
            timeframe,
            start_bar,
            num_bars
        )
        return bars
    
    def save(self, price_data: np.ndarray, path: str):
        price_df = pd.DataFrame(price_data)
        price_df['time']=pd.to_datetime(price_df['time'], unit='s')
        price_df.to_csv(path, index=False)



