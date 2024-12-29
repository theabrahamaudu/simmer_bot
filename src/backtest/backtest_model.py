from time import perf_counter, strftime, gmtime
import datetime
import pandas as pd
from backtesting import Backtest, Strategy
from src.backtest.backtest_utils import PREDICTIONS

#! Toggle this to switch between hardcoded backtest length of 9000
#! or dynamic based on the pipeline configuration
MANUAL_GAME = True
if MANUAL_GAME:
    PREDICTIONS = PREDICTIONS[-9000:]


class RunBacktest:
    def __init__(
            self,
            raw_data_path: str = "./data/interim/merged_scrapped_n_price.csv",
            predictions: list = None,
            strategy: Strategy = None,
            cash: float = 300,
            margin: float = 1/5,
            commission: float = 0.0002,
            report_path: str = "./reports/",
            ):
        self.raw_data = pd.read_csv(raw_data_path)
        self.predictions = predictions
        self.strategy = strategy
        self.cash = cash
        self.margin = margin
        self.commission = commission
        self.report_path = report_path

        self.backtest_data = self.__prepare_data()

    def run(self) -> None:
        """
        Executes the backtest and optimization for the specified strategy.

        This method performs a backtest using the provided strategy, optimizes key parameters, 
        saves the results to CSV files, and outputs statistics and a heatmap.

        Args:
            None

        Returns:
            None

        Outputs:
            - Saves backtest statistics to `{self.report_path}backtest_stats.csv`.
            - Saves optimization heatmap to `{self.report_path}backtest_heatmap.csv`.
            - Prints backtest runtime, statistics, and heatmap.

        Optimization Parameters:
            - `close_days`: Range of days to close overdue trades (1 to 2).
            - `sl_perc`: Stop loss percentages (0.001 to 0.008 in steps of 0.001).
            - `tp_perc`: Take profit percentages (0.001 to 0.008 in steps of 0.001).
            - `buy_threshold`: Buy thresholds (0.001 to 0.005 in steps of 0.001).
            - `sell_threshold`: Sell thresholds (0.001 to 0.005 in steps of 0.001).

        Example:
            RunBacktest().run()
        """
        print("Running backtest for {} predictions...".format(len(self.predictions)))
        
        start = perf_counter()
        backtest = Backtest(
            data=self.backtest_data,
            strategy=self.strategy,
            cash=self.cash,
            margin=self.margin,
            commission=self.commission
        )

        stats, heatmap = backtest.optimize(
            close_days=[i for i in range(1, 3)],
            sl_perc=[i/1000 for i in range(1, 9)],  # Expanding the range for stop loss
            tp_perc=[i/1000 for i in range(1, 9)],  # Expanding the range for take profit
            buy_threshold=[i/1000 for i in range(1, 6)],  # Testing different buy thresholds
            sell_threshold=[i/1000 for i in range(1, 6)],  # Testing different sell thresholds
            maximize='Return [%]', max_tries=5000,
            random_state=0,
            return_heatmap=True
        )

        backtest_run_time = perf_counter() - start

        heatmap = heatmap.sort_values()

        stats.to_csv(
            "{}backtest_stats.csv".format(self.report_path),
            index=True
        )

        heatmap.to_csv(
            "{}backtest_heatmap.csv".format(self.report_path),
            index=True
        )

        formatted_time = strftime("%H:%M:%S", gmtime(backtest_run_time))
        print(f"\nBacktest completed in {formatted_time}\n")
        print(f"Backtest results saved to {self.report_path}")

        print(stats)
        print(heatmap)

    def __prepare_data(self):
        """
        Prepares the raw data for backtesting by aligning it with predictions.

        This method processes the raw data to match the length of the predictions, 
        sets the time column as the index, and renames columns to a predefined format.

        Returns:
            pd.DataFrame: Processed DataFrame with properly formatted columns and time-based index.

        Columns:
            - 'time': Timestamp column, set as the index.
            - 'Open': Opening price.
            - 'High': Highest price.
            - 'Low': Lowest price.
            - 'Close': Closing price.
            - 'Volume': Trading volume.
            - 'spread': Bid-ask spread.
            - 'real_volume': Real traded volume.
            - 'impact': News impact score.
            - 'link_text': Associated text or metadata.

        Example:
            prepared_data = self.__prepare_data()
        """
        proper_cols = [
            'time', 'Open', 'High', 'Low', 'Close', 'Volume', 'spread',
            'real_volume', 'impact', 'link_text'
        ]
        bt_data = self.raw_data.copy()
        bt_data = bt_data.iloc[-(len(self.predictions)):,:]
        # bt_data.reset_index(drop=True, inplace=True)
        bt_data.set_index(pd.to_datetime(bt_data['time']), drop=False, inplace=True)
        bt_data.columns=proper_cols
        bt_data.index.name=None

        return bt_data


class MyStrategy(Strategy):
    sl_perc = 0.002
    tp_perc = 0.006
    buy_threshold = 0.004
    sell_threshold = 0.004
    close_days = 1

    def init(self):
        super().init()
        self.predictions = PREDICTIONS
        self.idx = 0
        self.trade_size = 0.2



    def next(self):
        """
        Executes the next step in the trading strategy simulation.

        This method uses forecasted prices to manage trades, including opening new trades 
        or closing overdue trades based on specified conditions such as stop loss, 
        take profit, and buy/sell thresholds.

        Steps:
            1. Close trades that have been open longer than `close_days`.
            2. Open buy or sell trades based on the forecasted price and current thresholds.
            3. Increment the internal index for tracking predictions.

        Logic:
            - Buy if the forecasted price exceeds the current price by `buy_threshold`
            and there is no open position.
            - Sell if the forecasted price is below the current price by `sell_threshold`
            and there is no open position.

        Attributes:
            forecast (float): The predicted price from the model for the current step.
            current_price (float): The most recent closing price in the data.
            close_days (int): The maximum number of days a trade can remain open.
            buy_threshold (float): The percentage increase threshold for triggering a buy.
            sell_threshold (float): The percentage decrease threshold for triggering a sell.
            sl_perc (float): Stop loss percentage.
            trade_size (float): Size of the trade.
            tp (float): Take profit target price.
            sl (float): Stop loss price.

        Example:
            self.next()
        """
        super().next()
        forecast = self.predictions[self.idx]

        current_price = self.data['Close'][-1]

        # Check and close trades open for more than 1 day
        for trade in self.trades:
            if self.data.index[-1] - trade.entry_time >= datetime.timedelta(days=int(self.close_days)):
                trade.close()
                # print("closed overdue trade >>", trade.value)
        
        # Buy logic
        if forecast > current_price * (1 + self.buy_threshold) and not self.position:
            sl = current_price * (1 - self.sl_perc)
            tp = forecast  # Use model's forecast as target price
            self.buy(size=self.trade_size, tp=tp, sl=sl)

        # Sell logic
        elif forecast < current_price * (1 - self.sell_threshold) and not self.position:
            sl = current_price * (1 + self.sl_perc)
            tp = forecast  # Use model's forecast as target price
            self.sell(size=self.trade_size, tp=tp, sl=sl)

        # increment the index
        self.idx += 1

if __name__ == "__main__":
    backtest_pipeline = RunBacktest(
        predictions=PREDICTIONS,
        strategy=MyStrategy
    )

    backtest_pipeline.run()