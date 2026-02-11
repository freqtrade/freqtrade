# pragma pylint: disable=missing-docstring, invalid-name, stateless-class
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
)


class GridStrategy(IStrategy):
    """
    This is a Grid Strategy template.
    It places buy/sell signals based on a grid of prices.

    How it works:
    - Calculates a grid range based on recent High/Low or fixed parameters.
    - Divides the range into N levels.
    - Buys when price crosses below a level.
    - Sells when price crosses above a level.

    You can adjust the parameters in the config or hyperopt them.
    """

    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 100  # Exit is managed by grid logic, set high ROI to avoid interference
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.99  # Effectively disabled, managed by grid

    # Trailing stoploss
    trailing_stop = False

    # Timeframe
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before:
    # - the strategy runs for the first time (backtesting)
    # - the strategy runs after start up (live/dry-run)
    startup_candle_count: int = 200

    # Grid Parameters
    grid_levels = IntParameter(5, 20, default=10, space="buy")
    grid_range = DecimalParameter(0.01, 0.2, default=0.05, space="buy")  # 5% range

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate grid boundaries
        # For simplicity, we use a moving average as the center of the grid
        dataframe['grid_center'] = dataframe['close'].rolling(window=20).mean()

        # Calculate upper and lower bounds of the grid
        # The range is defined as a percentage around the center
        range_pct = self.grid_range.value
        dataframe['grid_upper'] = dataframe['grid_center'] * (1 + range_pct)
        dataframe['grid_lower'] = dataframe['grid_center'] * (1 - range_pct)

        # We can visualize the grid levels if needed, but for signals we just check relative position
        # A more complex grid would track state, but for a strategy template we use reactive signals

        # Calculate relative position within the grid (0 to 1)
        dataframe['grid_position'] = (
            (dataframe['close'] - dataframe['grid_lower']) /
            (dataframe['grid_upper'] - dataframe['grid_lower'])
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Buy if price is in the lower part of the grid
        # E.g. below 0.2 (bottom 20% of the range)

        dataframe.loc[
            (
                (dataframe['grid_position'] < 0.2) &  # Buy low
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Sell if price is in the upper part of the grid
        # E.g. above 0.8 (top 20% of the range)

        dataframe.loc[
            (
                (dataframe['grid_position'] > 0.8) &  # Sell high
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 1

        return dataframe
