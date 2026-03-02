# isort: skip_file
import numpy as np
from pandas import DataFrame
from typing import Optional
from freqtrade.strategy import IStrategy, Trade
import talib.abstract as ta


class RSIDivergence5mStrategy(IStrategy):
    """RSI Divergence — price makes lower low but RSI makes higher low = reversal."""
    INTERFACE_VERSION = 3
    can_short = False
    minimal_roi = {"0": 0.025, "15": 0.015, "30": 0.008, "60": 0.004, "120": 0.002}
    stoploss = -0.009
    use_custom_stoploss = False
    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True
    timeframe = "5m"
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    startup_candle_count = 30

    @property
    def protections(self):
        return []

    order_types = {"entry": "limit", "exit": "limit", "stoploss": "market", "stoploss_on_exchange": False}
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema55"] = ta.EMA(dataframe, timeperiod=55)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_sma"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["green"] = np.where(dataframe["close"] > dataframe["open"], 1.0, 0.0)

        # Bullish divergence: price lower low, RSI higher low
        dataframe["price_lower_low"] = np.where(
            (dataframe["low"] < dataframe["low"].shift(3)) & (dataframe["low"].shift(3) < dataframe["low"].shift(6)),
            1.0, 0.0
        )
        dataframe["rsi_higher_low"] = np.where(
            (dataframe["rsi"] > dataframe["rsi"].shift(3)) & (dataframe["low"].shift(3) < dataframe["low"].shift(6)),
            1.0, 0.0
        )
        dataframe["bull_divergence"] = np.where(
            (dataframe["price_lower_low"] == 1) & (dataframe["rsi_higher_low"] == 1),
            1.0, 0.0
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["bull_divergence"] == 1)
                & (dataframe["rsi"] > 25)
                & (dataframe["rsi"] < 55)
                & (dataframe["green"] == 1)
                & (dataframe["close"] > dataframe["low"].shift(1))
                & (dataframe["volume"] > dataframe["volume_sma"])
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            ((dataframe["rsi"] > 70) | (dataframe["close"] < dataframe["ema55"])) & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
