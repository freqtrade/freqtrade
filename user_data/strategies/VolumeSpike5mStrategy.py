# isort: skip_file
import numpy as np
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional
from freqtrade.strategy import IStrategy, Trade
import talib.abstract as ta
from technical import qtpylib


class VolumeSpike5mStrategy(IStrategy):
    """Volume Spike — when big money moves, follow it."""
    INTERFACE_VERSION = 3
    can_short = False
    minimal_roi = {"0": 0.02, "15": 0.012, "30": 0.007, "60": 0.004, "120": 0.002}
    stoploss = -0.009
    use_custom_stoploss = False
    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.012
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
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_sma"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_sma"]
        dataframe["green"] = np.where(dataframe["close"] > dataframe["open"], 1.0, 0.0)
        dataframe["candle_body"] = abs(dataframe["close"] - dataframe["open"]) / dataframe["open"] * 100
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema21"])
                & (dataframe["volume_ratio"] > 4.0)        # 3x volume spike
                & (dataframe["green"] == 1)                 # green candle
                & (dataframe["candle_body"] > 0.3)          # strong body
                & (dataframe["rsi"] > 45)
                & (dataframe["rsi"] < 70)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            ((dataframe["rsi"] > 75) | (dataframe["close"] < dataframe["ema21"])) & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
