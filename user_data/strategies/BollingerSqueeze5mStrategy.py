# isort: skip_file
import numpy as np
from pandas import DataFrame
from typing import Optional
from freqtrade.strategy import IStrategy, Trade
import talib.abstract as ta
from technical import qtpylib


class BollingerSqueeze5mStrategy(IStrategy):
    """Bollinger Squeeze — tight bands then breakout = explosive move."""
    INTERFACE_VERSION = 3
    can_short = False
    minimal_roi = {"0": 0.025, "15": 0.015, "30": 0.008, "60": 0.005, "120": 0.002}
    stoploss = -0.009
    use_custom_stoploss = False
    trailing_stop = True
    trailing_stop_positive = 0.004
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
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_upper"] = bollinger["upper"]
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_mid"] = bollinger["mid"]
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"] * 100
        dataframe["bb_width_min"] = dataframe["bb_width"].rolling(window=20).min()
        dataframe["squeeze"] = np.where(dataframe["bb_width"] < dataframe["bb_width_min"] * 1.1, 1.0, 0.0)
        dataframe["volume_sma"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["green"] = np.where(dataframe["close"] > dataframe["open"], 1.0, 0.0)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["squeeze"].shift(1) == 1)         # was in squeeze
                & (dataframe["close"] > dataframe["bb_upper"])  # broke above upper band
                & (dataframe["close"] > dataframe["ema21"])
                & (dataframe["rsi"] > 50)
                & (dataframe["rsi"] < 70)
                & (dataframe["green"] == 1)
                & (dataframe["volume"] > dataframe["volume_sma"] * 1.5)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            ((dataframe["rsi"] > 75) | (dataframe["close"] < dataframe["bb_mid"])) & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
