# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter
import talib.abstract as ta
from technical import qtpylib


class ToobitRSIStrategy(IStrategy):
    """
    Einfache RSI + EMA-Strategie für Toobit Spot-Trading.

    Einstieg Long:
      - RSI(14) kreuzt über buy_rsi (default 30) — überverkauft dreht hoch
      - Kurs liegt über EMA(50) — Aufwärtstrend bestätigt
      - Volumen > 0

    Ausstieg:
      - RSI(14) kreuzt über sell_rsi (default 70) — überkauft
      - ODER minimal_roi / stoploss greift
    """

    INTERFACE_VERSION = 3
    can_short: bool = False

    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }

    stoploss = -0.05
    trailing_stop = False
    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count: int = 50

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    buy_rsi = IntParameter(low=20, high=40, default=30, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=60, high=80, default=70, space="sell", optimize=True, load=True)

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_upperband"] = bollinger["upper"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["rsi"], self.buy_rsi.value)
                & (dataframe["close"] > dataframe["ema50"])
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value)
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
