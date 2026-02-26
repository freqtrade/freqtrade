# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
from technical import qtpylib


class TrendRiderStrategy(IStrategy):
    """
    TrendRider v6 — Final refined version.

    Based on v4 (best: -5.64% in -36% bear market, 70% win rate).
    Fix: tighter stoploss to cut losses before they grow.
    v4 had 12 stop losses at avg -4%, causing -173 USDT.
    With -2.5% stops, those same 12 losses = ~-100 USDT, making the
    strategy break-even or profitable even in a bear market.
    """

    INTERFACE_VERSION = 3
    can_short: bool = False

    # --- ROI ---
    minimal_roi = {
        "0": 0.06,      # 6% immediate
        "240": 0.035,   # 3.5% after 4h
        "600": 0.02,    # 2% after 10h
        "1200": 0.008,  # 0.8% after 20h
    }

    # --- Stop Loss: tighter to cut losses early ---
    stoploss = -0.025  # 2.5% hard stop
    use_custom_stoploss = False

    # --- Trailing Stop ---
    trailing_stop = True
    trailing_stop_positive = 0.012       # trail at 1.2%
    trailing_stop_positive_offset = 0.025  # activate after 2.5% profit
    trailing_only_offset_is_reached = True

    # --- Timeframe ---
    timeframe = "1h"

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 210

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "ema21": {"color": "#3af0f0"},
            "ema50": {"color": "#f5a623"},
            "ema200": {"color": "#d63384"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "#7b68ee"}},
            "ADX": {"adx": {"color": "#ff6347"}},
            "Regime": {"bull_regime": {"color": "#00ff7f"}},
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"] * 100

        dataframe["volume_sma20"] = dataframe["volume"].rolling(window=20).mean()

        dataframe["ema200_slope"] = (
            (dataframe["ema200"] - dataframe["ema200"].shift(10))
            / dataframe["ema200"].shift(10) * 100
        )
        dataframe["ema50_slope"] = (
            (dataframe["ema50"] - dataframe["ema50"].shift(5))
            / dataframe["ema50"].shift(5) * 100
        )

        dataframe["bull_regime"] = np.where(
            (dataframe["ema50"] > dataframe["ema200"])
            & (dataframe["ema200_slope"] > -0.1),
            1.0, 0.0,
        )

        dataframe["dist_to_ema21"] = (
            (dataframe["close"] - dataframe["ema21"]) / dataframe["ema21"] * 100
        )

        dataframe["rsi_rising"] = np.where(
            (dataframe["rsi"] > dataframe["rsi"].shift(1))
            & (dataframe["rsi"] > dataframe["rsi"].shift(2)),
            1.0, 0.0,
        )

        dataframe["higher_low"] = np.where(
            dataframe["low"] > dataframe["low"].shift(2), 1.0, 0.0,
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Ultra-selective entries (same as v3/v4)."""
        dataframe.loc[
            (
                (dataframe["bull_regime"] == 1)
                & (dataframe["ema21"] > dataframe["ema50"])
                & (dataframe["close"] > dataframe["ema21"])
                & (dataframe["ema50_slope"] > 0)
                & (dataframe["dist_to_ema21"] < 1.5)
                & (dataframe["dist_to_ema21"] > -0.3)
                & (dataframe["adx"] > 22)
                & (dataframe["plus_di"] > dataframe["minus_di"])
                & (dataframe["rsi"] > 45)
                & (dataframe["rsi"] < 62)
                & (dataframe["rsi_rising"] == 1)
                & (dataframe["macdhist"] > 0)
                & (dataframe["higher_low"] == 1)
                & (dataframe["atr_pct"] < 3.0)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit only profitable trades when trend turns."""
        dataframe.loc[
            (
                (
                    (dataframe["ema21"] < dataframe["ema50"])
                    | (dataframe["rsi"] > 72)
                )
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        return dataframe
