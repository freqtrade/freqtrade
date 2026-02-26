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


class DailyProfitStrategy(IStrategy):
    """
    DailyProfit v3 — MEXC-optimized for consistent accumulation.

    Based on proven TrendRider v8 core (profitable in -48% bear market)
    with slightly relaxed bull entries for more trades + careful bear scalps.

    Two modes:
    - Bull trend pullback (proven profitable)
    - Bear deep-oversold bounce (ultra-selective)
    """

    INTERFACE_VERSION = 3
    can_short: bool = False

    minimal_roi = {
        "0": 0.05,
        "120": 0.03,
        "360": 0.018,
        "720": 0.008,
    }

    stoploss = -0.020
    use_custom_stoploss = False

    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    timeframe = "1h"

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 210

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 24},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 72,
                "trade_limit": 2,
                "stop_duration_candles": 24,
                "only_per_pair": False,
            },
        ]

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
            "bb_lowerband": {"color": "#cc0000"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "#7b68ee"}},
            "Mode": {"bull_regime": {"color": "#00ff7f"}},
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2.5
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_pct"] = (
            (dataframe["close"] - dataframe["bb_lowerband"])
            / (bollinger["upper"] - dataframe["bb_lowerband"])
        )

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macdhist"] = macd["macdhist"]

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"] * 100

        dataframe["volume_sma"] = dataframe["volume"].rolling(window=20).mean()

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
            1.0, 0.0
        )

        dataframe["higher_low"] = np.where(
            dataframe["low"] > dataframe["low"].shift(2), 1.0, 0.0,
        )

        dataframe["green_candle"] = np.where(
            dataframe["close"] > dataframe["open"], 1.0, 0.0,
        )
        dataframe["prev_green"] = dataframe["green_candle"].shift(1)

        ema_above = (dataframe["ema21"] > dataframe["ema50"]).astype(int)
        streak = ema_above * 0
        for i in range(1, len(ema_above)):
            if ema_above.iloc[i] == 1:
                streak.iloc[i] = streak.iloc[i - 1] + 1
            else:
                streak.iloc[i] = 0
        dataframe["trend_age"] = streak

        # Volume surge
        dataframe["vol_surge"] = np.where(
            dataframe["volume"] > dataframe["volume_sma"] * 2.0, 1.0, 0.0
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        BULL MODE: TrendRider v8 proven entries (slightly relaxed for more trades)
        BEAR MODE: Ultra-selective deep oversold bounce with volume surge
        """
        # --- BULL: Trend pullback (relaxed from v8) ---
        dataframe.loc[
            (
                (dataframe["bull_regime"] == 1)
                & (dataframe["ema21"] > dataframe["ema50"])
                & (dataframe["trend_age"] >= 3)
                & (dataframe["close"] > dataframe["ema21"])
                & (dataframe["ema50_slope"] > 0)
                & (dataframe["dist_to_ema21"] < 1.8)
                & (dataframe["dist_to_ema21"] > -0.5)
                & (dataframe["adx"] > 20)
                & (dataframe["plus_di"] > dataframe["minus_di"])
                & (dataframe["rsi"] > 42)
                & (dataframe["rsi"] < 63)
                & (dataframe["rsi_rising"] == 1)
                & (dataframe["macdhist"] > 0)
                & (dataframe["higher_low"] == 1)
                & (dataframe["green_candle"] == 1)
                & (dataframe["prev_green"] == 1)
                & (dataframe["volume"] > dataframe["volume_sma"])
                & (dataframe["atr_pct"] < 3.0)
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "bull_pullback")

        # --- BEAR: Deep oversold + volume surge (ultra safe) ---
        dataframe.loc[
            (
                (dataframe["bull_regime"] == 0)
                & (dataframe["bb_pct"] < 0.05)
                & (dataframe["rsi"] < 25)
                & (dataframe["rsi"] > dataframe["rsi"].shift(1))
                & (dataframe["green_candle"] == 1)
                & (dataframe["close"].shift(1) < dataframe["open"].shift(1))
                & (dataframe["vol_surge"] == 1)
                & (dataframe["close"] > dataframe["bb_lowerband"])
                & (dataframe["atr_pct"] < 4.0)
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "bear_bounce")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe["rsi"] > 72)
                    | (
                        (dataframe["ema21"] < dataframe["ema50"])
                        & (dataframe["rsi"] > 50)
                    )
                )
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
