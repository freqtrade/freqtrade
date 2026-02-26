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


class MomentumScalpStrategy(IStrategy):
    """
    MomentumScalp v1 — Aggressive 15m scalping for $50→$500 challenge.

    Designed for SPOT trading with compound growth.
    Uses 15m timeframe for more trading opportunities.

    Logic: catch strong momentum moves early and ride them.
    - RSI reversal from oversold in a short-term uptrend
    - Quick take-profit (2-4%)
    - Tight stop (1.5%)
    - High frequency → compound many small wins

    With 2% avg win and 60% win rate:
    - 100 trades per month → net ~20% monthly
    - $50 → $500 in ~12 months (compounding)
    """

    INTERFACE_VERSION = 3
    can_short: bool = False

    # --- ROI: quick scalp exits ---
    minimal_roi = {
        "0": 0.04,      # 4% immediate
        "30": 0.025,    # 2.5% after 30min
        "60": 0.015,    # 1.5% after 1h
        "120": 0.008,   # 0.8% after 2h
        "240": 0.004,   # 0.4% after 4h
    }

    # --- Stop Loss ---
    stoploss = -0.010  # 1.5% tight stop
    use_custom_stoploss = False

    # --- Trailing Stop ---
    trailing_stop = True
    trailing_stop_positive = 0.006
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    # --- Timeframe ---
    timeframe = "1h"

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 100

    # --- Protections ---
    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 8},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 3,
                "stop_duration_candles": 16,
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
            "ema8": {"color": "#3af0f0"},
            "ema21": {"color": "#f5a623"},
            "ema55": {"color": "#d63384"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "#7b68ee"}},
            "MACD": {
                "macdhist": {"color": "#ff6347", "type": "bar"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # --- Fast EMAs for scalping ---
        dataframe["ema8"] = ta.EMA(dataframe, timeperiod=8)
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema55"] = ta.EMA(dataframe, timeperiod=55)

        # --- RSI ---
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=7)

        # --- MACD fast ---
        macd = ta.MACD(dataframe, fastperiod=8, slowperiod=21, signalperiod=5)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # --- Stochastic RSI ---
        stochrsi = ta.STOCHRSI(dataframe, timeperiod=14)
        dataframe["stochrsi_k"] = stochrsi["fastk"]
        dataframe["stochrsi_d"] = stochrsi["fastd"]

        # --- ATR ---
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"] * 100

        # --- Volume ---
        dataframe["volume_sma"] = dataframe["volume"].rolling(window=20).mean()

        # --- Trend ---
        dataframe["uptrend"] = np.where(
            (dataframe["ema8"] > dataframe["ema21"])
            & (dataframe["ema21"] > dataframe["ema55"]),
            1.0, 0.0,
        )

        # --- EMA8 slope (momentum) ---
        dataframe["ema8_slope"] = (
            (dataframe["ema8"] - dataframe["ema8"].shift(3))
            / dataframe["ema8"].shift(3) * 100
        )

        # --- RSI turning up from oversold ---
        dataframe["rsi_bounce"] = np.where(
            (dataframe["rsi"] > dataframe["rsi"].shift(1))
            & (dataframe["rsi"].shift(1) < 40)
            & (dataframe["rsi"] > 30),
            1.0, 0.0,
        )

        # --- MACD histogram turning positive ---
        dataframe["macd_cross_up"] = np.where(
            (dataframe["macdhist"] > 0)
            & (dataframe["macdhist"].shift(1) <= 0),
            1.0, 0.0,
        )

        # --- Green candle with momentum ---
        dataframe["strong_green"] = np.where(
            (dataframe["close"] > dataframe["open"])
            & ((dataframe["close"] - dataframe["open"]) / dataframe["open"] > 0.003),
            1.0, 0.0,
        )

        # --- Price near EMA support ---
        dataframe["near_ema21"] = np.where(
            (dataframe["close"] > dataframe["ema21"])
            & ((dataframe["close"] - dataframe["ema21"]) / dataframe["ema21"] < 0.01),
            1.0, 0.0,
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Two entry types:

        1. RSI Bounce: Buy the bounce from oversold in uptrend
        2. MACD Crossover: Buy when MACD turns positive with trend support
        """
        # Entry Type 1: RSI Bounce in uptrend
        dataframe.loc[
            (
                (dataframe["uptrend"] == 1)
                & (dataframe["rsi_bounce"] == 1)
                & (dataframe["close"] > dataframe["ema21"])
                & (dataframe["ema8_slope"] > -0.1)
                & (dataframe["volume"] > dataframe["volume_sma"] * 0.8)
                & (dataframe["atr_pct"] < 2.5)
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "rsi_bounce")

        # Entry Type 2: MACD cross + near EMA support
        dataframe.loc[
            (
                (dataframe["uptrend"] == 1)
                & (dataframe["macd_cross_up"] == 1)
                & (dataframe["near_ema21"] == 1)
                & (dataframe["rsi"] > 35)
                & (dataframe["rsi"] < 60)
                & (dataframe["strong_green"] == 1)
                & (dataframe["volume"] > dataframe["volume_sma"])
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "macd_cross")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit profitable trades when momentum fades."""
        dataframe.loc[
            (
                (
                    (dataframe["rsi"] > 70)
                    | (
                        (dataframe["macdhist"] < 0)
                        & (dataframe["macdhist"] < dataframe["macdhist"].shift(1))
                        & (dataframe["rsi"] > 55)
                    )
                )
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        return dataframe
