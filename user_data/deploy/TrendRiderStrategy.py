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
    TrendRider v8 — v6 base + anti-false-signal filters.

    v6 had 25 wins and 15 stop-loss losses. If we can eliminate
    even 5 of those 15 bad trades, we go from -6.6% to profitable.

    Key additions over v6:
    - Cooldown: 48h lockout per pair after stop loss (prevents re-entering false trends)
    - Confirmation: previous candle must also be green (momentum confirmation)
    - Volume: must be above 1.2x average (conviction)
    - Trend maturity: EMA21 must have been above EMA50 for 5+ candles
    - Max 2 open trades (less exposure in uncertain markets)
    - Simple stoploss, no custom (proven cleaner in v6)
    """

    INTERFACE_VERSION = 3
    can_short: bool = False

    # --- ROI ---
    minimal_roi = {
        "0": 0.06,
        "240": 0.035,
        "600": 0.02,
        "1200": 0.008,
    }

    # --- Stop Loss ---
    stoploss = -0.018
    use_custom_stoploss = False

    # --- Trailing Stop ---
    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.030
    trailing_only_offset_is_reached = True

    # --- Timeframe ---
    timeframe = "1h"

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 210

    # --- Cooldown protection: lock pair for 48h after stop loss ---
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 48,  # 48 hours on 1h timeframe
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 72,  # 3 days
                "trade_limit": 2,  # if 2 stops in 3 days, lock ALL pairs 24h
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

        # --- EMA slopes ---
        dataframe["ema200_slope"] = (
            (dataframe["ema200"] - dataframe["ema200"].shift(10))
            / dataframe["ema200"].shift(10) * 100
        )
        dataframe["ema50_slope"] = (
            (dataframe["ema50"] - dataframe["ema50"].shift(5))
            / dataframe["ema50"].shift(5) * 100
        )

        # --- Bull regime ---
        dataframe["bull_regime"] = np.where(
            (dataframe["ema50"] > dataframe["ema200"])
            & (dataframe["ema200_slope"] > -0.1),
            1.0, 0.0,
        )

        # --- Pullback ---
        dataframe["dist_to_ema21"] = (
            (dataframe["close"] - dataframe["ema21"]) / dataframe["ema21"] * 100
        )

        # --- RSI rising ---
        dataframe["rsi_rising"] = np.where(
            (dataframe["rsi"] > dataframe["rsi"].shift(1))
            & (dataframe["rsi"] > dataframe["rsi"].shift(2)),
            1.0, 0.0,
        )

        # --- Higher lows ---
        dataframe["higher_low"] = np.where(
            dataframe["low"] > dataframe["low"].shift(2), 1.0, 0.0,
        )

        # --- Trend maturity: how many consecutive candles EMA21 > EMA50 ---
        ema21_above = (dataframe["ema21"] > dataframe["ema50"]).astype(int)
        streak = ema21_above * 0
        for i in range(1, len(ema21_above)):
            if ema21_above.iloc[i] == 1:
                streak.iloc[i] = streak.iloc[i - 1] + 1
            else:
                streak.iloc[i] = 0
        dataframe["trend_maturity"] = streak

        # --- Confirmation: previous candle was green ---
        dataframe["prev_green"] = np.where(
            dataframe["close"].shift(1) > dataframe["open"].shift(1),
            1.0, 0.0,
        )

        # --- Current candle green ---
        dataframe["green_candle"] = np.where(
            dataframe["close"] > dataframe["open"], 1.0, 0.0,
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Ultra-selective entries with v8 additions:
        - Trend maturity: uptrend running for 5+ candles (not fresh/fragile)
        - Double green: current AND previous candle must be green
        - Volume conviction: above 1.2x average
        - All v6 trend filters still apply
        """
        dataframe.loc[
            (
                # Bull regime
                (dataframe["bull_regime"] == 1)
                # Established uptrend (not just started)
                & (dataframe["ema21"] > dataframe["ema50"])
                & (dataframe["trend_maturity"] >= 5)
                & (dataframe["close"] > dataframe["ema21"])
                & (dataframe["ema50_slope"] > 0)
                # Pullback entry zone
                & (dataframe["dist_to_ema21"] < 1.5)
                & (dataframe["dist_to_ema21"] > -0.3)
                # Momentum
                & (dataframe["adx"] > 22)
                & (dataframe["plus_di"] > dataframe["minus_di"])
                & (dataframe["rsi"] > 45)
                & (dataframe["rsi"] < 62)
                & (dataframe["rsi_rising"] == 1)
                # MACD confirmation
                & (dataframe["macdhist"] > 0)
                # Structure
                & (dataframe["higher_low"] == 1)
                # Double green confirmation
                & (dataframe["green_candle"] == 1)
                & (dataframe["prev_green"] == 1)
                # Volume conviction
                & (dataframe["volume"] > dataframe["volume_sma20"] * 1.2)
                # Low volatility
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
