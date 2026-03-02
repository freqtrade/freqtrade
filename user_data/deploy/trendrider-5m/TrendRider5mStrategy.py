# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import os, logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union
from freqtrade.strategy import (
    IStrategy, Trade, Order, PairLocks, informative,
    stoploss_from_open,
)
import talib.abstract as ta
from technical import qtpylib

logger = logging.getLogger(__name__)


class TrendRider5mStrategy(IStrategy):
    """
    TrendRider 5m v2 — Loosened entries, tighter stops, faster profit lock.

    Changes from v1:
    - Dropped: double green, MACD rising, volume > 1.2x (too restrictive)
    - Stop: -0.8% (was -1%)
    - Profit lock: breakeven at +0.3% (saves trades like SUI)
    - Target: 2-3x more trades with same win rate
    """

    INTERFACE_VERSION = 3
    can_short: bool = False

    minimal_roi = {
        "0": 0.02,      # 2% immediate
        "15": 0.012,    # 1.2% after 15min
        "30": 0.007,    # 0.7% after 30min
        "60": 0.004,    # 0.4% after 1h
        "120": 0.003,   # 0.2% after 2h
    }

    stoploss = -0.009  # 0.8% stop — clean, no custom override
    use_custom_stoploss = False

    trailing_stop = True
    trailing_stop_positive = 0.003       # trail at 0.3%
    trailing_stop_positive_offset = 0.012  # only activate at 1.2% profit
    trailing_only_offset_is_reached = True

    timeframe = "5m"
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 60

    @property
    def protections(self):
        return []

    order_types = {
        "entry": "limit", "exit": "limit",
        "stoploss": "market", "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss

        last = dataframe.iloc[-1]
        trend_intact = last["ema9"] > last["ema21"]

        # Only lock profits above 0.8% — don't lock tiny gains
        if current_profit > 0.015:
            return -0.004  # up 1.5%+ → lock +1.1%
        if current_profit > 0.008:
            return -0.005  # up 0.8%+ → lock +0.3%

        # Trend-based stop — wider when trend is healthy
        if trend_intact:
            return -0.012  # patient
        return self.stoploss  # -0.8% when trend weak

    def custom_exit(self, pair, trade, current_time, current_rate,
                    current_profit, **kwargs) -> Optional[str]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None
        last = dataframe.iloc[-1]

        candle_drop = (last["open"] - last["close"]) / last["open"]
        if candle_drop > 0.02 and last["volume"] > last["volume_sma"] * 1.5:
            return "crash_exit"
        if last["rsi"] < 20 and last["ema9"] < last["ema21"]:
            return "panic_exit"
        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema55"] = ta.EMA(dataframe, timeperiod=55)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=8, slowperiod=21, signalperiod=5)
        dataframe["macdhist"] = macd["macdhist"]
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"] * 100
        dataframe["volume_sma"] = dataframe["volume"].rolling(window=20).mean()

        dataframe["ema21_slope"] = (
            (dataframe["ema21"] - dataframe["ema21"].shift(3))
            / dataframe["ema21"].shift(3) * 100
        )
        dataframe["dist_to_ema9"] = (
            (dataframe["close"] - dataframe["ema9"]) / dataframe["ema9"] * 100
        )
        dataframe["rsi_rising"] = np.where(
            dataframe["rsi"] > dataframe["rsi"].shift(1), 1.0, 0.0
        )
        dataframe["higher_low"] = np.where(
            dataframe["low"] > dataframe["low"].shift(2), 1.0, 0.0
        )
        dataframe["green_candle"] = np.where(
            dataframe["close"] > dataframe["open"], 1.0, 0.0
        )

        ema_above = (dataframe["ema9"] > dataframe["ema21"]).astype(int)
        streak = ema_above * 0
        for i in range(1, len(ema_above)):
            if ema_above.iloc[i] == 1:
                streak.iloc[i] = streak.iloc[i - 1] + 1
            else:
                streak.iloc[i] = 0
        dataframe["trend_age"] = streak

        # Freshness filter: how far has price risen from recent low?
        dataframe["recent_low"] = dataframe["low"].rolling(window=12).min()
        dataframe["rise_from_low"] = (
            (dataframe["close"] - dataframe["recent_low"]) / dataframe["recent_low"] * 100
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        v2 entries — dropped 4 conditions for more trades:
        - Removed: prev_green (double green candle)
        - Removed: macdhist rising
        - Removed: volume > 1.2x avg (just needs > 0)
        - Relaxed: near EMA9 from ±0.1/0.3 to ±0.3/0.6
        - Relaxed: trend_age from 5 to 3
        """
        dataframe.loc[
            (
                (dataframe["ema21"] > dataframe["ema55"])      # bull regime
                & (dataframe["ema9"] > dataframe["ema21"])     # short trend up
                & (dataframe["trend_age"] >= 3)                # trend running 3+ candles
                & (dataframe["close"] > dataframe["ema9"])     # price above fast EMA
                & (dataframe["ema21_slope"] > 0)               # trend gaining
                & (dataframe["dist_to_ema9"] < 0.6)           # not too far from EMA9
                & (dataframe["dist_to_ema9"] > -0.3)          # not too far below
                & (dataframe["adx"] > 20)                      # trend has strength
                & (dataframe["plus_di"] > dataframe["minus_di"]) # buyers > sellers
                & (dataframe["rsi"] > 45)                      # healthy momentum
                & (dataframe["rsi"] < 65)                      # not overbought
                & (dataframe["rsi_rising"] == 1)               # momentum building
                & (dataframe["macdhist"] > 0)                  # MACD positive
                & (dataframe["macdhist"] > dataframe["macdhist"].shift(1))  # MACD rising
                & (dataframe["higher_low"] == 1)               # structure intact
                & (dataframe["green_candle"] == 1)             # current green
                & (dataframe["close"].shift(1) > dataframe["open"].shift(1))  # prev green
                & (dataframe["rise_from_low"] < 0.8)          # FRESHNESS: enter early, not late
                & (dataframe["atr_pct"] < 1.2)                # calmer markets
                & (dataframe["volume"] > dataframe["volume_sma"])  # above avg volume
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe["ema9"] < dataframe["ema21"])
                    | (dataframe["rsi"] > 72)
                )
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
