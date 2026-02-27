# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import os
import logging
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

logger = logging.getLogger(__name__)


class TrendRider5mStrategy(IStrategy):
    """
    TrendRider 5m — fast scalping version.

    Same logic as 1h but adapted for 5-minute candles:
    - Faster EMAs (9/21/55 instead of 21/50/200)
    - Tighter ROI targets (1-3% instead of 2-6%)
    - Faster trailing stop
    - Same protections and early warning exits
    """

    INTERFACE_VERSION = 3
    can_short: bool = False

    # --- ROI: original targets that backtested near break-even ---
    minimal_roi = {
        "0": 0.03,
        "15": 0.02,
        "30": 0.012,
        "60": 0.007,
        "120": 0.003,
    }

    stoploss = -0.010  # 1% hard stop
    use_custom_stoploss = True

    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    timeframe = "5m"

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 60

    @property
    def protections(self):
        return []  # No cooldowns, no lockouts — trade freely

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # ─── Smart Stoploss: wider when trend intact, tight when broken ──

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs) -> float:
        """
        SMART STOP: distinguishes dips from reversals.

        - If trend is INTACT (EMAs aligned), give the trade room to breathe
        - If trend is BROKEN (EMAs crossed), tighten stop immediately
        - First 30 min: wider stop (let the trade develop)
        - Use ATR to adapt to volatility
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss

        last = dataframe.iloc[-1]
        mins_open = (current_time - trade.open_date_utc).total_seconds() / 60

        trend_intact = last["ema9"] > last["ema21"] and last["ema21"] > last["ema55"]
        atr_stop = last["atr_pct"] / 100 * 2  # 2x ATR as dynamic stop

        # Lock gains progressively
        if current_profit > 0.015:
            return -0.003  # up 1.5%+ → lock +1.2%
        if current_profit > 0.01:
            return -0.005  # up 1.0%+ → lock +0.5%
        if current_profit > 0.005:
            return -0.007  # up 0.5%+ → near breakeven

        # First 30 min — give the trade time to develop
        if mins_open < 30:
            if trend_intact:
                return -max(atr_stop, 0.015)  # ATR-based, min -1.5%
            else:
                return -0.01  # trend broke early, tighter stop

        # After 30 min — adapt based on trend health
        if trend_intact:
            return -max(atr_stop, 0.015)  # ATR-based, min -1.5%, trend = patience
        else:
            return -0.012  # trend weakening, but give it a chance to recover

    # ─── Smart Exit: only exit on REAL reversals ────────────────

    def custom_exit(self, pair, trade, current_time, current_rate,
                    current_profit, **kwargs) -> Optional[str]:
        """
        Distinguish between dips and reversals:

        HOLD through dips if:
        - EMAs still aligned (trend intact)
        - Dip is on low volume (no conviction to sell)
        - RSI is just pulling back, not crashing

        EXIT on reversals if:
        - EMAs have crossed (real trend change)
        - High volume selling (conviction behind the dump)
        - RSI collapsing below 25 (panic)
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last = dataframe.iloc[-1]
        trend_intact = last["ema9"] > last["ema21"]
        high_volume = last["volume"] > last["volume_sma"] * 1.5

        # CRASH: huge red candle (>2%) WITH volume = real crash, not a dip
        candle_drop = (last["open"] - last["close"]) / last["open"]
        if candle_drop > 0.02 and high_volume:
            return "crash_exit"

        # PANIC: RSI < 20 = extreme fear, not a normal dip
        if last["rsi"] < 20 and not trend_intact:
            return "panic_exit"

        # REVERSAL: trend broken AND we're losing AND volume confirms
        if not trend_intact and current_profit < -0.005 and high_volume:
            return "reversal_exit"

        # DIP with trend intact? HOLD — don't exit
        # (the custom_stoploss gives wider room when trend is intact)

        return None

    # ─── Indicators ─────────────────────────────────────────────

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Fast EMAs for 5m
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

        dataframe["bull_regime"] = np.where(
            dataframe["ema21"] > dataframe["ema55"], 1.0, 0.0,
        )

        dataframe["dist_to_ema9"] = (
            (dataframe["close"] - dataframe["ema9"]) / dataframe["ema9"] * 100
        )

        dataframe["rsi_rising"] = np.where(
            (dataframe["rsi"] > dataframe["rsi"].shift(1))
            & (dataframe["rsi"] > dataframe["rsi"].shift(2)),
            1.0, 0.0,
        )

        dataframe["higher_low"] = np.where(
            dataframe["low"] > dataframe["low"].shift(2), 1.0, 0.0,
        )

        dataframe["green_candle"] = np.where(
            dataframe["close"] > dataframe["open"], 1.0, 0.0,
        )

        ema_above = (dataframe["ema9"] > dataframe["ema21"]).astype(int)
        streak = ema_above * 0
        for i in range(1, len(ema_above)):
            if ema_above.iloc[i] == 1:
                streak.iloc[i] = streak.iloc[i - 1] + 1
            else:
                streak.iloc[i] = 0
        dataframe["trend_age"] = streak

        return dataframe

    # ─── Entry ──────────────────────────────────────────────────

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["bull_regime"] == 1)
                & (dataframe["ema9"] > dataframe["ema21"])
                & (dataframe["trend_age"] >= 5)
                & (dataframe["close"] > dataframe["ema9"])
                & (dataframe["ema21_slope"] > 0)
                & (dataframe["dist_to_ema9"] < 0.3)
                & (dataframe["dist_to_ema9"] > -0.1)
                & (dataframe["adx"] > 22)
                & (dataframe["plus_di"] > dataframe["minus_di"])
                & (dataframe["rsi"] > 48)
                & (dataframe["rsi"] < 62)
                & (dataframe["rsi_rising"] == 1)
                & (dataframe["macdhist"] > 0)
                & (dataframe["macdhist"] > dataframe["macdhist"].shift(1))
                & (dataframe["higher_low"] == 1)
                & (dataframe["green_candle"] == 1)
                & (dataframe["volume"] > dataframe["volume_sma"] * 1.2)
                & (dataframe["atr_pct"] < 1.0)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    # ─── Exit ───────────────────────────────────────────────────

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
