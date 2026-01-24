"""Deterministic smoke strategy for India equities."""

from __future__ import annotations

from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IStrategy


class IndiaEquitySmokeStrategy(IStrategy):
    """Simple deterministic strategy used for plumbing validation."""

    INTERFACE_VERSION = 3

    timeframe = "5m"
    startup_candle_count = 50

    minimal_roi = {"0": 0.01}
    stoploss = -0.02

    process_only_new_candles = True
    can_short: bool = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add EMA and RSI indicators needed for entry/exit rules."""
        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define deterministic entry conditions."""
        dataframe.loc[
            (dataframe["ema_9"] > dataframe["ema_21"]) & (dataframe["rsi_14"] > 52),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define deterministic exit conditions."""
        dataframe.loc[
            (dataframe["ema_9"] < dataframe["ema_21"]) | (dataframe["rsi_14"] < 48),
            "exit_long",
        ] = 1
        return dataframe
