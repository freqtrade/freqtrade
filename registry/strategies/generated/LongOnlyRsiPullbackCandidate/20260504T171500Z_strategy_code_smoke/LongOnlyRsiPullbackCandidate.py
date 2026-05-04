from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import DecimalParameter, IStrategy, IntParameter, timeframe_to_minutes


class LongOnlyRsiPullbackCandidate(IStrategy):
    """
    Generated Bot Factory long-only RSI pullback strategy.

    Candidate ID: 20260504T171500Z_strategy_code_smoke
    Source proposal hash: c64c39b4977a36aaf2bf2c823c3493a77524a4f86d9b5f9f1308c5c3c36570c6
    """

    INTERFACE_VERSION = 3

    can_short = False
    timeframe = "5m"
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 120

    minimal_roi = {"0": 0.03, "120": 0.01, "360": 0.0}
    stoploss = -0.05
    trailing_stop = False

    buy_rsi_window = IntParameter(8, 30, default=14, space="buy", optimize=True, load=True)
    buy_pullback_lookback = IntParameter(2, 12, default=5, space="buy", optimize=True, load=True)
    buy_rsi_pullback = IntParameter(20, 45, default=32, space="buy", optimize=True, load=True)
    buy_rsi_recovery = IntParameter(35, 55, default=42, space="buy", optimize=True, load=True)
    buy_ema_fast = IntParameter(8, 24, default=12, space="buy", optimize=True, load=True)
    buy_ema_slow = IntParameter(32, 96, default=48, space="buy", optimize=True, load=True)
    buy_volume_window = IntParameter(12, 60, default=24, space="buy", optimize=True, load=True)
    buy_volume_factor = DecimalParameter(
        0.80, 2.00, decimals=2, default=1.00, space="buy", optimize=True, load=True
    )
    sell_rsi_exit = IntParameter(55, 80, default=65, space="sell", optimize=True, load=True)
    sell_timeout_candles = IntParameter(
        24, 288, default=96, space="sell", optimize=True, load=True
    )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=int(self.buy_rsi_window.value))
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=int(self.buy_ema_fast.value))
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=int(self.buy_ema_slow.value))
        dataframe["volume_mean"] = dataframe["volume"].rolling(
            int(self.buy_volume_window.value), min_periods=1
        ).mean()
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pullback_seen = (
            dataframe["rsi"].rolling(
                int(self.buy_pullback_lookback.value), min_periods=1
            ).min()
            <= self.buy_rsi_pullback.value
        )
        rsi_recovered = qtpylib.crossed_above(
            dataframe["rsi"], self.buy_rsi_recovery.value
        )
        trend_filter = dataframe["ema_fast"] >= dataframe["ema_slow"]
        volume_filter = dataframe["volume"] > (
            dataframe["volume_mean"] * self.buy_volume_factor.value
        )
        entry_condition = (
            pullback_seen
            & rsi_recovered
            & trend_filter
            & volume_filter
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[entry_condition, ["enter_long", "enter_tag"]] = (
            1,
            "rsi_pullback_recovery",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        mean_reversion_target = dataframe["rsi"] >= self.sell_rsi_exit.value
        momentum_failure = dataframe["ema_fast"] < dataframe["ema_slow"]
        exit_condition = (
            (mean_reversion_target | momentum_failure)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[exit_condition, ["exit_long", "exit_tag"]] = (
            1,
            "mean_reversion_or_momentum_failure",
        )
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | None:
        hold_minutes = int(self.sell_timeout_candles.value) * timeframe_to_minutes(
            self.timeframe
        )
        if current_time - trade.open_date_utc >= timedelta(minutes=hold_minutes):
            return "timeout_exit"
        return None
