from __future__ import annotations

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class LongOnlyFreqAIStrategy(IStrategy):
    """Phase 2-safe FreqAI regression strategy for historical backtests only."""

    INTERFACE_VERSION = 3

    minimal_roi = {"0": 0.03, "120": 0.01, "360": 0}
    stoploss = -0.05
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count: int = 120
    can_short = False

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-ema-distance-period"] = (
            ta.EMA(dataframe, timeperiod=period) / dataframe["close"] - 1.0
        )
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw-volume"] = dataframe["volume"]
        dataframe["%-raw-close"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-day-of-week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour-of-day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        label_period = int(
            self.config["freqai"]["feature_parameters"].get("label_period_candles", 12)
        )
        target_close = (
            dataframe["close"].shift(-label_period).rolling(label_period).mean()
        )
        dataframe["&-long_return"] = target_close / dataframe["close"] - 1.0
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=48)

        self.freqai_info = self.config["freqai"]
        dataframe = self.freqai.start(dataframe, metadata, self)

        if "&-long_return_mean" in dataframe and "&-long_return_std" in dataframe:
            dataframe["target_roi"] = (
                dataframe["&-long_return_mean"] + dataframe["&-long_return_std"] * 0.8
            )
            dataframe["exit_roi"] = (
                dataframe["&-long_return_mean"] - dataframe["&-long_return_std"] * 0.5
            )
        else:
            dataframe["target_roi"] = 0.002
            dataframe["exit_roi"] = -0.001

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        prediction = dataframe.get("&-long_return", pd.Series(0, index=dataframe.index))
        do_predict = dataframe.get("do_predict", pd.Series(0, index=dataframe.index))
        trend_ok = dataframe["ema_fast"] > dataframe["ema_slow"]

        enter_long = (do_predict == 1) & (prediction > dataframe["target_roi"]) & trend_ok
        dataframe.loc[enter_long, ["enter_long", "enter_tag"]] = (1, "freqai_long")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        prediction = dataframe.get("&-long_return", pd.Series(0, index=dataframe.index))
        do_predict = dataframe.get("do_predict", pd.Series(0, index=dataframe.index))
        trend_exit = dataframe["ema_fast"] < dataframe["ema_slow"]

        exit_long = ((do_predict == 1) & (prediction < dataframe["exit_roi"])) | trend_exit
        dataframe.loc[exit_long, "exit_long"] = 1
        return dataframe
