from pandas import DataFrame, Series
from technical import qtpylib
from typing import Dict
import logging
import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta

from freqtrade.strategy import IntParameter, IStrategy, merge_informative_pair  # noqa


logger = logging.getLogger(__name__)


class FreqaiExampleHybridStrategy(IStrategy):
    minimal_roi = {
        # "60": 0.01,
        # "30": 0.02,
        # "0": 0.04
    }

    plot_config = {
        "main_plot": {
            "tema": {},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
            "Labels": {
                "&-s_labels": {"color": "green"},
            },
        },
    }

    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count: int = 30
    can_short = False

    trailing_stop = True
    stoploss = -0.05

    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell", optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space="sell", optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: Dict, **kwargs
    ) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]
        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = dataframe["close"] / dataframe["bb_lowerband-period"]
        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)
        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        def triple_barrier_labeling(
            df: DataFrame,
            upper_barrier: float,
            lower_barrier: float,
            time_barrier: int,
            name: str = "close",
        ) -> Series:
            upper = df[name] + df[name] * upper_barrier
            lower = df[name] - df[name] * lower_barrier
            barriers = DataFrame(index=df.index)
            barriers["upper"] = upper
            barriers["lower"] = lower
            barriers = barriers.shift(-time_barrier)
            labels = Series(index=df.index).astype(str)
            labels[(df[name] >= barriers["upper"]).shift(time_barrier).fillna(False)] = "up"
            labels[(df[name] <= barriers["lower"]).shift(time_barrier).fillna(False)] = "down"
            return labels

        LABELING_UPPER_BARRIER = 0.0025
        LABELING_LOWER_BARRIER = 0.0025
        LABELING_TIME_BARRIER = 5
        dataframe["&-s_labels"] = triple_barrier_labeling(
            dataframe,
            name="close",
            upper_barrier=LABELING_UPPER_BARRIER,
            lower_barrier=LABELING_LOWER_BARRIER,
            time_barrier=LABELING_TIME_BARRIER,
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        dataframe["rsi"] = ta.RSI(dataframe)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        )
        dataframe["bb_width"] = (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe[
            "bb_middleband"
        ]
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Signal: RSI crosses above 30
                (qtpylib.crossed_above(dataframe["rsi"], self.buy_rsi.value))
                & (dataframe["tema"] <= dataframe["bb_middleband"])
                & (dataframe["tema"] > dataframe["tema"].shift(1))  # Guard: tema below BB middle
                & (dataframe["volume"] > 0)  # Guard: tema is raising
                & (dataframe["do_predict"] == 1)  # Make sure Volume is not 0
                &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                # (dataframe["&s-up_or_down"] == "up")
                (dataframe["&-s_labels"] == "up")
            ),
            "enter_long",
        ] = 1
        dataframe.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(dataframe["rsi"], self.short_rsi.value))
                & (dataframe["tema"] > dataframe["bb_middleband"])
                & (dataframe["tema"] < dataframe["tema"].shift(1))  # Guard: tema above BB middle
                & (dataframe["volume"] > 0)  # Guard: tema is falling
                & (dataframe["do_predict"] == 1)  # Make sure Volume is not 0
                &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                # (dataframe["&s-up_or_down"] == "down")
                (dataframe["&-s_labels"] == "down")
            ),
            "enter_short",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value))
                & (dataframe["tema"] > dataframe["bb_middleband"])
                & (dataframe["tema"] < dataframe["tema"].shift(1))  # Guard: tema above BB middle
                & (dataframe["volume"] > 0)  # Guard: tema is falling  # Make sure Volume is not 0
            ),
            "exit_long",
        ] = 1
        dataframe.loc[
            (
                # Signal: RSI crosses above 30
                (qtpylib.crossed_above(dataframe["rsi"], self.exit_short_rsi.value))
                &
                # Guard: tema below BB middle
                (dataframe["tema"] <= dataframe["bb_middleband"])
                & (dataframe["tema"] > dataframe["tema"].shift(1))
                & (dataframe["volume"] > 0)  # Guard: tema is raising  # Make sure Volume is not 0
            ),
            "exit_short",
        ] = 1
        return dataframe
