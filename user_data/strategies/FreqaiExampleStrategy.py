import logging
from functools import reduce
from typing import Dict

import numpy as np
import talib.abstract as ta
from pandas import DataFrame, Series
from technical import qtpylib

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class FreqaiExampleStrategy(IStrategy):
    minimal_roi = {"0": 0.1, "240": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "&-s_close": {"&-s_close": {"color": "blue"}},
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    # this is the maximum period fed to talib (timeframe independent)
    startup_candle_count: int = 40
    can_short = True

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
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
        )

        # Classifiers are typically set up with strings as targets:
        dataframe["&s-up_or_down"] = np.where(
            dataframe["close"].shift(-100) > dataframe["close"], "up", "down"
        )

        # If user wishes to use multiple targets, they can add more by
        # appending more columns with '&'. User should keep in mind that multi targets
        # requires a multioutput prediction model such as
        # freqai/prediction_models/CatboostRegressorMultiTarget.py,
        # freqtrade trade --freqaimodel CatboostRegressorMultiTarget

        # df["&-s_range"] = (
        #     df["close"]
        #     .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .max()
        #     -
        #     df["close"]
        #     .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
        #     .min()
        # )

        # self.freqai.class_names = ["SHORT", "HOLD", "LONG"]

        # def triple_barrier_labeling(
        #     df: DataFrame,
        #     upper_barrier: float,
        #     lower_barrier: float,
        #     time_barrier: int,
        #     name: str = "close",
        # ) -> Series:
        #     labels = Series(index=df.index).astype(str)
        #     for i in range(len(df)):
        #         upper = df.iloc[i][name] + df.iloc[i][name] * upper_barrier
        #         lower = df.iloc[i][name] - df.iloc[i][name] * lower_barrier
        #         end_time = i + time_barrier
        #         for j in range(i + 1, min(end_time, len(df))):
        #             if df.iloc[j][name] >= upper:
        #                 labels.loc[df.index[i]] = "LONG"
        #                 break
        #             elif df.iloc[j][name] <= lower:
        #                 labels.loc[df.index[i]] = "SHORT"
        #                 break
        #         else:
        #             labels.loc[df.index[i]] = "HOLD"
        #     return labels

        # LABELING_UPPER_BARRIER = 0.025
        # LABELING_LOWER_BARRIER = 0.025
        # LABELING_TIME_BARRIER = 10
        # dataframe["&-s_labels"] = triple_barrier_labeling(
        #     dataframe,
        #     name="close",
        #     upper_barrier=LABELING_UPPER_BARRIER,
        #     lower_barrier=LABELING_LOWER_BARRIER,
        #     time_barrier=LABELING_TIME_BARRIER,
        # )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-s_close"] > 0.01,
        ]

        if enter_long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-s_close"] < -0.01,
        ]

        if enter_short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [dataframe["do_predict"] == 1, dataframe["&-s_close"] < 0]
        if exit_long_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [dataframe["do_predict"] == 1, dataframe["&-s_close"] > 0]
        if exit_short_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True
