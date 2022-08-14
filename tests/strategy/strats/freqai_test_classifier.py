import logging
from functools import reduce

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy, merge_informative_pair


logger = logging.getLogger(__name__)


class freqai_test_classifier(IStrategy):
    """
    Test strategy - used for testing freqAI functionalities.
    DO not use in production.
    """

    minimal_roi = {"0": 0.1, "240": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "prediction": {"prediction": {"color": "blue"}},
            "target_roi": {
                "target_roi": {"color": "brown"},
            },
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 300
    can_short = False

    linear_roi_offset = DecimalParameter(
        0.00, 0.02, default=0.005, space="sell", optimize=False, load=True
    )
    max_roi_time_long = IntParameter(0, 800, default=400, space="sell", optimize=False, load=True)

    def informative_pairs(self):
        whitelist_pairs = self.dp.current_whitelist()
        corr_pairs = self.config["freqai"]["feature_parameters"]["include_corr_pairlist"]
        informative_pairs = []
        for tf in self.config["freqai"]["feature_parameters"]["include_timeframes"]:
            for pair in whitelist_pairs:
                informative_pairs.append((pair, tf))
            for pair in corr_pairs:
                if pair in whitelist_pairs:
                    continue  # avoid duplication
                informative_pairs.append((pair, tf))
        return informative_pairs

    def populate_any_indicators(
        self, pair, df, tf, informative=None, set_generalized_indicators=False
    ):

        coin = pair.split('/')[0]

        if informative is None:
            informative = self.dp.get_pair_dataframe(pair, tf)

        # first loop is automatically duplicating indicators for time periods
        for t in self.freqai_info["feature_parameters"]["indicator_periods_candles"]:

            t = int(t)
            informative[f"%-{coin}rsi-period_{t}"] = ta.RSI(informative, timeperiod=t)
            informative[f"%-{coin}mfi-period_{t}"] = ta.MFI(informative, timeperiod=t)
            informative[f"%-{coin}adx-period_{t}"] = ta.ADX(informative, window=t)

        informative[f"%-{coin}pct-change"] = informative["close"].pct_change()
        informative[f"%-{coin}raw_volume"] = informative["volume"]
        informative[f"%-{coin}raw_price"] = informative["close"]

        indicators = [col for col in informative if col.startswith("%")]
        # This loop duplicates and shifts all indicators to add a sense of recency to data
        for n in range(self.freqai_info["feature_parameters"]["include_shifted_candles"] + 1):
            if n == 0:
                continue
            informative_shift = informative[indicators].shift(n)
            informative_shift = informative_shift.add_suffix("_shift-" + str(n))
            informative = pd.concat((informative, informative_shift), axis=1)

        df = merge_informative_pair(df, informative, self.config["timeframe"], tf, ffill=True)
        skip_columns = [
            (s + "_" + tf) for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        df = df.drop(columns=skip_columns)

        # Add generalized indicators here (because in live, it will call this
        # function to populate indicators during training). Notice how we ensure not to
        # add them multiple times
        if set_generalized_indicators:
            df["%-day_of_week"] = (df["date"].dt.dayofweek + 1) / 7
            df["%-hour_of_day"] = (df["date"].dt.hour + 1) / 25

            # user adds targets here by prepending them with &- (see convention below)
            # If user wishes to use multiple targets, a multioutput prediction model
            # needs to be used such as templates/CatboostPredictionMultiModel.py
        df['&s-up_or_down'] = np.where(df["close"].shift(-100) > df["close"], 'up', 'down')

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.freqai_info = self.config["freqai"]

        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        enter_long_conditions = [df['&s-up_or_down'] == 'up']

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [df['&s-up_or_down'] == 'down']

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        return df
