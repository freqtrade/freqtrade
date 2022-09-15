import logging
from functools import reduce

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy, merge_informative_pair


logger = logging.getLogger(__name__)


class freqai_test_spice_rack(IStrategy):
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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Example of how to use the freqai.spice_rack. User treats it the same as any
        # typical talib indicator. They set a new column in their dataframe

        dataframe['dissimilarity_index'] = self.freqai.spice_rack(
            'DI_values', dataframe, metadata, self)
        dataframe['maxima'] = self.freqai.spice_rack(
            '&s-maxima', dataframe, metadata, self)
        dataframe['minima'] = self.freqai.spice_rack(
            '&s-minima', dataframe, metadata, self)
        self.freqai.close_spice_rack()  # user must close the spicerack

        dataframe['rsi'] = ta.RSI(dataframe)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['dissimilarity_index'] < 1) &
                (df['maxima'] > 0.1)
            ),
            'enter_long'] = 1

        df.loc[
            (
                (df['tema'] < df['tema'].shift(1)) &  # Guard: tema is falling
                (df['dissimilarity_index'] < 1) &
                (df['minima'] > 0.1)
            ),
            'enter_short'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['tema'] < df['tema'].shift(1)) &  # Guard: tema is falling
                (df['dissimilarity_index'] < 1) &
                (df['maxima'] > 0.1)
            ),

            'exit_long'] = 1

        df.loc[
            (
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['dissimilarity_index'] < 1) &
                (df['minima'] > 0.1)
            ),
            'exit_short'] = 1

        return df
