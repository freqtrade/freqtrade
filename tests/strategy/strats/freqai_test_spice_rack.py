import logging

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class freqai_test_spice_rack(IStrategy):
    """
    Test strategy - used for testing freqAI functionalities.
    DO not use in production.
    """

    minimal_roi = {"0": 0.1, "240": -1}

    process_only_new_candles = True
    startup_candle_count: int = 30

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
                (df['rsi'] > df['rsi'].shift(1)) &  # Guard: tema is raising
                (df['dissimilarity_index'] < 1) &
                (df['maxima'] > 0.1)
            ),
            'enter_long'] = 1

        df.loc[
            (
                (df['rsi'] < df['rsi'].shift(1)) &  # Guard: tema is falling
                (df['dissimilarity_index'] < 1) &
                (df['minima'] > 0.1)
            ),
            'enter_short'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['rsi'] < df['rsi'].shift(1)) &  # Guard: tema is falling
                (df['dissimilarity_index'] < 1) &
                (df['maxima'] > 0.1)
            ),

            'exit_long'] = 1

        df.loc[
            (
                (df['rsi'] > df['rsi'].shift(1)) &  # Guard: tema is raising
                (df['dissimilarity_index'] < 1) &
                (df['minima'] > 0.1)
            ),
            'exit_short'] = 1

        return df
