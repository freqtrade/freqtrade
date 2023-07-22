import logging

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IntParameter, IStrategy


logger = logging.getLogger(__name__)


class FreqaiExampleSpiceRack(IStrategy):
    """
    Example strategy showing how the user can incorporate spice_rack
    """

    minimal_roi = {"0": 0.1, "240": -1}

    plot_config = {
        "main_plot": {},
        "subplots": {
            "dissimilarity_index": {"dissimilarity_index": {"color": "blue"}},
            "minima": {
                "minima": {"color": "brown"},
            },
            "maxima": {
                "maxima": {"color": "brown"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    # this is the maximum period fed to talib (timeframe independent)
    startup_candle_count: int = 40
    can_short = False

    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space='sell', optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)

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

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                # Signal: RSI crosses above 30
                (qtpylib.crossed_above(df['rsi'], self.buy_rsi.value)) &
                (df['tema'] <= df['bb_middleband']) &  # Guard: tema below BB middle
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['dissimilarity_index'] < 1) &
                (df['maxima'] > 0.1) &
                (df['volume'] > 0)  # Make sure Volume is not 0
                # (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                # (df['&s-up_or_down'] == 'up')
            ),
            'enter_long'] = 1

        df.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(df['rsi'], self.short_rsi.value)) &
                (df['tema'] > df['bb_middleband']) &  # Guard: tema above BB middle
                (df['tema'] < df['tema'].shift(1)) &  # Guard: tema is falling
                (df['dissimilarity_index'] < 1) &
                (df['minima'] > 0.1) &
                (df['volume'] > 0)  # Make sure Volume is not 0
                # (df['do_predict'] == 1) &  # Make sure Freqai is confident in the prediction
                # Only enter trade if Freqai thinks the trend is in this direction
                # (df['&s-up_or_down'] == 'down')
            ),
            'enter_short'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(df['rsi'], self.sell_rsi.value)) &
                (df['tema'] > df['bb_middleband']) &  # Guard: tema above BB middle
                (df['tema'] < df['tema'].shift(1)) &  # Guard: tema is falling
                (df['dissimilarity_index'] < 1) &
                (df['maxima'] > 0.1) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),

            'exit_long'] = 1

        df.loc[
            (
                # Signal: RSI crosses above 30
                (qtpylib.crossed_above(df['rsi'], self.exit_short_rsi.value)) &
                # Guard: tema below BB middle
                (df['tema'] <= df['bb_middleband']) &
                (df['tema'] > df['tema'].shift(1)) &  # Guard: tema is raising
                (df['dissimilarity_index'] < 1) &
                (df['minima'] > 0.1) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 1

        return df
