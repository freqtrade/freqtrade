# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from hyperopt import hp
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Quickie(IStrategy):
    """

    author@: Gert Wohlgemuth

    idea:
        momentum based strategie. The main idea is that it closes trades very quickly, while avoiding excessive losses. Hence a rather moderate stop loss in this case
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "100": 0.01,
        "45": 0.02,
        "30": 0.03,
        "15": 0.06,
        "10": 0.15,
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.25

    # Optimal ticker interval for the strategy
    ticker_interval = 5

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)


        # required for graphing
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                    (
                            (dataframe['adx'] > 30) &
                            (dataframe['tema'] < dataframe['bb_middleband']) &
                            (dataframe['tema'] > dataframe['tema'].shift(1)) &
                            (dataframe['sma_200'] > dataframe['close'])
                    )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['adx'] > 70) &
                    (dataframe['tema'] > dataframe['bb_middleband']) &
                    (dataframe['tema'] < dataframe['tema'].shift(1))
                )
            ),
            'sell'] = 1
        return dataframe
