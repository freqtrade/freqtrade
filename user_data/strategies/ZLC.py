# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from hyperopt import hp
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class ZLC(IStrategy):
    """

    author@: Gert Wohlgemuth
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "60": 0.01,
        "30": 0.03,
        "20": 0.04,
        "0": 0.01
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.3

    # Optimal ticker interval for the strategy
    ticker_interval = 5

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe['cci-slow'] = ta.CCI(dataframe, timeperiod=25)
        dataframe['cci-fast'] = ta.CCI(dataframe, timeperiod=50)
        dataframe['expo'] = ta.EMA(dataframe, timeperiod=35)

        # required for graphing
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                #don't buy on peak tops
                        (dataframe['close'] < dataframe['bb_middleband'])
                        # this is the main concept of evaluating buys
                        & (dataframe['cci-fast'] > 0)
                        & (dataframe['cci-slow'] > 0)
                        & (dataframe['close'] > dataframe['expo'])

            )
            ,
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (dataframe['close'] >= dataframe['bb_upperband']) |
            (
                    (dataframe['cci-fast'] < 0)
                    & (dataframe['cci-slow'] < 0)
                    & (dataframe['close'] < dataframe['expo'])

            )
            ,
            'sell'] = 0
        return dataframe
