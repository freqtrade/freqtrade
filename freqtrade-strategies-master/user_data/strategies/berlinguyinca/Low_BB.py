# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, DatetimeIndex, merge
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# import numpy as np # noqa

class Low_BB(IStrategy):
    """

    author@: Thorsten

    works on new objectify branch!

    idea:
        buy after crossing .98 * lower_bb and sell if trailing stop loss is hit
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.9,
        "1": 0.05,
        "10": 0.04,
        "15": 0.5
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.015

    # Optimal timeframe for the strategy
    timeframe = '1m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ##################################################################################
        # buy and sell indicators

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # dataframe['cci'] = ta.CCI(dataframe)
        # dataframe['mfi'] = ta.MFI(dataframe)
        # dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)

        # dataframe['canbuy'] = np.NaN
        # dataframe['canbuy2'] = np.NaN
        # dataframe.loc[dataframe.close.rolling(49).min() <= 1.1 * dataframe.close, 'canbuy'] == 1
        # dataframe.loc[dataframe.close.rolling(600).max() < 1.2 * dataframe.close, 'canbuy'] = 1
        # dataframe.loc[dataframe.close.rolling(600).max() * 0.8 >  dataframe.close, 'canbuy2'] = 1
        ##################################################################################
        # required for graphing
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (

                (dataframe['close'] <= 0.98 * dataframe['bb_lowerband'])

            )
            ,
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (),
            'sell'] = 1
        return dataframe
