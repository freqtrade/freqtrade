"""
IStrategy interface
This module defines the interface to apply for strategies
"""
import warnings
from typing import Dict

from abc import ABC
from pandas import DataFrame


class IStrategy(ABC):
    """
    Interface for freqtrade strategies
    Defines the mandatory structure must follow any custom strategies

    Attributes you can use:
        minimal_roi -> Dict: Minimal ROI designed for the strategy
        stoploss -> float: optimal stoploss designed for the strategy
        ticker_interval -> str: value of the ticker interval to use for the strategy
    """

    # associated minimal roi
    minimal_roi: Dict

    # associated stoploss
    stoploss: float

    # associated ticker interval
    ticker_interval: str

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Populate indicators that will be used in the Buy and Sell strategy
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        warnings.warn("deprecated - please replace this method with advise_indicators!",
                      DeprecationWarning)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        warnings.warn("deprecated - please replace this method with advise_buy!",
                      DeprecationWarning)
        dataframe.loc[(), 'buy'] = 0
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with sell column
        """
        warnings.warn("deprecated - please replace this method with advise_sell!",
                      DeprecationWarning)
        dataframe.loc[(), 'sell'] = 0
        return dataframe

    def advise_indicators(self, dataframe: DataFrame, pair: str) -> DataFrame:
        """

        This wraps around the internal method

        Populate indicators that will be used in the Buy and Sell strategy
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param pair: The currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        return self.populate_indicators(dataframe)

    def advise_buy(self, dataframe: DataFrame, pair: str) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :param pair: The currently traded pair
        :return: DataFrame with buy column
        """

        return self.populate_buy_trend(dataframe)

    def advise_sell(self, dataframe: DataFrame, pair: str) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :param pair: The currently traded pair
        :return: DataFrame with sell column
        """
        return self.populate_sell_trend(dataframe)
