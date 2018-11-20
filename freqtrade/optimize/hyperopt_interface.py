"""
IHyperOpt interface
This module defines the interface to apply for hyperopts
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable

from pandas import DataFrame


class IHyperOpt(ABC):
    """
    Interface for freqtrade hyperopts
    Defines the mandatory structure must follow any custom strategies

    Attributes you can use:
        minimal_roi -> Dict: Minimal ROI designed for the strategy
        stoploss -> float: optimal stoploss designed for the strategy
        ticker_interval -> int: value of the ticker interval to use for the strategy
    """

    @abstractmethod
    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Populate indicators that will be used in the Buy and Sell strategy
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :return: a Dataframe with all mandatory indicators for the strategies
        """

    @abstractmethod
    def buy_strategy_generator(self, params: Dict[str, Any]) -> Callable:
        """
        Create a buy strategy generator
        """

    @abstractmethod
    def indicator_space(self) -> Dict[str, Any]:
        """
        Create an indicator space
        """

    @abstractmethod
    def generate_roi_table(self, params: Dict) -> Dict[int, float]:
        """
        Create an roi table
        """

    @abstractmethod
    def stoploss_space(self) -> Dict[str, Any]:
        """
        Create a stoploss space
        """

    @abstractmethod
    def roi_space(self) -> Dict[str, Any]:
        """
        Create a roi space
        """
