"""
IHyperOpt interface
This module defines the interface to apply for hyperopts
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List

from pandas import DataFrame
from skopt.space import Dimension


class IHyperOpt(ABC):
    """
    Interface for freqtrade hyperopts
    Defines the mandatory structure must follow any custom strategies

    Attributes you can use:
        minimal_roi -> Dict: Minimal ROI designed for the strategy
        stoploss -> float: optimal stoploss designed for the strategy
        ticker_interval -> int: value of the ticker interval to use for the strategy
    """

    @staticmethod
    @abstractmethod
    def populate_indicators(dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators that will be used in the Buy and Sell strategy
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :return: a Dataframe with all mandatory indicators for the strategies
        """

    @staticmethod
    @abstractmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Create a buy strategy generator
        """

    @staticmethod
    @abstractmethod
    def indicator_space() -> List[Dimension]:
        """
        Create an indicator space
        """

    @staticmethod
    @abstractmethod
    def generate_roi_table(params: Dict) -> Dict[int, float]:
        """
        Create an roi table
        """

    @staticmethod
    @abstractmethod
    def stoploss_space() -> List[Dimension]:
        """
        Create a stoploss space
        """

    @staticmethod
    @abstractmethod
    def roi_space() -> List[Dimension]:
        """
        Create a roi space
        """
