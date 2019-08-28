"""
IHyperOpt interface
This module defines the interface to apply for hyperopts
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List

from pandas import DataFrame
from skopt.space import Dimension, Integer, Real


class IHyperOpt(ABC):
    """
    Interface for freqtrade hyperopts
    Defines the mandatory structure must follow any custom strategies

    Attributes you can use:
        minimal_roi -> Dict: Minimal ROI designed for the strategy
        stoploss -> float: optimal stoploss designed for the strategy
        ticker_interval -> int: value of the ticker interval to use for the strategy
    """
    ticker_interval: str

    @staticmethod
    @abstractmethod
    def populate_indicators(dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators that will be used in the Buy and Sell strategy.
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe().
        :return: A Dataframe with all mandatory indicators for the strategies.
        """

    @staticmethod
    @abstractmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Create a buy strategy generator.
        """

    @staticmethod
    @abstractmethod
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Create a sell strategy generator.
        """

    @staticmethod
    @abstractmethod
    def indicator_space() -> List[Dimension]:
        """
        Create an indicator space.
        """

    @staticmethod
    @abstractmethod
    def sell_indicator_space() -> List[Dimension]:
        """
        Create a sell indicator space.
        """

    @staticmethod
    def generate_roi_table(params: Dict) -> Dict[int, float]:
        """
        Create a ROI table.

        Generates the ROI table that will be used by Hyperopt.
        You may override it in your custom Hyperopt class.
        """
        roi_table = {}
        roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
        roi_table[params['roi_t3']] = params['roi_p1'] + params['roi_p2']
        roi_table[params['roi_t3'] + params['roi_t2']] = params['roi_p1']
        roi_table[params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0

        return roi_table

    @staticmethod
    def stoploss_space() -> List[Dimension]:
        """
        Create a stoploss space.

        Defines range of stoploss values to search.
        You may override it in your custom Hyperopt class.
        """
        return [
            Real(-0.5, -0.02, name='stoploss'),
        ]

    @staticmethod
    def roi_space() -> List[Dimension]:
        """
        Create a ROI space.

        Defines values to search for each ROI steps.
        You may override it in your custom Hyperopt class.
        """
        return [
            Integer(10, 120, name='roi_t1'),
            Integer(10, 60, name='roi_t2'),
            Integer(10, 40, name='roi_t3'),
            Real(0.01, 0.04, name='roi_p1'),
            Real(0.01, 0.07, name='roi_p2'),
            Real(0.01, 0.20, name='roi_p3'),
        ]
