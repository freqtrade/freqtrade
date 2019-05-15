"""
IHyperOpt interface
This module defines the interface to apply for hyperopts
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List

from pandas import DataFrame

#
# Filter warnings from importing scikit-learn via skopt.
# scikit-learn specifically forces warnings to be displayed, which we don't like.
#
# See:
# https://github.com/scikit-learn/scikit-learn/issues/2531
# https://github.com/ims-tcl/DeRE/commit/cb865fedddf6977cb7536b8c4334dc2325d09e53
# https://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
# for more details.
#
# Use
# from freqtrade.optimize.hyperopt_interface import Categorical, Dimension, Integer, Real
# instead of
# from skopt.space import Categorical, Dimension, Integer, Real
# in the custom HyperOpts to get rid of this deprecation warning
# (refer to default_hyperopt.py for example) without implementing this ugly
# workaround in every custom HyperOpts.
#
# Let's wait till scikit-learn v0.23 and skopt using it
# where it will probably be eliminated...
#
import warnings


def warn(*args, **kwargs):
    pass


old_warn = warnings.showwarning
warnings.showwarning = warn
from skopt.space import Categorical, Dimension, Integer, Real  # noqa: F401
warnings.showwarning = old_warn


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
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Create a sell strategy generator
        """

    @staticmethod
    @abstractmethod
    def indicator_space() -> List[Dimension]:
        """
        Create an indicator space
        """

    @staticmethod
    @abstractmethod
    def sell_indicator_space() -> List[Dimension]:
        """
        Create a sell indicator space
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
