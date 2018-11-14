"""
IExecute interface
This module defines the execution strategy
"""

from abc import ABC, abstractmethod
from freqtrade.exchange import Exchange


class MarketOrder():
    """
    Class encapsulating an order to be placed at the current market price.
    """

    def __init__(self, exchange=None):
        self._exchange = exchange

    def get_limit_price(self, _is_buy):
        return None

    def get_stop_price(self, _is_buy):
        return None


class LimitOrder():
    """
    Execution style representing an order to be executed at a price equal to or
    better than a specified limit price.
    """

    def __init__(self, limit_price, asset=None, exchange=None):
        """
        Store the given price.
        """

        self.limit_price = limit_price
        self._exchange = exchange
        self.asset = asset


class StopOrder():
    """
    Execution style representing an order to be placed once the market price
    reaches a specified stop price.
    """

    def __init__(self, stop_price, asset=None, exchange=None):
        """
        Store the given price.
        """

        self.stop_price = stop_price
        self._exchange = exchange
        self.asset = asset


class StopLimitOrder():
    """
    Execution style representing a limit order to be placed with a specified
    limit price once the market reaches a specified stop price.
    """

    def __init__(self, limit_price, stop_price, asset=None, exchange=None):
        """
        Store the given prices
        """

        self.limit_price = limit_price
        self.stop_price = stop_price
        self._exchange = exchange
        self.asset = asset


class IExecution(ABC):

    def __init__(self, config: dict, exchange: Exchange) -> None:
        self.config = config
        self.exchange = exchange

    @abstractmethod
    def execute_buy(self, pair: str, stake_amount: float = None, price: float = None) -> str:
        """
        Populate indicators that will be used in the Buy and Sell strategy
        :param pair: pair to buy
        :param stake_amount: Amount to buy (Optional)
        :param price: Price at which the asset should be bought (Optional)
        :return: Order id if execution was successful, otherwise None
        """
        raise NotImplementedError
