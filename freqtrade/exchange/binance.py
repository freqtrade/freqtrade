""" Binance exchange subclass """
import logging
from typing import Dict

from freqtrade.exchange import Exchange

logger = logging.getLogger(__name__)


class Binance(Exchange):

    _ft_has: Dict = {
        "stoploss_on_exchange": True,
    }

    def get_order_book(self, pair: str, limit: int = 100) -> dict:
        """
        get order book level 2 from exchange

        20180619: binance support limits but only on specific range
        """
        limit_range = [5, 10, 20, 50, 100, 500, 1000]
        # get next-higher step in the limit_range list
        limit = min(list(filter(lambda x: limit <= x, limit_range)))

        return super().get_order_book(pair, limit)

    def validate_order_time_in_force(self, order_time_in_force: Dict) -> None:
        """
        Checks if order time in force configured in strategy/config are supported
        """
        pass
