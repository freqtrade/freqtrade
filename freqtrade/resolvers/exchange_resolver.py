"""
This module loads custom exchanges
"""
import logging

from freqtrade.exchange import Exchange
import freqtrade.exchange as exchanges
from freqtrade.resolvers import IResolver

logger = logging.getLogger(__name__)


class ExchangeResolver(IResolver):
    """
    This class contains all the logic to load a custom exchange class
    """

    __slots__ = ['exchange']

    def __init__(self, exchange_name: str, config: dict) -> None:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary
        """
        exchange_name = exchange_name.title()
        try:
            self.exchange = self._load_exchange(exchange_name, kwargs={'config': config})
        except ImportError:
            logger.info(
                f"No {exchange_name} specific subclass found. Using the generic class instead.")
        if not hasattr(self, "exchange"):
            self.exchange = Exchange(config)

    def _load_exchange(
            self, exchange_name: str, kwargs: dict) -> Exchange:
        """
        Loads the specified exchange.
        Only checks for exchanges exported in freqtrade.exchanges
        :param exchange_name: name of the module to import
        :return: Exchange instance or None
        """

        try:
            ex_class = getattr(exchanges, exchange_name)

            exchange = ex_class(kwargs['config'])
            if exchange:
                logger.info("Using resolved exchange %s", exchange_name)
                return exchange
        except AttributeError:
            # Pass and raise ImportError instead
            pass

        raise ImportError(
            "Impossible to load Exchange '{}'. This class does not exist"
            " or contains Python code errors".format(exchange_name)
        )
