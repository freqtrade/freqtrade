"""
This module loads custom exchanges
"""
import logging

import freqtrade.exchange as exchanges
from freqtrade.constants import Config
from freqtrade.exchange import MAP_EXCHANGE_CHILDCLASS, Exchange
from freqtrade.resolvers import IResolver


logger = logging.getLogger(__name__)


class ExchangeResolver(IResolver):
    """
    This class contains all the logic to load a custom exchange class
    """
    object_type = Exchange

    @staticmethod
    def load_exchange(exchange_name: str, config: Config, validate: bool = True,
                      load_leverage_tiers: bool = False) -> Exchange:
        """
        Load the custom class from config parameter
        :param exchange_name: name of the Exchange to load
        :param config: configuration dictionary
        """
        # Map exchange name to avoid duplicate classes for identical exchanges
        exchange_name = MAP_EXCHANGE_CHILDCLASS.get(exchange_name, exchange_name)
        exchange_name = exchange_name.title()
        exchange = None
        try:
            exchange = ExchangeResolver._load_exchange(
                exchange_name,
                kwargs={
                    'config': config,
                    'validate': validate,
                    'load_leverage_tiers': load_leverage_tiers}
            )
        except ImportError:
            logger.info(
                f"No {exchange_name} specific subclass found. Using the generic class instead.")
        if not exchange:
            exchange = Exchange(config, validate=validate)
        return exchange

    @staticmethod
    def _load_exchange(exchange_name: str, kwargs: dict) -> Exchange:
        """
        Loads the specified exchange.
        Only checks for exchanges exported in freqtrade.exchanges
        :param exchange_name: name of the module to import
        :return: Exchange instance or None
        """

        try:
            ex_class = getattr(exchanges, exchange_name)

            exchange = ex_class(**kwargs)
            if exchange:
                logger.info(f"Using resolved exchange '{exchange_name}'...")
                return exchange
        except AttributeError:
            # Pass and raise ImportError instead
            pass

        raise ImportError(
            f"Impossible to load Exchange '{exchange_name}'. This class does not exist "
            "or contains Python code errors."
        )
