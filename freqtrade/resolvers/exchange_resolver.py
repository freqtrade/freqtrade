"""
This module loads custom exchanges
"""
import logging
from pathlib import Path

from freqtrade.exchange import Exchange
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
        :param config: configuration dictionary or None
        """
        try:
            self.exchange = self._load_exchange(exchange_name, kwargs={'config': config})
        except ImportError:
            logger.info(
                f"No {exchange_name} specific subclass found. Using the generic class instead.")
            self.exchange = Exchange(config)

    def _load_exchange(
            self, exchange_name: str, kwargs: dict) -> Exchange:
        """
        Search and loads the specified exchange.
        :param exchange_name: name of the module to import
        :param extra_dir: additional directory to search for the given exchange
        :return: Exchange instance or None
        """
        abs_path = Path(__file__).parent.parent.joinpath('exchange').resolve()

        try:
            exchange = self._search_object(directory=abs_path, object_type=Exchange,
                                           object_name=exchange_name,
                                           kwargs=kwargs)
            if exchange:
                logger.info("Using resolved exchange %s from '%s'", exchange_name, abs_path)
                return exchange
        except FileNotFoundError:
            logger.warning('Path "%s" does not exist', abs_path.relative_to(Path.cwd()))

        raise ImportError(
            "Impossible to load Exchange '{}'. This class does not exist"
            " or contains Python code errors".format(exchange_name)
        )
