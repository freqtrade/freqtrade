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

    def __init__(self, exchange_name: str, freqtrade, config: dict) -> None:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary or None
        """
        self.pairlist = self._load_exchange(exchange_name, kwargs={'freqtrade': freqtrade,
                                                                   'config': config})

    def _load_exchange(
            self, exchange_name: str, kwargs: dict) -> Exchange:
        """
        Search and loads the specified exchange.
        :param exchange_name: name of the module to import
        :param extra_dir: additional directory to search for the given exchange
        :return: Exchange instance or None
        """
        current_path = Path(__file__).parent.parent.joinpath('exchange').resolve()

        abs_paths = [
            current_path.parent.parent.joinpath('user_data/exchange'),
            current_path,
        ]

        for _path in abs_paths:
            try:
                pairlist = self._search_object(directory=_path, object_type=Exchange,
                                               object_name=exchange_name,
                                               kwargs=kwargs)
                if pairlist:
                    logger.info('Using resolved exchange %s from \'%s\'', exchange_name, _path)
                    return pairlist
            except FileNotFoundError:
                logger.warning('Path "%s" does not exist', _path.relative_to(Path.cwd()))

        raise ImportError(
            "Impossible to load Exchange '{}'. This class does not exist"
            " or contains Python code errors".format(exchange_name)
        )
