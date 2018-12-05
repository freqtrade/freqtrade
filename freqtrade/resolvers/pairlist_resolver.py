# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom hyperopts
"""
import logging
from pathlib import Path

from freqtrade.pairlist.IPairList import IPairList
from freqtrade.resolvers import IResolver

logger = logging.getLogger(__name__)


class PairListResolver(IResolver):
    """
    This class contains all the logic to load custom hyperopt class
    """

    __slots__ = ['pairlist']

    def __init__(self, pairlist_name: str, freqtrade, config: dict) -> None:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary or None
        """
        self.pairlist = self._load_pairlist(pairlist_name, kwargs={'freqtrade': freqtrade,
                                                                   'config': config})

    def _load_pairlist(
            self, pairlist_name: str, kwargs: dict) -> IPairList:
        """
        Search and loads the specified pairlist.
        :param pairlist_name: name of the module to import
        :param extra_dir: additional directory to search for the given pairlist
        :return: PairList instance or None
        """
        current_path = Path(__file__).parent.parent.joinpath('pairlist').resolve()

        abs_paths = [
            current_path.parent.parent.joinpath('user_data/pairlist'),
            current_path,
        ]

        for _path in abs_paths:
            try:
                pairlist = self._search_object(directory=_path, object_type=IPairList,
                                               object_name=pairlist_name,
                                               kwargs=kwargs)
                if pairlist:
                    logger.info('Using resolved pairlist %s from \'%s\'', pairlist_name, _path)
                    return pairlist
            except FileNotFoundError:
                logger.warning('Path "%s" does not exist', _path.relative_to(Path.cwd()))

        raise ImportError(
            "Impossible to load Pairlist '{}'. This class does not exist"
            " or contains Python code errors".format(pairlist_name)
        )
