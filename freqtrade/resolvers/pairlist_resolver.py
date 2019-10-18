# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom pairlists
"""
import logging
from pathlib import Path

from freqtrade import OperationalException
from freqtrade.pairlist.IPairList import IPairList
from freqtrade.resolvers import IResolver

logger = logging.getLogger(__name__)


class PairListResolver(IResolver):
    """
    This class contains all the logic to load custom PairList class
    """

    __slots__ = ['pairlist']

    def __init__(self, pairlist_name: str, freqtrade, config: dict) -> None:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary or None
        """
        self.pairlist = self._load_pairlist(pairlist_name, config, kwargs={'freqtrade': freqtrade,
                                                                           'config': config})

    def _load_pairlist(
            self, pairlist_name: str, config: dict, kwargs: dict) -> IPairList:
        """
        Search and loads the specified pairlist.
        :param pairlist_name: name of the module to import
        :param config: configuration dictionary
        :param extra_dir: additional directory to search for the given pairlist
        :return: PairList instance or None
        """
        current_path = Path(__file__).parent.parent.joinpath('pairlist').resolve()

        abs_paths = self.build_search_paths(config, current_path=current_path,
                                            user_subdir='pairlist', extra_dir=None)

        pairlist = self._load_object(paths=abs_paths, object_type=IPairList,
                                     object_name=pairlist_name, kwargs=kwargs)
        if pairlist:
            return pairlist
        raise OperationalException(
            f"Impossible to load Pairlist '{pairlist_name}'. This class does not exist "
            "or contains Python code errors."
        )
