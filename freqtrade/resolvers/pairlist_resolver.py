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

    @staticmethod
    def load_pairlist(pairlist_name: str, exchange, pairlistmanager,
                      config: dict, pairlistconfig: dict, pairlist_pos: int) -> IPairList:
        """
        Load the pairlist with pairlist_name
        :param pairlist_name: Classname of the pairlist
        :param exchange: Initialized exchange class
        :param pairlistmanager: Initialized pairlist manager
        :param config: configuration dictionary
        :param pairlistconfig: Configuration dedicated to this pairlist
        :param pairlist_pos: Position of the pairlist in the list of pairlists
        :return: initialized Pairlist class
        """

        return PairListResolver._load_pairlist(pairlist_name, config,
                                               kwargs={'exchange': exchange,
                                                       'pairlistmanager': pairlistmanager,
                                                       'config': config,
                                                       'pairlistconfig': pairlistconfig,
                                                       'pairlist_pos': pairlist_pos})

    @staticmethod
    def _load_pairlist(pairlist_name: str, config: dict, kwargs: dict) -> IPairList:
        """
        Search and loads the specified pairlist.
        :param pairlist_name: name of the module to import
        :param config: configuration dictionary
        :param extra_dir: additional directory to search for the given pairlist
        :return: PairList instance or None
        """
        current_path = Path(__file__).parent.parent.joinpath('pairlist').resolve()

        abs_paths = IResolver.build_search_paths(config, current_path=current_path,
                                                 user_subdir=None, extra_dir=None)

        pairlist = IResolver._load_object(paths=abs_paths, object_type=IPairList,
                                          object_name=pairlist_name, kwargs=kwargs)
        if pairlist:
            return pairlist
        raise OperationalException(
            f"Impossible to load Pairlist '{pairlist_name}'. This class does not exist "
            "or contains Python code errors."
        )
