# pragma pylint: disable=attribute-defined-outside-init

"""
This module load custom pairlists
"""
import logging
from pathlib import Path

from freqtrade.constants import Config
from freqtrade.plugins.pairlist.IPairList import IPairList
from freqtrade.resolvers import IResolver


logger = logging.getLogger(__name__)


class PairListResolver(IResolver):
    """
    This class contains all the logic to load custom PairList class
    """
    object_type = IPairList
    object_type_str = "Pairlist"
    user_subdir = None
    initial_search_path = Path(__file__).parent.parent.joinpath('plugins/pairlist').resolve()

    @staticmethod
    def load_pairlist(pairlist_name: str, exchange, pairlistmanager,
                      config: Config, pairlistconfig: dict, pairlist_pos: int) -> IPairList:
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
        return PairListResolver.load_object(pairlist_name, config,
                                            kwargs={'exchange': exchange,
                                                    'pairlistmanager': pairlistmanager,
                                                    'config': config,
                                                    'pairlistconfig': pairlistconfig,
                                                    'pairlist_pos': pairlist_pos},
                                            )
