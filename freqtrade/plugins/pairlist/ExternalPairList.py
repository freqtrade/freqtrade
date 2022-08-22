"""
External Pair List provider

Provides pair list from Leader data
"""
import logging
from threading import Event
from typing import Any, Dict, List

from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class ExternalPairList(IPairList):
    """
    PairList plugin for use with replicate follower mode.
    Will use pairs given from leader data.

    Usage:
        "pairlists": [
            {
                "method": "ExternalPairList",
                "number_assets": 5, # We can limit the amount of pairs to use from leader
            }
        ],
    """

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        # Not sure how to enforce ExternalPairList as the only PairList

        self._num_assets = self._pairlistconfig.get('number_assets')

        self._leader_pairs: List[str] = []
        self._has_data = Event()

    def _clamped_pairlist(self):
        """
        Return the self._leader_pairs pairlist limited to the maximum set num_assets
        or the length of it.
        """
        length = len(self._leader_pairs)
        if self._num_assets:
            return self._leader_pairs[:min(length, self._num_assets)]
        else:
            return self._leader_pairs

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return False

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        -> Please overwrite in subclasses
        """
        return f"{self.name}"

    def add_pairlist_data(self, pairlist: List[str]):
        """
        Add pairs from Leader
        """

        # If some pairs were removed on Leader, remove them here
        for pair in self._leader_pairs:
            if pair not in pairlist:
                logger.debug(f"Leader removed pair: {pair}")
                self._leader_pairs.remove(pair)

        # Only add new pairs
        seen = set(self._leader_pairs)
        for pair in pairlist:
            if pair in seen:
                continue
            self._leader_pairs.append(pair)

        if not self._has_data.is_set() and len(self._leader_pairs) > 0:
            self._has_data.set()

    def gen_pairlist(self, tickers: Dict) -> List[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: List of pairs
        """
        if not self._has_data.is_set():
            logger.info("Waiting on pairlists from Leaders...")
            self._has_data.wait()
            logger.info("Pairlist data received...")

        return self._clamped_pairlist()

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        return self._clamped_pairlist()
