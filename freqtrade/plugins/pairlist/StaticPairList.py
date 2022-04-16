"""
Static Pair List provider

Provides pair white list as it configured in config
"""
import logging
from copy import deepcopy
from typing import Any, Dict, List

from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class StaticPairList(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._allow_inactive = self._pairlistconfig.get('allow_inactive', False)

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

    def gen_pairlist(self, tickers: Dict) -> List[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: List of pairs
        """
        if self._allow_inactive:
            return self.verify_whitelist(
                self._config['exchange']['pair_whitelist'], logger.info, keep_invalid=True
            )
        else:
            return self._whitelist_for_active_markets(
                self.verify_whitelist(self._config['exchange']['pair_whitelist'], logger.info))

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        pairlist_ = deepcopy(pairlist)
        for pair in self._config['exchange']['pair_whitelist']:
            if pair not in pairlist_:
                pairlist_.append(pair)
        return pairlist_
