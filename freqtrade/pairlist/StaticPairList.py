"""
Static List provider

Provides lists as configured in config.json

 """
import logging
from typing import List, Dict

from freqtrade.pairlist.IPairList import IPairList

logger = logging.getLogger(__name__)


class StaticPairList(IPairList):

    def __init__(self, exchange, config, pairlistconfig: dict) -> None:
        super().__init__(exchange, config, pairlistconfig)

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requries tickers, an empty List is passed
        as tickers argument to filter_pairlist
        """
        return False

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        -> Please overwrite in subclasses
        """
        return f"{self.name}: {self.whitelist}"

    def filter_pairlist(self, pairlist: List[str], tickers: List[Dict]) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        return self._config['exchange']['pair_whitelist']
