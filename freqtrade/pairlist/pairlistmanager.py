"""
Static List provider

Provides lists as configured in config.json

 """
import logging
from copy import deepcopy
from typing import List

from freqtrade.pairlist.IPairList import IPairList
from freqtrade.resolvers import PairListResolver

logger = logging.getLogger(__name__)


class PairListManager():

    def __init__(self, exchange, config: dict) -> None:
        self._exchange = exchange
        self._config = config
        self._whitelist = self._config['exchange'].get('pair_whitelist')
        self._blacklist = self._config['exchange'].get('pair_blacklist', [])
        self._pairlists: List[IPairList] = []
        self._tickers_needed = False
        for pl in self._config.get('pairlists', [{'method': "StaticPairList"}]):
            pairl = PairListResolver(pl.get('method'),
                                     exchange, config,
                                     pl.get('config')).pairlist
            self._tickers_needed = pairl.needstickers or self._tickers_needed
            self._pairlists.append(pairl)

    @property
    def whitelist(self) -> List[str]:
        """
        Has the current whitelist
        """
        return self._whitelist

    @property
    def blacklist(self) -> List[str]:
        """
        Has the current blacklist
        -> no need to overwrite in subclasses
        """
        return self._blacklist

    def refresh_pairlist(self) -> None:
        """
        Run pairlist through all pairlists.
        """

        pairlist = self._whitelist.copy()

        # tickers should be cached to avoid calling the exchange on each call.
        tickers = []
        if self._tickers_needed:
            tickers = self._exchange.get_tickers()

        for pl in self._pairlists:
            pl.filter_pairlist(pairlist, tickers)

        # Validation against blacklist happens after the pairlists to ensure blacklist is respected.
        pairlist = self.verify_blacklist(pairlist, self.blacklist)
        self._whitelist = pairlist

    @staticmethod
    def verify_blacklist(pairlist: List[str], blacklist: List[str]) -> List[str]:

        for pair in deepcopy(pairlist):
            if pair in blacklist:
                logger.warning(f"Pair {pair} in your blacklist. Removing it from whitelist...")
                pairlist.remove(pair)
        return pairlist
