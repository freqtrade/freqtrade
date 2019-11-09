"""
Static List provider

Provides lists as configured in config.json

 """
import logging
from typing import Dict, List

from freqtrade import OperationalException
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

        for pl in self._config.get('pairlists', None):
            if 'method' not in pl:
                logger.warning(f"No method in {pl}")
                continue
            pairl = PairListResolver(pl.get('method'),
                                     exchange, self, config,
                                     pl.get('config')).pairlist
            self._tickers_needed = pairl.needstickers or self._tickers_needed
            self._pairlists.append(pairl)

        if not self._pairlists:
            raise OperationalException("No Pairlist defined!")

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

    @property
    def name_list(self) -> List[str]:
        """
        Get list of loaded pairlists names
        """
        return [p.name for p in self._pairlists]

    def short_desc(self) -> List[Dict]:
        """
        List of short_desc for each pairlist
        """
        return [{p.name: p.short_desc()} for p in self._pairlists]

    def refresh_pairlist(self) -> None:
        """
        Run pairlist through all pairlists.
        """

        pairlist = self._whitelist.copy()

        # tickers should be cached to avoid calling the exchange on each call.
        tickers: Dict = {}
        if self._tickers_needed:
            tickers = self._exchange.get_tickers()

        # Process all pairlists in chain
        for pl in self._pairlists:
            pairlist = pl.filter_pairlist(pairlist, tickers)

        # Validation against blacklist happens after the pairlists to ensure blacklist is respected.
        pairlist = IPairList.verify_blacklist(pairlist, self.blacklist)

        self._whitelist = pairlist
