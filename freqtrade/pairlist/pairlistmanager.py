"""
PairList manager class
"""
import logging
from copy import deepcopy
from typing import Dict, List

from cachetools import TTLCache, cached

from freqtrade.exceptions import OperationalException
from freqtrade.pairlist.IPairList import IPairList
from freqtrade.resolvers import PairListResolver
from freqtrade.constants import ListPairsWithTimeframes


logger = logging.getLogger(__name__)


class PairListManager():

    def __init__(self, exchange, config: dict) -> None:
        self._exchange = exchange
        self._config = config
        self._whitelist = self._config['exchange'].get('pair_whitelist')
        self._blacklist = self._config['exchange'].get('pair_blacklist', [])
        self._pairlist_handlers: List[IPairList] = []
        self._tickers_needed = False
        for pairlist_handler_config in self._config.get('pairlists', None):
            if 'method' not in pairlist_handler_config:
                logger.warning(f"No method found in {pairlist_handler_config}, ignoring.")
                continue
            pairlist_handler = PairListResolver.load_pairlist(
                    pairlist_handler_config['method'],
                    exchange=exchange,
                    pairlistmanager=self,
                    config=config,
                    pairlistconfig=pairlist_handler_config,
                    pairlist_pos=len(self._pairlist_handlers)
                    )
            self._tickers_needed |= pairlist_handler.needstickers
            self._pairlist_handlers.append(pairlist_handler)

        if not self._pairlist_handlers:
            raise OperationalException("No Pairlist Handlers defined")

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
        Get list of loaded Pairlist Handler names
        """
        return [p.name for p in self._pairlist_handlers]

    def short_desc(self) -> List[Dict]:
        """
        List of short_desc for each Pairlist Handler
        """
        return [{p.name: p.short_desc()} for p in self._pairlist_handlers]

    @cached(TTLCache(maxsize=1, ttl=1800))
    def _get_cached_tickers(self):
        return self._exchange.get_tickers()

    def refresh_pairlist(self) -> None:
        """
        Run pairlist through all configured Pairlist Handlers.
        """
        # Tickers should be cached to avoid calling the exchange on each call.
        tickers: Dict = {}
        if self._tickers_needed:
            tickers = self._get_cached_tickers()

        # Adjust whitelist if filters are using tickers
        pairlist = self._prepare_whitelist(self._whitelist.copy(), tickers)

        # Process all Pairlist Handlers in the chain
        for pairlist_handler in self._pairlist_handlers:
            pairlist = pairlist_handler.filter_pairlist(pairlist, tickers)

        # Validation against blacklist happens after the chain of Pairlist Handlers
        # to ensure blacklist is respected.
        pairlist = self.verify_blacklist(pairlist, logger.warning)

        self._whitelist = pairlist

    def _prepare_whitelist(self, pairlist: List[str], tickers) -> List[str]:
        """
        Prepare sanitized pairlist for Pairlist Handlers that use tickers data - remove
        pairs that do not have ticker available
        """
        if self._tickers_needed:
            # Copy list since we're modifying this list
            for p in deepcopy(pairlist):
                if p not in tickers:
                    pairlist.remove(p)

        return pairlist

    def verify_blacklist(self, pairlist: List[str], logmethod) -> List[str]:
        """
        Verify and remove items from pairlist - returning a filtered pairlist.
        Logs a warning or info depending on `aswarning`.
        Pairlist Handlers explicitly using this method shall use
        `logmethod=logger.info` to avoid spamming with warning messages
        :param pairlist: Pairlist to validate
        :param logmethod: Function that'll be called, `logger.info` or `logger.warning`.
        :return: pairlist - blacklisted pairs
        """
        for pair in deepcopy(pairlist):
            if pair in self._blacklist:
                logmethod(f"Pair {pair} in your blacklist. Removing it from whitelist...")
                pairlist.remove(pair)
        return pairlist

    def create_pair_list(self, pairs: List[str], timeframe: str = None) -> ListPairsWithTimeframes:
        """
        Create list of pair tuples with (pair, ticker_interval)
        """
        return [(pair, timeframe or self._config['ticker_interval']) for pair in pairs]
