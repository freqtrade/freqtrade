"""
PairList manager class
"""
import logging
from functools import partial
from typing import Dict, List, Optional

from cachetools import TTLCache, cached

from freqtrade.constants import Config, ListPairsWithTimeframes
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.types import Tickers
from freqtrade.mixins import LoggingMixin
from freqtrade.plugins.pairlist.IPairList import IPairList
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.resolvers import PairListResolver


logger = logging.getLogger(__name__)


class PairListManager(LoggingMixin):

    def __init__(
            self, exchange, config: Config, dataprovider: Optional[DataProvider] = None) -> None:
        self._exchange = exchange
        self._config = config
        self._whitelist = self._config['exchange'].get('pair_whitelist')
        self._blacklist = self._config['exchange'].get('pair_blacklist', [])
        self._pairlist_handlers: List[IPairList] = []
        self._tickers_needed = False
        self._dataprovider: Optional[DataProvider] = dataprovider
        for pairlist_handler_config in self._config.get('pairlists', []):
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

        if self._tickers_needed and not self._exchange.exchange_has('fetchTickers'):
            invalid = ". ".join([p.name for p in self._pairlist_handlers if p.needstickers])

            raise OperationalException(
                "Exchange does not support fetchTickers, therefore the following pairlists "
                "cannot be used. Please edit your config and restart the bot.\n"
                f"{invalid}."
            )

        refresh_period = config.get('pairlist_refresh_period', 3600)
        LoggingMixin.__init__(self, logger, refresh_period)

    @property
    def whitelist(self) -> List[str]:
        """The current whitelist"""
        return self._whitelist

    @property
    def blacklist(self) -> List[str]:
        """
        The current blacklist
        -> no need to overwrite in subclasses
        """
        return self._blacklist

    @property
    def expanded_blacklist(self) -> List[str]:
        """The expanded blacklist (including wildcard expansion)"""
        return expand_pairlist(self._blacklist, self._exchange.get_markets().keys())

    @property
    def name_list(self) -> List[str]:
        """Get list of loaded Pairlist Handler names"""
        return [p.name for p in self._pairlist_handlers]

    def short_desc(self) -> List[Dict]:
        """List of short_desc for each Pairlist Handler"""
        return [{p.name: p.short_desc()} for p in self._pairlist_handlers]

    @cached(TTLCache(maxsize=1, ttl=1800))
    def _get_cached_tickers(self) -> Tickers:
        return self._exchange.get_tickers()

    def refresh_pairlist(self) -> None:
        """Run pairlist through all configured Pairlist Handlers."""
        # Tickers should be cached to avoid calling the exchange on each call.
        tickers: Dict = {}
        if self._tickers_needed:
            tickers = self._get_cached_tickers()

        # Generate the pairlist with first Pairlist Handler in the chain
        pairlist = self._pairlist_handlers[0].gen_pairlist(tickers)

        # Process all Pairlist Handlers in the chain
        # except for the first one, which is the generator.
        for pairlist_handler in self._pairlist_handlers[1:]:
            pairlist = pairlist_handler.filter_pairlist(pairlist, tickers)

        # Validation against blacklist happens after the chain of Pairlist Handlers
        # to ensure blacklist is respected.
        pairlist = self.verify_blacklist(pairlist, logger.warning)

        self.log_once(f"Whitelist with {len(pairlist)} pairs: {pairlist}", logger.info)

        self._whitelist = pairlist

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
        try:
            blacklist = self.expanded_blacklist
        except ValueError as err:
            logger.error(f"Pair blacklist contains an invalid Wildcard: {err}")
            return []
        log_once = partial(self.log_once, logmethod=logmethod)
        for pair in pairlist.copy():
            if pair in blacklist:
                log_once(f"Pair {pair} in your blacklist. Removing it from whitelist...")
                pairlist.remove(pair)
        return pairlist

    def verify_whitelist(self, pairlist: List[str], logmethod,
                         keep_invalid: bool = False) -> List[str]:
        """
        Verify and remove items from pairlist - returning a filtered pairlist.
        Logs a warning or info depending on `aswarning`.
        Pairlist Handlers explicitly using this method shall use
        `logmethod=logger.info` to avoid spamming with warning messages
        :param pairlist: Pairlist to validate
        :param logmethod: Function that'll be called, `logger.info` or `logger.warning`
        :param keep_invalid: If sets to True, drops invalid pairs silently while expanding regexes.
        :return: pairlist - whitelisted pairs
        """
        try:
            whitelist = expand_pairlist(pairlist, self._exchange.get_markets().keys(), keep_invalid)
        except ValueError as err:
            logger.error(f"Pair whitelist contains an invalid Wildcard: {err}")
            return []
        return whitelist

    def create_pair_list(
            self, pairs: List[str], timeframe: Optional[str] = None) -> ListPairsWithTimeframes:
        """
        Create list of pair tuples with (pair, timeframe)
        """
        return [
            (
                pair,
                timeframe or self._config['timeframe'],
                self._config.get('candle_type_def', CandleType.SPOT)
            ) for pair in pairs
        ]
