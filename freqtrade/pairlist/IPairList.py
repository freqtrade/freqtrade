"""
PairList Handler base class
"""
import logging
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from typing import Any, Dict, List

from cachetools import TTLCache, cached

from freqtrade.exchange import market_is_active


logger = logging.getLogger(__name__)


class IPairList(ABC):

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        """
        :param exchange: Exchange instance
        :param pairlistmanager: Instantiated Pairlist manager
        :param config: Global bot configuration
        :param pairlistconfig: Configuration for this Pairlist Handler - can be empty.
        :param pairlist_pos: Position of the Pairlist Handler in the chain
        """
        self._enabled = True

        self._exchange = exchange
        self._pairlistmanager = pairlistmanager
        self._config = config
        self._pairlistconfig = pairlistconfig
        self._pairlist_pos = pairlist_pos
        self.refresh_period = self._pairlistconfig.get('refresh_period', 1800)
        self._last_refresh = 0
        self._log_cache = TTLCache(maxsize=1024, ttl=self.refresh_period)

    @property
    def name(self) -> str:
        """
        Gets name of the class
        -> no need to overwrite in subclasses
        """
        return self.__class__.__name__

    def log_on_refresh(self, logmethod, message: str) -> None:
        """
        Logs message - not more often than "refresh_period" to avoid log spamming
        Logs the log-message as debug as well to simplify debugging.
        :param logmethod: Function that'll be called. Most likely `logger.info`.
        :param message: String containing the message to be sent to the function.
        :return: None.
        """

        @cached(cache=self._log_cache)
        def _log_on_refresh(message: str):
            logmethod(message)

        # Log as debug first
        logger.debug(message)
        # Call hidden function.
        _log_on_refresh(message)

    @abstractproperty
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requries tickers, an empty List is passed
        as tickers argument to filter_pairlist
        """

    @abstractmethod
    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        -> Please overwrite in subclasses
        """

    def _validate_pair(self, ticker) -> bool:
        """
        Check one pair against Pairlist Handler's specific conditions.

        Either implement it in the Pairlist Handler or override the generic
        filter_pairlist() method.

        :param ticker: ticker dict as returned from ccxt.load_markets()
        :return: True if the pair can stay, false if it should be removed
        """
        raise NotImplementedError()

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.

        Called on each bot iteration - please use internal caching if necessary
        This generic implementation calls self._validate_pair() for each pair
        in the pairlist.

        Some Pairlist Handlers override this generic implementation and employ
        own filtration.

        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        if self._enabled:
            # Copy list since we're modifying this list
            for p in deepcopy(pairlist):
                # Filter out assets
                if not self._validate_pair(tickers[p]):
                    pairlist.remove(p)

        return pairlist

    def verify_blacklist(self, pairlist: List[str], logmethod) -> List[str]:
        """
        Proxy method to verify_blacklist for easy access for child classes.
        :param pairlist: Pairlist to validate
        :param logmethod: Function that'll be called, `logger.info` or `logger.warning`.
        :return: pairlist - blacklisted pairs
        """
        return self._pairlistmanager.verify_blacklist(pairlist, logmethod)

    def _whitelist_for_active_markets(self, pairlist: List[str]) -> List[str]:
        """
        Check available markets and remove pair from whitelist if necessary
        :param whitelist: the sorted list of pairs the user might want to trade
        :return: the list of pairs the user wants to trade without those unavailable or
        black_listed
        """
        markets = self._exchange.markets

        sanitized_whitelist: List[str] = []
        for pair in pairlist:
            # pair is not in the generated dynamic market or has the wrong stake currency
            if pair not in markets:
                logger.warning(f"Pair {pair} is not compatible with exchange "
                               f"{self._exchange.name}. Removing it from whitelist..")
                continue

            if self._exchange.get_pair_quote_currency(pair) != self._config['stake_currency']:
                logger.warning(f"Pair {pair} is not compatible with your stake currency "
                               f"{self._config['stake_currency']}. Removing it from whitelist..")
                continue

            # Check if market is active
            market = markets[pair]
            if not market_is_active(market):
                logger.info(f"Ignoring {pair} from whitelist. Market is not active.")
                continue
            if pair not in sanitized_whitelist:
                sanitized_whitelist.append(pair)

        # We need to remove pairs that are unknown
        return sanitized_whitelist
