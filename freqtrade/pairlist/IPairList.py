"""
Static List provider

Provides lists as configured in config.json

 """
import logging
from abc import ABC, abstractmethod
from typing import Dict, List

from freqtrade.exchange import market_is_active
from freqtrade.pairlist.IPairListFilter import IPairListFilter
from freqtrade.resolvers.pairlistfilter_resolver import PairListFilterResolver

logger = logging.getLogger(__name__)


class IPairList(ABC):

    def __init__(self, freqtrade, config: dict) -> None:
        self._freqtrade = freqtrade
        self._config = config
        self._whitelist = self._config['exchange']['pair_whitelist']
        self._blacklist = self._config['exchange'].get('pair_blacklist', [])
        self._filters = self._config.get('pairlist', {}).get('filters', {})
        self._pairlistfilters: List[IPairListFilter] = []
        for pl_filter in self._filters.keys():
            self._pairlistfilters.append(
                PairListFilterResolver(pl_filter, freqtrade, self._config).pairlistfilter
                )

    @property
    def name(self) -> str:
        """
        Gets name of the class
        -> no need to overwrite in subclasses
        """
        return self.__class__.__name__

    @property
    def whitelist(self) -> List[str]:
        """
        Has the current whitelist
        -> no need to overwrite in subclasses
        """
        return self._whitelist

    @property
    def blacklist(self) -> List[str]:
        """
        Has the current blacklist
        -> no need to overwrite in subclasses
        """
        return self._blacklist

    @abstractmethod
    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        -> Please overwrite in subclasses
        """

    @abstractmethod
    def refresh_pairlist(self) -> None:
        """
        Refreshes pairlists and assigns them to self._whitelist and self._blacklist respectively
        -> Please overwrite in subclasses
        """

    def validate_whitelist(self, pairlist: List[str],
                           tickers: List[Dict] = []) -> List[str]:
        """
        Validate pairlist against active markets and blacklist.
        Run PairlistFilters if these are configured.
        """
        pairlist = self._whitelist_for_active_markets(pairlist)

        if not tickers:
            # Refresh tickers if they are not used by the parent Pairlist
            tickers = self._freqtrade.exchange.get_tickers()

        for pl_filter in self._pairlistfilters:
            pairlist = pl_filter.filter_pairlist(pairlist, tickers)
        return pairlist

    def _whitelist_for_active_markets(self, whitelist: List[str]) -> List[str]:
        """
        Check available markets and remove pair from whitelist if necessary
        :param whitelist: the sorted list of pairs the user might want to trade
        :return: the list of pairs the user wants to trade without those unavailable or
        black_listed
        """
        markets = self._freqtrade.exchange.markets

        sanitized_whitelist: List[str] = []
        for pair in whitelist:
            # pair is not in the generated dynamic market, or in the blacklist ... ignore it
            if (pair in self.blacklist or pair not in markets
                    or not pair.endswith(self._config['stake_currency'])):
                logger.warning(f"Pair {pair} is not compatible with exchange "
                               f"{self._freqtrade.exchange.name} or contained in "
                               f"your blacklist. Removing it from whitelist..")
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
