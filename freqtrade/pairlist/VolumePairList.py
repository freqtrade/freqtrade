"""
Volume PairList provider

Provides lists as configured in config.json

 """
import logging
from typing import Dict, List

from cachetools import TTLCache, cached

from freqtrade import OperationalException
from freqtrade.pairlist.IPairList import IPairList

logger = logging.getLogger(__name__)

SORT_VALUES = ['askVolume', 'bidVolume', 'quoteVolume']


class VolumePairList(IPairList):

    def __init__(self, exchange, config, pairlistconfig: dict) -> None:
        super().__init__(exchange, config, pairlistconfig)

        if 'number_assets' not in self._pairlistconfig:
            raise OperationalException(
                f'`number_assets` not specified. Please check your configuration '
                'for "pairlist.config.number_assets"')
        self._number_pairs = self._pairlistconfig['number_assets']
        self._sort_key = self._pairlistconfig.get('sort_key', 'quoteVolume')

        if not self._exchange.exchange_has('fetchTickers'):
            raise OperationalException(
                'Exchange does not support dynamic whitelist.'
                'Please edit your config and restart the bot'
            )
        if not self._validate_keys(self._sort_key):
            raise OperationalException(
                f'key {self._sort_key} not in {SORT_VALUES}')

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requries tickers, an empty List is passed
        as tickers argument to filter_pairlist
        """
        return True

    def _validate_keys(self, key):
        return key in SORT_VALUES

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return f"{self.name} - top {self._pairlistconfig['number_assets']} volume pairs."

    def filter_pairlist(self, pairlist: List[str], tickers: List[Dict]) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        # Generate dynamic whitelist
        return self._gen_pair_whitelist(self._config['stake_currency'], self._sort_key)

    @cached(TTLCache(maxsize=1, ttl=1800))
    def _gen_pair_whitelist(self, base_currency: str, key: str) -> List[str]:
        """
        Updates the whitelist with with a dynamically generated list
        :param base_currency: base currency as str
        :param key: sort key (defaults to 'quoteVolume')
        :param tickers: Tickers (from exchange.get_tickers()).
        :return: List of pairs
        """

        tickers = self._exchange.get_tickers()
        # check length so that we make sure that '/' is actually in the string
        tickers = [v for k, v in tickers.items()
                   if (len(k.split('/')) == 2 and k.split('/')[1] == base_currency
                       and v[key] is not None)]
        sorted_tickers = sorted(tickers, reverse=True, key=lambda t: t[key])
        # Validate whitelist to only have active market pairs
        pairs = self._whitelist_for_active_markets([s['symbol'] for s in sorted_tickers])

        logger.info(f"Searching {self._number_pairs} pairs: {pairs[:self._number_pairs]}")

        return pairs
