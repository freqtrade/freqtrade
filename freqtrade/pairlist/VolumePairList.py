"""
Volume PairList provider

Provides lists as configured in config.json

 """
import logging
from typing import List

from cachetools import TTLCache, cached

from freqtrade import OperationalException
from freqtrade.pairlist.IPairList import IPairList

logger = logging.getLogger(__name__)

SORT_VALUES = ['askVolume', 'bidVolume', 'quoteVolume']


class VolumePairList(IPairList):

    def __init__(self, freqtrade, config: dict) -> None:
        super().__init__(freqtrade, config)
        self._whitelistconf = self._config.get('pairlist', {}).get('config')
        if 'number_assets' not in self._whitelistconf:
            raise OperationalException(
                f'`number_assets` not specified. Please check your configuration '
                'for "pairlist.config.number_assets"')
        self._number_pairs = self._whitelistconf['number_assets']
        self._sort_key = self._whitelistconf.get('sort_key', 'quoteVolume')
        self._precision_filter = self._whitelistconf.get('precision_filter', True)

        if not self._freqtrade.exchange.exchange_has('fetchTickers'):
            raise OperationalException(
                'Exchange does not support dynamic whitelist.'
                'Please edit your config and restart the bot'
            )
        if not self._validate_keys(self._sort_key):
            raise OperationalException(
                f'key {self._sort_key} not in {SORT_VALUES}')

    def _validate_keys(self, key):
        return key in SORT_VALUES

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        -> Please overwrite in subclasses
        """
        return f"{self.name} - top {self._whitelistconf['number_assets']} volume pairs."

    def refresh_pairlist(self) -> None:
        """
        Refreshes pairlists and assigns them to self._whitelist and self._blacklist respectively
        -> Please overwrite in subclasses
        """
        # Generate dynamic whitelist
        self._whitelist = self._gen_pair_whitelist(
            self._config['stake_currency'], self._sort_key)

    @cached(TTLCache(maxsize=1, ttl=1800))
    def _gen_pair_whitelist(self, base_currency: str, key: str) -> List[str]:
        """
        Updates the whitelist with with a dynamically generated list
        :param base_currency: base currency as str
        :param key: sort key (defaults to 'quoteVolume')
        :return: List of pairs
        """

        tickers = self._freqtrade.exchange.get_tickers()
        # check length so that we make sure that '/' is actually in the string
        tickers = [v for k, v in tickers.items()
                   if (len(k.split('/')) == 2 and k.split('/')[1] == base_currency
                       and v[key] is not None)]
        sorted_tickers = sorted(tickers, reverse=True, key=lambda t: t[key])
        # Validate whitelist to only have active market pairs
        pairs = self.validate_whitelist([s['symbol'] for s in sorted_tickers], tickers)

        logger.info(f"Searching {self._number_pairs} pairs: {pairs[:self._number_pairs]}")

        return pairs
