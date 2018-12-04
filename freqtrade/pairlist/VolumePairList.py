"""
Static List provider

Provides lists as configured in config.json

 """
import logging
from typing import List
from cachetools import TTLCache, cached

from freqtrade.pairlist.StaticPairList import StaticPairList
from freqtrade import OperationalException
logger = logging.getLogger(__name__)

SORT_VALUES = ['askVolume', 'bidVolume', 'quoteVolume']


class VolumePairList(StaticPairList):

    def __init__(self, freqtrade, config: dict) -> None:
        self._freqtrade = freqtrade
        self._config = config
        self._whitelistconf = self._config.get('pairlist', {}).get('config')
        self._whitelist = self._config['exchange']['pair_whitelist']
        self._blacklist = self._config['exchange'].get('pair_blacklist', [])
        self._number_pairs = self._whitelistconf['number_assets']
        self._sort_key = self._whitelistconf.get('sort_key', 'quoteVolume')

        if not self._freqtrade.exchange.exchange_has('fetchTickers'):
            raise OperationalException(
                'Exchange does not support dynamic whitelist.'
                'Please edit your config and restart the bot'
            )
        if not self.validate_keys(self._sort_key):
            raise OperationalException(
                f'key {self._sort_key} not in {SORT_VALUES}')
        # self.refresh_whitelist()

    def validate_keys(self, key):
        return key in SORT_VALUES

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        -> Please overwrite in subclasses
        """
        return f"{self.name} - top {self._whitelistconf['number_assets']} volume pairs."

    def refresh_whitelist(self) -> None:
        """
        Refreshes whitelist and assigns it to self._whitelist
        """
        # Generate dynamic whitelist
        pairs = self._gen_pair_whitelist(self._config['stake_currency'], self._sort_key)
        # Validate whitelist to only have active market pairs
        self._whitelist = self._validate_whitelist(pairs)[:self._number_pairs]

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
                   if len(k.split('/')) == 2 and k.split('/')[1] == base_currency]

        sorted_tickers = sorted(tickers, reverse=True, key=lambda t: t[key])
        pairs = [s['symbol'] for s in sorted_tickers]
        return pairs

    def _validate_whitelist(self, whitelist: List[str]) -> List[str]:
        """
        Check available markets and remove pair from whitelist if necessary
        :param whitelist: the sorted list (based on BaseVolume) of pairs the user might want to
        trade
        :return: the list of pairs the user wants to trade without the one unavailable or
        black_listed
        """
        sanitized_whitelist = whitelist
        markets = self._freqtrade.exchange.get_markets()

        # Filter to markets in stake currency
        markets = [m for m in markets if m['quote'] == self._config['stake_currency']]
        known_pairs = set()

        for market in markets:
            pair = market['symbol']
            # pair is not int the generated dynamic market, or in the blacklist ... ignore it
            if pair not in whitelist or pair in self.blacklist:
                continue
            # else the pair is valid
            known_pairs.add(pair)
            # Market is not active
            if not market['active']:
                sanitized_whitelist.remove(pair)
                logger.info(
                    'Ignoring %s from whitelist. Market is not active.',
                    pair
                )

        # We need to remove pairs that are unknown
        return [x for x in sanitized_whitelist if x in known_pairs]
