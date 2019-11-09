"""
Volume PairList provider

Provides lists as configured in config.json

 """
import logging
from datetime import datetime
from typing import Dict, List

from freqtrade import OperationalException
from freqtrade.pairlist.IPairList import IPairList

logger = logging.getLogger(__name__)

SORT_VALUES = ['askVolume', 'bidVolume', 'quoteVolume']


class VolumePairList(IPairList):

    def __init__(self, exchange, pairlistmanager, config, pairlistconfig: dict) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig)

        if 'number_assets' not in self._pairlistconfig:
            raise OperationalException(
                f'`number_assets` not specified. Please check your configuration '
                'for "pairlist.config.number_assets"')
        self._number_pairs = self._pairlistconfig['number_assets']
        self._sort_key = self._pairlistconfig.get('sort_key', 'quoteVolume')
        self._ttl = self._pairlistconfig.get('ttl', 1800)

        if not self._exchange.exchange_has('fetchTickers'):
            raise OperationalException(
                'Exchange does not support dynamic whitelist.'
                'Please edit your config and restart the bot'
            )
        if not self._validate_keys(self._sort_key):
            raise OperationalException(
                f'key {self._sort_key} not in {SORT_VALUES}')
        self._last_refresh = 0

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

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        # Generate dynamic whitelist
        if self._last_refresh + self._ttl < datetime.now().timestamp():
            self._last_refresh = datetime.now().timestamp()
            return self._gen_pair_whitelist(pairlist,
                                            tickers,
                                            self._config['stake_currency'],
                                            self._sort_key,
                                            )
        else:
            return pairlist

    def _gen_pair_whitelist(self, pairlist, tickers, base_currency: str, key: str) -> List[str]:
        """
        Updates the whitelist with with a dynamically generated list
        :param base_currency: base currency as str
        :param key: sort key (defaults to 'quoteVolume')
        :param tickers: Tickers (from exchange.get_tickers()).
        :return: List of pairs
        """

        # check length so that we make sure that '/' is actually in the string
        tickers = [v for k, v in tickers.items()
                   if (len(k.split('/')) == 2 and k.split('/')[1] == base_currency
                       and v[key] is not None)]
        sorted_tickers = sorted(tickers, reverse=True, key=lambda t: t[key])
        # Validate whitelist to only have active market pairs
        pairs = self._whitelist_for_active_markets([s['symbol'] for s in sorted_tickers])
        pairs = self._verify_blacklist(pairs)
        # Limit to X number of pairs
        pairs = pairs[:self._number_pairs]
        logger.info(f"Searching {self._number_pairs} pairs: {pairs}")

        return pairs
