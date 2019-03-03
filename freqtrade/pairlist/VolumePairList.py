"""
Volume PairList provider

Provides lists as configured in config.json

 """
import logging
from typing import List
from cachetools import TTLCache, cached

from freqtrade.pairlist.IPairList import IPairList
from freqtrade import OperationalException
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
        self._precision_filter = self._whitelistconf.get('precision_filter', False)

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
            self._config['stake_currency'], self._sort_key)[:self._number_pairs]
        logger.info(f"Searching pairs: {self._whitelist}")

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
        # Validate whitelist to only have active market pairs
        valid_pairs = self._validate_whitelist([s['symbol'] for s in sorted_tickers])
        valid_tickers = [t for t in sorted_tickers if t["symbol"] in valid_pairs]

        if self._freqtrade.strategy.stoploss is not None and self._precision_filter:

            stop_prices = [self._freqtrade.get_target_bid(t["symbol"], t)
                           * (1 - abs(self._freqtrade.strategy.stoploss)) for t in valid_tickers]
            rates = [sp * 0.99 for sp in stop_prices]
            logger.debug("\n".join([f"{sp} : {r}" for sp, r in zip(stop_prices[:10], rates[:10])]))
            for i, t in enumerate(valid_tickers):
                sp = self._freqtrade.exchange.symbol_price_prec(t["symbol"], stop_prices[i])
                r = self._freqtrade.exchange.symbol_price_prec(t["symbol"], rates[i])
                logger.debug(f"{t['symbol']} - {sp} : {r}")
                if sp <= r:
                    logger.info(f"Removed {t['symbol']} from whitelist, "
                                f"because stop price {sp} would be <= stop limit {r}")
                    valid_tickers.remove(t)

        pairs = [s['symbol'] for s in valid_tickers]
        return pairs
