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
        self._precision_filter = self._whitelistconf.get('precision_filter', True)
        self._low_price_percent_filter = self._whitelistconf.get('low_price_percent_filter', None)
        print(self._whitelistconf)

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

    def _validate_precision_filter(self, ticker: dict, stoploss: float) -> bool:
        """
        Check if pair has enough room to add a stoploss to avoid "unsellable" buys of very
        low value pairs.
        :param ticker: ticker dict as returned from ccxt.load_markets()
        :param stoploss: stoploss value as set in the configuration
                        (already cleaned to be guaranteed negative)
        :return: True if the pair can stay, false if it should be removed
        """
        stop_price = (self._freqtrade.get_target_bid(ticker["symbol"], ticker) * stoploss)
        # Adjust stop-prices to precision
        sp = self._freqtrade.exchange.symbol_price_prec(ticker["symbol"], stop_price)
        stop_gap_price = self._freqtrade.exchange.symbol_price_prec(ticker["symbol"],
                                                                    stop_price * 0.99)
        logger.debug(f"{ticker['symbol']} - {sp} : {stop_gap_price}")
        if sp <= stop_gap_price:
            logger.info(f"Removed {ticker['symbol']} from whitelist, "
                        f"because stop price {sp} would be <= stop limit {stop_gap_price}")
            return False
        return True

    def _validate_precision_filter_lowprice(self, ticker) -> bool:
        """
        Check if if one price-step is > than a certain barrier.
        :param ticker: ticker dict as returned from ccxt.load_markets()
        :param precision: Precision
        :return: True if the pair can stay, false if it should be removed
        """
        precision = self._freqtrade.exchange.markets[ticker['symbol']]['precision']['price']

        compare = ticker['last'] + 1 / pow(10, precision)
        changeperc = (compare - ticker['last']) / ticker['last']
        if changeperc > self._low_price_percent_filter:
            logger.info(f"Removed {ticker['symbol']} from whitelist, "
                        f"because 1 unit is {changeperc * 100:.3f}%")
            return False
        return True

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
        valid_pairs = self._validate_whitelist([s['symbol'] for s in sorted_tickers])
        valid_tickers = [t for t in sorted_tickers if t["symbol"] in valid_pairs]

        stoploss = None
        if self._freqtrade.strategy.stoploss is not None:
            # Precalculate sanitized stoploss value to avoid recalculation for every pair
            stoploss = 1 - abs(self._freqtrade.strategy.stoploss)

        for t in valid_tickers:
            # Filter out assets which would not allow setting a stoploss
            if (stoploss and self._precision_filter
                and not self._validate_precision_filter(t, stoploss)):
                valid_tickers.remove(t)
                continue
            if self._low_price_percent_filter and not self._validate_precision_filter_lowprice(t,):
                valid_tickers.remove(t)
                continue

        pairs = [s['symbol'] for s in valid_tickers]
        logger.info(f"Searching pairs: {pairs[:self._number_pairs]}")

        return pairs
