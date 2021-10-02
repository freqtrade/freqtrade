"""
Volume PairList provider

Provides dynamic pair list based on trade volumes
"""
import logging
from functools import partial
from typing import Any, Dict, List

import arrow
from cachetools.ttl import TTLCache

from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.misc import format_ms_time
from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


SORT_VALUES = ['quoteVolume']


class VolumePairList(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        if 'number_assets' not in self._pairlistconfig:
            raise OperationalException(
                '`number_assets` not specified. Please check your configuration '
                'for "pairlist.config.number_assets"')

        self._stake_currency = config['stake_currency']
        self._number_pairs = self._pairlistconfig['number_assets']
        self._sort_key = self._pairlistconfig.get('sort_key', 'quoteVolume')
        self._min_value = self._pairlistconfig.get('min_value', 0)
        self._refresh_period = self._pairlistconfig.get('refresh_period', 1800)
        self._pair_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._lookback_days = self._pairlistconfig.get('lookback_days', 0)
        self._lookback_timeframe = self._pairlistconfig.get('lookback_timeframe', '1d')
        self._lookback_period = self._pairlistconfig.get('lookback_period', 0)

        if (self._lookback_days > 0) & (self._lookback_period > 0):
            raise OperationalException(
                'Ambigous configuration: lookback_days and lookback_period both set in pairlist '
                'config. Please set lookback_days only or lookback_period and lookback_timeframe '
                'and restart the bot.'
            )

        # overwrite lookback timeframe and days when lookback_days is set
        if self._lookback_days > 0:
            self._lookback_timeframe = '1d'
            self._lookback_period = self._lookback_days

        # get timeframe in minutes and seconds
        self._tf_in_min = timeframe_to_minutes(self._lookback_timeframe)
        self._tf_in_sec = self._tf_in_min * 60

        # wether to use range lookback or not
        self._use_range = (self._tf_in_min > 0) & (self._lookback_period > 0)

        if self._use_range & (self._refresh_period < self._tf_in_sec):
            raise OperationalException(
                f'Refresh period of {self._refresh_period} seconds is smaller than one '
                f'timeframe of {self._lookback_timeframe}. Please adjust refresh_period '
                f'to at least {self._tf_in_sec} and restart the bot.'
            )

        if not self._exchange.exchange_has('fetchTickers'):
            raise OperationalException(
                'Exchange does not support dynamic whitelist. '
                'Please edit your config and restart the bot.'
            )

        if not self._validate_keys(self._sort_key):
            raise OperationalException(
                f'key {self._sort_key} not in {SORT_VALUES}')

        if self._lookback_period < 0:
            raise OperationalException("VolumeFilter requires lookback_period to be >= 0")
        if self._lookback_period > exchange.ohlcv_candle_limit(self._lookback_timeframe):
            raise OperationalException("VolumeFilter requires lookback_period to not "
                                       "exceed exchange max request size "
                                       f"({exchange.ohlcv_candle_limit(self._lookback_timeframe)})")

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
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

    def gen_pairlist(self, tickers: Dict) -> List[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: List of pairs
        """
        # Generate dynamic whitelist
        # Must always run if this pairlist is not the first in the list.
        pairlist = self._pair_cache.get('pairlist')
        if pairlist:
            # Item found - no refresh necessary
            return pairlist.copy()
        else:
            # Use fresh pairlist
            # Check if pair quote currency equals to the stake currency.
            filtered_tickers = [
                v for k, v in tickers.items()
                if (self._exchange.get_pair_quote_currency(k) == self._stake_currency
                    and (self._use_range or v[self._sort_key] is not None))]
            pairlist = [s['symbol'] for s in filtered_tickers]

            pairlist = self.filter_pairlist(pairlist, tickers)
            self._pair_cache['pairlist'] = pairlist.copy()

        return pairlist

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        # Use the incoming pairlist.
        filtered_tickers = [v for k, v in tickers.items() if k in pairlist]

        # get lookback period in ms, for exchange ohlcv fetch
        if self._use_range:
            since_ms = int(arrow.utcnow()
                           .floor('minute')
                           .shift(minutes=-(self._lookback_period * self._tf_in_min)
                                  - self._tf_in_min)
                           .int_timestamp) * 1000

            to_ms = int(arrow.utcnow()
                        .floor('minute')
                        .shift(minutes=-self._tf_in_min)
                        .int_timestamp) * 1000

            # todo: utc date output for starting date
            self.log_once(f"Using volume range of {self._lookback_period} candles, timeframe: "
                          f"{self._lookback_timeframe}, starting from {format_ms_time(since_ms)} "
                          f"till {format_ms_time(to_ms)}", logger.info)
            needed_pairs = [
                (p, self._lookback_timeframe) for p in
                [
                    s['symbol'] for s in filtered_tickers
                ] if p not in self._pair_cache
            ]

            # Get all candles
            candles = {}
            if needed_pairs:
                candles = self._exchange.refresh_latest_ohlcv(
                    needed_pairs, since_ms=since_ms, cache=False
                )
            for i, p in enumerate(filtered_tickers):
                pair_candles = candles[
                    (p['symbol'], self._lookback_timeframe)
                ] if (p['symbol'], self._lookback_timeframe) in candles else None
                # in case of candle data calculate typical price and quoteVolume for candle
                if pair_candles is not None and not pair_candles.empty:
                    pair_candles['typical_price'] = (pair_candles['high'] + pair_candles['low']
                                                     + pair_candles['close']) / 3
                    pair_candles['quoteVolume'] = (
                        pair_candles['volume'] * pair_candles['typical_price']
                    )

                    # ensure that a rolling sum over the lookback_period is built
                    # if pair_candles contains more candles than lookback_period
                    quoteVolume = (pair_candles['quoteVolume']
                                   .rolling(self._lookback_period)
                                   .sum()
                                   .iloc[-1])

                    # replace quoteVolume with range quoteVolume sum calculated above
                    filtered_tickers[i]['quoteVolume'] = quoteVolume
                else:
                    filtered_tickers[i]['quoteVolume'] = 0

        if self._min_value > 0:
            filtered_tickers = [
                v for v in filtered_tickers if v[self._sort_key] > self._min_value]

        sorted_tickers = sorted(filtered_tickers, reverse=True, key=lambda t: t[self._sort_key])

        # Validate whitelist to only have active market pairs
        pairs = self._whitelist_for_active_markets([s['symbol'] for s in sorted_tickers])
        pairs = self.verify_blacklist(pairs, partial(self.log_once, logmethod=logger.info))
        # Limit pairlist to the requested number of pairs
        pairs = pairs[:self._number_pairs]

        self.log_once(f"Searching {self._number_pairs} pairs: {pairs}", logger.info)

        return pairs
