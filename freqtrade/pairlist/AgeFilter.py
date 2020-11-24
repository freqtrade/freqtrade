"""
Minimum age (days listed) pair list filter
"""
import logging
from typing import Any, Dict

import arrow

from freqtrade.exceptions import OperationalException
from freqtrade.misc import plural
from freqtrade.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class AgeFilter(IPairList):

    # Checked symbols cache (dictionary of ticker symbol => timestamp)
    _symbolsChecked: Dict[str, int] = {}

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._min_days_listed = pairlistconfig.get('min_days_listed', 10)

        if self._min_days_listed < 1:
            raise OperationalException("AgeFilter requires min_days_listed to be >= 1")
        if self._min_days_listed > exchange.ohlcv_candle_limit:
            raise OperationalException("AgeFilter requires min_days_listed to not exceed "
                                       "exchange max request size "
                                       f"({exchange.ohlcv_candle_limit})")

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return True

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return (f"{self.name} - Filtering pairs with age less than "
                f"{self._min_days_listed} {plural(self._min_days_listed, 'day')}.")

    def _validate_pair(self, ticker: dict) -> bool:
        """
        Validate age for the ticker
        :param ticker: ticker dict as returned from ccxt.load_markets()
        :return: True if the pair can stay, False if it should be removed
        """

        # Check symbol in cache
        if ticker['symbol'] in self._symbolsChecked:
            return True

        since_ms = int(arrow.utcnow()
                       .floor('day')
                       .shift(days=-self._min_days_listed)
                       .float_timestamp) * 1000

        daily_candles = self._exchange.get_historic_ohlcv(pair=ticker['symbol'],
                                                          timeframe='1d',
                                                          since_ms=since_ms)

        if daily_candles is not None:
            if len(daily_candles) > self._min_days_listed:
                # We have fetched at least the minimum required number of daily candles
                # Add to cache, store the time we last checked this symbol
                self._symbolsChecked[ticker['symbol']] = int(arrow.utcnow().float_timestamp) * 1000
                return True
            else:
                self.log_on_refresh(logger.info, f"Removed {ticker['symbol']} from whitelist, "
                                                 f"because age {len(daily_candles)} is less than "
                                                 f"{self._min_days_listed} "
                                                 f"{plural(self._min_days_listed, 'day')}")
                return False
        return False
