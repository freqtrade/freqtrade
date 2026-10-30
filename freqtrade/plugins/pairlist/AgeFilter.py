"""
Minimum age (days listed) pair list filter
"""

import logging
from datetime import datetime, timedelta

from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.misc import plural
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import PeriodicCache, dt_floor_day, dt_now, dt_ts


logger = logging.getLogger(__name__)


class AgeFilter(IPairList):
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Date of the first daily candle per pair - the listing date for young pairs,
        # the start of the fetched window for older pairs.
        self._listing_dates: dict[str, datetime] = {}
        # Pairs without any completed daily candle - recheck on the next UTC day
        self._symbols_no_data = PeriodicCache(maxsize=1000, ttl=86_400)

        self._min_days_listed = self._pairlistconfig.get("min_days_listed", 10)
        self._max_days_listed = self._pairlistconfig.get("max_days_listed")
        self._def_candletype = self._config["candle_type_def"]

        candle_limit = self._exchange.ohlcv_candle_limit("1d", self._def_candletype)
        if self._min_days_listed < 1:
            raise OperationalException("AgeFilter requires min_days_listed to be >= 1")
        if self._min_days_listed > candle_limit:
            raise OperationalException(
                "AgeFilter requires min_days_listed to not exceed "
                "exchange max request size "
                f"({candle_limit})"
            )
        if self._max_days_listed and self._max_days_listed <= self._min_days_listed:
            raise OperationalException("AgeFilter max_days_listed <= min_days_listed not permitted")
        if self._max_days_listed and self._max_days_listed > candle_limit:
            raise OperationalException(
                "AgeFilter requires max_days_listed to not exceed "
                "exchange max request size "
                f"({candle_limit})"
            )

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return (
            f"{self.name} - Filtering pairs with age less than "
            f"{self._min_days_listed} {plural(self._min_days_listed, 'day')}"
        ) + (
            (f" or more than {self._max_days_listed} {plural(self._max_days_listed, 'day')}")
            if self._max_days_listed
            else ""
        )

    @staticmethod
    def description() -> str:
        return "Filter pairs by age (days listed)."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "min_days_listed": {
                "type": "number",
                "default": 10,
                "description": "Minimum Days Listed",
                "help": "Minimum number of days a pair must have been listed on the exchange.",
            },
            "max_days_listed": {
                "type": "number",
                "default": None,
                "description": "Maximum Days Listed",
                "help": "Maximum number of days a pair must have been listed on the exchange.",
            },
        }

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new allowlist
        """
        needed_pairs: ListPairsWithTimeframes = [
            (p, "1d", self._def_candletype)
            for p in pairlist
            if p not in self._listing_dates and p not in self._symbols_no_data
        ]
        today = dt_floor_day(dt_now())
        if needed_pairs:
            # Fetch a few days beyond the limit
            since_days = (self._max_days_listed or self._min_days_listed) + 3
            since_ms = dt_ts(today - timedelta(days=since_days))
            candles = self._exchange.refresh_latest_ohlcv(
                needed_pairs, since_ms=since_ms, cache=False
            )
            for pair, _, _ in needed_pairs:
                daily_candles = candles.get((pair, "1d", self._def_candletype), None)
                if daily_candles is None:
                    # Fetch failed - retry on the next iteration
                    continue
                if daily_candles.empty:
                    self._symbols_no_data[pair] = dt_ts()
                else:
                    self._listing_dates[pair] = daily_candles.iloc[0]["date"]

        if self._enabled:
            pairlist = [p for p in pairlist if self._validate_pair_loc(p, today)]
        self.log_once(f"Validated {len(pairlist)} pairs.", logger.info)
        return pairlist

    def _validate_pair_loc(self, pair: str, today: datetime) -> bool:
        """
        Validate age for the ticker
        :param pair: Pair that's currently validated
        :param today: Start of the current UTC day
        :return: True if the pair can stay, false if it should be removed
        """
        listing_date = self._listing_dates.get(pair)
        if listing_date is None:
            # Failed fetches are logged by the exchange
            if pair in self._symbols_no_data:
                self.log_once(
                    f"Removed {pair} from whitelist, because no daily candles are available yet",
                    logger.info,
                )
            return False

        age = (today - listing_date).days
        if age < self._min_days_listed:
            reason = (
                f"age {age} is less than {self._min_days_listed} "
                f"{plural(self._min_days_listed, 'day')}"
            )
        elif self._max_days_listed and age > self._max_days_listed:
            reason = (
                f"it has been listed for more than {self._max_days_listed} "
                f"{plural(self._max_days_listed, 'day')}"
            )
        else:
            return True

        self.log_once(f"Removed {pair} from whitelist, because {reason}", logger.info)
        return False
