"""
Minimum age (days listed) pair list filter
"""

import logging
from copy import deepcopy
from datetime import timedelta
from typing import Optional

from pandas import DataFrame

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

        # Checked symbols cache (dictionary of ticker symbol => timestamp)
        self._symbolsChecked: dict[str, int] = {}
        self._symbolsCheckFailed = PeriodicCache(maxsize=1000, ttl=86_400)

        self._min_days_listed = self._pairlistconfig.get("min_days_listed", 10)
        self._max_days_listed = self._pairlistconfig.get("max_days_listed")

        candle_limit = self._exchange.ohlcv_candle_limit("1d", self._config["candle_type_def"])
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

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return False

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return (
            f"{self.name} - Filtering pairs with age less than "
            f"{self._min_days_listed} {plural(self._min_days_listed, 'day')}"
        ) + (
            (" or more than {self._max_days_listed} {plural(self._max_days_listed, 'day')}")
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
            (p, "1d", self._config["candle_type_def"])
            for p in pairlist
            if p not in self._symbolsChecked and p not in self._symbolsCheckFailed
        ]
        if not needed_pairs:
            # Remove pairs that have been removed before
            return [p for p in pairlist if p not in self._symbolsCheckFailed]

        since_days = (
            -(self._max_days_listed if self._max_days_listed else self._min_days_listed) - 1
        )
        since_ms = dt_ts(dt_floor_day(dt_now()) + timedelta(days=since_days))
        candles = self._exchange.refresh_latest_ohlcv(needed_pairs, since_ms=since_ms, cache=False)
        if self._enabled:
            for p in deepcopy(pairlist):
                daily_candles = (
                    candles[(p, "1d", self._config["candle_type_def"])]
                    if (p, "1d", self._config["candle_type_def"]) in candles
                    else None
                )
                if not self._validate_pair_loc(p, daily_candles):
                    pairlist.remove(p)
        self.log_once(f"Validated {len(pairlist)} pairs.", logger.info)
        return pairlist

    def _validate_pair_loc(self, pair: str, daily_candles: Optional[DataFrame]) -> bool:
        """
        Validate age for the ticker
        :param pair: Pair that's currently validated
        :param daily_candles: Downloaded daily candles
        :return: True if the pair can stay, false if it should be removed
        """
        # Check symbol in cache
        if pair in self._symbolsChecked:
            return True

        if daily_candles is not None:
            if len(daily_candles) >= self._min_days_listed and (
                not self._max_days_listed or len(daily_candles) <= self._max_days_listed
            ):
                # We have fetched at least the minimum required number of daily candles
                # Add to cache, store the time we last checked this symbol
                self._symbolsChecked[pair] = dt_ts()
                return True
            else:
                self.log_once(
                    (
                        f"Removed {pair} from whitelist, because age "
                        f"{len(daily_candles)} is less than {self._min_days_listed} "
                        f"{plural(self._min_days_listed, 'day')}"
                    )
                    + (
                        (
                            " or more than "
                            f"{self._max_days_listed} {plural(self._max_days_listed, 'day')}"
                        )
                        if self._max_days_listed
                        else ""
                    ),
                    logger.info,
                )
                self._symbolsCheckFailed[pair] = dt_ts()
                return False
        return False
