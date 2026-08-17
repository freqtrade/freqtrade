"""
Rate of change pairlist filter
"""

import logging

from pandas import DataFrame

from freqtrade.constants import ListPairsWithTimeframes
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.misc import plural
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import FtTTLCache


logger = logging.getLogger(__name__)


class RangeStabilityFilter(IPairList):
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        _lookback_days = self._pairlistconfig.get("lookback_days", 0)
        self._lookback_timeframe = self._pairlistconfig.get("lookback_timeframe", "1d")
        _lookback_period: int | None = self._pairlistconfig.get("lookback_period", None)
        self._min_rate_of_change = self._pairlistconfig.get("min_rate_of_change", 0.01)
        self._max_rate_of_change = self._pairlistconfig.get("max_rate_of_change")
        self._refresh_period = self._pairlistconfig.get("refresh_period", 86400)
        self._def_candletype = self._config["candle_type_def"]
        self._sort_direction: str | None = self._pairlistconfig.get("sort_direction", None)

        self._pair_cache: FtTTLCache = FtTTLCache(maxsize=1000, ttl=self._refresh_period)

        if (_lookback_days > 0) and ((_lookback_period or 0) > 0):
            raise OperationalException(
                "Ambiguous configuration: lookback_days and lookback_period both set in pairlist "
                "config. Please set lookback_days only or lookback_period and lookback_timeframe "
                "and restart the bot."
            )
        if "lookback_days" in self._pairlistconfig and _lookback_days < 1:
            raise OperationalException(f"{self.name} requires lookback_days to be >= 1")

        # overwrite lookback timeframe and period when lookback_days is set
        if _lookback_days > 0:
            self._lookback_timeframe = "1d"
            _lookback_period = _lookback_days
        if _lookback_period is None:
            if "lookback_timeframe" in self._pairlistconfig:
                raise OperationalException(
                    f"{self.name} requires lookback_period to be set when using lookback_timeframe."
                )
            logger.warning(
                f"DEPRECATED: Using {self.name} without lookback_days or lookback_period is "
                "deprecated and will result in an error in a future version. "
                "Please set either lookback_days or lookback_period and lookback_timeframe. "
                "Falling back to lookback_days: 10."
            )
            _lookback_period = 10
        self._lookback_period: int = _lookback_period

        candle_limit = self._exchange.ohlcv_candle_limit(
            self._lookback_timeframe, self._def_candletype
        )
        if self._lookback_period < 1:
            raise OperationalException(f"{self.name} requires lookback_period to be >= 1")
        if self._lookback_period > candle_limit:
            raise OperationalException(
                f"{self.name} requires lookback_period to not "
                f"exceed exchange max request size ({candle_limit})"
            )
        if self._sort_direction not in [None, "asc", "desc"]:
            raise OperationalException(
                f"{self.name} requires sort_direction to be "
                "either None (undefined), 'asc' or 'desc'"
            )

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        max_rate_desc = ""
        if self._max_rate_of_change:
            max_rate_desc = f" and above {self._max_rate_of_change}"
        return (
            f"{self.name} - Filtering pairs with rate of change below "
            f"{self._min_rate_of_change}{max_rate_desc} over the "
            f"last {self._lookback_period} {plural(self._lookback_period, 'candle')} of "
            f"{self._lookback_timeframe}."
        )

    @staticmethod
    def description() -> str:
        return "Filters pairs by their rate of change."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "lookback_days": {
                "type": "number",
                "default": 0,
                "description": "Lookback Days",
                "help": "Number of days to look back at. Implies a lookback_timeframe of 1d.",
            },
            "lookback_timeframe": {
                "type": "string",
                "default": "1d",
                "description": "Lookback Timeframe",
                "help": "Timeframe to use for lookback.",
            },
            "lookback_period": {
                "type": "number",
                "default": 10,
                "description": "Lookback Period",
                "help": "Number of periods to look back at.",
            },
            "min_rate_of_change": {
                "type": "number",
                "default": 0.01,
                "description": "Minimum Rate of Change",
                "help": "Minimum rate of change to filter pairs.",
            },
            "max_rate_of_change": {
                "type": "number",
                "default": None,
                "description": "Maximum Rate of Change",
                "help": "Maximum rate of change to filter pairs.",
            },
            "sort_direction": {
                "type": "option",
                "default": None,
                "options": ["", "asc", "desc"],
                "description": "Sort pairlist",
                "help": "Sort Pairlist ascending or descending by rate of change.",
            },
            **IPairList.refresh_period_parameter(),
        }

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        """
        Validate trading range
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new allowlist
        """
        needed_pairs: ListPairsWithTimeframes = [
            (p, self._lookback_timeframe, self._def_candletype)
            for p in pairlist
            if p not in self._pair_cache
        ]

        candles = self._exchange.refresh_ohlcv_with_cache(
            needed_pairs, lookback_period=self._lookback_period
        )

        resulting_pairlist: list[str] = []
        pct_changes: dict[str, float] = {}

        for p in pairlist:
            pair_candles = candles.get((p, self._lookback_timeframe, self._def_candletype), None)

            pct_change = self._calculate_rate_of_change(p, pair_candles)

            if pct_change is not None:
                if self._validate_pair_loc(p, pct_change):
                    resulting_pairlist.append(p)
                    pct_changes[p] = pct_change
            else:
                self.log_once(f"Removed {p} from whitelist, no candles found.", logger.info)

        if self._sort_direction:
            resulting_pairlist = sorted(
                resulting_pairlist,
                key=lambda p: pct_changes[p],
                reverse=self._sort_direction == "desc",
            )
        return resulting_pairlist

    def _calculate_rate_of_change(self, pair: str, pair_candles: DataFrame) -> float | None:
        # Check symbol in cache
        if (pct_change := self._pair_cache.get(pair, None)) is not None:
            return pct_change
        if pair_candles is not None and not pair_candles.empty:
            highest_high = pair_candles["high"].max()
            lowest_low = pair_candles["low"].min()
            pct_change = ((highest_high - lowest_low) / lowest_low) if lowest_low > 0 else 0
            self._pair_cache[pair] = pct_change
            return pct_change
        else:
            return None

    def _validate_pair_loc(self, pair: str, pct_change: float) -> bool:
        """
        Validate trading range
        :param pair: Pair that's currently validated
        :param pct_change: Rate of change
        :return: True if the pair can stay, false if it should be removed
        """

        result = True
        if pct_change < self._min_rate_of_change:
            self.log_once(
                f"Removed {pair} from whitelist, because rate of change "
                f"over {self._lookback_period} {plural(self._lookback_period, 'candle')} of "
                f"{self._lookback_timeframe} is {pct_change:.3f}, "
                f"which is below the threshold of {self._min_rate_of_change}.",
                logger.info,
            )
            result = False
        if self._max_rate_of_change:
            if pct_change > self._max_rate_of_change:
                self.log_once(
                    f"Removed {pair} from whitelist, because rate of change "
                    f"over {self._lookback_period} {plural(self._lookback_period, 'candle')} of "
                    f"{self._lookback_timeframe} is {pct_change:.3f}, "
                    f"which is above the threshold of {self._max_rate_of_change}.",
                    logger.info,
                )
                result = False
        return result
