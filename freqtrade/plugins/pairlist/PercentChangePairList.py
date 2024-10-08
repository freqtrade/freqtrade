"""
Percent Change PairList provider

Provides dynamic pair list based on trade change
sorted based on percentage change in price over a
defined period or as coming from ticker
"""

import logging
from datetime import timedelta
from typing import Any, Optional

from cachetools import TTLCache
from pandas import DataFrame

from freqtrade.constants import ListPairsWithTimeframes, PairWithTimeframe
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.exchange.exchange_types import Ticker, Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import dt_now, format_ms_time


logger = logging.getLogger(__name__)


class PercentChangePairList(IPairList):
    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.NO

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if "number_assets" not in self._pairlistconfig:
            raise OperationalException(
                "`number_assets` not specified. Please check your configuration "
                'for "pairlist.config.number_assets"'
            )

        self._stake_currency = self._config["stake_currency"]
        self._number_pairs = self._pairlistconfig["number_assets"]
        self._min_value = self._pairlistconfig.get("min_value", None)
        self._max_value = self._pairlistconfig.get("max_value", None)
        self._refresh_period = self._pairlistconfig.get("refresh_period", 1800)
        self._pair_cache: TTLCache = TTLCache(maxsize=1, ttl=self._refresh_period)
        self._lookback_days = self._pairlistconfig.get("lookback_days", 0)
        self._lookback_timeframe = self._pairlistconfig.get("lookback_timeframe", "1d")
        self._lookback_period = self._pairlistconfig.get("lookback_period", 0)
        self._sort_direction: Optional[str] = self._pairlistconfig.get("sort_direction", "desc")
        self._def_candletype = self._config["candle_type_def"]

        if (self._lookback_days > 0) & (self._lookback_period > 0):
            raise OperationalException(
                "Ambiguous configuration: lookback_days and lookback_period both set in pairlist "
                "config. Please set lookback_days only or lookback_period and lookback_timeframe "
                "and restart the bot."
            )

        # overwrite lookback timeframe and days when lookback_days is set
        if self._lookback_days > 0:
            self._lookback_timeframe = "1d"
            self._lookback_period = self._lookback_days

        # get timeframe in minutes and seconds
        self._tf_in_min = timeframe_to_minutes(self._lookback_timeframe)
        _tf_in_sec = self._tf_in_min * 60

        # whether to use range lookback or not
        self._use_range = (self._tf_in_min > 0) & (self._lookback_period > 0)

        if self._use_range & (self._refresh_period < _tf_in_sec):
            raise OperationalException(
                f"Refresh period of {self._refresh_period} seconds is smaller than one "
                f"timeframe of {self._lookback_timeframe}. Please adjust refresh_period "
                f"to at least {_tf_in_sec} and restart the bot."
            )

        if not self._use_range and not (
            self._exchange.exchange_has("fetchTickers")
            and self._exchange.get_option("tickers_have_percentage")
        ):
            raise OperationalException(
                "Exchange does not support dynamic whitelist in this configuration. "
                "Please edit your config and either remove PercentChangePairList, "
                "or switch to using candles. and restart the bot."
            )

        candle_limit = self._exchange.ohlcv_candle_limit(
            self._lookback_timeframe, self._config["candle_type_def"]
        )

        if self._lookback_period > candle_limit:
            raise OperationalException(
                "ChangeFilter requires lookback_period to not "
                f"exceed exchange max request size ({candle_limit})"
            )

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return not self._use_range

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return f"{self.name} - top {self._pairlistconfig['number_assets']} percent change pairs."

    @staticmethod
    def description() -> str:
        return "Provides dynamic pair list based on percentage change."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "number_assets": {
                "type": "number",
                "default": 30,
                "description": "Number of assets",
                "help": "Number of assets to use from the pairlist",
            },
            "min_value": {
                "type": "number",
                "default": None,
                "description": "Minimum value",
                "help": "Minimum value to use for filtering the pairlist.",
            },
            "max_value": {
                "type": "number",
                "default": None,
                "description": "Maximum value",
                "help": "Maximum value to use for filtering the pairlist.",
            },
            "sort_direction": {
                "type": "option",
                "default": "desc",
                "options": ["", "asc", "desc"],
                "description": "Sort pairlist",
                "help": "Sort Pairlist ascending or descending by rate of change.",
            },
            **IPairList.refresh_period_parameter(),
            "lookback_days": {
                "type": "number",
                "default": 0,
                "description": "Lookback Days",
                "help": "Number of days to look back at.",
            },
            "lookback_timeframe": {
                "type": "string",
                "default": "1d",
                "description": "Lookback Timeframe",
                "help": "Timeframe to use for lookback.",
            },
            "lookback_period": {
                "type": "number",
                "default": 0,
                "description": "Lookback Period",
                "help": "Number of periods to look back at.",
            },
        }

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: List of pairs
        """
        pairlist = self._pair_cache.get("pairlist")
        if pairlist:
            # Item found - no refresh necessary
            return pairlist.copy()
        else:
            # Use fresh pairlist
            # Check if pair quote currency equals to the stake currency.
            _pairlist = [
                k
                for k in self._exchange.get_markets(
                    quote_currencies=[self._stake_currency], tradable_only=True, active_only=True
                ).keys()
            ]

            # No point in testing for blacklisted pairs...
            _pairlist = self.verify_blacklist(_pairlist, logger.info)
            if not self._use_range:
                filtered_tickers = [
                    v
                    for k, v in tickers.items()
                    if (
                        self._exchange.get_pair_quote_currency(k) == self._stake_currency
                        and (self._use_range or v.get("percentage") is not None)
                        and v["symbol"] in _pairlist
                    )
                ]
                pairlist = [s["symbol"] for s in filtered_tickers]
            else:
                pairlist = _pairlist

            pairlist = self.filter_pairlist(pairlist, tickers)
            self._pair_cache["pairlist"] = pairlist.copy()

        return pairlist

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new whitelist
        """
        filtered_tickers: list[dict[str, Any]] = [{"symbol": k} for k in pairlist]
        if self._use_range:
            # calculating using lookback_period
            self.fetch_percent_change_from_lookback_period(filtered_tickers)
        else:
            # Fetching 24h change by default from supported exchange tickers
            self.fetch_percent_change_from_tickers(filtered_tickers, tickers)

        if self._min_value is not None:
            filtered_tickers = [v for v in filtered_tickers if v["percentage"] > self._min_value]
        if self._max_value is not None:
            filtered_tickers = [v for v in filtered_tickers if v["percentage"] < self._max_value]

        sorted_tickers = sorted(
            filtered_tickers,
            reverse=self._sort_direction == "desc",
            key=lambda t: t["percentage"],
        )

        # Validate whitelist to only have active market pairs
        pairs = self._whitelist_for_active_markets([s["symbol"] for s in sorted_tickers])
        pairs = self.verify_blacklist(pairs, logmethod=logger.info)
        # Limit pairlist to the requested number of pairs
        pairs = pairs[: self._number_pairs]

        return pairs

    def fetch_candles_for_lookback_period(
        self, filtered_tickers: list[dict[str, str]]
    ) -> dict[PairWithTimeframe, DataFrame]:
        since_ms = (
            int(
                timeframe_to_prev_date(
                    self._lookback_timeframe,
                    dt_now()
                    + timedelta(
                        minutes=-(self._lookback_period * self._tf_in_min) - self._tf_in_min
                    ),
                ).timestamp()
            )
            * 1000
        )
        to_ms = (
            int(
                timeframe_to_prev_date(
                    self._lookback_timeframe, dt_now() - timedelta(minutes=self._tf_in_min)
                ).timestamp()
            )
            * 1000
        )
        # todo: utc date output for starting date
        self.log_once(
            f"Using change range of {self._lookback_period} candles, timeframe: "
            f"{self._lookback_timeframe}, starting from {format_ms_time(since_ms)} "
            f"till {format_ms_time(to_ms)}",
            logger.info,
        )
        needed_pairs: ListPairsWithTimeframes = [
            (p, self._lookback_timeframe, self._def_candletype)
            for p in [s["symbol"] for s in filtered_tickers]
            if p not in self._pair_cache
        ]
        candles = self._exchange.refresh_ohlcv_with_cache(needed_pairs, since_ms)
        return candles

    def fetch_percent_change_from_lookback_period(self, filtered_tickers: list[dict[str, Any]]):
        # get lookback period in ms, for exchange ohlcv fetch
        candles = self.fetch_candles_for_lookback_period(filtered_tickers)

        for i, p in enumerate(filtered_tickers):
            pair_candles = (
                candles[(p["symbol"], self._lookback_timeframe, self._def_candletype)]
                if (p["symbol"], self._lookback_timeframe, self._def_candletype) in candles
                else None
            )

            # in case of candle data calculate typical price and change for candle
            if pair_candles is not None and not pair_candles.empty:
                current_close = pair_candles["close"].iloc[-1]
                previous_close = pair_candles["close"].shift(self._lookback_period).iloc[-1]
                pct_change = (
                    ((current_close - previous_close) / previous_close) if previous_close > 0 else 0
                )

                # replace change with a range change sum calculated above
                filtered_tickers[i]["percentage"] = pct_change
            else:
                filtered_tickers[i]["percentage"] = 0

    def fetch_percent_change_from_tickers(self, filtered_tickers: list[dict[str, Any]], tickers):
        for i, p in enumerate(filtered_tickers):
            # Filter out assets
            if not self._validate_pair(
                p["symbol"], tickers[p["symbol"]] if p["symbol"] in tickers else None
            ):
                filtered_tickers.remove(p)
            else:
                filtered_tickers[i]["percentage"] = tickers[p["symbol"]]["percentage"]

    def _validate_pair(self, pair: str, ticker: Optional[Ticker]) -> bool:
        """
        Check if one price-step (pip) is > than a certain barrier.
        :param pair: Pair that's currently validated
        :param ticker: ticker dict as returned from ccxt.fetch_ticker
        :return: True if the pair can stay, false if it should be removed
        """
        if not ticker or "percentage" not in ticker or ticker["percentage"] is None:
            self.log_once(
                f"Removed {pair} from whitelist, because "
                "ticker['percentage'] is empty (Usually no trade in the last 24h).",
                logger.info,
            )
            return False

        return True
