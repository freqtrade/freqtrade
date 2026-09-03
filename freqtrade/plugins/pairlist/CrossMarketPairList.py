"""Cross Market pair list filter"""

import logging

from freqtrade.constants import PairPrefixes
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import FtTTLCache


logger = logging.getLogger(__name__)


class CrossMarketPairList(IPairList):
    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.BIASED

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._pairs_exist_on: str = self._pairlistconfig.get("pairs_exist_on", "both_markets")
        self._stake_currency: str = self._config["stake_currency"]
        self._target_mode = "spot" if self._config["trading_mode"] == "futures" else "futures"
        self._refresh_period = self._pairlistconfig.get("refresh_period", 1800)
        self._pair_cache: FtTTLCache = FtTTLCache(maxsize=1, ttl=self._refresh_period)

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        pairs_exist_on = self._pairs_exist_on
        msg = f"{self.name} - Pairs that exists on {pairs_exist_on.capitalize()}."
        return msg

    @staticmethod
    def description() -> str:
        return "Filter pairs if they exist or not on another market."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "pairs_exist_on": {
                "type": "option",
                "default": "both_markets",
                "options": ["current_market_only", "both_markets"],
                "description": "Mode of operation",
                "help": "Mode of operation (current_market_only/both_markets)",
            },
            **IPairList.refresh_period_parameter(),
        }

    def get_base_list(self) -> list[str]:
        target_mode = self._target_mode
        spot_only = target_mode == "spot"
        futures_only = target_mode == "futures"
        bases = [
            v.get("base", "")
            for _, v in self._exchange.get_markets(
                quote_currencies=[self._stake_currency],
                tradable_only=False,
                active_only=True,
                spot_only=spot_only,
                futures_only=futures_only,
            ).items()
        ]
        return bases

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: List of pairs
        """
        # Generate dynamic whitelist
        # Must always run if this pairlist is the first in the list.
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
                )
            ]

            _pairlist = self.verify_blacklist(_pairlist, logger.info)

            pairlist = self.filter_pairlist(_pairlist, tickers)
            self._pair_cache["pairlist"] = pairlist.copy()

        return pairlist

    def filter_pairlist(self, pairlist: list[str], tickers: Tickers) -> list[str]:
        bases = self.get_base_list()
        pairs_exist_on = self._pairs_exist_on
        is_whitelist_mode = pairs_exist_on == "both_markets"
        whitelisted_pairlist: list[str] = []
        filtered_pairlist = pairlist.copy()

        for pair in pairlist:
            base = self._exchange.get_pair_base_currency(pair)
            if not base:
                self.log_once(
                    f"Unable to get base currency for pair {pair}, skipping it.", logger.warning
                )
                filtered_pairlist.remove(pair)
                continue
            found_in_bases = base in bases
            if not found_in_bases:
                for prefix in PairPrefixes:
                    # Check in case of PEPE needs to be changed into 1000PEPE for example
                    test_prefix = f"{prefix}{base}"
                    found_in_bases = test_prefix in bases
                    if found_in_bases:
                        break

                    # Avoid false positive since there are KAVA and AVA pairs, which aren't related
                    # Check in case of 1000PEPE needs to be changed into PEPE for example
                    if prefix != "K" and base.startswith(prefix):
                        temp_base = base.removeprefix(prefix)
                        found_in_bases = temp_base in bases
                        if found_in_bases:
                            break
            if found_in_bases:
                whitelisted_pairlist.append(pair)
                filtered_pairlist.remove(pair)

        return whitelisted_pairlist if is_whitelist_mode else filtered_pairlist
