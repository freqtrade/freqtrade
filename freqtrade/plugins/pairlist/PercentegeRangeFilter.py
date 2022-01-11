import logging
from copy import deepcopy
from typing import Any, Dict, List
from functools import partial

from freqtrade.plugins.pairlist.IPairList import IPairList

import ccxt
from freqtrade.exceptions import OperationalException

"""
__author__  = kunthar@gmail.com
__license__ = "GPL3"
__version__ = "1.0.1"

Motivation: When you need to pull a definitive price range for pairs between certain 
percentage ratios. Pairs can be selected to whatever percentage  range needed by your strat.

Yes yes, docker image is also built. Please see here for pull
instructions and image details:
https://hub.docker.com/u/kunthar

How to use this filter.
1. Add below code under "pairlists": section in your config file:
Example:

    {"method": "PercentegeRangeFilter",
        "min_price_change": -1,
        "max_price_change": 2
    },

2. Add this filter definition to freqtrade/constants.py 
AVAILABLE_PAIRLISTS = [ 'PercentegeRangeFilter', ...

Todo:
* More exchanges will be implemented in time.

"""

logger = logging.getLogger(__name__)

# Supported Exchanges listed below. You should implement your exchange,
# if it is not exist in the list!
exchange_name_list = ["binance", "kucoin", "gateio"]


class PercentegeRangeFilter(IPairList):
    def __init__(
        self,
        exchange,
        pairlistmanager,
        config: Dict[str, Any],
        pairlistconfig: Dict[str, Any],
        pairlist_pos: int,
    ) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)
        exchange.close()
        self._allow_inactive = self._pairlistconfig.get("allow_inactive", False)

        self._min_price_change = pairlistconfig.get("min_price_change", None)
        self._max_price_change = pairlistconfig.get("max_price_change", None)
        self._exchange_name = config["exchange"]["name"]
        self.stake_currency = config["stake_currency"]

        if self._min_price_change is None:
            raise OperationalException(
                "PercentageRangeFilter requires min_price_change is not NULL. Please check config file!"
            )

        if self._max_price_change is None:
            raise OperationalException(
                "PercentageRangeFilter requires max_price_change is not NULL. Please check config file!"
            )

        if self._exchange_name not in exchange_name_list:
            raise OperationalException(
                "PercentageRangeFilter warning. This exchange is not implemented. Please check config file!"
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
        -> Please overwrite in subclasses
        """

        return f"{self.name} - Filtering pairs between >>> {self._min_price_change} <----> {self._max_price_change}"

    def gen_pairlist(self, tickers: Dict) -> List[str]:

        if self._exchange_name == "binance":
            _exchange = ccxt.binance()
            _percentageHandler = "priceChangePercent"

        if self._exchange_name == "kucoin":
            _exchange = ccxt.kucoin()
            _percentageHandler = "changeRate"

        if self._exchange_name == "gateio":
            _exchange = ccxt.gateio()
            _percentageHandler = "change_percentage"

        # price range from config shouldn't be empty
        minPrice = self._min_price_change
        maxPrice = self._max_price_change

        pair_list = []

        if _exchange.has["fetchTickers"]:

            temp_res = _exchange.fetch_tickers()

            for key in temp_res.keys():
                if str(key).endswith(self.stake_currency):
                    _temp = temp_res.get(key)
                    _tmpPriceChange = float(_temp["info"][_percentageHandler])

                    if _tmpPriceChange > minPrice and _tmpPriceChange < maxPrice:
                        pair_list.append(key)

        return pair_list

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        pairlist_ = deepcopy(pairlist)
        for pair in self._config["exchange"]["pair_whitelist"]:
            if pair not in pairlist_:
                pairlist_.append(pair)

        pairs = self.verify_blacklist(pairlist_)

        # Limit pairlist to the requested number of pairs
        pairs = pairs[: self._number_pairs]

        return pairs
