"""
Market Cap PairList provider

Provides dynamic pair list based on Market Cap
"""

import logging
import math

from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter, SupportsBacktesting
from freqtrade.util import FtTTLCache
from freqtrade.util.coin_gecko import FtCoinGeckoApi


logger = logging.getLogger(__name__)


class MarketCapPairList(IPairList):
    is_pairlist_generator = True
    supports_backtesting = SupportsBacktesting.BIASED

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._mode = self._pairlistconfig.get("mode", "whitelist")

        if (self._mode == "whitelist") and ("number_assets" not in self._pairlistconfig):
            raise OperationalException(
                "`number_assets` not specified. Please check your configuration "
                'for "pairlist.config.number_assets"'
            )

        self._stake_currency = self._config["stake_currency"]
        self._number_assets = self._pairlistconfig.get("number_assets", 30)
        self._max_rank = self._pairlistconfig.get("max_rank", 30)
        self._refresh_period = self._pairlistconfig.get("refresh_period", 86400)
        self._categories = self._pairlistconfig.get("categories", [])
        self._marketcap_cache: FtTTLCache = FtTTLCache(maxsize=1, ttl=self._refresh_period)

        _coingecko_config = self._config.get("coingecko", {})

        self._coingecko: FtCoinGeckoApi = FtCoinGeckoApi(
            api_key=_coingecko_config.get("api_key", ""),
            is_demo=_coingecko_config.get("is_demo", True),
        )

        if self._categories:
            categories = self._coingecko.get_coins_categories_list()
            category_ids = [cat["category_id"] for cat in categories]

            for category in self._categories:
                if category not in category_ids:
                    raise OperationalException(
                        f"Category {category} not in coingecko category list. "
                        f"You can choose from {category_ids}"
                    )

        if self._max_rank > 250:
            self.logger.warning(
                f"The max rank you have set ({self._max_rank}) is quite high. "
                "This may lead to coingecko API rate limit issues. "
                "Please ensure this value is necessary for your use case.",
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
        num = self._number_assets
        rank = self._max_rank
        mode = self._mode
        pair_text = num if (mode == "whitelist") else "blacklisting"
        msg = f"{self.name} - {pair_text} pairs placed within top {rank} market cap."
        return msg

    @staticmethod
    def description() -> str:
        return "Provides pair list based on CoinGecko's market cap rank."

    @staticmethod
    def available_parameters() -> dict[str, PairlistParameter]:
        return {
            "number_assets": {
                "type": "number",
                "default": 30,
                "description": "Number of assets",
                "help": "Number of assets to use from the pairlist",
            },
            "max_rank": {
                "type": "number",
                "default": 30,
                "description": "Max rank of assets",
                "help": "Maximum rank of assets to use from the pairlist",
            },
            "categories": {
                "type": "list",
                "default": [],
                "description": "Coin Categories",
                "help": (
                    "The Category of the coin e.g layer-1 default [] "
                    "(https://www.coingecko.com/en/categories)"
                ),
            },
            "refresh_period": {
                "type": "number",
                "default": 86400,
                "description": "Refresh period",
                "help": "Refresh period in seconds",
            },
            "mode": {
                "type": "option",
                "default": "whitelist",
                "options": ["whitelist", "blacklist"],
                "description": "Mode of operation",
                "help": "Mode of operation (whitelist/blacklist)",
            },
        }

    def get_markets_exchange(self):
        markets = [
            k
            for k in self._exchange.get_markets(
                quote_currencies=[self._stake_currency], tradable_only=True, active_only=True
            ).keys()
        ]

        return markets

    def gen_pairlist(self, tickers: Tickers) -> list[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: List of pairs
        """
        # Generate dynamic whitelist
        # Must always run if this pairlist is the first in the list.
        pairlist = self._marketcap_cache.get("pairlist_mc")
        if pairlist:
            # Item found - no refresh necessary
            return pairlist.copy()
        else:
            # Use fresh pairlist
            # Check if pair quote currency equals to the stake currency.
            _pairlist = self.get_markets_exchange()

            # No point in testing for blacklisted pairs...
            _pairlist = self.verify_blacklist(_pairlist, logger.info)

            pairlist = self.filter_pairlist(_pairlist, tickers)
            self._marketcap_cache["pairlist_mc"] = pairlist.copy()

        return pairlist

    # Prefixes to test to discover coins like 1000PEPE/USDDT:USDT or KPEPE/USDC (hyperliquid)
    prefixes = ("1000", "K")

    def resolve_marketcap_pair(
        self,
        pair: str,
        pairlist: list[str],
        markets: list[str],
        filtered_pairlist: list[str],
    ) -> str | None:
        if pair in filtered_pairlist:
            return None

        if pair in pairlist:
            return pair

        if pair not in markets:
            for prefix in self.prefixes:
                test_prefix = f"{prefix}{pair}"

                if test_prefix in pairlist:
                    return test_prefix

        return None

    def filter_pairlist(self, pairlist: list[str], tickers: dict) -> list[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers). May be cached.
        :return: new whitelist
        """
        marketcap_list = self._marketcap_cache.get("marketcap")
        mode = self._mode
        is_whitelist_mode = mode == "whitelist"
        filtered_pairlist: list[str] = []

        default_kwargs = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": "250",
            "page": "1",
            "sparkline": "false",
            "locale": "en",
        }

        if marketcap_list is None:
            data = []

            if not self._categories:
                pages_required = math.ceil(self._max_rank / 250)
                for page in range(1, pages_required + 1):
                    default_kwargs["page"] = str(page)
                    page_data = self._coingecko.get_coins_markets(**default_kwargs)
                    data.extend(page_data)
            else:
                for category in self._categories:
                    category_data = self._coingecko.get_coins_markets(
                        **default_kwargs, **({"category": category} if category else {})
                    )
                    data += category_data

            data.sort(key=lambda d: float(d.get("market_cap") or 0.0), reverse=True)

            if data:
                marketcap_list = [row["symbol"] for row in data]
                self._marketcap_cache["marketcap"] = marketcap_list

        if marketcap_list:
            market = self._exchange._config["trading_mode"]
            pair_format = f"{self._stake_currency.upper()}" + (
                f":{self._stake_currency.upper()}" if market == "futures" else ""
            )

            top_marketcap = marketcap_list[: self._max_rank :]
            markets = self.get_markets_exchange()

            for mc_pair in top_marketcap:
                pair = f"{mc_pair.upper()}/{pair_format}"
                resolved = self.resolve_marketcap_pair(pair, pairlist, markets, filtered_pairlist)

                if resolved:
                    if not is_whitelist_mode:
                        pairlist.remove(resolved)
                        continue

                    filtered_pairlist.append(resolved)
                    if len(filtered_pairlist) == self._number_assets:
                        break

        if not is_whitelist_mode:
            return pairlist

        # If no pairs are found, return the original pairlist
        return filtered_pairlist
