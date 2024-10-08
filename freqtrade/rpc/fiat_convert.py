"""
Module that define classes to convert Crypto-currency to FIAT
e.g BTC to USD
"""

import logging
from datetime import datetime
from typing import Any

from cachetools import TTLCache
from requests.exceptions import RequestException

from freqtrade.constants import SUPPORTED_FIAT, Config
from freqtrade.mixins.logging_mixin import LoggingMixin
from freqtrade.util.coin_gecko import FtCoinGeckoApi


logger = logging.getLogger(__name__)


# Manually map symbol to ID for some common coins
# with duplicate coingecko entries
coingecko_mapping = {
    "eth": "ethereum",
    "bnb": "binancecoin",
    "sol": "solana",
    "usdt": "tether",
    "busd": "binance-usd",
    "tusd": "true-usd",
    "usdc": "usd-coin",
    "btc": "bitcoin",
}


class CryptoToFiatConverter(LoggingMixin):
    """
    Main class to initiate Crypto to FIAT.
    This object contains a list of pair Crypto, FIAT
    This object is also a Singleton
    """

    __instance = None

    _coinlistings: list[dict] = []
    _backoff: float = 0.0

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Singleton pattern to ensure only one instance is created.
        """
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self, config: Config) -> None:
        # Timeout: 6h
        self._pair_price: TTLCache = TTLCache(maxsize=500, ttl=6 * 60 * 60)

        _coingecko_config = config.get("coingecko", {})
        self._coingecko = FtCoinGeckoApi(
            api_key=_coingecko_config.get("api_key", ""),
            is_demo=_coingecko_config.get("is_demo", True),
            retries=1,
        )
        LoggingMixin.__init__(self, logger, 3600)
        self._load_cryptomap()

    def _load_cryptomap(self) -> None:
        try:
            # Use list-comprehension to ensure we get a list.
            self._coinlistings = [x for x in self._coingecko.get_coins_list()]
        except RequestException as request_exception:
            if "429" in str(request_exception):
                logger.warning(
                    "Too many requests for CoinGecko API, backing off and trying again later."
                )
                # Set backoff timestamp to 60 seconds in the future
                self._backoff = datetime.now().timestamp() + 60
                return
            # If the request is not a 429 error we want to raise the normal error
            logger.error(
                "Could not load FIAT Cryptocurrency map for the following problem: "
                f"{request_exception}"
            )
        except Exception as exception:
            logger.error(
                f"Could not load FIAT Cryptocurrency map for the following problem: {exception}"
            )

    def _get_gecko_id(self, crypto_symbol):
        if not self._coinlistings:
            if self._backoff <= datetime.now().timestamp():
                self._load_cryptomap()
                # Still not loaded.
                if not self._coinlistings:
                    return None
            else:
                return None
        found = [x for x in self._coinlistings if x["symbol"].lower() == crypto_symbol]

        if crypto_symbol in coingecko_mapping.keys():
            found = [x for x in self._coinlistings if x["id"] == coingecko_mapping[crypto_symbol]]

        if len(found) == 1:
            return found[0]["id"]

        if len(found) > 0:
            # Wrong!
            logger.warning(f"Found multiple mappings in CoinGecko for {crypto_symbol}.")
            return None

    def convert_amount(self, crypto_amount: float, crypto_symbol: str, fiat_symbol: str) -> float:
        """
        Convert an amount of crypto-currency to fiat
        :param crypto_amount: amount of crypto-currency to convert
        :param crypto_symbol: crypto-currency used
        :param fiat_symbol: fiat to convert to
        :return: float, value in fiat of the crypto-currency amount
        """
        if crypto_symbol == fiat_symbol:
            return float(crypto_amount)
        price = self.get_price(crypto_symbol=crypto_symbol, fiat_symbol=fiat_symbol)
        return float(crypto_amount) * float(price)

    def get_price(self, crypto_symbol: str, fiat_symbol: str) -> float:
        """
        Return the price of the Crypto-currency in Fiat
        :param crypto_symbol: Crypto-currency you want to convert (e.g BTC)
        :param fiat_symbol: FIAT currency you want to convert to (e.g USD)
        :return: Price in FIAT
        """
        crypto_symbol = crypto_symbol.lower()
        fiat_symbol = fiat_symbol.lower()
        inverse = False

        if crypto_symbol == "usd":
            # usd corresponds to "uniswap-state-dollar" for coingecko.
            # We'll therefore need to "swap" the currencies
            logger.info(f"reversing Rates {crypto_symbol}, {fiat_symbol}")
            crypto_symbol = fiat_symbol
            fiat_symbol = "usd"
            inverse = True

        symbol = f"{crypto_symbol}/{fiat_symbol}"
        # Check if the fiat conversion you want is supported
        if not self._is_supported_fiat(fiat=fiat_symbol):
            raise ValueError(f"The fiat {fiat_symbol} is not supported.")

        price = self._pair_price.get(symbol, None)

        if not price:
            price = self._find_price(crypto_symbol=crypto_symbol, fiat_symbol=fiat_symbol)
            if inverse and price != 0.0:
                price = 1 / price
            self._pair_price[symbol] = price

        return price

    def _is_supported_fiat(self, fiat: str) -> bool:
        """
        Check if the FIAT your want to convert to is supported
        :param fiat: FIAT to check (e.g USD)
        :return: bool, True supported, False not supported
        """

        return fiat.upper() in SUPPORTED_FIAT

    def _find_price(self, crypto_symbol: str, fiat_symbol: str) -> float:
        """
        Call CoinGecko API to retrieve the price in the FIAT
        :param crypto_symbol: Crypto-currency you want to convert (e.g btc)
        :param fiat_symbol: FIAT currency you want to convert to (e.g usd)
        :return: float, price of the crypto-currency in Fiat
        """
        # Check if the fiat conversion you want is supported
        if not self._is_supported_fiat(fiat=fiat_symbol):
            raise ValueError(f"The fiat {fiat_symbol} is not supported.")

        # No need to convert if both crypto and fiat are the same
        if crypto_symbol == fiat_symbol:
            return 1.0

        _gecko_id = self._get_gecko_id(crypto_symbol)

        if not _gecko_id:
            # return 0 for unsupported stake currencies (fiat-convert should not break the bot)
            self.log_once(
                f"unsupported crypto-symbol {crypto_symbol.upper()} - returning 0.0", logger.warning
            )
            return 0.0

        try:
            return float(
                self._coingecko.get_price(ids=_gecko_id, vs_currencies=fiat_symbol)[_gecko_id][
                    fiat_symbol
                ]
            )
        except Exception as exception:
            logger.error("Error in _find_price: %s", exception)
            return 0.0
