"""
Module that define classes to convert Crypto-currency to FIAT
e.g BTC to USD
"""

import logging
from typing import Dict

from cachetools.ttl import TTLCache
from pycoingecko import CoinGeckoAPI

from freqtrade.constants import SUPPORTED_FIAT


logger = logging.getLogger(__name__)


class CryptoToFiatConverter:
    """
    Main class to initiate Crypto to FIAT.
    This object contains a list of pair Crypto, FIAT
    This object is also a Singleton
    """
    __instance = None
    _coingekko: CoinGeckoAPI = None

    _cryptomap: Dict = {}

    def __new__(cls):
        """
        This class is a singleton - cannot be instantiated twice.
        """
        if CryptoToFiatConverter.__instance is None:
            CryptoToFiatConverter.__instance = object.__new__(cls)
            try:
                CryptoToFiatConverter._coingekko = CoinGeckoAPI()
            except BaseException:
                CryptoToFiatConverter._coingekko = None
        return CryptoToFiatConverter.__instance

    def __init__(self) -> None:
        # Timeout: 6h
        self._pair_price: TTLCache = TTLCache(maxsize=500, ttl=6 * 60 * 60)

        self._load_cryptomap()

    def _load_cryptomap(self) -> None:
        try:
            coinlistings = self._coingekko.get_coins_list()
            # Create mapping table from synbol to coingekko_id
            self._cryptomap = {x['symbol']: x['id'] for x in coinlistings}
        except (Exception) as exception:
            logger.error(
                f"Could not load FIAT Cryptocurrency map for the following problem: {exception}")

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

        if crypto_symbol == 'usd':
            # usd corresponds to "uniswap-state-dollar" for coingecko.
            # We'll therefore need to "swap" the currencies
            logger.info(f"reversing Rates {crypto_symbol}, {fiat_symbol}")
            crypto_symbol = fiat_symbol
            fiat_symbol = 'usd'
            inverse = True

        symbol = f"{crypto_symbol}/{fiat_symbol}"
        # Check if the fiat convertion you want is supported
        if not self._is_supported_fiat(fiat=fiat_symbol):
            raise ValueError(f'The fiat {fiat_symbol} is not supported.')

        price = self._pair_price.get(symbol, None)

        if not price:
            price = self._find_price(
                crypto_symbol=crypto_symbol,
                fiat_symbol=fiat_symbol
            )
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
        Call CoinGekko API to retrieve the price in the FIAT
        :param crypto_symbol: Crypto-currency you want to convert (e.g btc)
        :param fiat_symbol: FIAT currency you want to convert to (e.g usd)
        :return: float, price of the crypto-currency in Fiat
        """
        # Check if the fiat convertion you want is supported
        if not self._is_supported_fiat(fiat=fiat_symbol):
            raise ValueError(f'The fiat {fiat_symbol} is not supported.')

        # No need to convert if both crypto and fiat are the same
        if crypto_symbol == fiat_symbol:
            return 1.0

        if crypto_symbol not in self._cryptomap:
            # return 0 for unsupported stake currencies (fiat-convert should not break the bot)
            logger.warning("unsupported crypto-symbol %s - returning 0.0", crypto_symbol)
            return 0.0

        try:
            _gekko_id = self._cryptomap[crypto_symbol]
            return float(
                self._coingekko.get_price(
                    ids=_gekko_id,
                    vs_currencies=fiat_symbol
                )[_gekko_id][fiat_symbol]
            )
        except Exception as exception:
            logger.error("Error in _find_price: %s", exception)
            return 0.0
