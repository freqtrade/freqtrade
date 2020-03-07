"""
Module that define classes to convert Crypto-currency to FIAT
e.g BTC to USD
"""

import logging
import time
from typing import Dict, List

from pycoingecko import CoinGeckoAPI

from freqtrade.constants import SUPPORTED_FIAT


logger = logging.getLogger(__name__)


class CryptoFiat:
    """
    Object to describe what is the price of Crypto-currency in a FIAT
    """
    # Constants
    CACHE_DURATION = 6 * 60 * 60  # 6 hours

    def __init__(self, crypto_symbol: str, fiat_symbol: str, price: float) -> None:
        """
        Create an object that will contains the price for a crypto-currency in fiat
        :param crypto_symbol: Crypto-currency you want to convert (e.g BTC)
        :param fiat_symbol: FIAT currency you want to convert to (e.g USD)
        :param price: Price in FIAT
        """

        # Public attributes
        self.crypto_symbol = None
        self.fiat_symbol = None
        self.price = 0.0

        # Private attributes
        self._expiration = 0.0

        self.crypto_symbol = crypto_symbol.lower()
        self.fiat_symbol = fiat_symbol.lower()
        self.set_price(price=price)

    def set_price(self, price: float) -> None:
        """
        Set the price of the Crypto-currency in FIAT and set the expiration time
        :param price: Price of the current Crypto currency in the fiat
        :return: None
        """
        self.price = price
        self._expiration = time.time() + self.CACHE_DURATION

    def is_expired(self) -> bool:
        """
        Return if the current price is still valid or needs to be refreshed
        :return: bool, true the price is expired and needs to be refreshed, false the price is
         still valid
        """
        return self._expiration - time.time() <= 0


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
        self._pairs: List[CryptoFiat] = []
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

        # Check if the fiat convertion you want is supported
        if not self._is_supported_fiat(fiat=fiat_symbol):
            raise ValueError(f'The fiat {fiat_symbol} is not supported.')

        # Get the pair that interest us and return the price in fiat
        for pair in self._pairs:
            if pair.crypto_symbol == crypto_symbol and pair.fiat_symbol == fiat_symbol:
                # If the price is expired we refresh it, avoid to call the API all the time
                if pair.is_expired():
                    pair.set_price(
                        price=self._find_price(
                            crypto_symbol=pair.crypto_symbol,
                            fiat_symbol=pair.fiat_symbol
                        )
                    )

                # return the last price we have for this pair
                return pair.price

        # The pair does not exist, so we create it and return the price
        return self._add_pair(
            crypto_symbol=crypto_symbol,
            fiat_symbol=fiat_symbol,
            price=self._find_price(
                crypto_symbol=crypto_symbol,
                fiat_symbol=fiat_symbol
            )
        )

    def _add_pair(self, crypto_symbol: str, fiat_symbol: str, price: float) -> float:
        """
        :param crypto_symbol: Crypto-currency you want to convert (e.g BTC)
        :param fiat_symbol: FIAT currency you want to convert to (e.g USD)
        :return: price in FIAT
        """
        self._pairs.append(
            CryptoFiat(
                crypto_symbol=crypto_symbol,
                fiat_symbol=fiat_symbol,
                price=price
            )
        )

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
