import logging
import time

from pymarketcap import Pymarketcap

logger = logging.getLogger(__name__)


class CryptoFiat():
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
        self._expiration = 0

        self.crypto_symbol = crypto_symbol.upper()
        self.fiat_symbol = fiat_symbol.upper()
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


class CryptoToFiatConverter(object):
    __instance = None
    _coinmarketcap = None

    # Constants
    SUPPORTED_FIAT = [
        "AUD", "BRL", "CAD", "CHF", "CLP", "CNY", "CZK", "DKK",
        "EUR", "GBP", "HKD", "HUF", "IDR", "ILS", "INR", "JPY",
        "KRW", "MXN", "MYR", "NOK", "NZD", "PHP", "PKR", "PLN",
        "RUB", "SEK", "SGD", "THB", "TRY", "TWD", "ZAR", "USD"
    ]

    def __new__(cls):
        if CryptoToFiatConverter.__instance is None:
            CryptoToFiatConverter.__instance = object.__new__(cls)
            try:
                CryptoToFiatConverter._coinmarketcap = Pymarketcap()
            except BaseException:
                CryptoToFiatConverter._coinmarketcap = None
        return CryptoToFiatConverter.__instance

    def __init__(self) -> None:
        self._pairs = []

    def convert_amount(self, crypto_amount: float, crypto_symbol: str, fiat_symbol: str) -> float:
        """
        Convert an amount of crypto-currency to fiat
        :param crypto_amount: amount of crypto-currency to convert
        :param crypto_symbol: crypto-currency used
        :param fiat_symbol: fiat to convert to
        :return: float, value in fiat of the crypto-currency amount
        """
        price = self.get_price(crypto_symbol=crypto_symbol, fiat_symbol=fiat_symbol)
        return float(crypto_amount) * float(price)

    def get_price(self, crypto_symbol: str, fiat_symbol: str) -> float:
        """
        Return the price of the Crypto-currency in Fiat
        :param crypto_symbol: Crypto-currency you want to convert (e.g BTC)
        :param fiat_symbol: FIAT currency you want to convert to (e.g USD)
        :return: Price in FIAT
        """
        crypto_symbol = crypto_symbol.upper()
        fiat_symbol = fiat_symbol.upper()

        # Check if the fiat convertion you want is supported
        if not self._is_supported_fiat(fiat=fiat_symbol):
            raise ValueError('The fiat {} is not supported.'.format(fiat_symbol))

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

        fiat = fiat.upper()

        return fiat in self.SUPPORTED_FIAT

    def _find_price(self, crypto_symbol: str, fiat_symbol: str) -> float:
        """
        Call CoinMarketCap API to retrieve the price in the FIAT
        :param crypto_symbol: Crypto-currency you want to convert (e.g BTC)
        :param fiat_symbol: FIAT currency you want to convert to (e.g USD)
        :return: float, price of the crypto-currency in Fiat
        """
        # Check if the fiat convertion you want is supported
        if not self._is_supported_fiat(fiat=fiat_symbol):
            raise ValueError('The fiat {} is not supported.'.format(fiat_symbol))
        try:
            return float(
                self._coinmarketcap.ticker(
                    currency=crypto_symbol,
                    convert=fiat_symbol
                )['price_' + fiat_symbol.lower()]
            )
        except BaseException:
            return 0.0
