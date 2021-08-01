# from decimal import Decimal
from enum import Enum
# from math import ceil
from typing import Optional

from freqtrade.enums import TradingMode
from freqtrade.exceptions import OperationalException


class LiqFormula(Enum):
    """Equations to calculate liquidation price"""

    BINANCE = "BINANCE"
    KRAKEN = "KRAKEN"
    FTX = "FTX"

    def __exception(self, trading_mode: TradingMode, freq_specific: Optional[bool] = True):
        """
            Raises an exception if exchange used doesn't support desired leverage mode
            :param trading_mode: cross, isolated, cross_futures or isolated_futures
            :param freq_specific:
                False if the exchange does not support this leverage mode
                True if only freqtrade doesn't support it
        """
        if freq_specific:
            raise OperationalException(
                f"Freqtrade does not support {trading_mode.value} on {self.name}")
        else:
            raise OperationalException(f"{self.name} does not support {trading_mode.value}")

    def __call__(self, **k):
        trading_mode: TradingMode = k.trading_mode

        # * Cross Margin
        if trading_mode == TradingMode.CROSS:
            if self.name == "BINANCE":
                # TODO: perform a calculation based on this formula
                # https://www.binance.com/en/support/faq/f6b010588e55413aa58b7d63ee0125ed
                self.__exception(trading_mode)
            elif self.name == "KRAKEN":
                # TODO: perform a calculation based on this formula
                # https://support.kraken.com/hc/en-us/articles/203325763-Margin-Call-Level-and-Margin-Liquidation-Level
                self.__exception(trading_mode)
            elif self.name == "FTX":
                self.__exception(trading_mode)

        # * Isolated Margin
        elif trading_mode == TradingMode.ISOLATED:
            if self.name == "KRAKEN":  # Kraken doesn't have isolated margin
                self.__exception(trading_mode, False)
            else:
                self.__exception(trading_mode)

        # * Cross Futures
        elif trading_mode == TradingMode.CROSS_FUTURES:
            if self.name == "BINANCE":
                # TODO: perform a calculation based on this formula
                # https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93
                self.__exception(trading_mode)
            else:
                self.__exception(trading_mode)

        # * Isolated Futures
        elif trading_mode == TradingMode.ISOLATED_FUTURES:
            if self.name == "BINANCE":
                # TODO: perform a calculation based on this formula
                # https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93
                self.__exception(trading_mode)
            elif self.name == "KRAKEN":  # Kraken doesn't have isolated margin
                self.__exception(trading_mode, False)
            else:
                self.__exception(trading_mode)
