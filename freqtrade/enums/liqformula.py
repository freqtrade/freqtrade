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
            raise OperationalException(f"{self.name} does not support {trading_mode.value} trading")

    def __binance(self, trading_mode: TradingMode):
        # TODO-mg: Additional arguments, fill in formulas

        if trading_mode == TradingMode.CROSS_MARGIN:
            # TODO-mg: perform a calculation based on this formula
            # https://www.binance.com/en/support/faq/f6b010588e55413aa58b7d63ee0125ed
            self.__exception(trading_mode)
        elif trading_mode == TradingMode.ISOLATED_MARGIN:
            self.__exception(trading_mode)  # Likely won't be implemented
        elif trading_mode == TradingMode.CROSS_FUTURES:
            # TODO-mg: perform a calculation based on this formula
            # https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93
            self.__exception(trading_mode)
        elif trading_mode == TradingMode.ISOLATED_FUTURES:
            # TODO-mg: perform a calculation based on this formula
            # https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93
            self.__exception(trading_mode)
        else:
            self.__exception(trading_mode)

    def __kraken(self, trading_mode: TradingMode):
        # TODO-mg: Additional arguments, fill in formulas

        if trading_mode == TradingMode.CROSS_MARGIN:
            self.__exception(trading_mode)
            # TODO-mg: perform a calculation based on this formula
            # https://support.kraken.com/hc/en-us/articles/203325763-Margin-Call-Level-and-Margin-Liquidation-Level
        elif trading_mode == TradingMode.CROSS_FUTURES:
            # TODO-mg: implement
            self.__exception(trading_mode)
        elif trading_mode == TradingMode.ISOLATED_MARGIN or \
                trading_mode == TradingMode.ISOLATED_FUTURES:
            self.__exception(trading_mode, True)
        else:
            self.__exception(trading_mode)

    def __ftx(self, trading_mode: TradingMode):
        # TODO-mg: Additional arguments, fill in formulas
        self.__exception(trading_mode)

    def __call__(self, **k):

        trading_mode: TradingMode = k['trading_mode']

        if trading_mode == TradingMode.SPOT:
            return None

        if self.name == "BINANCE":
            return self.__binance(trading_mode)
        elif self.name == "KRAKEN":
            return self.__kraken(trading_mode)
        elif self.name == "FTX":
            return self.__ftx(trading_mode)
        else:
            self.__exception(trading_mode)
