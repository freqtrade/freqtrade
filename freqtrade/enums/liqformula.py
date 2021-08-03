# from decimal import Decimal
from enum import Enum

from freqtrade.enums.collateral import Collateral
from freqtrade.enums.tradingmode import TradingMode
from freqtrade.exceptions import OperationalException


# from math import ceil


# from math import ceil


class LiqFormula(Enum):
    """Equations to calculate liquidation price"""

    BINANCE = "Binance"
    KRAKEN = "Kraken"
    FTX = "FTX"
    NONE = None

    def __call__(self, **k):

        trading_mode: TradingMode = k['trading_mode']
        if trading_mode == TradingMode.SPOT or self.name == "NONE":
            return None

        collateral: Collateral = k['collateral']

        if self.name == "BINANCE":
            return binance(trading_mode, collateral)
        elif self.name == "KRAKEN":
            return kraken(trading_mode, collateral)
        elif self.name == "FTX":
            return ftx(trading_mode, collateral)
        else:
            exception(self.name, trading_mode, collateral)


def exception(name: str, trading_mode: TradingMode, collateral: Collateral):
    """
        Raises an exception if exchange used doesn't support desired leverage mode
        :param name: Name of the exchange
        :param trading_mode: spot, margin, futures
        :param collateral: cross, isolated
    """
    raise OperationalException(
        f"{name} does not support {collateral.value} {trading_mode.value} trading")


def binance(name: str, trading_mode: TradingMode, collateral: Collateral):
    """
        Calculates the liquidation price on Binance
        :param name: Name of the exchange
        :param trading_mode: spot, margin, futures
        :param collateral: cross, isolated
    """
    # TODO-lev: Additional arguments, fill in formulas

    if trading_mode == TradingMode.MARGIN and collateral == Collateral.CROSS:
        # TODO-lev: perform a calculation based on this formula
        # https://www.binance.com/en/support/faq/f6b010588e55413aa58b7d63ee0125ed
        exception(name, trading_mode, collateral)
    elif trading_mode == TradingMode.FUTURES and collateral == Collateral.CROSS:
        # TODO-lev: perform a calculation based on this formula
        # https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93
        exception(name, trading_mode, collateral)
    elif trading_mode == TradingMode.FUTURES and collateral == Collateral.ISOLATED:
        # TODO-lev: perform a calculation based on this formula
        # https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93
        exception(name, trading_mode, collateral)

    # If nothing was returned
    exception(name, trading_mode, collateral)


def kraken(name: str, trading_mode: TradingMode, collateral: Collateral):
    """
        Calculates the liquidation price on Kraken
        :param name: Name of the exchange
        :param trading_mode: spot, margin, futures
        :param collateral: cross, isolated
    """
    # TODO-lev: Additional arguments, fill in formulas

    if collateral == Collateral.CROSS:
        if trading_mode == TradingMode.MARGIN:
            exception(name, trading_mode, collateral)
            # TODO-lev: perform a calculation based on this formula
            # https://support.kraken.com/hc/en-us/articles/203325763-Margin-Call-Level-and-Margin-Liquidation-Level
        elif trading_mode == TradingMode.FUTURES:
            exception(name, trading_mode, collateral)

    # If nothing was returned
    exception(name, trading_mode, collateral)


def ftx(name: str, trading_mode: TradingMode, collateral: Collateral):
    """
        Calculates the liquidation price on FTX
        :param name: Name of the exchange
        :param trading_mode: spot, margin, futures
        :param collateral: cross, isolated
    """
    if collateral == Collateral.CROSS:
        # TODO-lev: Additional arguments, fill in formulas
        exception(name, trading_mode, collateral)

    # If nothing was returned
    exception(name, trading_mode, collateral)
