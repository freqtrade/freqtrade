from typing import Optional

from freqtrade.enums import Collateral, TradingMode
from freqtrade.exceptions import OperationalException


def liquidation_price(
    exchange_name: str,
    open_rate: float,
    is_short: bool,
    leverage: float,
    trading_mode: TradingMode,
    collateral: Optional[Collateral]
) -> Optional[float]:

    leverage_exchanges = [
        'binance',
        'kraken',
        'ftx'
    ]
    if trading_mode == TradingMode.SPOT or exchange_name.lower() not in leverage_exchanges:
        return None

    if not collateral:
        raise OperationalException(
            "Parameter collateral is required by liquidation_price when trading_mode is "
            f"{trading_mode}"
        )

    if exchange_name.lower() == "binance":
        return binance(open_rate, is_short, leverage, trading_mode, collateral)
    elif exchange_name.lower() == "kraken":
        return kraken(open_rate, is_short, leverage, trading_mode, collateral)
    elif exchange_name.lower() == "ftx":
        return ftx(open_rate, is_short, leverage, trading_mode, collateral)
    raise OperationalException(
        f"liquidation_price is not yet implemented for {exchange_name}"
    )


def exception(
    exchange: str,
    trading_mode: TradingMode,
    collateral: Collateral
):
    """
        Raises an exception if exchange used doesn't support desired leverage mode
        :param name: Name of the exchange
        :param trading_mode: spot, margin, futures
        :param collateral: cross, isolated
    """
    raise OperationalException(
        f"{exchange} does not support {collateral.value} {trading_mode.value} trading")


def binance(
    open_rate: float,
    is_short: bool,
    leverage: float,
    trading_mode: TradingMode,
    collateral: Collateral
):
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
        exception("binance", trading_mode, collateral)
    elif trading_mode == TradingMode.FUTURES and collateral == Collateral.CROSS:
        # TODO-lev: perform a calculation based on this formula
        # https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93
        exception("binance", trading_mode, collateral)
    elif trading_mode == TradingMode.FUTURES and collateral == Collateral.ISOLATED:
        # TODO-lev: perform a calculation based on this formula
        # https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93
        exception("binance", trading_mode, collateral)

    # If nothing was returned
    exception("binance", trading_mode, collateral)


def kraken(
    open_rate: float,
    is_short: bool,
    leverage: float,
    trading_mode: TradingMode,
    collateral: Collateral
):
    """
        Calculates the liquidation price on Kraken
        :param name: Name of the exchange
        :param trading_mode: spot, margin, futures
        :param collateral: cross, isolated
    """
    # TODO-lev: Additional arguments, fill in formulas

    if collateral == Collateral.CROSS:
        if trading_mode == TradingMode.MARGIN:
            exception("kraken", trading_mode, collateral)
            # TODO-lev: perform a calculation based on this formula
            # https://support.kraken.com/hc/en-us/articles/203325763-Margin-Call-Level-and-Margin-Liquidation-Level
        elif trading_mode == TradingMode.FUTURES:
            exception("kraken", trading_mode, collateral)

    # If nothing was returned
    exception("kraken", trading_mode, collateral)


def ftx(
    open_rate: float,
    is_short: bool,
    leverage: float,
    trading_mode: TradingMode,
    collateral: Collateral
):
    """
        Calculates the liquidation price on FTX
        :param name: Name of the exchange
        :param trading_mode: spot, margin, futures
        :param collateral: cross, isolated
    """
    if collateral == Collateral.CROSS:
        # TODO-lev: Additional arguments, fill in formulas
        exception("ftx", trading_mode, collateral)

    # If nothing was returned
    exception("ftx", trading_mode, collateral)
