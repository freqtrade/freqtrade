from freqtrade.enums import Collateral, TradingMode
from freqtrade.exceptions import OperationalException


def liquidation_price(
    exchange_name: str,
    trading_mode: TradingMode,
    ** k
):

    leverage_exchanges = [
        'binance',
        'kraken',
        'ftx'
    ]
    if trading_mode == TradingMode.SPOT or exchange_name.lower() not in leverage_exchanges:
        return None

    collateral: Collateral = k['collateral']

    if exchange_name.lower() == "binance":
        # TODO-lev: Get more variables from **k and pass them to binance
        return binance(trading_mode, collateral)
    elif exchange_name.lower() == "kraken":
        # TODO-lev: Get more variables from **k and pass them to kraken
        return kraken(trading_mode, collateral)
    elif exchange_name.lower() == "ftx":
        return ftx(trading_mode, collateral)
    return


def exception(
    exchange_name: str,
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
        f"{exchange_name} does not support {collateral.value} {trading_mode.value} trading")


def binance(trading_mode: TradingMode, collateral: Collateral):
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


def kraken(trading_mode: TradingMode, collateral: Collateral):
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


def ftx(trading_mode: TradingMode, collateral: Collateral):
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
