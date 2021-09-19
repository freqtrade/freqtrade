from typing import Optional

from freqtrade.enums import Collateral, TradingMode
from freqtrade.exceptions import OperationalException


def liquidation_price(
    exchange_name: str,
    open_rate: float,
    is_short: bool,
    leverage: float,
    trading_mode: TradingMode,
    collateral: Optional[Collateral],
    wallet_balance: Optional[float],
    maintenance_margin_ex_1: Optional[float],
    unrealized_pnl_ex_1: Optional[float],
    maintenance_amount_both: Optional[float],
    position_1_both: Optional[float],
    entry_price_1_both: Optional[float],
    maintenance_margin_rate_both: Optional[float]
) -> Optional[float]:

    if trading_mode == TradingMode.SPOT:
        return None

    if not collateral:
        raise OperationalException(
            "Parameter collateral is required by liquidation_price when trading_mode is "
            f"{trading_mode}"
        )

    if exchange_name.lower() == "binance":
        if not wallet_balance or not maintenance_margin_ex_1 or not unrealized_pnl_ex_1 or not maintenance_amount_both \
                or not position_1_both or not entry_price_1_both or not maintenance_margin_rate_both:
            raise OperationalException(
                f"Parameters wallet_balance, maintenance_margin_ex_1, unrealized_pnl_ex_1, maintenance_amount_both, "
                f"position_1_both, entry_price_1_both, maintenance_margin_rate_both is required by liquidation_price "
                f"when exchange is {exchange_name.lower()}")

        return binance(open_rate, is_short, leverage, trading_mode, collateral, wallet_balance, maintenance_margin_ex_1,
                       unrealized_pnl_ex_1, maintenance_amount_both, position_1_both, entry_price_1_both,
                       maintenance_margin_rate_both)
    elif exchange_name.lower() == "kraken":
        return kraken(open_rate, is_short, leverage, trading_mode, collateral)
    elif exchange_name.lower() == "ftx":
        return ftx(open_rate, is_short, leverage, trading_mode, collateral)
    raise OperationalException(
        f"liquidation_price is not implemented for {exchange_name}"
    )


def exception(
    exchange: str,
    trading_mode: TradingMode,
    collateral: Collateral,
):
    """
        Raises an exception if exchange used doesn't support desired leverage mode
        :param exchange: Name of the exchange
        :param trading_mode: spot, margin, futures
        :param collateral: cross, isolated
    """

    raise OperationalException(
        f"{exchange} does not support {collateral.value} Mode {trading_mode.value} trading ")


def binance(
    open_rate: float,
    is_short: bool,
    leverage: float,
    trading_mode: TradingMode,
    collateral: Collateral,
    wallet_balance: float,
    maintenance_margin_ex_1: float,
    unrealized_pnl_ex_1: float,
    maintenance_amount_both: float,
    position_1_both: float,
    entry_price_1_both: float,
    maintenance_margin_rate_both: float,
):
    """
        Calculates the liquidation price on Binance
        :param open_rate: open_rate
        :param is_short: true or false
        :param leverage: leverage in float
        :param trading_mode: spot, margin, futures
        :param collateral: cross, isolated

        :param wallet_balance: Wallet Balance is crossWalletBalance in Cross-Margin Mode.
          Wallet Balance is isolatedWalletBalance in Isolated Margin Mode

        :param maintenance_margin_ex_1: Maintenance Margin of all other contracts, excluding Contract 1.
          If it is an isolated margin mode, then TMM=0

        :param unrealized_pnl_ex_1: Unrealized PNL of all other contracts, excluding Contract 1.
          If it is an isolated margin mode, then UPNL=0

        :param maintenance_amount_both: Maintenance Amount of BOTH position (one-way mode)

        :param position_1_both: Absolute value of BOTH position size (one-way mode)

        :param entry_price_1_both: Entry Price of BOTH position (one-way mode)

        :param maintenance_margin_rate_both: Maintenance margin rate of BOTH position (one-way mode)

    """
    # TODO-lev: Additional arguments, fill in formulas
    wb = wallet_balance
    tmm_1 = 0.0 if collateral == Collateral.ISOLATED else maintenance_margin_ex_1
    upnl_1 = 0.0 if collateral == Collateral.ISOLATED else unrealized_pnl_ex_1
    cum_b = maintenance_amount_both
    side_1_both = -1 if is_short else 1
    position_1_both = abs(position_1_both)
    ep1_both = entry_price_1_both
    mmr_b = maintenance_margin_rate_both

    if trading_mode == TradingMode.MARGIN and collateral == Collateral.CROSS:
        # TODO-lev: perform a calculation based on this formula
        # https://www.binance.com/en/support/faq/f6b010588e55413aa58b7d63ee0125ed
        exception("binance", trading_mode, collateral)
    elif trading_mode == TradingMode.FUTURES and collateral == Collateral.ISOLATED:
        # https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93
        # Liquidation Price of USDⓈ-M Futures Contracts Isolated

        # Isolated margin mode, then TMM=0，UPNL=0
        return (wb + cum_b - (side_1_both * position_1_both * ep1_both)) / (
            position_1_both * mmr_b - side_1_both * position_1_both)

    elif trading_mode == TradingMode.FUTURES and collateral == Collateral.CROSS:
        # TODO-lev: perform a calculation based on this formula
        # https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93
        # Liquidation Price of USDⓈ-M Futures Contracts Cross

        # Isolated margin mode, then TMM=0，UPNL=0
        return (wb - tmm_1 + upnl_1 + cum_b - (side_1_both * position_1_both * ep1_both)) / (
            position_1_both * mmr_b - side_1_both * position_1_both)

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
