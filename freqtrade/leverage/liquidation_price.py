from typing import Optional

from freqtrade.enums import Collateral, TradingMode
from freqtrade.exceptions import OperationalException


def liquidation_price(
    exchange_name: str,
    open_rate: float,   # Entry price of position
    is_short: bool,
    leverage: float,
    trading_mode: TradingMode,
    mm_ratio: float,
    collateral: Optional[Collateral] = Collateral.ISOLATED,

    # Binance
    maintenance_amt: Optional[float] = None,

    # Binance and Gateio
    wallet_balance: Optional[float] = None,
    position: Optional[float] = None,   # Absolute value of position size

    # Gateio & Okex
    taker_fee_rate: Optional[float] = None,

    # Okex
    liability: Optional[float] = None,
    interest: Optional[float] = None,
    position_assets: Optional[float] = None,  # * Might be same as position

    # * Cross only
    mm_ex_1: Optional[float] = 0.0,  # Cross only
    upnl_ex_1: Optional[float] = 0.0,  # Cross only
) -> Optional[float]:
    '''
    wallet_balance
        In Cross margin mode, WB is crossWalletBalance
        In Isolated margin mode, WB is isolatedWalletBalance of the isolated position,
        TMM=0, UPNL=0, substitute the position quantity, MMR, cum into the formula to calculate.
        Under the cross margin mode, the same ticker/symbol,
        both long and short position share the same liquidation price except in the isolated mode.
        Under the isolated mode, each isolated position will have different liquidation prices
        depending on the margin allocated to the positions.
    position
        Absolute value of position size (in base currency)

    # Binance
    maintenance_amt (cumb)
        Maintenance Amount of position

    # Gateio & okex & binance
    mm_ratio
        [assets in the position - (liability +interest) * mark price] /
            (maintenance margin + liquidation fee) (okex)
        # * Note: Binance's formula specifies maintenance margin rate which is mm_ratio * 100%

    # Gateio & okex
    taker_fee_rate

    # Okex
    liability
        Initial liabilities + deducted interest
            • Long positions: Liability is calculated in quote currency.
            • Short positions: Liability is calculated in trading currency.
    interest
        Interest that has not been deducted yet.
    position_assets
        Total position assets – on-hold by pending order

    # * Cross only
    mm_ex_1
        Maintenance Margin of all other contracts, excluding Contract 1
        If it is an isolated margin mode, then TMM=0，UPNL=0
    upnl_ex_1
        Unrealized PNL of all other contracts, excluding Contract 1
        If it is an isolated margin mode, then UPNL=0
    '''
    if trading_mode == TradingMode.SPOT:
        return None

    if not collateral:
        raise OperationalException(
            "Parameter collateral is required by liquidation_price when trading_mode is "
            f"{trading_mode}"
        )

    if exchange_name.lower() == "binance":
        if (
            wallet_balance is None or
            # mm_ex_1 is None or # * Cross only
            # upnl_ex_1 is None or # * Cross only
            maintenance_amt is None or
            position is None or
            mm_ratio is None
        ):
            raise OperationalException(
                f"Parameters wallet_balance, mm_ex_1, upnl_ex_1, "
                f"maintenance_amt, position, mm_ratio "
                f"is required by liquidation_price when exchange is {exchange_name.lower()}")

        # Suppress incompatible type "Optional[float]"; expected "float" as the check exists above.
        return binance(
            open_rate=open_rate,
            is_short=is_short,
            leverage=leverage,
            trading_mode=trading_mode,
            collateral=collateral,  # type: ignore
            wallet_balance=wallet_balance,
            # mm_ex_1=mm_ex_1,
            # upnl_ex_1=upnl_ex_1,
            maintenance_amt=maintenance_amt,  # type: ignore
            position=position,
            mm_ratio=mm_ratio,
        )  # type: ignore
    elif exchange_name.lower() == "kraken":
        return kraken(open_rate, is_short, leverage, trading_mode, collateral)
    elif exchange_name.lower() == "ftx":
        return ftx(open_rate, is_short, leverage, trading_mode, collateral)
    elif exchange_name.lower() == "gateio":
        if (
            not wallet_balance or
            not position or
            not mm_ratio or
            not taker_fee_rate
        ):
            raise OperationalException(
                f"{exchange_name} {collateral} {trading_mode} requires parameters "
                f"wallet_balance, contract_size, num_contracts, mm_ratio and taker_fee"
            )
        else:
            return gateio(
                open_rate=open_rate,
                is_short=is_short,
                trading_mode=trading_mode,
                collateral=collateral,
                wallet_balance=wallet_balance,
                position=position,
                mm_ratio=mm_ratio,
                taker_fee_rate=taker_fee_rate
            )
    elif exchange_name.lower() == "okex":
        if (
            not mm_ratio or
            not liability or
            not interest or
            not taker_fee_rate or
            not position_assets
        ):
            raise OperationalException(
                f"{exchange_name} {collateral} {trading_mode} requires parameters "
                f"mm_ratio, liability, interest, taker_fee_rate, position_assets"
            )
        else:
            return okex(
                is_short=is_short,
                trading_mode=trading_mode,
                collateral=collateral,
                mm_ratio=mm_ratio,
                liability=liability,
                interest=interest,
                taker_fee_rate=taker_fee_rate,
                position_assets=position_assets,
            )
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
    mm_ex_1: float,
    upnl_ex_1: float,
    maintenance_amt: float,
    position: float,
    mm_ratio: float,
):
    """
    Calculates the liquidation price on Binance
    :param is_short: true or false
    :param leverage: leverage in float
    :param trading_mode: spot, margin, futures
    :param collateral: cross, isolated
    :param wallet_balance: Wallet Balance is crossWalletBalance in Cross-Margin Mode.
        Wallet Balance is isolatedWalletBalance in Isolated Margin Mode
    :param mm_ex_1: Maintenance Margin of all other contracts,
        excluding Contract 1. If it is an isolated margin mode, then TMM=0
    :param upnl_ex_1: Unrealized PNL of all other contracts, excluding Contract 1.
        If it is an isolated margin mode, then UPNL=0
    :param maintenance_amt: Maintenance Amount of position (one-way mode)
    :param position: Absolute value of position size (one-way mode)
    :param open_rate: Entry Price of position (one-way mode)
    :param mm_ratio: Maintenance margin rate of position (one-way mode)
    """
    # TODO-lev: Additional arguments, fill in formulas
    wb = wallet_balance
    tmm_1 = 0.0 if collateral == Collateral.ISOLATED else mm_ex_1
    upnl_1 = 0.0 if collateral == Collateral.ISOLATED else upnl_ex_1
    cum_b = maintenance_amt
    side_1 = -1 if is_short else 1
    position = abs(position)
    ep1 = open_rate
    mmr_b = mm_ratio

    if trading_mode == TradingMode.MARGIN and collateral == Collateral.CROSS:
        # TODO-lev: perform a calculation based on this formula
        # https://www.binance.com/en/support/faq/f6b010588e55413aa58b7d63ee0125ed
        exception("binance", trading_mode, collateral)
    elif trading_mode == TradingMode.FUTURES and collateral == Collateral.ISOLATED:
        # https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93
        # Liquidation Price of USDⓈ-M Futures Contracts Isolated

        # Isolated margin mode, then TMM=0，UPNL=0
        return (wb + cum_b - side_1 * position * ep1) / (
            position * mmr_b - side_1 * position)

    elif trading_mode == TradingMode.FUTURES and collateral == Collateral.CROSS:
        # TODO-lev: perform a calculation based on this formula
        # https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93
        # Liquidation Price of USDⓈ-M Futures Contracts Cross

        # Isolated margin mode, then TMM=0，UPNL=0
        return (wb - tmm_1 + upnl_1 + cum_b - side_1 * position * ep1) / (
            position * mmr_b - side_1 * position)

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
    :param trading_mode: spot, margin, futures
    :param collateral: cross, isolated
    """
    if collateral == Collateral.CROSS:
        # TODO-lev: Additional arguments, fill in formulas
        exception("ftx", trading_mode, collateral)

    # If nothing was returned
    exception("ftx", trading_mode, collateral)


def gateio(
    open_rate: float,
    is_short: bool,
    trading_mode: TradingMode,
    collateral: Collateral,
    wallet_balance: float,
    position: float,
    mm_ratio: float,
    taker_fee_rate: float,
    is_inverse: bool = False
):
    """
    Calculates the liquidation price on Gate.io
    :param open_rate: Entry Price of position
    :param is_short: True for short trades
    :param trading_mode: spot, margin, futures
    :param collateral: cross, isolated
    :param wallet_balance: Also called margin
    :param position: size of position in base currency
        contract_size / num_contracts
        contract_size: How much one contract is worth
        num_contracts: Also called position
    :param mm_ratio: Viewed in contract details
    :param taker_fee_rate:
    :param is_inverse: True if settle currency matches base currency

    ( Opening Price ± Margin/Contract Multiplier/Position ) / [ 1 ± ( MMR + Taker Fee)]
    '±' in the formula refers to the direction of the contract,
        go long refers to '-'
        go short refers to '+'
    Position refers to the number of contracts.
    Maintenance Margin Ratio and Contract Multiplier can be viewed in the Contract Details.

    https://www.gate.io/help/futures/perpetual/22160/calculation-of-liquidation-price
    """

    if trading_mode == TradingMode.FUTURES and collateral == Collateral.ISOLATED:
        if is_inverse:
            raise OperationalException(
                "Freqtrade does not support inverse contracts at the moment")
        value = wallet_balance / position

        mm_ratio_taker = (mm_ratio + taker_fee_rate)
        if is_short:
            return (open_rate + value) / (1 + mm_ratio_taker)
        else:
            return (open_rate - value) / (1 - mm_ratio_taker)
    else:
        exception("gatio", trading_mode, collateral)


def okex(
    is_short: bool,
    trading_mode: TradingMode,
    collateral: Collateral,
    liability: float,
    interest: float,
    mm_ratio: float,
    taker_fee_rate: float,
    position_assets: float
):
    '''
    https://www.okex.com/support/hc/en-us/articles/
    360053909592-VI-Introduction-to-the-isolated-mode-of-Single-Multi-currency-Portfolio-margin

    Initial liabilities + deducted interest
        Long positions: Liability is calculated in quote currency.
        Short positions: Liability is calculated in trading currency.
    interest: Interest that has not been deducted yet.
    Margin ratio
        Long: [position_assets - (liability + interest) / mark_price] / (maintenance_margin + fees)
        Short: [position_assets - (liability + interest) * mark_price] / (maintenance_margin + fees)
    '''
    if trading_mode == TradingMode.FUTURES and collateral == Collateral.ISOLATED:
        if is_short:
            return (liability + interest) * (1 + mm_ratio) * (1 + taker_fee_rate)
        else:
            return (liability + interest) * (1 + mm_ratio) * (1 + taker_fee_rate) / position_assets
    else:
        exception("okex", trading_mode, collateral)

# if __name__ == '__main__':
#     print(liquidation_price(
#         "binance",
#         32481.980,
#         False,
#         1,
#         TradingMode.FUTURES,
#         Collateral.ISOLATED,
#         1535443.01,
#         356512.508,
#         0.0,
#         16300.000,
#         109.488,
#         0.025
#     ))
