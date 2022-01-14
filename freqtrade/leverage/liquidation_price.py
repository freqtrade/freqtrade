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
    maintenance_amt: Optional[float] = None,  # (Binance)
    position: Optional[float] = None,  # (Binance and Gateio) Absolute value of position size
    wallet_balance: Optional[float] = None,  # (Binance and Gateio)
    taker_fee_rate: Optional[float] = None,  # (Gateio & Okex)
    liability: Optional[float] = None,  # (Okex)
    interest: Optional[float] = None,  # (Okex)
    position_assets: Optional[float] = None,  # * (Okex) Might be same as position
    mm_ex_1: Optional[float] = 0.0,  # (Binance) Cross only
    upnl_ex_1: Optional[float] = 0.0,  # (Binance) Cross only
) -> Optional[float]:
    """
    :param exchange_name:
    :param open_rate: (EP1) Entry price of position
    :param is_short: True if the trade is a short, false otherwise
    :param leverage: The amount of leverage on the trade
    :param trading_mode: SPOT, MARGIN, FUTURES, etc.
    :param position: Absolute value of position size (in base currency)
    :param mm_ratio: (MMR)
        Okex: [assets in the position - (liability +interest) * mark price] /
            (maintenance margin + liquidation fee)
        # * Note: Binance's formula specifies maintenance margin rate which is mm_ratio * 100%
    :param collateral: Either ISOLATED or CROSS

    # * Binance
    :param maintenance_amt: (CUM) Maintenance Amount of position

    # * Binance and Gateio
    :param wallet_balance: (WB)
        Cross-Margin Mode: crossWalletBalance
        Isolated-Margin Mode: isolatedWalletBalance
    :param position: Absolute value of position size (in base currency)

    # * Gateio & Okex
    :param taker_fee_rate:

    # * Okex
    :param liability:
        Initial liabilities + deducted interest
            • Long positions: Liability is calculated in quote currency.
            • Short positions: Liability is calculated in trading currency.
    :param interest:
        Interest that has not been deducted yet.
    :param position_assets:
        Total position assets – on-hold by pending order

    # * Cross only (Binance)
    :param mm_ex_1: (TMM)
        Cross-Margin Mode: Maintenance Margin of all other contracts, excluding Contract 1
        Isolated-Margin Mode: 0
    :param upnl_ex_1: (UPNL)
        Cross-Margin Mode: Unrealized PNL of all other contracts, excluding Contract 1.
        Isolated-Margin Mode: 0
    """
    if trading_mode == TradingMode.SPOT:
        return None

    if not collateral:
        raise OperationalException(
            "Parameter collateral is required by liquidation_price when trading_mode is "
            f"{trading_mode}"
        )

    if exchange_name.lower() == "binance":
        if (wallet_balance is None or maintenance_amt is None or position is None):
            # mm_ex_1 is None or # * Cross only
            # upnl_ex_1 is None or # * Cross only
            raise OperationalException(
                f"Parameters wallet_balance, maintenance_amt, position"
                f"are required by liquidation_price when exchange is {exchange_name.lower()}"
            )
        # Suppress incompatible type "Optional[...]"; expected "..." as the check exists above.
        return binance(
            open_rate=open_rate,
            is_short=is_short,
            leverage=leverage,
            trading_mode=trading_mode,
            collateral=collateral,  # type: ignore
            wallet_balance=wallet_balance,
            mm_ex_1=mm_ex_1,  # type: ignore
            upnl_ex_1=upnl_ex_1,  # type: ignore
            maintenance_amt=maintenance_amt,  # type: ignore
            position=position,
            mm_ratio=mm_ratio,
        )
    elif exchange_name.lower() == "gateio":
        if (not wallet_balance or not position or not taker_fee_rate):
            raise OperationalException(
                f"Parameters wallet_balance, position, taker_fee_rate"
                f"are required by liquidation_price when exchange is {exchange_name.lower()}"
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
        if (not liability or not interest or not taker_fee_rate or not position_assets):
            raise OperationalException(
                f"Parameters liability, interest, taker_fee_rate, position_assets"
                f"are required by liquidation_price when exchange is {exchange_name.lower()}"
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
    elif exchange_name.lower() == "ftx":
        return ftx(open_rate, is_short, leverage, trading_mode, collateral)
    elif exchange_name.lower() == "kraken":
        return kraken(open_rate, is_short, leverage, trading_mode, collateral)
    raise OperationalException(f"liquidation_price is not implemented for {exchange_name}")


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
    mm_ratio: float,
    collateral: Collateral,
    maintenance_amt: float,
    wallet_balance: float,
    position: float,
    mm_ex_1: float,
    upnl_ex_1: float,
):
    """
    MARGIN: https://www.binance.com/en/support/faq/f6b010588e55413aa58b7d63ee0125ed
    PERPETUAL: https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93

    :param open_rate: (EP1) Entry Price of position (one-way mode)
    :param is_short: true or false
    :param leverage: leverage in float
    :param trading_mode: SPOT, MARGIN, FUTURES
    :param mm_ratio: (MMR) Maintenance margin rate of position (one-way mode)
    :param collateral: CROSS, ISOLATED
    :param maintenance_amt: (CUM) Maintenance Amount of position (one-way mode)
    :param position: Absolute value of position size (one-way mode)
    :param wallet_balance: (WB)
        Cross-Margin Mode: crossWalletBalance
        Isolated-Margin Mode: isolatedWalletBalance
            TMM=0, UPNL=0, substitute the position quantity, MMR, cum into the formula to calculate.
            Under the cross margin mode, the same ticker/symbol,
            both long and short position share the same liquidation price except in isolated mode.
            Under the isolated mode, each isolated position will have different liquidation prices
            depending on the margin allocated to the positions.
    :param mm_ex_1: (TMM)
        Cross-Margin Mode: Maintenance Margin of all other contracts, excluding Contract 1
        Isolated-Margin Mode: 0
    :param upnl_ex_1: (UPNL)
        Cross-Margin Mode: Unrealized PNL of all other contracts, excluding Contract 1.
        Isolated-Margin Mode: 0
    """
    side_1 = -1 if is_short else 1
    position = abs(position)
    cross_vars = upnl_ex_1 - mm_ex_1 if collateral == Collateral.CROSS else 0.0

    if trading_mode == TradingMode.MARGIN and collateral == Collateral.CROSS:
        # ! Not Implemented
        exception("binance", trading_mode, collateral)
    if trading_mode == TradingMode.FUTURES:
        return (wallet_balance + cross_vars + maintenance_amt - (side_1 * position * open_rate)) / (
            (position * mm_ratio) - (side_1 * position))

    exception("binance", trading_mode, collateral)


def gateio(
    open_rate: float,
    is_short: bool,
    trading_mode: TradingMode,
    mm_ratio: float,
    collateral: Collateral,
    position: float,
    wallet_balance: float,
    taker_fee_rate: float,
    is_inverse: bool = False
):
    """
    PERPETUAL: https://www.gate.io/help/futures/perpetual/22160/calculation-of-liquidation-price

    :param open_rate: Entry Price of position
    :param is_short: True for short trades
    :param trading_mode: SPOT, MARGIN, FUTURES
    :param mm_ratio: Viewed in contract details
    :param collateral: CROSS, ISOLATED
    :param position: size of position in base currency
        contract_size / num_contracts
        contract_size: How much one contract is worth
        num_contracts: Also called position
    :param wallet_balance: Also called margin
    :param taker_fee_rate:
    :param is_inverse: True if settle currency matches base currency

    ( Opening Price ± Margin/Contract Multiplier/Position ) / [ 1 ± ( MMR + Taker Fee)]
    '±' in the formula refers to the direction of the contract,
        go long refers to '-'
        go short refers to '+'

    """

    if trading_mode == TradingMode.FUTURES and collateral == Collateral.ISOLATED:
        if is_inverse:
            # ! Not implemented
            raise OperationalException("Freqtrade does not support inverse contracts at the moment")
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
    mm_ratio: float,
    collateral: Collateral,
    taker_fee_rate: float,
    liability: float,
    interest: float,
    position_assets: float
):
    """
    PERPETUAL: https://www.okex.com/support/hc/en-us/articles/
    360053909592-VI-Introduction-to-the-isolated-mode-of-Single-Multi-currency-Portfolio-margin

    :param is_short: True if the position is short, false otherwise
    :param trading_mode: SPOT, MARGIN, FUTURES
    :param mm_ratio:
        long: [position_assets - (liability + interest) / mark_price] / (maintenance_margin + fees)
        short: [position_assets - (liability + interest) * mark_price] / (maintenance_margin + fees)
    :param collateral: CROSS, ISOLATED
    :param taker_fee_rate:
    :param liability: Initial liabilities + deducted interest
        long: Liability is calculated in quote currency
        short: Liability is calculated in trading currency
    :param interest: Interest that has not been deducted yet
    :param position_assets: Total position assets - on-hold by pending order

    Total: The number of positive assets on the position (including margin).
        long: with trading currency as position asset.
        short: with quote currency as position asset.

    Est. liquidation price
        long: (liability + interest）* (1 + maintenance margin ratio) *
            (1 + taker fee rate) / position assets
        short: (liability + interest）* (1 + maintenance margin ratio) *
            (1 + taker fee rate)

    """
    if trading_mode == TradingMode.FUTURES and collateral == Collateral.ISOLATED:
        if is_short:
            return (liability + interest) * (1 + mm_ratio) * (1 + taker_fee_rate)
        else:
            return (liability + interest) * (1 + mm_ratio) * (1 + taker_fee_rate) / position_assets
    else:
        exception("okex", trading_mode, collateral)


def ftx(
    open_rate: float,
    is_short: bool,
    leverage: float,
    trading_mode: TradingMode,
    collateral: Collateral
    # ...
):
    """
    # ! Not Implemented
    Calculates the liquidation price on FTX
    :param open_rate: Entry price of position
    :param is_short: True if the trade is a short, false otherwise
    :param leverage: The amount of leverage on the trade
    :param trading_mode: SPOT, MARGIN, FUTURES, etc.
    :param collateral: Either ISOLATED or CROSS
    """
    if collateral == Collateral.CROSS:
        exception("ftx", trading_mode, collateral)

    # If nothing was returned
    exception("ftx", trading_mode, collateral)


def kraken(
    open_rate: float,
    is_short: bool,
    leverage: float,
    trading_mode: TradingMode,
    collateral: Collateral
    # ...
):
    """
    # ! Not Implemented
    MARGIN:
    https://support.kraken.com/hc/en-us/articles/203325763-Margin-Call-Level-and-Margin-Liquidation-Level

    :param open_rate: Entry price of position
    :param is_short: True if the trade is a short, false otherwise
    :param leverage: The amount of leverage on the trade
    :param trading_mode: SPOT, MARGIN, FUTURES, etc.
    :param collateral: Either ISOLATED or CROSS
    """

    if collateral == Collateral.CROSS:
        if trading_mode == TradingMode.MARGIN:
            exception("kraken", trading_mode, collateral)
        elif trading_mode == TradingMode.FUTURES:
            exception("kraken", trading_mode, collateral)

    # If nothing was returned
    exception("kraken", trading_mode, collateral)
