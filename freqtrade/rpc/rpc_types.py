from datetime import datetime
from typing import Any, Literal, TypedDict

from freqtrade.constants import PairWithTimeframe
from freqtrade.enums import RPCMessageType


ProfitLossStr = Literal["profit", "loss"]


class RPCSendMsgBase(TypedDict):
    pass


class RPCStatusMsg(RPCSendMsgBase):
    """Used for Status, Startup and Warning messages"""

    type: Literal[RPCMessageType.STATUS, RPCMessageType.STARTUP, RPCMessageType.WARNING]
    status: str


class RPCStrategyMsg(RPCSendMsgBase):
    """Used for Status, Startup and Warning messages"""

    type: Literal[RPCMessageType.STRATEGY_MSG]
    msg: str


class RPCProtectionMsg(RPCSendMsgBase):
    type: Literal[RPCMessageType.PROTECTION_TRIGGER, RPCMessageType.PROTECTION_TRIGGER_GLOBAL]
    id: int
    pair: str
    base_currency: str | None
    lock_time: str
    lock_timestamp: int
    lock_end_time: str
    lock_end_timestamp: int
    reason: str
    side: str
    active: bool


class RPCLiquidationWarningMsg(RPCSendMsgBase):
    """Sent when open position(s) approach freqtrade's liquidation"""

    type: Literal[RPCMessageType.LIQUIDATION_WARNING]
    exchange: str
    margin_mode: str
    # Details of the position closest to its liquidation stop
    trade_id: int
    pair: str
    base_currency: str
    quote_currency: str
    direction: str
    leverage: float | None
    current_rate: float
    liquidation_price: float
    # Distance to the liquidation stop, as a share of the price move that would use up the
    # position's margin (10% at 10x leverage). About 1.0 when freshly opened, 0.0 at the stop.
    remaining_ratio: float
    # Configured `liquidation_warn_ratio` that triggered this message
    warn_ratio: float
    # Positions within warn_ratio, and open trades in total
    positions_at_risk: int
    open_positions: int


class RPCWhitelistMsg(RPCSendMsgBase):
    type: Literal[RPCMessageType.WHITELIST]
    data: list[str]


class __RPCEntryExitMsgBase(RPCSendMsgBase):
    trade_id: int
    buy_tag: str | None
    enter_tag: str | None
    exchange: str
    pair: str
    base_currency: str
    quote_currency: str
    leverage: float | None
    direction: str
    limit: float  # Deprecated, use order_rate instead
    order_rate: float
    open_rate: float
    order_type: str
    stake_amount: float
    stake_currency: str
    fiat_currency: str | None
    amount: float
    open_date: datetime
    current_rate: float | None
    sub_trade: bool


class RPCEntryMsg(__RPCEntryExitMsgBase):
    type: Literal[RPCMessageType.ENTRY, RPCMessageType.ENTRY_FILL]


class RPCCancelMsg(__RPCEntryExitMsgBase):
    type: Literal[RPCMessageType.ENTRY_CANCEL]
    reason: str


class RPCExitMsg(__RPCEntryExitMsgBase):
    type: Literal[RPCMessageType.EXIT, RPCMessageType.EXIT_FILL]
    cumulative_profit: float
    gain: ProfitLossStr
    close_rate: float
    profit_amount: float
    profit_ratio: float
    exit_reason: str | None
    close_date: datetime
    # current_rate: float | None
    final_profit_ratio: float | None
    is_final_exit: bool


class RPCExitCancelMsg(__RPCEntryExitMsgBase):
    type: Literal[RPCMessageType.EXIT_CANCEL]
    reason: str
    gain: ProfitLossStr
    profit_amount: float
    profit_ratio: float
    exit_reason: str | None
    close_date: datetime


class _AnalyzedDFData(TypedDict):
    key: PairWithTimeframe
    df: Any
    la: datetime


class RPCAnalyzedDFMsg(RPCSendMsgBase):
    """New Analyzed dataframe message"""

    type: Literal[RPCMessageType.ANALYZED_DF]
    data: _AnalyzedDFData


class RPCNewCandleMsg(RPCSendMsgBase):
    """New candle ping message, issued once per new candle/pair"""

    type: Literal[RPCMessageType.NEW_CANDLE]
    data: PairWithTimeframe


RPCOrderMsg = RPCEntryMsg | RPCExitMsg | RPCExitCancelMsg | RPCCancelMsg


RPCSendMsg = (
    RPCStatusMsg
    | RPCStrategyMsg
    | RPCProtectionMsg
    | RPCLiquidationWarningMsg
    | RPCWhitelistMsg
    | RPCEntryMsg
    | RPCCancelMsg
    | RPCExitMsg
    | RPCExitCancelMsg
    | RPCAnalyzedDFMsg
    | RPCNewCandleMsg
)
