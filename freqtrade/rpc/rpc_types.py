from datetime import datetime
from typing import Any, List, Literal, Optional, TypedDict, Union

from freqtrade.constants import PairWithTimeframe
from freqtrade.enums import RPCMessageType


ProfitLossStr = Literal["profit", "loss"]


class RPCSendMsgBase(TypedDict):
    pass
    # ty1pe: Literal[RPCMessageType]


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
    base_currency: Optional[str]
    lock_time: str
    lock_timestamp: int
    lock_end_time: str
    lock_end_timestamp: int
    reason: str
    side: str
    active: bool


class RPCWhitelistMsg(RPCSendMsgBase):
    type: Literal[RPCMessageType.WHITELIST]
    data: List[str]


class __RPCEntryExitMsgBase(RPCSendMsgBase):
    trade_id: int
    buy_tag: Optional[str]
    enter_tag: Optional[str]
    exchange: str
    pair: str
    base_currency: str
    quote_currency: str
    leverage: Optional[float]
    direction: str
    limit: float
    open_rate: float
    order_type: str
    stake_amount: float
    stake_currency: str
    fiat_currency: Optional[str]
    amount: float
    open_date: datetime
    current_rate: Optional[float]
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
    exit_reason: Optional[str]
    close_date: datetime
    # current_rate: Optional[float]
    order_rate: Optional[float]
    final_profit_ratio: Optional[float]
    is_final_exit: bool


class RPCExitCancelMsg(__RPCEntryExitMsgBase):
    type: Literal[RPCMessageType.EXIT_CANCEL]
    reason: str
    gain: ProfitLossStr
    profit_amount: float
    profit_ratio: float
    exit_reason: Optional[str]
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


RPCOrderMsg = Union[RPCEntryMsg, RPCExitMsg, RPCExitCancelMsg, RPCCancelMsg]


RPCSendMsg = Union[
    RPCStatusMsg,
    RPCStrategyMsg,
    RPCProtectionMsg,
    RPCWhitelistMsg,
    RPCEntryMsg,
    RPCCancelMsg,
    RPCExitMsg,
    RPCExitCancelMsg,
    RPCAnalyzedDFMsg,
    RPCNewCandleMsg
    ]
