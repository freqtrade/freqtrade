from datetime import datetime
from typing import Optional, TypedDict, Union

from freqtrade.enums import RPCMessageType


class RPCSendMsgBase(TypedDict):
    type: RPCMessageType


class RPCStatusMsg(RPCSendMsgBase):
    """Used for Status, Startup and Warning messages"""
    status: str


class RPCProtectionMsg(RPCSendMsgBase):
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


class RPCBuyMsg(RPCSendMsgBase):
    trade_id: str
    buy_tag: str
    enter_tag: str
    exchange: str
    pair: str
    leverage: float
    direction: str
    limit: float
    open_rate: float
    order_type: Optional[str]  # TODO: why optional??
    stake_amount: float
    stake_currency: str
    fiat_currency: Optional[str]
    amount: float
    open_date: datetime
    current_rate: float
    sub_trade: bool


class RPCCancelMsg(RPCBuyMsg):
    reason: str


class RPCSellMsg(RPCBuyMsg):
    cumulative_profit: float
    gain: str  # Literal["profit", "loss"]
    close_rate: float
    profit_amount: float
    profit_ratio: float
    sell_reason: str
    exit_reason: str
    close_date: datetime
    current_rate: Optional[float]


class RPCSellCancelMsg(RPCBuyMsg):
    reason: str
    gain: str  # Literal["profit", "loss"]
    profit_amount: float
    profit_ratio: float
    sell_reason: str
    exit_reason: str
    close_date: datetime


RPCSendMsg = Union[
    RPCStatusMsg,
    RPCProtectionMsg,
    RPCBuyMsg,
    RPCCancelMsg,
    RPCSellMsg,
    RPCSellCancelMsg
    ]
