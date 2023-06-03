# Used for list-exchanges
from typing import List

from typing_extensions import TypedDict


class TradeModeType(TypedDict):
    trading_mode: str
    margin_mode: str


class ValidExchangesType(TypedDict):
    name: str
    valid: bool
    supported: bool
    comment: str
    trade_modes: List[TradeModeType]
