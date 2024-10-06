# Used for list-exchanges
from typing import Optional

from typing_extensions import TypedDict


class TradeModeType(TypedDict):
    trading_mode: str
    margin_mode: str


class ValidExchangesType(TypedDict):
    name: str
    classname: str
    valid: bool
    supported: bool
    comment: str
    dex: bool
    is_alias: bool
    alias_for: Optional[str]
    trade_modes: list[TradeModeType]
