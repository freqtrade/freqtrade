# Used for list-exchanges

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
    comment_futures: str
    dex: bool
    is_alias: bool
    alias_for: str | None
    trade_modes: list[TradeModeType]
