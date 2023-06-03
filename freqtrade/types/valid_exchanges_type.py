# Used for list-exchanges
from typing import List, TypedDict


class ValidExchangesType(TypedDict):
    name: str
    valid: bool
    supported: bool
    comment: str
    trade_modes: List[str]
