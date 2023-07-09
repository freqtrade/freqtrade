from dataclasses import dataclass
from typing import Optional
from strenum import StrEnum

class SignalType(StrEnum):
    """
    Enum to distinguish between enter and exit signals
    """
    ENTER_LONG = "enter_long"
    EXIT_LONG = "exit_long"
    ENTER_SHORT = "enter_short"
    EXIT_SHORT = "exit_short"
    TP_PRICE = "tp_price"
    SL_PRICE = "sl_price"


class SignalTagType(StrEnum):
    """
    Enum for signal columns
    """
    ENTER_TAG = "enter_tag"
    EXIT_TAG = "exit_tag"


class SignalDirection(StrEnum):
    LONG = 'long'
    SHORT = 'short'

@dataclass(frozen=True)
class EntrySignal:
    """
    Class to hold an entry signal and its associated metadata
    """
    direction: SignalDirection
    tag: str
    TP_price: Optional[float] = None
    SL_price: Optional[float] = None
    ttl_ms: Optional[int] = None  # Time to live in milliseconds
