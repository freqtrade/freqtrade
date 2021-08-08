from enum import Enum


class SignalType(Enum):
    """
    Enum to distinguish between buy and sell signals
    """
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    EXIT_SHORT = "exit_short"


class SignalTagType(Enum):
    """
    Enum for signal columns
    """
    BUY_TAG = "buy_tag"
    SHORT_TAG = "short_tag"
