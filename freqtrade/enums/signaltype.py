from enum import Enum


class SignalType(Enum):
    """
    Enum to distinguish between buy and sell signals
    """
    BUY = "buy"   # To be renamed to enter_long
    SELL = "sell"  # To be renamed to exit_long
    SHORT = "short"  # Should be "enter_short"
    EXIT_SHORT = "exit_short"


class SignalTagType(Enum):
    """
    Enum for signal columns
    """
    BUY_TAG = "buy_tag"
    SHORT_TAG = "short_tag"


class SignalDirection(Enum):
    LONG = 'long'
    SHORT = 'short'
