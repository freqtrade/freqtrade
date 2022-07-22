from enum import Enum


class SignalType(Enum):
    """
    Enum to distinguish between enter and exit signals
    """
    ENTER_LONG = "enter_long"
    EXIT_LONG = "exit_long"
    ENTER_SHORT = "enter_short"
    EXIT_SHORT = "exit_short"


class SignalTagType(Enum):
    """
    Enum for signal columns
    """
    ENTER_TAG = "enter_tag"
    EXIT_TAG = "exit_tag"


class SignalDirection(str, Enum):
    LONG = 'long'
    SHORT = 'short'
