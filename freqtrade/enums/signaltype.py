from enum import StrEnum


class SignalType(StrEnum):
    """
    Enum to distinguish between enter and exit signals
    """

    ENTER_LONG = "enter_long"
    EXIT_LONG = "exit_long"
    ENTER_SHORT = "enter_short"
    EXIT_SHORT = "exit_short"


class SignalTagType(StrEnum):
    """
    Enum for signal columns
    """

    ENTER_TAG = "enter_tag"
    EXIT_TAG = "exit_tag"


class SignalDirection(StrEnum):
    LONG = "long"
    SHORT = "short"
