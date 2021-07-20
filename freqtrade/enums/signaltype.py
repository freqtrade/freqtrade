from enum import Enum


class SignalType(Enum):
    """
    Enum to distinguish between buy and sell signals
    """
    BUY = "buy"
    SELL = "sell"


class SignalNameType(Enum):
    """
    Enum for signal name
    """
    BUY_SIGNAL_NAME = "buy_signal_name"
