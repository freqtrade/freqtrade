from enum import StrEnum


class TradingMode(StrEnum):
    """
    Enum to distinguish between
    spot, margin, futures or any other trading method
    """

    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"
