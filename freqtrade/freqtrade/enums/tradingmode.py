from enum import Enum


class TradingMode(str, Enum):
    """
    Enum to distinguish between
    spot, margin, futures or any other trading method
    """

    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"

    def __str__(self):
        return f"{self.name.lower()}"
