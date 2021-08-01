from enum import Enum


class TradingMode(Enum):
    """
    Enum to distinguish between
    spot, cross margin, isolated margin, futures or any other trading method
    """
    SPOT = "spot"
    CROSS_MARGIN = "cross margin"
    ISOLATED_MARGIN = "isolated margin"
    CROSS_FUTURES = "cross futures"
    ISOLATED_FUTURES = "isolated futures"
