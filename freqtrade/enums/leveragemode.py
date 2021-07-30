from enum import Enum


class LeverageMode(Enum):
    """
    Enum to distinguish between cross margin, isolated margin, and futures
    """
    CROSS = "cross"
    ISOLATED = "isolated"
    CROSS_FUTURES = "cross_futures"
    ISOLATED_FUTURES = "cross_futures"
