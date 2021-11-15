from enum import Enum


class Collateral(Enum):
    """
    Enum to distinguish between
    cross margin/futures collateral and
    isolated margin/futures collateral
    """
    CROSS = "cross"
    ISOLATED = "isolated"
