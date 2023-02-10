from enum import Enum


class PriceType(str, Enum):
    """Enum to distinguish possible trigger prices for stoplosses"""
    LAST = "last"
    MARK = "mark"
    INDEX = "index"
