from enum import StrEnum


class PriceType(StrEnum):
    """Enum to distinguish possible trigger prices for stoplosses"""

    LAST = "last"
    MARK = "mark"
    INDEX = "index"
