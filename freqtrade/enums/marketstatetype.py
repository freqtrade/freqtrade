from enum import Enum


class MarketDirection(Enum):
    """
    Enum for various market directions.
    """
    LONG = "long"
    SHORT = "short"
    EVEN = "even"
    NONE = ''
