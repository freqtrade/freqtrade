from enum import Enum


class MarketDirection(Enum):
    """
    Enum for various market directions.
    """
    LONG = "long"
    SHORT = "short"
    EVEN = "even"
    NONE = ''

    @staticmethod
    def string_to_enum(label : str) -> str:
        match label:
            case "long":
                return MarketDirection.LONG
            case "short":
                return MarketDirection.SHORT
            case "even":
                return MarketDirection.EVEN
            case 'none':
                return MarketDirection.NONE
            case _:
                return None

