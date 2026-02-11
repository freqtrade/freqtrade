from enum import StrEnum


class MarginMode(StrEnum):
    """
    Enum to distinguish between
    cross margin/futures margin_mode and
    isolated margin/futures margin_mode
    """

    CROSS = "cross"
    ISOLATED = "isolated"
    NONE = ""
