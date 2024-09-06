from enum import Enum


class MarginMode(str, Enum):
    """
    Enum to distinguish between
    cross margin/futures margin_mode and
    isolated margin/futures margin_mode
    """

    CROSS = "cross"
    ISOLATED = "isolated"
    NONE = ""

    def __str__(self):
        return f"{self.name.lower()}"
