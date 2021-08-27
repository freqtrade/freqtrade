from enum import Enum


class MarginMode(Enum):
    """
        Enum to distinguish between
        one-way mode or hedge mode in Futures (Cross and Isolated) or Margin Trading
    """
    ONE_WAY = "one-way"
    HEDGE = "hedge"
