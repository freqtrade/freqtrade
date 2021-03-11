from enum import Enum


class BacktestState(Enum):
    """
    Bot application states
    """
    STARTUP = 1
    DATALOAD = 2
    ANALYZE = 3
    BACKTEST = 4

    def __str__(self):
        return f"{self.name.lower()}"
