# pragma pylint: disable=too-few-public-methods

"""
Bot state constant
"""
from enum import Enum


class State(Enum):
    """
    Bot application states
    """
    RUNNING = 1
    STOPPED = 2
    RELOAD_CONF = 3

    def __str__(self):
        return f"{self.name.lower()}"


class RunMode(Enum):
    """
    Bot running mode (backtest, hyperopt, ...)
    can be "live", "dry-run", "backtest", "edge", "hyperopt".
    """
    LIVE = "live"
    DRY_RUN = "dry_run"
    BACKTEST = "backtest"
    EDGE = "edge"
    HYPEROPT = "hyperopt"
    UTIL_EXCHANGE = "util_exchange"
    UTIL_NO_EXCHANGE = "util_no_exchange"
    PLOT = "plot"
    OTHER = "other"


TRADING_MODES = [RunMode.LIVE, RunMode.DRY_RUN]
OPTIMIZE_MODES = [RunMode.BACKTEST, RunMode.EDGE, RunMode.HYPEROPT]
NON_UTIL_MODES = TRADING_MODES + OPTIMIZE_MODES
