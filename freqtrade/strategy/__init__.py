# flake8: noqa: F401
from freqtrade.exchange import (
    timeframe_to_minutes,
    timeframe_to_msecs,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    timeframe_to_seconds,
)
from freqtrade.persistence import Order, PairLocks, Trade
from freqtrade.strategy.informative_decorator import informative
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.parameters import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
)
from freqtrade.strategy.strategy_helper import (
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)


# Imports to be used for `from freqtrade.strategy import *`
__all__ = [
    "IStrategy",
    "Trade",
    "Order",
    "PairLocks",
    "informative",
    # Parameters
    "BooleanParameter",
    "CategoricalParameter",
    "DecimalParameter",
    "IntParameter",
    "RealParameter",
    # timeframe helpers
    "timeframe_to_minutes",
    "timeframe_to_next_date",
    "timeframe_to_prev_date",
    # Strategy helper functions
    "merge_informative_pair",
    "stoploss_from_absolute",
    "stoploss_from_open",
]
