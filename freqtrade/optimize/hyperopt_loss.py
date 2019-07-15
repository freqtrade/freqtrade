from datetime import datetime
from math import exp

import numpy as np
from pandas import DataFrame

# Define some constants:

# set TARGET_TRADES to suit your number concurrent trades so its realistic
# to the number of days
TARGET_TRADES = 600
# This is assumed to be expected avg profit * expected trade count.
# For example, for 0.35% avg per trade (or 0.0035 as ratio) and 1100 trades,
# self.expected_max_profit = 3.85
# Check that the reported Σ% values do not exceed this!
# Note, this is ratio. 3.85 stated above means 385Σ%.
EXPECTED_MAX_PROFIT = 3.0

# max average trade duration in minutes
# if eval ends with higher value, we consider it a failed eval
MAX_ACCEPTED_TRADE_DURATION = 300


def hyperopt_loss_legacy(results: DataFrame, trade_count: int,
                         *args, **kwargs) -> float:
    """
    Objective function, returns smaller number for better results
    This is the legacy algorithm (used until now in freqtrade).
    Weights are distributed as follows:
    * 0.4 to trade duration
    * 0.25: Avoiding trade loss
    """
    total_profit = results.profit_percent.sum()
    trade_duration = results.trade_duration.mean()

    trade_loss = 1 - 0.25 * exp(-(trade_count - TARGET_TRADES) ** 2 / 10 ** 5.8)
    profit_loss = max(0, 1 - total_profit / EXPECTED_MAX_PROFIT)
    duration_loss = 0.4 * min(trade_duration / MAX_ACCEPTED_TRADE_DURATION, 1)
    result = trade_loss + profit_loss + duration_loss
    return result


def hyperopt_loss_sharpe(results: DataFrame, trade_count: int,
                         min_date: datetime, max_date: datetime, *args, **kwargs) -> float:
    """
    Objective function, returns smaller number for more optimal results
    Using sharpe ratio calculation
    """
    total_profit = results.profit_percent
    days_period = (max_date - min_date).days

    # adding slippage of 0.1% per trade
    total_profit = total_profit - 0.0005
    expected_yearly_return = total_profit.sum() / days_period

    if (np.std(total_profit) != 0.):
        sharp_ratio = expected_yearly_return / np.std(total_profit) * np.sqrt(365)
    else:
        sharp_ratio = 1.

    # print(expected_yearly_return, np.std(total_profit), sharp_ratio)

    # Negate sharp-ratio so lower is better (??)
    return -sharp_ratio
