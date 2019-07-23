"""
OnlyProfitHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss


# This is assumed to be expected avg profit * expected trade count.
# For example, for 0.35% avg per trade (or 0.0035 as ratio) and 1100 trades,
# expected max profit = 3.85
# Check that the reported Σ% values do not exceed this!
# Note, this is ratio. 3.85 stated above means 385Σ%.
EXPECTED_MAX_PROFIT = 3.0


class OnlyProfitHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation takes only profit into account.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for better results.
        """
        total_profit = results.profit_percent.sum()
        return max(0, 1 - total_profit / EXPECTED_MAX_PROFIT)
