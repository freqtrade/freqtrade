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
#
# Note, this is ratio. 3.85 stated above means 385Σ%, 3.0 means 300Σ%.
#
# In this implementation it's only used in calculation of the resulting value
# of the objective function as a normalization coefficient and does not
# represent any limit for profits as in the Freqtrade legacy default loss function.
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
        return 1 - total_profit / EXPECTED_MAX_PROFIT
