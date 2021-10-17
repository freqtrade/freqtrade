"""
OnlyProfitHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss


class OnlyProfitHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation takes only absolute profit into account, not looking at any other indicator.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for better results.
        """
        total_profit = results['profit_abs'].sum()
        return -1 * total_profit
