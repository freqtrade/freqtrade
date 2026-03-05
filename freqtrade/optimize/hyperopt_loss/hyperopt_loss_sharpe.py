"""
SharpeHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""

from datetime import datetime

from pandas import DataFrame

from freqtrade.data.metrics import calculate_sharpe
from freqtrade.optimize.hyperopt import IHyperOptLoss


class SharpeHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation uses the Sharpe Ratio calculation.
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        min_date: datetime,
        max_date: datetime,
        starting_balance: float,
        *args,
        **kwargs,
    ) -> float:
        """
        Objective function, returns smaller number for more optimal results.

        Uses Sharpe Ratio calculation.
        """
        sharpe_ratio = calculate_sharpe(results, min_date, max_date, starting_balance)
        return -sharpe_ratio
