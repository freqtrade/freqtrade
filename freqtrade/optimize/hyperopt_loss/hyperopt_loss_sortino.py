"""
SortinoHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""

from datetime import datetime

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_sortino
from freqtrade.optimize.hyperopt import IHyperOptLoss


class SortinoHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation uses the Sortino Ratio calculation.
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        *args,
        **kwargs,
    ) -> float:
        """
        Objective function, returns smaller number for more optimal results.

        Uses Sortino Ratio calculation.
        """
        starting_balance = config["dry_run_wallet"]
        sortino_ratio = calculate_sortino(results, min_date, max_date, starting_balance)
        # print(expected_returns_mean, down_stdev, sortino_ratio)
        return -sortino_ratio
