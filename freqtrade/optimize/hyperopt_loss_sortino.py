"""
SortinoHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
from datetime import datetime

from pandas import DataFrame
import numpy as np

from freqtrade.optimize.hyperopt import IHyperOptLoss


class SortinoHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation uses the Sortino Ratio calculation.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for more optimal results.

        Uses Sortino Ratio calculation.
        """
        total_profit = results["profit_percent"]
        days_period = (max_date - min_date).days

        # adding slippage of 0.1% per trade
        total_profit = total_profit - 0.0005
        expected_returns_mean = total_profit.sum() / days_period

        results['downside_returns'] = 0
        results.loc[total_profit < 0, 'downside_returns'] = results['profit_percent']
        down_stdev = np.std(results['downside_returns'])

        if down_stdev != 0:
            sortino_ratio = expected_returns_mean / down_stdev * np.sqrt(365)
        else:
            # Define high (negative) sortino ratio to be clear that this is NOT optimal.
            sortino_ratio = -20.

        # print(expected_returns_mean, down_stdev, sortino_ratio)
        return -sortino_ratio
