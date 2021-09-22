"""
CalmarHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
from datetime import datetime

import numpy as np
from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss
from freqtrade.data.btanalysis import calculate_max_drawdown


class CalmarHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation uses the Calmar Ratio calculation.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for more optimal results.

        Uses Calmar Ratio calculation.
        """
        total_profit = results["profit_ratio"]
        days_period = (max_date - min_date).days

        # adding slippage of 0.1% per trade
        total_profit = total_profit - 0.0005
        expected_returns_mean = total_profit.sum() / days_period

        # calculate max drawdown
        try:
            _, _, _, high_val, low_val = calculate_max_drawdown(results)
            max_drawdown = -(high_val - low_val) / high_val
        except ValueError:
            max_drawdown = 0

        if max_drawdown > 0:
            calmar_ratio = expected_returns_mean / max_drawdown * np.sqrt(365)
        else:
            calmar_ratio = -20.

        # print(calmar_ratio)
        return -calmar_ratio
