"""
CalmarHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
from datetime import datetime
from math import sqrt as msqrt
from typing import Any, Dict

from pandas import DataFrame

from freqtrade.data.btanalysis import calculate_max_drawdown
from freqtrade.optimize.hyperopt import IHyperOptLoss


class CalmarHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation uses the Calmar Ratio calculation.
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Dict,
        processed: Dict[str, DataFrame],
        backtest_stats: Dict[str, Any],
        *args,
        **kwargs
    ) -> float:
        """
        Objective function, returns smaller number for more optimal results.

        Uses Calmar Ratio calculation.
        """
        total_profit = backtest_stats["profit_total"]
        days_period = (max_date - min_date).days

        # adding slippage of 0.1% per trade
        total_profit = total_profit - 0.0005
        expected_returns_mean = total_profit.sum() / days_period * 100

        # calculate max drawdown
        try:
            _, _, _, high_val, low_val = calculate_max_drawdown(
                results, value_col="profit_abs"
            )
            max_drawdown = (high_val - low_val) / high_val
        except ValueError:
            max_drawdown = 0

        if max_drawdown != 0:
            calmar_ratio = expected_returns_mean / max_drawdown * msqrt(365)
        else:
            # Define high (negative) calmar ratio to be clear that this is NOT optimal.
            calmar_ratio = -20.0

        # print(expected_returns_mean, max_drawdown, calmar_ratio)
        return -calmar_ratio
