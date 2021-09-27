"""
CalmarHyperOptLossDaily

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
from datetime import datetime
from math import sqrt as msqrt
from typing import Any, Dict

from pandas import DataFrame, date_range

from freqtrade.optimize.hyperopt import IHyperOptLoss


class CalmarHyperOptLossDaily(IHyperOptLoss):
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
        resample_freq = "1D"
        slippage_per_trade_ratio = 0.0005
        days_in_year = 365

        # create the index within the min_date and end max_date
        t_index = date_range(
            start=min_date, end=max_date, freq=resample_freq, normalize=True
        )

        # apply slippage per trade to profit_total
        results.loc[:, "profit_ratio_after_slippage"] = (
            results["profit_ratio"] - slippage_per_trade_ratio
        )

        sum_daily = (
            results.resample(resample_freq, on="close_date")
            .agg({"profit_ratio_after_slippage": sum})
            .reindex(t_index)
            .fillna(0)
        )

        total_profit = sum_daily["profit_ratio_after_slippage"]
        expected_returns_mean = total_profit.mean() * 100

        # calculate max drawdown
        try:
            high_val = total_profit.max()
            low_val = total_profit.min()
            max_drawdown = (high_val - low_val) / high_val

        except (ValueError, ZeroDivisionError):
            max_drawdown = 0

        if max_drawdown != 0:
            calmar_ratio = expected_returns_mean / max_drawdown * msqrt(days_in_year)
        else:
            # Define high (negative) calmar ratio to be clear that this is NOT optimal.
            calmar_ratio = -20.0

        # print(t_index, sum_daily, total_profit)
        # print(expected_returns_mean, max_drawdown, calmar_ratio)
        return -calmar_ratio
