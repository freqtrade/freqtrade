"""
SharpeHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
from datetime import datetime

from pandas import DataFrame
import numpy as np

from freqtrade.optimize.hyperopt import IHyperOptLoss


class SharpeHyperOptLossDaily(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation uses the Sharpe Ratio calculation.
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        *args,
        **kwargs
    ) -> float:
        """
        Objective function, returns smaller number for more optimal results.

        Uses Sharpe Ratio calculation.
        """
        total_profit = results.profit_percent
        # days_period = (max_date - min_date).days

        # adding slippage of 0.1% per trade
        total_profit = total_profit - 0.0005
        # expected_yearly_return = total_profit.sum() / days_period

        # if np.std(total_profit) != 0.0:
        #     sharp_ratio = expected_yearly_return / np.std(total_profit) * np.sqrt(365)
        # else:
        #     # Define high (negative) sharpe ratio to be clear that this is NOT optimal.
        #     sharp_ratio = -20.0

        # # print(expected_yearly_return, np.std(total_profit), sharp_ratio)
        # return -sharp_ratio

        sum_daily = (
            results.resample("D", on="close_time").agg(
                {"profit_percent": sum, "profit_abs": sum}
            )
            * 100.0
        )

        if np.std(total_profit) != 0.0:
            sharp_ratio = (
                sum_daily["profit_percent"].mean()
                / sum_daily["profit_percent"].std()
                * np.sqrt(365)
            )
        else:
            # Define high (negative) sharpe ratio to be clear that this is NOT optimal.
            sharp_ratio = -20.0

        return -sharp_ratio
