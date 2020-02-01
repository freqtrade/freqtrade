"""
SharpeHyperOptLossDaily

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
        # get profit_percent and apply slippage of 0.1% per trade
        results.loc[:, 'profit_percent'] = results['profit_percent'] - 0.0005

        sum_daily = (
            results.resample("D", on="close_time").agg(
                {"profit_percent": sum}
            )
            * 100.0
        )

        if (np.std(sum_daily.profit_percent) != 0.):
            sharp_ratio = (
                sum_daily["profit_percent"].mean()
                / np.std(sum_daily["profit_percent"])
                * np.sqrt(365)
            )
        else:
            # Define high (negative) sharpe ratio to be clear that this is NOT optimal.
            sharp_ratio = -20.0

        return -sharp_ratio
