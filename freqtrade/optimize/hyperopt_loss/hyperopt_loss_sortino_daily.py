"""
SortinoHyperOptLossDaily

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
import math
from datetime import datetime

from pandas import DataFrame, date_range

from freqtrade.optimize.hyperopt import IHyperOptLoss


class SortinoHyperOptLossDaily(IHyperOptLoss):
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

        Sortino Ratio calculated as described in
        http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
        """
        resample_freq = '1D'
        slippage_per_trade_ratio = 0.0005
        days_in_year = 365
        minimum_acceptable_return = 0.0

        # apply slippage per trade to profit_ratio
        results.loc[:, 'profit_ratio_after_slippage'] = \
            results['profit_ratio'] - slippage_per_trade_ratio

        # create the index within the min_date and end max_date
        t_index = date_range(start=min_date, end=max_date, freq=resample_freq,
                             normalize=True)

        sum_daily = (
            results.resample(resample_freq, on='close_date').agg(
                {"profit_ratio_after_slippage": sum}).reindex(t_index).fillna(0)
        )

        total_profit = sum_daily["profit_ratio_after_slippage"] - minimum_acceptable_return
        expected_returns_mean = total_profit.mean()

        sum_daily['downside_returns'] = 0
        sum_daily.loc[total_profit < 0, 'downside_returns'] = total_profit
        total_downside = sum_daily['downside_returns']
        # Here total_downside contains min(0, P - MAR) values,
        # where P = sum_daily["profit_ratio_after_slippage"]
        down_stdev = math.sqrt((total_downside**2).sum() / len(total_downside))

        if down_stdev != 0:
            sortino_ratio = expected_returns_mean / down_stdev * math.sqrt(days_in_year)
        else:
            # Define high (negative) sortino ratio to be clear that this is NOT optimal.
            sortino_ratio = -20.

        # print(t_index, sum_daily, total_profit)
        # print(minimum_acceptable_return, expected_returns_mean, down_stdev, sortino_ratio)
        return -sortino_ratio
