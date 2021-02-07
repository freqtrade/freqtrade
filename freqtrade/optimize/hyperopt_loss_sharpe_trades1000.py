"""
SharpeHyperOptLossTrades

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.

The MINIMUM_TRADES and SLIPPAGE_PER_TRADE_RATIO can be altered to whatever you like.
The values that make up the maximum trade_grade can be altered as well.
"""
import math
from datetime import datetime

from pandas import DataFrame, date_range

from freqtrade.optimize.hyperopt import IHyperOptLoss


class SharpeHyperOptLossTrades1000(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation uses the Sharpe Ratio Daily calculation and the Trade Grade calculation.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for more optimal results.

        Uses the Sharpe Ratio Daily calculation and the Trade Grade calculation.
        """

        # CONSTANTS
        MINIMUM_TRADES = 1000
        SLIPPAGE_PER_TRADE_RATIO = 0.001
        NUMERATOR_MAX_TRADEGRADE = 80
        DENOMINATOR_MAX_TRADEGRADE = 8
        RESAMPLE_FREQ = '1D'
        DAYS_IN_YEAR = 365
        ANNUAL_RISK_FREE_RATE = 0.0

        risk_free_rate = ANNUAL_RISK_FREE_RATE / DAYS_IN_YEAR

        """
        Sharpe Ratio Calculation
        """
        # apply slippage per trade to profit_percent
        results.loc[:, 'profit_percent_after_slippage'] = \
            results['profit_percent'] - SLIPPAGE_PER_TRADE_RATIO

        # create the index within the min_date and end max_date
        t_index = date_range(start=min_date, end=max_date, freq=RESAMPLE_FREQ,
                             normalize=True)

        sum_daily = (
            results.resample(RESAMPLE_FREQ, on='close_time').agg(
                {"profit_percent_after_slippage": sum}).reindex(t_index).fillna(0)
        )

        total_profit = sum_daily["profit_percent_after_slippage"] - risk_free_rate
        expected_returns_mean = total_profit.mean()
        up_stdev = total_profit.std()

        if up_stdev != 0:
            sharp_ratio = expected_returns_mean / up_stdev * math.sqrt(DAYS_IN_YEAR)
        else:
            # Define high (negative) sharpe ratio to be clear that this is NOT optimal.
            sharp_ratio = -30.

        """
        Trade Grade Calculation
        This function has a maximum grade of 80/DENOMINATOR_MAX_TRADEGRADE.
        A minimum of 1005 trades.
        """

        if trade_count <= (MINIMUM_TRADES + 5):
            # Define high (negative) trade grade tp be clear that this is NOT optimal
            trade_grade = -30
        else:
            trade_grade = ((1 / (-0.001 * (trade_count - MINIMUM_TRADES))) +
                           NUMERATOR_MAX_TRADEGRADE) / DENOMINATOR_MAX_TRADEGRADE

        return -(sharp_ratio + trade_grade)
