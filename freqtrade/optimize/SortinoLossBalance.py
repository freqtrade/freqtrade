"""
SortinoHyperOptLoss
This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
import logging
import os
from datetime import datetime

import numpy as np
from pandas import DataFrame, Timedelta

from freqtrade.data.btanalysis import calculate_outstanding_balance
from freqtrade.optimize.hyperopt import IHyperOptLoss


logger = logging.getLogger(__name__)

interval = os.getenv("FQT_TIMEFRAME") or "5m"
slippage = 0.0005
target = 0
annualize = np.sqrt(365 * (Timedelta("1D") / Timedelta(interval)))

logger.info(f"SortinoLossBalance target is set to: {target}")


class SortinoLossBalance(IHyperOptLoss):
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
        *args,
        **kwargs,
    ) -> float:
        """
        Objective function, returns smaller number for more optimal results.
        Uses Sortino Ratio calculation.
        """
        hloc = kwargs["processed"]
        timeframe = SortinoLossBalance.timeframe

        balance_total = calculate_outstanding_balance(results, timeframe, hloc)

        returns = balance_total.mean()
        # returns = balance_total.values.mean()

        downside_returns = np.where(balance_total < 0, balance_total, 0)
        downside_risk = np.sqrt((downside_returns ** 2).sum() / len(hloc))

        if downside_risk != 0.0:
            sortino_ratio = (returns - target) / downside_risk * annualize
        else:
            sortino_ratio = -np.iinfo(np.int32).max

        # print(expected_returns_mean, down_stdev, sortino_ratio)
        return -sortino_ratio
