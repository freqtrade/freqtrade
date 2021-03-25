"""
SortinoHyperOptLoss
This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
import logging
from datetime import datetime
from typing import Dict

import numpy as np
from pandas import DataFrame, Timedelta

from freqtrade.data.btanalysis import calculate_outstanding_balance
from freqtrade.optimize.hyperopt import IHyperOptLoss


logger = logging.getLogger(__name__)

target = 0
logger.info(f"SortinoLossBalance target is set to: {target}")


class SortinoLossBalance(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.
    This implementation uses the Sortino Ratio calculation.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for more optimal results.
        Uses Sortino Ratio calculation.
        """
        timeframe = SortinoLossBalance.timeframe
        annualize = np.sqrt(365 * (Timedelta("1D") / Timedelta(timeframe)))

        balance_total = calculate_outstanding_balance(results, timeframe, processed)

        returns = balance_total.mean()
        # returns = balance_total.values.mean()

        downside_returns = np.where(balance_total < 0, balance_total, 0)
        downside_risk = np.sqrt((downside_returns ** 2).sum() / len(processed))

        if downside_risk != 0.0:
            sortino_ratio = (returns - target) / downside_risk * annualize
        else:
            sortino_ratio = -np.iinfo(np.int32).max

        # print(expected_returns_mean, down_stdev, sortino_ratio)
        return -sortino_ratio
