"""
MaxDrawDownRelativeHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_underwater
from freqtrade.optimize.hyperopt import IHyperOptLoss


class MaxDrawDownRelativeHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation optimizes for max draw down and profit
    Less max drawdown more profit -> Lower return value
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, config: Config, *args, **kwargs) -> float:
        """
        Objective function.

        Uses profit ratio weighted max_drawdown when drawdown is available.
        Otherwise directly optimizes profit ratio.
        """
        total_profit = results["profit_abs"].sum()
        try:
            drawdown_df = calculate_underwater(
                results, value_col="profit_abs", starting_balance=config["dry_run_wallet"]
            )
            max_drawdown = abs(min(drawdown_df["drawdown"]))
            relative_drawdown = max(drawdown_df["drawdown_relative"])
            if max_drawdown == 0:
                return -total_profit
            return -total_profit / max_drawdown / relative_drawdown
        except (Exception, ValueError):
            return -total_profit
