"""
MaxDrawDownPerPairHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""

from typing import Any

from freqtrade.optimize.hyperopt import IHyperOptLoss


class MaxDrawDownPerPairHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation calculates the profit/drawdown ratio per pair and
    returns the worst result as objective, forcing hyperopt to optimize
    the parameters for all pairs in the pairlist.

    This way, we prevent one or more pairs with good results from inflating
    the metrics, while the rest of the pairs with poor results are not
    represented and therefore not optimized.
    """

    @staticmethod
    def hyperopt_loss_function(backtest_stats: dict[str, Any], *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for better results.
        """

        ##############################################
        # Configurable parameters
        ##############################################
        # Minimum acceptable profit/drawdown per pair
        min_acceptable_profit_dd = 1.0
        # Penalty when acceptable minimum are not met
        penalty = 20
        ##############################################

        score_per_pair = []
        for p in backtest_stats["results_per_pair"]:
            if p["key"] != "TOTAL":
                profit = p.get("profit_total_abs", 0)
                drawdown = p.get("max_drawdown_abs", 0)

                if drawdown != 0 and profit != 0:
                    profit_dd = profit / drawdown
                else:
                    profit_dd = profit

                if profit_dd < min_acceptable_profit_dd:
                    score = profit_dd - penalty
                else:
                    score = profit_dd

                score_per_pair.append(score)

        return -min(score_per_pair)
