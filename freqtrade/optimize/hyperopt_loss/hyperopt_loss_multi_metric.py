"""
MultiMetricHyperOptLoss

This module defines the alternative HyperOptLoss class based on:
  - Profit
  - Drawdown
  - Profit Factor
  - Expectancy Ratio
  - Winrate
  - Amount of trades

Possible to change:
  - `DRAWDOWN_MULT` to penalize drawdown objective for individual needs;
  - `TARGET_TRADE_AMOUNT` to adjust amount of trades impact.
  - `EXPECTANCY_CONST` to adjust expectancy ratio impact.
  - `PF_CONST` to adjust profit factor impact.
  - `WINRATE_CONST` to adjust winrate impact.


DRAWDOWN_MULT variable within the hyperoptloss file can be adjusted to be stricter or more
              flexible on drawdown purposes. Smaller numbers penalize drawdowns more severely.
PF_CONST variable adjusts the impact of the Profit Factor on the optimization.
EXPECTANCY_CONST variable controls the influence of the Expectancy Ratio.
WINRATE_CONST variable can be adjusted to increase or decrease impact of winrate.

PF_CONST, EXPECTANCY_CONST, WINRATE_CONST all operate in a similar manner:
        a higher value means that the metric has a lesser impact on the objective,
        while a lower value means that it has a greater impact.
TARGET_TRADE_AMOUNT variable sets the minimum number of trades required to avoid penalties.
            If the trade amount falls below this threshold, the penalty is applied.
"""

import numpy as np
from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_expectancy, calculate_max_drawdown
from freqtrade.optimize.hyperopt import IHyperOptLoss


# smaller numbers penalize drawdowns more severely
DRAWDOWN_MULT = 0.055
# A very large number to use as a replacement for infinity
LARGE_NUMBER = 1e6
# Target trade amount, if higher that TARGET_TRADE_AMOUNT - no penalty
TARGET_TRADE_AMOUNT = 50
# Coefficient to adjust impact of expectancy
EXPECTANCY_CONST = 2.0
# Coefficient to adjust profit factor impact
PF_CONST = 1.0
# Coefficient to adjust winrate impact
WINRATE_CONST = 1.2


class MultiMetricHyperOptLoss(IHyperOptLoss):
    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        config: Config,
        **kwargs,
    ) -> float:
        total_profit = results["profit_abs"].sum()

        # Calculate profit factor
        winning_profit = results.loc[results["profit_abs"] > 0, "profit_abs"].sum()
        losing_profit = results.loc[results["profit_abs"] < 0, "profit_abs"].sum()
        profit_factor = winning_profit / (abs(losing_profit) + 1e-6)
        log_profit_factor = np.log(profit_factor + PF_CONST)

        # Calculate expectancy
        expectancy, expectancy_ratio = calculate_expectancy(results)
        if expectancy_ratio > 10:
            log_expectancy_ratio = np.log(1.01)
        else:
            log_expectancy_ratio = np.log(expectancy_ratio + EXPECTANCY_CONST)

        # Calculate winrate
        winning_trades = results.loc[results["profit_abs"] > 0]
        winrate = len(winning_trades) / len(results)
        log_winrate_coef = np.log(WINRATE_CONST + winrate)

        # Calculate drawdown
        try:
            drawdown = calculate_max_drawdown(
                results, starting_balance=config["dry_run_wallet"], value_col="profit_abs"
            )
            relative_account_drawdown = drawdown.relative_account_drawdown
        except ValueError:
            relative_account_drawdown = 0

        # Trade Count Penalty
        trade_count_penalty = 1.0  # Default: no penalty
        if trade_count < TARGET_TRADE_AMOUNT:
            trade_count_penalty = 1 - (abs(trade_count - TARGET_TRADE_AMOUNT) / TARGET_TRADE_AMOUNT)
            trade_count_penalty = max(trade_count_penalty, 0.1)

        profit_draw_function = total_profit - (relative_account_drawdown * total_profit) * (
            1 - DRAWDOWN_MULT
        )

        return -1 * (
            profit_draw_function
            * log_profit_factor
            * log_expectancy_ratio
            * log_winrate_coef
            * trade_count_penalty
        )
