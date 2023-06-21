import math
import traceback
from datetime import datetime

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_sortino, calculate_underwater
from freqtrade.optimize.hyperopt import IHyperOptLoss


"""
ProfitSquare with Sortino-Spring Efficiency (PSSE) is a reward function for optimizing
financial portfolios. It aims to maximize profits while accounting for downside risk and
drawdowns. PSSE combines three components:

1. Quadratic Profit: Represents the square of profits, exponentially emphasizing higher
   gains. It allows portfolios to explore a broader space of investment opportunities.

2. Sortino Ratio: A risk-adjustment metric similar to the Sharpe ratio but focuses on
   downside risk. It serves as a navigation system that guides the portfolio through risky
   terrains by measuring the performance relative to the downside.

3. Hybrid Drawdown Penalty: A combination of geometric and harmonic drawdown penalties.
   It acts like a damping system, ensuring the portfolio is robust enough to withstand
   market downturns but also flexible to take advantage of positive trends.

The PSSE formula is:
PSSE = profit^2 + (profit * Sortino ratio) / hybrid drawdown penalty

PSSE integrates mathematical rigor with empirical validation. It resonates with theories
like the Efficient Market Hypothesis and Behavioral Finance, and represents a self-adaptive,
antifragile approach to investment strategy. PSSE acknowledges the complexity of the markets,
striving for a balance between risk and reward.
"""


def calculate_average_drawdown(drawdown_df):
    """
    Calculate average drawdown based on a DataFrame containing drawdown information.

    Parameters:
    - drawdown_df: DataFrame containing columns 'high_value', 'cumulative' and
      'drawdown_relative'

    Returns:
    - Tuple of average absolute drawdown and average relative drawdown
    """
    # Copy input DataFrame to avoid modifying the original
    dd_df = drawdown_df.copy()

    # Calculate drawdowns
    dd_df['calculated_drawdowns'] = dd_df['high_value'] - dd_df['cumulative']

    # Calculate the average drawdown (absolute)
    average_drawdown = dd_df['calculated_drawdowns'].mean()

    # Calculate the average drawdown (relative)
    average_drawdown_relative = dd_df['drawdown_relative'].mean()

    return average_drawdown, average_drawdown_relative


class ProfitSquareSortinoSpringEfficiencyHyperOptLoss(IHyperOptLoss):
    """
    Custom loss function for hyperopt optimization combining quadratic profits, Sortino ratio,
    and hybrid drawdown penalty. The loss function is designed to optimize the trading strategies
    by prioritizing higher profits and lower drawdowns, while adjusting for risk through Sortino
    ratio.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Config, *args, **kwargs) -> float:
        """
        Calculate the hyperopt loss based on trade results.

        Parameters:
        - results: DataFrame containing the trading results
        - trade_count: int representing the number of trades
        - min_date: datetime object representing the minimum date in the dataset
        - max_date: datetime object representing the maximum date in the dataset
        - config: Config object containing the configuration

        Returns:
        - The calculated loss as a float
        """

        try:
            # Total profit
            profit = results['profit_abs'].sum()

            # Sortino ratio
            sortino_ratio = calculate_sortino(
                results, min_date, max_date, config['dry_run_wallet'])

            # Underwater DataFrame (max value to date, drawdown, drawdown relative)
            drawdown_df = calculate_underwater(
                results,
                value_col='profit_abs',
                starting_balance=config['dry_run_wallet']
            )

            # Calculate Max Drawdown
            max_drawdown = abs(min(drawdown_df['drawdown']))

            # Calculate Average Drawdown (absolute and relative)
            avg_drawdown, _ = calculate_average_drawdown(drawdown_df)

            # PSSE Loss Function
            if profit > 0:
                # Sortino ratio requires at least 2 trades for meaningful results
                if trade_count < 2:
                    sortino_ratio = 1

                # Geometric mean of drawdown penalties
                geometric_drawdown = math.sqrt((1 + max_drawdown) * (1 + avg_drawdown))

                # Harmonic mean of drawdown penalties
                harmonic_drawdown = (2 / ((1 / (1 + max_drawdown)) + (1 / (1 + avg_drawdown))))

                # Hybrid drawdown penalty (geometric mean combined with harmonic mean)
                hybrid_drawdown = math.sqrt(
                    geometric_drawdown * harmonic_drawdown)

                # Combine quadratic profit with Sortino ratio and hybrid drawdown penalty
                loss = -1 * ((profit ** 2) + (profit * sortino_ratio) / hybrid_drawdown)
            else:  # profit < 0
                loss = -profit

            return loss

        # Handle any exception that may occur during loss calculation
        except (Exception, ValueError):
            return -profit
