"""
OmegaHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.

Possible to change `OMEGA_THRESHOLD` (the daily return that counts as neither
a gain nor a loss) for individual needs.
"""

from datetime import datetime

from pandas import DataFrame, date_range

from freqtrade.optimize.hyperopt import IHyperOptLoss


# Daily return threshold L of the Omega ratio. 0.0 means every losing day is a loss.
OMEGA_THRESHOLD = 0.0


class OmegaHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation uses the Omega ratio (Keating and Shadwick, 2002) of the
    daily returns of the account balance.
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        min_date: datetime,
        max_date: datetime,
        starting_balance: float,
        *args,
        **kwargs,
    ) -> float:
        """
        Objective function, returns smaller number for more optimal results.

        Omega(L) = sum(max(r - L, 0)) / sum(max(L - r, 0)) over the daily returns r
        of the account balance, where L is `OMEGA_THRESHOLD`. Unlike the Sharpe and
        Sortino ratios it uses the whole return distribution instead of its first two
        moments, so fat tails and skew are priced in.
        """
        if len(results) == 0 or min_date == max_date:
            return 0.0

        t_index = date_range(start=min_date, end=max_date, freq="1D", normalize=True)
        daily_profit = (
            results.resample("1D", on="close_date")["profit_abs"].sum().reindex(t_index).fillna(0)
        )
        # Balance at the start of each day, so a day's return is relative to the
        # account it was earned on.
        balance = starting_balance + daily_profit.cumsum().shift(1, fill_value=0.0)
        if (balance <= 0).any():
            # The account was wiped out: as bad as it gets.
            return 100.0

        excess = daily_profit / balance - OMEGA_THRESHOLD
        gains = excess.clip(lower=0).sum()
        losses = (-excess).clip(lower=0).sum()

        if losses == 0:
            # No day below the threshold leaves the ratio undefined. Like the Sortino
            # losses with a zero downside deviation, treat it as NOT optimal.
            return 100.0

        return -float(gains / losses)
