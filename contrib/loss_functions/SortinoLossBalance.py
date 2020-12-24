"""
SortinoHyperOptLoss
This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
import logging
import os
from datetime import datetime

import numpy as np
from pandas import DataFrame, DatetimeIndex, Timedelta, date_range
from scipy.ndimage.interpolation import shift

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
        timeframe = SortinoLossBalance.ticker_interval
        timedelta = Timedelta(timeframe)

        date_index: DatetimeIndex = date_range(
            start=min_date, end=max_date, freq=timeframe, normalize=True
        )
        balance_total: np.ndarray = []
        for pair in hloc:
            pair_candles = hloc[pair].set_index("date").reindex(date_index)
            # index becomes open_time
            pair_trades = (
                results.loc[results["pair"].values == pair]
                .set_index("open_time")
                .resample(timeframe)
                .asfreq()
                .reindex(date_index)
            )
            open_rate = pair_trades["open_rate"].fillna(0).values
            open_time = pair_trades.index.values
            close_time = pair_trades["close_time"].values
            close = pair_candles["close"].values
            profits = pair_trades["profit_percent"].values - slippage
            # at the open_time candle, the balance is matched to the close of the candle
            pair_balance = np.where(
                # only the rows with actual trades
                (open_rate > 0)
                # only if the trade is not also closed on the same candle
                & (open_time != close_time),
                1 - open_rate / close - slippage,
                # or initialize to 0
                0,
            )
            # at the close_time candle, the balance just uses the profits col
            pair_balance = pair_balance + np.where(
                # only rows with actual trades
                (open_rate > 0)
                # the rows where a close happens
                & (open_time == close_time),
                # use to profits
                profits,
                # otherwise leave unchanged
                pair_balance,
            )

            # how much time each trade was open, close - open time
            periods = close_time - open_time
            # how many candles each trade was open, set as a counter at each trade open_time index
            hops = np.nan_to_num(periods / timedelta).astype(int)

            # each loop update one timeframe forward, the balance on each timeframe
            # where there is at least one hop left to do (>0)
            for _ in range(1, hops.max() + 1):
                # move hops and open_rate by one
                hops = shift(hops, 1, cval=0)
                open_rate = shift(open_rate, 1, cval=0)
                pair_balance = np.where(
                    hops > 0, pair_balance + (1 - open_rate / close) - slippage, pair_balance
                )
                hops -= 1

            # same as above but one loop per pair
            # trades_indexes = np.nonzero(hops)[0]
            # for i in trades_indexes:
            #     # start from 1 because counters are set at the open_time balance
            #     # which was already added previously
            #     for c in range(1, hops[i]):
            #         offset = i + c
            #         # the open rate is always for the current date, not the offset
            #         pair_balance[offset] += 1 - open_rate[i] / close[offset] - slippage

            # add the pair balance to the total
            balance_total.append(pair_balance)
        balance_total = np.array(balance_total).sum(axis=0)

        returns = balance_total.mean()
        # returns = balance_total.values.mean()

        downside_returns = np.where(balance_total < 0, balance_total, 0)
        downside_risk = np.sqrt((downside_returns ** 2).sum() / len(date_index))

        if downside_risk != 0.0:
            sortino_ratio = (returns - target) / downside_risk * annualize
        else:
            sortino_ratio = -np.iinfo(np.int32).max

        # print(expected_returns_mean, down_stdev, sortino_ratio)
        return -sortino_ratio
