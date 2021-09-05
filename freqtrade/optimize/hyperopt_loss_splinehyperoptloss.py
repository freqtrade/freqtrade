import logging
from datetime import datetime

from freqtrade.data.btanalysis import calculate_max_drawdown
from freqtrade.optimize.hyperopt import IHyperOptLoss
from pandas import DataFrame
from scipy import interpolate

logger = logging.getLogger(__name__)

import numpy as np

# HyperOpt Loss function, it is difficult to put the various optimizations into one number.
# To make it easier, you can specify one table per optimization here. Missing values are interpolated.
# In this way it is possible to create your own loss curves for each value quite easily.

#
# this function also removes extrem vales from the results, to optimize on the major mass to avoid
# single trades/pairs to dominate
#

class SplineHyperOptLoss(IHyperOptLoss):

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               *args, **kwargs) -> float:

        def remove_outliners(x) -> np.array:
            x = x[x.between(x.quantile(.15), x.quantile(.85))]
            return x

        def execute_table(table: dict, x: float) -> float:
            x_points = [float(k) for k in table.keys()]
            y_points = list(table.values())
            tck = interpolate.splrep(x_points, y_points, k=2)
            return float(interpolate.splev(x, tck))

        try:
            max_drawdown = calculate_max_drawdown(results)[0]
        except:
            max_drawdown = 0.0

        #  profit_sum = results['profit_ratio'].sum()
        profit = remove_outliners(results['profit_ratio']).sum()
        profit_mean = remove_outliners(results['profit_ratio']).mean()
        profit_median = remove_outliners(results['profit_ratio']).median()
        trade_duration = remove_outliners(results['trade_duration']).mean()
        trade_count = len(results)

        ret = 0
        profit_table = {
            '-1.0': -1000,
            '0.2': 50,
            '0.3': 100,
            '1.0': 500,
            '2.0': 1000,
            '100.0': 10000,
        }
        r = execute_table(profit_table, profit)
        logger.debug(f'profit: {profit}')
        logger.debug(f'profit ret: {r}')
        ret += r

        profit_mean_table = {
            '-1.0': -1000,
            '0.01': 50,
            '0.02': 500,
            '0.03': 1000,
            '0.04': 500,
            '0.05': 50,
        }
        r = execute_table(profit_mean_table, profit_mean)
        logger.debug(f'profit_mean: {profit_mean}')
        logger.debug(f'profit_mean ret: {r}')
        ret += r

        profit_median_table = {
            '-1.0': -1000,
            '0.01': 50,
            '0.02': 1000,
            '0.03': 1000,
            '0.04': 50,
            '0.05': 10,
        }
        r = execute_table(profit_median_table, profit_median)
        logger.debug(f'profit_median: {profit_median}')
        logger.debug(f'profit_median ret: {r}')
        ret += r

        trade_duration_table = {
            str(1): -10,
            str(5): 5,
            str(60): 500,
            str(60 * 5): 250,
            str(60 * 10): 150,
            str(60 * 20): 50,
            str(60 * 2000): -10,
        }
        r = execute_table(trade_duration_table, trade_duration)
        logger.debug(f'trade_duration: {trade_duration}')
        logger.debug(f'trade_duration ret: {r}')
        ret += r

        trade_count_table = {
            '0': 0,
            '20': 100,
            '100': 500,
            '300': 1000,
            '500': 500,
            '1000': 50,
        }
        r = execute_table(trade_count_table, trade_count)
        logger.debug(f'trade_count: {trade_count}')
        logger.debug(f'trade_count ret: {r}')
        ret += r

        max_drawdown_table = {
            '0': 1000,
            '0.1': 3000,
            '0.4': 500,
            '0.8': -5000,
        }
        r = execute_table(max_drawdown_table, max_drawdown)
        logger.debug(f'max_drawdown: {max_drawdown}')
        logger.debug(f'max_drawdown ret: {r}')
        ret += r

        return -ret
