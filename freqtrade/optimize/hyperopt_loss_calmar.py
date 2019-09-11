from datetime import datetime
from pandas import DataFrame, Series
from freqtrade.optimize.hyperopt import IHyperOptLoss
import numpy as np
from scipy.stats import norm

NB_SIMULATIONS = 1000
SIMULATION_YEAR_DURATION = 3
HIGH_NUMBER = 100000
CALMAR_LOSS_WEIGHT = 1
SLIPPAGE_PERCENT = 0.001


class CalmarHyperOptLoss(IHyperOptLoss):
    """
    Defines the default loss function for hyperopt
    This is intended to give you some inspiration for your own loss function.
    The Function needs to return a number (float) - which becomes for better backtest results.
    """

    @classmethod
    def hyperopt_loss_function(cls, results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               *args, **kwargs) -> float:

        """
        Objective function, returns smaller number for better results
        """

        simulated_drawdowns = []

        backtest_duration_years = ((max_date-min_date).days/365.2425)
        trade_count_average_per_year = trade_count/backtest_duration_years

        # add slipage to be closed to live
        results['profit_percent'] -= SLIPPAGE_PERCENT

        return_avg_per_year = (results.profit_percent.sum() / backtest_duration_years)
        return_avg_simulation_duration = return_avg_per_year * SIMULATION_YEAR_DURATION

        sample_size = round(trade_count_average_per_year * SIMULATION_YEAR_DURATION)

        # exclude the case when no trade was lost
        if(results.profit_percent.min() >= 0):
            return HIGH_NUMBER

        # simulate n years of run to define a median max drawdown
        for i in range(0, NB_SIMULATIONS):
            randomized_result = results.profit_percent.sample(n=sample_size,
                                                              random_state=np.random.RandomState(),
                                                              replace=True)
            simulated_drawdown = cls.abs_max_drawdown(randomized_result)
            simulated_drawdowns.append(simulated_drawdown)

        abs_mediam_simulated_drawdowns = Series(simulated_drawdowns).median()
        calmar_ratio = return_avg_simulation_duration/abs_mediam_simulated_drawdowns

        # float between ]0,1[
        calmar_loss = 1 - (norm.cdf(calmar_ratio, 0, 100))

        # feel free to add other criterias (e.g avg expected time duration)
        loss = (calmar_loss * CALMAR_LOSS_WEIGHT)

        # print('calmar_ratio {}'.format(calmar_ratio))

        return loss

    @staticmethod
    def abs_max_drawdown(profit_percent_results: Series) -> float:

        max_drawdown_df = DataFrame()
        max_drawdown_df['cumulative'] = profit_percent_results.cumsum()
        max_drawdown_df['high_value'] = max_drawdown_df['cumulative'].cummax()
        max_drawdown_df['drawdown'] = max_drawdown_df['cumulative'] - max_drawdown_df['high_value']

        return abs(max_drawdown_df['drawdown'].min())
