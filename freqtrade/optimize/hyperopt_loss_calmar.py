from datetime import datetime
from pandas import DataFrame, Series
import numpy as np
from scipy.stats import norm

from freqtrade.optimize.hyperopt import IHyperOptLoss, MAX_LOSS

NB_SIMULATIONS = 1000
SIMULATION_YEAR_DURATION = 3
CALMAR_LOSS_WEIGHT = 1
SLIPPAGE_PERCENT = 0.001


class CalmarHyperOptLoss(IHyperOptLoss):
    """
    Defines the calmar loss function for hyperopt (you maybe need to add other criterias
    like the number of trades expected and the max duration expected).
    Calmar ratio is based on  average annual rate of return for the last 36 months divided by the
    maximum drawdown for the last 36 months.
    But you maybe don't have running hyperopt with 36 months of data so we will simulate 36 months
    of trading with a montecarlo simulation and find the median drawdown (what's happenned if the
    trades orders changes, the max drawdown change ?)
    shorturl.at/ioAK2
    """

    @classmethod
    def hyperopt_loss_function(cls, results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               *args, **kwargs) -> float:

        """
        Objective function, returns smaller number for better results
        """

        # exclude the case when no trade was lost
        if results.profit_percent.min() >= 0:
            return MAX_LOSS

        simulated_drawdowns = []
        simulated_annualized_returns = []

        backtest_duration_years = ((max_date-min_date).days/365)
        trade_count_average_per_year = trade_count/backtest_duration_years

        # add slipage to be closed to live
        results['profit_percent'] -= SLIPPAGE_PERCENT

        sample_size = round(trade_count_average_per_year * SIMULATION_YEAR_DURATION)

        # simulate n years of run to define a median max drawdown and median annual return
        for i in range(0, NB_SIMULATIONS):
            randomized_result = results.profit_percent.sample(n=sample_size,
                                                              random_state=np.random.RandomState(),
                                                              replace=True)
            simulated_drawdown = cls.abs_max_drawdown(randomized_result)
            simulated_drawdowns.append(simulated_drawdown)
            simulated_annualized_returns.append(randomized_result.sum()/SIMULATION_YEAR_DURATION)

        abs_mediam_simulated_drawdowns = Series(simulated_drawdowns).median()
        mediam_simulated_annualized_returns = Series(simulated_annualized_returns).median()

        calmar_ratio = mediam_simulated_annualized_returns/abs_mediam_simulated_drawdowns

        # Normalize loss value to be float between (0, 1) :  0.5 value mean no profit
        calmar_loss = 1 - (norm.cdf(calmar_ratio, 0, 10))

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
