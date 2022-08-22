from datetime import datetime
from typing import Any, Dict

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss

MIN_ACCEPTED_WIN_PERCENTAGE = 0.30
MIN_ACCEPTED_AV_PROFIT = 0.007
MAX_ACCEPTED_WIN_TRADE_DURATION = 120
MAX_ACCEPTED_LOST_TRADE_DURATION = 90


class WAOHyperOptLoss1(IHyperOptLoss):

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               backtest_stats: Dict[str, Any],
                               *args, **kwargs) -> float:

        """We can put that into a function above"""
        x = 0
        win_trade_duration = []
        lost_trade_duration = []
        win_trades_number = 0
        for profit_ratio in results['profit_ratio']:
            if profit_ratio > 0:
                win_trades_number += 1
            #     win_trade_duration.append(results['trade_duration'][x])
            #
            # else:
            #     lost_trade_duration.append(results['trade_duration'][x])

            x += 1
        win_trade_percentage = win_trades_number / trade_count
        """We can put that into a function above"""
        # y = 0
        # profit_sell_signal = []
        # for exit_reason in results['exit_reason']:
        #     if exit_reason == 'exit_signal':
        #         profit_sell_signal.append(results['profit_ratio'][y])
        #     y += 1

        # av_profit_sell_signal = profit_sell_signal.mean()
        # win_trades_duration_av = win_trades_duration.mean()
        # lost_trades_duration_av = lost_trades_duration.mean()

        win_trade_percentage_loss = min(0, 1 - win_trades_percentage / MIN_ACCEPTED_WIN_PERCENTAGE)
        # av_profit_sell_signal_loss = min(0, 1 - av_profit_sell_signal / MIN_ACCEPTED_AV_PROFIT)
        # win_duration_loss = min(win_trades_duration_av / MAX_ACCEPTED_WIN_TRADE_DURATION, 1)
        # lost_duration_loss = min(lost_trades_duration_av / MAX_ACCEPTED_LOST_TRADE_DURATION, 1)

        """
        Objective function, returns smaller number for better results
        Weights are distributed as follows:
        * 0.15 to win_trade_percentage_loss
        * 0.4 to av_profit_sell_signal_loss
        * 0.2 to win_duration_loss
        * 0.25 to lost_duration_loss
        The most important thing is to increase our small profit trades ==> profit weight is higher. 
        We loose money when trades durations are big (noise), specially for the lost ones, so the quicker the lost trades is, the better it is ===> lost teade duration weight is the second most important weight
        Then follow the win trades duration and finally we want, at least, win % > 30% but it's not the most important thing ==> the smallest weight 
        """
        result = 1 * win_trade_percentage_loss

        return result
