from datetime import datetime
from typing import Any, Dict
import statistics

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss

MIN_ACCEPTED_WIN_PERCENTAGE = 0.37
MIN_ACCEPTED_AV_PROFIT = 0.007
MAX_ACCEPTED_WIN_TRADE_DURATION = 120
MAX_ACCEPTED_LOST_TRADE_DURATION = 90
MIN_TRADE_COUNT = 1210


class WAOHyperOptLoss(IHyperOptLoss):

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               backtest_stats: Dict[str, Any],
                               *args, **kwargs) -> float:

        """We can put that into a function above"""
        x = 0
        win_trades_duration = []
        lost_trades_duration = []
        win_trades_number = 0
        for profit_ratio in results['profit_ratio']:
            if profit_ratio > 0:
                win_trades_number += 1
                win_trades_duration.append(results['trade_duration'][x])

            else:
                lost_trades_duration.append(results['trade_duration'][x])

            x += 1

        if trade_count == 0 or win_trades_number == 0:
            win_trades_percentage = 0.0000001
        else:
            win_trades_percentage = win_trades_number / trade_count
        """We can put that into a function above"""
        y = 0
        profit_sell_signal = []
        for exit_reason in results['exit_reason']:
            if exit_reason == 'exit_signal':
                profit_sell_signal.append(results['profit_ratio'][y])
            y += 1
        if len(profit_sell_signal) > 0 and statistics.mean(profit_sell_signal) > 0:
            av_profit_sell_signal = statistics.mean(profit_sell_signal)
            win_trades_duration_av = statistics.mean(win_trades_duration)
        else:
            av_profit_sell_signal = 0.0000001
            win_trades_duration_av = 0.0000001

        win_trades_percentage_loss = MIN_ACCEPTED_WIN_PERCENTAGE / win_trades_percentage
        av_profit_sell_signal_loss = MIN_ACCEPTED_AV_PROFIT / av_profit_sell_signal
        win_duration_loss = win_trades_duration_av / MAX_ACCEPTED_WIN_TRADE_DURATION
        # lost_duration_loss = lost_trades_duration_av / MAX_ACCEPTED_LOST_TRADE_DURATION
        trades_number_loss = MIN_TRADE_COUNT / trade_count

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
        result = 0.15 * win_trades_percentage_loss + 0.4 * av_profit_sell_signal_loss + 0.1 * win_duration_loss + 0.35 * trades_number_loss

        return result
