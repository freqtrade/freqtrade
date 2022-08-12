import logging
import random
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from sklearn.decomposition import PCA, KernelPCA


logger = logging.getLogger(__name__)

# from bokeh.io import output_notebook
# from bokeh.plotting import figure, show
# from bokeh.models import (
#     CustomJS,
#     ColumnDataSource,
#     NumeralTickFormatter,
#     Span,
#     HoverTool,
#     Range1d,
#     DatetimeTickFormatter,
#     Scatter,
#     Label, LabelSet
# )

class Actions(Enum):
    Short = 0
    Long = 1
    Neutral = 2

class Actions_v2(Enum):
    Neutral = 0
    Long_buy = 1
    Long_sell = 2
    Short_buy = 3
    Short_sell = 4


class Positions(Enum):
    Short = 0
    Long = 1
    Neutral = 0.5

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long

def mean_over_std(x):
    std = np.std(x, ddof=1)
    mean = np.mean(x)
    return mean / std if std > 0 else 0

class DEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, prices, reward_kwargs, window_size=10, starting_point=True, ):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.signal_features = self.df
        self.prices = prices
        self.window_size = window_size
        self.starting_point = starting_point
        self.rr = reward_kwargs["rr"]
        self.profit_aim = reward_kwargs["profit_aim"]

        self.fee=0.0015

        # # spaces
        self.shape = (window_size, self.signal_features.shape[1])
        self.action_space = spaces.Discrete(len(Actions_v2))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = Positions.Neutral
        self._position_history = None
        self.total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None
        self.trade_history = []

        # self.A_t, self.B_t = 0.000639, 0.00001954
        self.r_t_change = 0.

        self.returns_report = []


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):

        self._done = False

        if self.starting_point == True:
            self._position_history = (self._start_tick* [None]) + [self._position]
        else:
            self._position_history = (self.window_size * [None]) + [self._position]

        self._current_tick = self._start_tick
        self._last_trade_tick = None
        #self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Neutral

        self.total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        self.trade_history = []
        self.portfolio_log_returns = np.zeros(len(self.prices))


        self._profits = [(self._start_tick, 1)]
        self.close_trade_profit = []
        self.r_t_change = 0.

        self.returns_report = []

        return self._get_observation()


    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        self.update_portfolio_log_returns(action)

        self._update_profit(action)
        step_reward = self._calculate_reward(action)
        self.total_reward += step_reward





        trade_type = None
        if self.is_tradesignal_v2(action): # exclude 3 case not trade
            # Update position
            """
            Action: Neutral, position: Long ->  Close Long
            Action: Neutral, position: Short -> Close Short

            Action: Long, position: Neutral -> Open Long
            Action: Long, position: Short -> Close Short and Open Long

            Action: Short, position: Neutral -> Open Short
            Action: Short, position: Long -> Close Long and Open Short
            """


            temp_position = self._position
            if action == Actions_v2.Neutral.value:
                self._position = Positions.Neutral
                trade_type = "neutral"
            elif action == Actions_v2.Long_buy.value:
                self._position = Positions.Long
                trade_type = "long"
            elif action == Actions_v2.Short_buy.value:
                self._position = Positions.Short
                trade_type = "short"
            elif action == Actions_v2.Long_sell.value:
                self._position = Positions.Neutral
                trade_type = "neutral"
            elif action == Actions_v2.Short_sell.value:
                self._position = Positions.Neutral
                trade_type = "neutral"
            else:
                print("case not defined")

            # Update last trade tick
            self._last_trade_tick = self._current_tick

            if trade_type != None:
                self.trade_history.append(
                    {'price': self.current_price(), 'index': self._current_tick, 'type': trade_type})

        if self._total_profit < 0.2:
            self._done = True

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            tick = self._current_tick,
            total_reward = self.total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info


    def processState(self, state):
        return state.to_numpy()

    def convert_mlp_Policy(self, obs_):
        pass

    def _get_observation(self):
        return self.signal_features[(self._current_tick - self.window_size):self._current_tick]


    def get_unrealized_profit(self):

        if self._last_trade_tick == None:
            return 0.

        if self._position == Positions.Neutral:
            return 0.
        elif self._position == Positions.Short:
            current_price = self.add_buy_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price = self.add_sell_fee(self.prices.iloc[self._last_trade_tick].open)
            return  (last_trade_price - current_price)/last_trade_price
        elif self._position == Positions.Long:
            current_price = self.add_sell_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price = self.add_buy_fee(self.prices.iloc[self._last_trade_tick].open)
            return (current_price - last_trade_price)/last_trade_price
        else:
            return 0.


    def is_tradesignal(self, action):
        # trade signal
        """
        not trade signal is :
        Action: Neutral, position: Neutral -> Nothing
        Action: Long, position: Long -> Hold Long
        Action: Short, position: Short -> Hold Short
        """
        return not ((action == Actions.Neutral.value and self._position == Positions.Neutral)
                    or (action == Actions.Short.value and self._position == Positions.Short)
                    or (action == Actions.Long.value and self._position == Positions.Long))

    def is_tradesignal_v2(self, action):
        # trade signal
        """
        not trade signal is :
        Action: Neutral, position: Neutral -> Nothing
        Action: Long, position: Long -> Hold Long
        Action: Short, position: Short -> Hold Short
        """
        return not ((action == Actions_v2.Neutral.value and self._position == Positions.Neutral) or
                    (action == Actions_v2.Short_buy.value and self._position == Positions.Short) or
                    (action == Actions_v2.Short_sell.value and self._position == Positions.Short) or
                    (action == Actions_v2.Short_buy.value and self._position == Positions.Long) or
                    (action == Actions_v2.Short_sell.value and self._position == Positions.Long) or

                    (action == Actions_v2.Long_buy.value and self._position == Positions.Long) or
                    (action == Actions_v2.Long_sell.value and self._position == Positions.Long) or
                    (action == Actions_v2.Long_buy.value and self._position == Positions.Short) or
                    (action == Actions_v2.Long_sell.value and self._position == Positions.Short))



    def _is_trade(self, action: Actions):
        return ((action == Actions.Long.value and self._position == Positions.Short) or
        (action == Actions.Short.value and self._position == Positions.Long) or
        (action == Actions.Neutral.value and self._position == Positions.Long) or
        (action == Actions.Neutral.value and self._position == Positions.Short)
        )

    def _is_trade_v2(self, action: Actions_v2):
        return ((action == Actions_v2.Long_buy.value and self._position == Positions.Short) or
        (action == Actions_v2.Short_buy.value and self._position == Positions.Long) or
        (action == Actions_v2.Neutral.value and self._position == Positions.Long) or
        (action == Actions_v2.Neutral.value and self._position == Positions.Short) or

        (action == Actions_v2.Neutral.Short_sell and self._position == Positions.Long) or
        (action == Actions_v2.Neutral.Long_sell and self._position == Positions.Short)
        )


    def is_hold(self, action):
        return ((action == Actions.Short.value and self._position == Positions.Short)
                or (action == Actions.Long.value and self._position == Positions.Long))

    def is_hold_v2(self, action):
        return ((action == Actions_v2.Short_buy.value and self._position == Positions.Short)
                or (action == Actions_v2.Long_buy.value and self._position == Positions.Long))


    def add_buy_fee(self, price):
        return price * (1 + self.fee)

    def add_sell_fee(self, price):
        return price / (1 + self.fee)

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices.loc[tick].open, color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        plt.cla()
        plt.plot(self.prices)
        _plot_position(self._position, self._current_tick)

        plt.suptitle("Total Reward: %.6f" % self.total_reward + ' ~ ' + "Total Profit: %.6f" % self._total_profit)
        plt.pause(0.01)


    def render_all(self):
        plt.figure()
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices['open'], alpha=0.5)

        short_ticks = []
        long_ticks = []
        neutral_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick - 1)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick - 1)
            elif self._position_history[i] == Positions.Neutral:
                neutral_ticks.append(tick - 1)

        plt.plot(neutral_ticks, self.prices.loc[neutral_ticks].open,
                 'o', color='grey', ms=3, alpha=0.1)
        plt.plot(short_ticks, self.prices.loc[short_ticks].open,
                 'o', color='r', ms=3, alpha=0.8)
        plt.plot(long_ticks, self.prices.loc[long_ticks].open,
                 'o', color='g', ms=3, alpha=0.8)

        plt.suptitle("Generalising")
        fig = plt.gcf()
        fig.set_size_inches(15, 10)




    def close_trade_report(self):
        small_trade = 0
        positive_big_trade = 0
        negative_big_trade = 0
        small_profit = 0.003
        for i in self.close_trade_profit:
            if i < small_profit and i > -small_profit:
                small_trade+=1
            elif i > small_profit:
                positive_big_trade += 1
            elif i < -small_profit:
                negative_big_trade += 1
        print(f"small trade={small_trade/len(self.close_trade_profit)}; positive_big_trade={positive_big_trade/len(self.close_trade_profit)}; negative_big_trade={negative_big_trade/len(self.close_trade_profit)}")


    def report(self):

        # get total trade
        long_trade = 0
        short_trade = 0
        neutral_trade = 0
        for trade in self.trade_history:
            if trade['type'] == 'long':
                long_trade += 1

            elif trade['type'] == 'short':
                short_trade += 1
            else:
                neutral_trade += 1

        negative_trade = 0
        positive_trade = 0
        for tr in self.close_trade_profit:
            if tr < 0.:
                negative_trade += 1

            if tr > 0.:
                positive_trade += 1

        total_trade_lr = negative_trade+positive_trade


        total_trade = long_trade + short_trade
        sharp_ratio = self.sharpe_ratio()
        sharp_log = self.get_sharpe_ratio()

        from tabulate import tabulate

        headers = ["Performance", ""]
        performanceTable = [["Total Trade", "{0:.2f}".format(total_trade)],
                         ["Total reward", "{0:.3f}".format(self.total_reward)],
                         ["Start profit(unit)", "{0:.2f}".format(1.)],
                         ["End profit(unit)", "{0:.3f}".format(self._total_profit)],
                         ["Sharp ratio", "{0:.3f}".format(sharp_ratio)],
                         ["Sharp log", "{0:.3f}".format(sharp_log)],
                         # ["Sortino ratio", "{0:.2f}".format(0) + '%'],
                         ["winrate", "{0:.2f}".format(positive_trade*100/total_trade_lr) + '%']
                         ]
        tabulation = tabulate(performanceTable, headers, tablefmt="fancy_grid", stralign="center")
        print(tabulation)

        result = {
            "Start": "{0:.2f}".format(1.),
            "End": "{0:.2f}".format(self._total_profit),
            "Sharp": "{0:.3f}".format(sharp_ratio),
            "Winrate": "{0:.2f}".format(positive_trade*100/total_trade_lr)
        }
        return result

    def close(self):
        plt.close()

    def get_sharpe_ratio(self):
        return mean_over_std(self.get_portfolio_log_returns())


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()


    def _calculate_reward(self, action):
        # rw = self.transaction_profit_reward(action)
        #rw = self.reward_rr_profit_config(action)
        rw = self.reward_rr_profit_config_v2(action)
        return rw


    def _update_profit(self, action):
        #if self._is_trade(action) or self._done:
        if self._is_trade_v2(action) or self._done:
            pnl = self.get_unrealized_profit()

            if self._position == Positions.Long:
                self._total_profit = self._total_profit + self._total_profit*pnl
                self._profits.append((self._current_tick, self._total_profit))
                self.close_trade_profit.append(pnl)

            if self._position == Positions.Short:
                self._total_profit = self._total_profit + self._total_profit*pnl
                self._profits.append((self._current_tick, self._total_profit))
                self.close_trade_profit.append(pnl)


    def most_recent_return(self, action):
        """
        We support Long, Neutral and Short positions.
        Return is generated from rising prices in Long
        and falling prices in Short positions.
        The actions Sell/Buy or Hold during a Long position trigger the sell/buy-fee.
        """
        # Long positions
        if self._position == Positions.Long:
            current_price = self.prices.iloc[self._current_tick].open
            #if action == Actions.Short.value or action == Actions.Neutral.value:
            if action == Actions_v2.Short_buy.value or action == Actions_v2.Neutral.value:
                current_price = self.add_sell_fee(current_price)

            previous_price = self.prices.iloc[self._current_tick - 1].open

            if (self._position_history[self._current_tick - 1] == Positions.Short
                    or self._position_history[self._current_tick - 1] == Positions.Neutral):
                previous_price = self.add_buy_fee(previous_price)

            return np.log(current_price) - np.log(previous_price)

        # Short positions
        if self._position == Positions.Short:
            current_price = self.prices.iloc[self._current_tick].open
            #if action == Actions.Long.value or action == Actions.Neutral.value:
            if action == Actions_v2.Long_buy.value or action == Actions_v2.Neutral.value:
                current_price = self.add_buy_fee(current_price)

            previous_price = self.prices.iloc[self._current_tick - 1].open
            if (self._position_history[self._current_tick - 1] == Positions.Long
                    or self._position_history[self._current_tick - 1] == Positions.Neutral):
                previous_price = self.add_sell_fee(previous_price)

            return np.log(previous_price) - np.log(current_price)

        return 0

    def get_portfolio_log_returns(self):
        return self.portfolio_log_returns[1:self._current_tick + 1]


    def get_trading_log_return(self):
        return self.portfolio_log_returns[self._start_tick:]

    def update_portfolio_log_returns(self, action):
        self.portfolio_log_returns[self._current_tick] = self.most_recent_return(action)

    def current_price(self) -> float:
        return self.prices.iloc[self._current_tick].open

    def prev_price(self) -> float:
        return self.prices.iloc[self._current_tick-1].open



    def sharpe_ratio(self):
        if len(self.close_trade_profit) == 0:
            return 0.
        returns = np.array(self.close_trade_profit)
        reward = (np.mean(returns) - 0. + 1e-9) / (np.std(returns) + 1e-9)
        return reward

    def get_bnh_log_return(self):
        return np.diff(np.log(self.prices['open'][self._start_tick:]))


    def transaction_profit_reward(self, action):
        rw = 0.

        pt  = self.prev_price()
        pt_1 = self.current_price()


        if self._position == Positions.Long:
            a_t = 1
        elif self._position == Positions.Short:
            a_t = -1
        else:
            a_t = 0

        # close long
        if (action == Actions.Short.value or action == Actions.Neutral.value) and self._position == Positions.Long:
            pt_1 = self.add_sell_fee(self.current_price())
            po = self.add_buy_fee(self.prices.iloc[self._last_trade_tick].open)

            rw = a_t*(pt_1 - po)/po
            #rw = rw*2
        # close short
        elif (action == Actions.Long.value or action == Actions.Neutral.value) and self._position == Positions.Short:
            pt_1 = self.add_buy_fee(self.current_price())
            po = self.add_sell_fee(self.prices.iloc[self._last_trade_tick].open)
            rw = a_t*(pt_1 - po)/po
            #rw = rw*2
        else:
            rw = a_t*(pt_1 - pt)/pt

        return np.clip(rw, 0, 1)



    def reward_rr_profit_config_v2(self, action):
        rw = 0.

        pt_1 = self.current_price()


        if len(self.close_trade_profit) > 0:
            # long
            if self._position == Positions.Long:
                pt_1 = self.add_sell_fee(self.current_price())
                po = self.add_buy_fee(self.prices.iloc[self._last_trade_tick].open)

                if action == Actions_v2.Short_buy.value:
                    if self.close_trade_profit[-1] > self.profit_aim * self.rr:
                        rw = 10 * 2
                    elif self.close_trade_profit[-1] > 0 and self.close_trade_profit[-1] < self.profit_aim * self.rr:
                        rw = 10 * 1 * 1
                    elif self.close_trade_profit[-1] < 0:
                        rw = 10 * -1
                    elif self.close_trade_profit[-1] < (self.profit_aim * -1) * self.rr:
                        rw = 10 * 3 * -1

                if action == Actions_v2.Long_sell.value:
                    if self.close_trade_profit[-1] > self.profit_aim * self.rr:
                        rw = 10 * 5
                    elif self.close_trade_profit[-1] > 0 and self.close_trade_profit[-1] < self.profit_aim * self.rr:
                        rw = 10 * 1 * 3
                    elif self.close_trade_profit[-1] < 0:
                        rw = 10 * -1
                    elif self.close_trade_profit[-1] < (self.profit_aim * -1) * self.rr:
                        rw = 10 * 3 * -1

                if action == Actions_v2.Neutral.value:
                    if self.close_trade_profit[-1] > 0:
                        rw = 2
                    elif self.close_trade_profit[-1] < 0:
                        rw = 2 * -1

            # short
            if self._position == Positions.Short:
                pt_1 = self.add_sell_fee(self.current_price())
                po = self.add_buy_fee(self.prices.iloc[self._last_trade_tick].open)

                if action == Actions_v2.Long_buy.value:
                    if self.close_trade_profit[-1] > self.profit_aim * self.rr:
                        rw = 10 * 2
                    elif self.close_trade_profit[-1] > 0 and self.close_trade_profit[-1] < (self.profit_aim * -1) * self.rr:
                        rw = 10 * 1 * 1
                    elif self.close_trade_profit[-1] < 0:
                        rw = 10 * -1
                    elif self.close_trade_profit[-1] < (self.profit_aim * -1) * self.rr:
                        rw = 10 * 3 * -1

                if action == Actions_v2.Short_sell.value:
                    if self.close_trade_profit[-1] > self.profit_aim * self.rr:
                        rw = 10 * 5
                    elif self.close_trade_profit[-1] > 0 and self.close_trade_profit[-1] < (self.profit_aim * -1) * self.rr:
                        rw = 10 * 1 * 3
                    elif self.close_trade_profit[-1] < 0:
                        rw = 10 * -1
                    elif self.close_trade_profit[-1] < (self.profit_aim * -1) * self.rr:
                        rw = 10 * 3 * -1

                if action == Actions_v2.Neutral.value:
                    if self.close_trade_profit[-1] > 0:
                        rw = 2
                    elif self.close_trade_profit[-1] < 0:
                        rw = 2 * -1

        return np.clip(rw, 0, 1)
