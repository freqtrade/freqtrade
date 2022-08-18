import logging
from enum import Enum
# from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from pandas import DataFrame

logger = logging.getLogger(__name__)


class Actions(Enum):
    Short = 0
    Long = 1
    Neutral = 2


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


class Base3ActionRLEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df: DataFrame = DataFrame(), prices: DataFrame = DataFrame(),
                 reward_kwargs: dict = {}, window_size=10, starting_point=True,
                 id: str = 'baseenv-1', seed: int = 1):
        assert df.ndim == 2

        self.id = id
        self.seed(seed)
        self.df = df
        self.signal_features = self.df
        self.prices = prices
        self.window_size = window_size
        self.starting_point = starting_point
        self.rr = reward_kwargs["rr"]
        self.profit_aim = reward_kwargs["profit_aim"]

        self.fee = 0.0015

        # # spaces
        self.shape = (window_size, self.signal_features.shape[1])
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

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

    def seed(self, seed: int = 1):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self._done = False

        if self.starting_point is True:
            self._position_history = (self._start_tick * [None]) + [self._position]
        else:
            self._position_history = (self.window_size * [None]) + [self._position]

        self._current_tick = self._start_tick
        self._last_trade_tick = None
        self._position = Positions.Neutral

        self.total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        self.trade_history = []
        self.portfolio_log_returns = np.zeros(len(self.prices))

        self._profits = [(self._start_tick, 1)]
        self.close_trade_profit = []

        return self._get_observation()

    def step(self, action: int):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        self.update_portfolio_log_returns(action)

        self._update_profit(action)
        step_reward = self.calculate_reward(action)
        self.total_reward += step_reward

        trade_type = None
        if self.is_tradesignal(action):  # exclude 3 case not trade
            # Update position
            """
            Action: Neutral, position: Long ->  Close Long
            Action: Neutral, position: Short -> Close Short

            Action: Long, position: Neutral -> Open Long
            Action: Long, position: Short -> Close Short and Open Long

            Action: Short, position: Neutral -> Open Short
            Action: Short, position: Long -> Close Long and Open Short
            """

            if action == Actions.Neutral.value:
                self._position = Positions.Neutral
                trade_type = "neutral"
            elif action == Actions.Long.value:
                self._position = Positions.Long
                trade_type = "long"
            elif action == Actions.Short.value:
                self._position = Positions.Short
                trade_type = "short"
            else:
                print("case not defined")

            # Update last trade tick
            self._last_trade_tick = self._current_tick

            if trade_type is not None:
                self.trade_history.append(
                    {'price': self.current_price(), 'index': self._current_tick,
                     'type': trade_type})

        if self._total_profit < 0.2:
            self._done = True

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            tick=self._current_tick,
            total_reward=self.total_reward,
            total_profit=self._total_profit,
            position=self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _get_observation(self):
        return self.signal_features[(self._current_tick - self.window_size):self._current_tick]

    def get_unrealized_profit(self):

        if self._last_trade_tick is None:
            return 0.

        if self._position == Positions.Neutral:
            return 0.
        elif self._position == Positions.Short:
            current_price = self.add_buy_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price = self.add_sell_fee(self.prices.iloc[self._last_trade_tick].open)
            return (last_trade_price - current_price) / last_trade_price
        elif self._position == Positions.Long:
            current_price = self.add_sell_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price = self.add_buy_fee(self.prices.iloc[self._last_trade_tick].open)
            return (current_price - last_trade_price) / last_trade_price
        else:
            return 0.

    def is_tradesignal(self, action: int):
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

    def _is_trade(self, action: Actions):
        return ((action == Actions.Long.value and self._position == Positions.Short) or
                (action == Actions.Short.value and self._position == Positions.Long) or
                (action == Actions.Neutral.value and self._position == Positions.Long) or
                (action == Actions.Neutral.value and self._position == Positions.Short)
                )

    def is_hold(self, action):
        return ((action == Actions.Short.value and self._position == Positions.Short)
                or (action == Actions.Long.value and self._position == Positions.Long))

    def add_buy_fee(self, price):
        return price * (1 + self.fee)

    def add_sell_fee(self, price):
        return price / (1 + self.fee)

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def get_sharpe_ratio(self):
        return mean_over_std(self.get_portfolio_log_returns())

    def calculate_reward(self, action):

        if self._last_trade_tick is None:
            return 0.

        # close long
        if (action == Actions.Short.value or
                action == Actions.Neutral.value) and self._position == Positions.Long:
            last_trade_price = self.add_buy_fee(self.prices.iloc[self._last_trade_tick].open)
            current_price = self.add_sell_fee(self.prices.iloc[self._current_tick].open)
            return float(np.log(current_price) - np.log(last_trade_price))

        # close short
        if (action == Actions.Long.value or
                action == Actions.Neutral.value) and self._position == Positions.Short:
            last_trade_price = self.add_sell_fee(self.prices.iloc[self._last_trade_tick].open)
            current_price = self.add_buy_fee(self.prices.iloc[self._current_tick].open)
            return float(np.log(last_trade_price) - np.log(current_price))

        return 0.

    def _update_profit(self, action):
        if self._is_trade(action) or self._done:
            pnl = self.get_unrealized_profit()

            if self._position == Positions.Long:
                self._total_profit = self._total_profit + self._total_profit * pnl
                self._profits.append((self._current_tick, self._total_profit))
                self.close_trade_profit.append(pnl)

            if self._position == Positions.Short:
                self._total_profit = self._total_profit + self._total_profit * pnl
                self._profits.append((self._current_tick, self._total_profit))
                self.close_trade_profit.append(pnl)

    def most_recent_return(self, action: int):
        """
        We support Long, Neutral and Short positions.
        Return is generated from rising prices in Long
        and falling prices in Short positions.
        The actions Sell/Buy or Hold during a Long position trigger the sell/buy-fee.
        """
        # Long positions
        if self._position == Positions.Long:
            current_price = self.prices.iloc[self._current_tick].open
            if action == Actions.Short.value or action == Actions.Neutral.value:
                current_price = self.add_sell_fee(current_price)

            previous_price = self.prices.iloc[self._current_tick - 1].open

            if (self._position_history[self._current_tick - 1] == Positions.Short
                    or self._position_history[self._current_tick - 1] == Positions.Neutral):
                previous_price = self.add_buy_fee(previous_price)

            return np.log(current_price) - np.log(previous_price)

        # Short positions
        if self._position == Positions.Short:
            current_price = self.prices.iloc[self._current_tick].open
            if action == Actions.Long.value or action == Actions.Neutral.value:
                current_price = self.add_buy_fee(current_price)

            previous_price = self.prices.iloc[self._current_tick - 1].open
            if (self._position_history[self._current_tick - 1] == Positions.Long
                    or self._position_history[self._current_tick - 1] == Positions.Neutral):
                previous_price = self.add_sell_fee(previous_price)

            return np.log(previous_price) - np.log(current_price)

        return 0

    def get_portfolio_log_returns(self):
        return self.portfolio_log_returns[1:self._current_tick + 1]

    def update_portfolio_log_returns(self, action):
        self.portfolio_log_returns[self._current_tick] = self.most_recent_return(action)

    def current_price(self) -> float:
        return self.prices.iloc[self._current_tick].open

    def prev_price(self) -> float:
        return self.prices.iloc[self._current_tick - 1].open

    def sharpe_ratio(self):
        if len(self.close_trade_profit) == 0:
            return 0.
        returns = np.array(self.close_trade_profit)
        reward = (np.mean(returns) - 0. + 1e-9) / (np.std(returns) + 1e-9)
        return reward
