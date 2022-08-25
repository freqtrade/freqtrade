import logging
from enum import Enum
from typing import Optional

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from pandas import DataFrame
import pandas as pd
from abc import abstractmethod
logger = logging.getLogger(__name__)


class Actions(Enum):
    Neutral = 0
    Long_enter = 1
    Long_exit = 2
    Short_enter = 3
    Short_exit = 4


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


class Base5ActionRLEnv(gym.Env):
    """
    Base class for a 5 action environment
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df: DataFrame = DataFrame(), prices: DataFrame = DataFrame(),
                 reward_kwargs: dict = {}, window_size=10, starting_point=True,
                 id: str = 'baseenv-1', seed: int = 1, config: dict = {}):

        self.rl_config = config['freqai']['rl_config']
        self.id = id
        self.seed(seed)
        self.reset_env(df, prices, window_size, reward_kwargs, starting_point)

    def reset_env(self, df: DataFrame, prices: DataFrame, window_size: int,
                  reward_kwargs: dict, starting_point=True):
        self.df = df
        self.signal_features = self.df
        self.prices = prices
        self.window_size = window_size
        self.starting_point = starting_point
        self.rr = reward_kwargs["rr"]
        self.profit_aim = reward_kwargs["profit_aim"]

        self.fee = 0.0015

        # # spaces
        self.shape = (window_size, self.signal_features.shape[1] + 3)
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self._start_tick: int = self.window_size
        self._end_tick: int = len(self.prices) - 1
        self._done: bool = False
        self._current_tick: int = self._start_tick
        self._last_trade_tick: Optional[int] = None
        self._position = Positions.Neutral
        self._position_history: list = [None]
        self.total_reward: float = 0
        self._total_profit: float = 1
        self.history: dict = {}
        self.trade_history: list = []

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
        if self.is_tradesignal(action):
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
                self._last_trade_tick = None
            elif action == Actions.Long_enter.value:
                self._position = Positions.Long
                trade_type = "long"
                self._last_trade_tick = self._current_tick
            elif action == Actions.Short_enter.value:
                self._position = Positions.Short
                trade_type = "short"
                self._last_trade_tick = self._current_tick
            elif action == Actions.Long_exit.value:
                self._position = Positions.Neutral
                trade_type = "neutral"
                self._last_trade_tick = None
            elif action == Actions.Short_exit.value:
                self._position = Positions.Neutral
                trade_type = "neutral"
                self._last_trade_tick = None
            else:
                print("case not defined")

            if trade_type is not None:
                self.trade_history.append(
                    {'price': self.current_price(), 'index': self._current_tick,
                     'type': trade_type})

        if self._total_profit < 1 - self.rl_config.get('max_training_drawdown_pct', 0.8):
            self._done = True

        self._position_history.append(self._position)

        info = dict(
            tick=self._current_tick,
            total_reward=self.total_reward,
            total_profit=self._total_profit,
            position=self._position.value
        )

        observation = self._get_observation()

        self._update_history(info)

        return observation, step_reward, self._done, info

    def _get_observation(self):
        features_window = self.signal_features[(
            self._current_tick - self.window_size):self._current_tick]
        features_and_state = DataFrame(np.zeros((len(features_window), 3)),
                                       columns=['current_profit_pct', 'position', 'trade_duration'],
                                       index=features_window.index)

        features_and_state['current_profit_pct'] = self.get_unrealized_profit()
        features_and_state['position'] = self._position.value
        features_and_state['trade_duration'] = self.get_trade_duration()
        features_and_state = pd.concat([features_window, features_and_state], axis=1)
        return features_and_state

    def get_trade_duration(self):
        if self._last_trade_tick is None:
            return 0
        else:
            return self._current_tick - self._last_trade_tick

    def get_unrealized_profit(self):

        if self._last_trade_tick is None:
            return 0.

        if self._position == Positions.Neutral:
            return 0.
        elif self._position == Positions.Short:
            current_price = self.add_entry_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price = self.add_exit_fee(self.prices.iloc[self._last_trade_tick].open)
            return (last_trade_price - current_price) / last_trade_price
        elif self._position == Positions.Long:
            current_price = self.add_exit_fee(self.prices.iloc[self._current_tick].open)
            last_trade_price = self.add_entry_fee(self.prices.iloc[self._last_trade_tick].open)
            return (current_price - last_trade_price) / last_trade_price
        else:
            return 0.

    def is_tradesignal(self, action: int):
        # trade signal
        """
        Determine if the signal is a trade signal
        e.g.: agent wants a Actions.Long_exit while it is in a Positions.short
        """
        return not ((action == Actions.Neutral.value and self._position == Positions.Neutral) or
                    (action == Actions.Neutral.value and self._position == Positions.Short) or
                    (action == Actions.Neutral.value and self._position == Positions.Long) or
                    (action == Actions.Short_enter.value and self._position == Positions.Short) or
                    (action == Actions.Short_enter.value and self._position == Positions.Long) or
                    (action == Actions.Short_exit.value and self._position == Positions.Long) or
                    (action == Actions.Short_exit.value and self._position == Positions.Neutral) or
                    (action == Actions.Long_enter.value and self._position == Positions.Long) or
                    (action == Actions.Long_enter.value and self._position == Positions.Short) or
                    (action == Actions.Long_exit.value and self._position == Positions.Short) or
                    (action == Actions.Long_exit.value and self._position == Positions.Neutral))

    def _is_valid(self, action: int):
        # trade signal
        """
        Determine if the signal is valid.
        e.g.: agent wants a Actions.Long_exit while it is in a Positions.short
        """
        # Agent should only try to exit if it is in position
        if action in (Actions.Short_exit.value, Actions.Long_exit.value):
            if self._position not in (Positions.Short, Positions.Long):
                return False

        # Agent should only try to enter if it is not in position
        if action in (Actions.Short_enter.value, Actions.Long_enter.value):
            if self._position != Positions.Neutral:
                return False

        return True

    def _is_trade(self, action: Actions):
        return ((action == Actions.Long_enter.value and self._position == Positions.Neutral) or
                (action == Actions.Short_enter.value and self._position == Positions.Neutral))

    def is_hold(self, action):
        return ((action == Actions.Short_enter.value and self._position == Positions.Short) or
                (action == Actions.Long_enter.value and self._position == Positions.Long) or
                (action == Actions.Neutral.value and self._position == Positions.Long) or
                (action == Actions.Neutral.value and self._position == Positions.Short) or
                (action == Actions.Neutral.value and self._position == Positions.Neutral))

    def add_entry_fee(self, price):
        return price * (1 + self.fee)

    def add_exit_fee(self, price):
        return price / (1 + self.fee)

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def get_sharpe_ratio(self):
        return mean_over_std(self.get_portfolio_log_returns())

    @abstractmethod
    def calculate_reward(self, action):
        """
        Reward is created by BaseReinforcementLearningModel and can
        be inherited/edited by the user made ReinforcementLearner file.
        """

        return 0.

    def _update_profit(self, action):
        if self._is_trade(action) or self._done:
            pnl = self.get_unrealized_profit()

            if self._position in (Positions.Long, Positions.Short):
                self._total_profit *= (1 + pnl)
                self._profits.append((self._current_tick, self._total_profit))
                self.close_trade_profit.append(pnl)

    def most_recent_return(self, action: int):
        """
        Calculate the tick to tick return if in a trade.
        Return is generated from rising prices in Long
        and falling prices in Short positions.
        The actions Sell/Buy or Hold during a Long position trigger the sell/buy-fee.
        """
        # Long positions
        if self._position == Positions.Long:
            current_price = self.prices.iloc[self._current_tick].open
            previous_price = self.prices.iloc[self._current_tick - 1].open

            if (self._position_history[self._current_tick - 1] == Positions.Short
                    or self._position_history[self._current_tick - 1] == Positions.Neutral):
                previous_price = self.add_entry_fee(previous_price)

            return np.log(current_price) - np.log(previous_price)

        # Short positions
        if self._position == Positions.Short:
            current_price = self.prices.iloc[self._current_tick].open
            previous_price = self.prices.iloc[self._current_tick - 1].open
            if (self._position_history[self._current_tick - 1] == Positions.Long
                    or self._position_history[self._current_tick - 1] == Positions.Neutral):
                previous_price = self.add_exit_fee(previous_price)

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
