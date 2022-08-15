import logging
from typing import Any, Dict  # Optional
from enum import Enum
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import SubprocVecEnv
from freqtrade.freqai.RL.BaseRLEnv import BaseRLEnv
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel
from freqtrade.freqai.RL.TDQNagent import TDQN
from stable_baselines3.common.buffers import ReplayBuffer
from gym import spaces
from gym.utils import seeding

logger = logging.getLogger(__name__)


class ReinforcementLearningTDQN(BaseReinforcementLearningModel):
    """
    User created Reinforcement Learning Model prediction model.
    """

    def fit(self, data_dictionary: Dict[str, Any], pair: str = ''):

        agent_params = self.freqai_info['model_training_parameters']
        reward_params = self.freqai_info['model_reward_parameters']
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]
        eval_freq = agent_params["eval_cycles"] * len(test_df)
        total_timesteps = agent_params["train_cycles"] * len(train_df)

        # price data for model training and evaluation
        price = self.dd.historic_data[pair][f"{self.config['timeframe']}"].tail(len(train_df.index))
        price_test = self.dd.historic_data[pair][f"{self.config['timeframe']}"].tail(
            len(test_df.index))

        # environments
        train_env = MyRLEnv(df=train_df, prices=price, window_size=self.CONV_WIDTH,
                            reward_kwargs=reward_params)
        eval = MyRLEnv(df=test_df, prices=price_test,
                       window_size=self.CONV_WIDTH, reward_kwargs=reward_params)
        eval_env = Monitor(eval, ".")
        eval_env.reset()

        path = self.dk.data_path
        eval_callback = EvalCallback(eval_env, best_model_save_path=f"{path}/",
                                     log_path=f"{path}/tdqn/logs/", eval_freq=int(eval_freq),
                                     deterministic=True, render=False)

        # model arch
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=[256, 256, 128])

        model = TDQN('TMultiInputPolicy', train_env,
                     policy_kwargs=policy_kwargs,
                     tensorboard_log=f"{path}/tdqn/tensorboard/",
                     learning_rate=0.00025, gamma=0.9,
                     target_update_interval=5000, buffer_size=50000,
                     exploration_initial_eps=1, exploration_final_eps=0.1,
                     replay_buffer_class=ReplayBuffer
                     )

        model.learn(
            total_timesteps=int(total_timesteps),
            callback=eval_callback
        )

        print('Training finished!')

        return model


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


class MyRLEnv(BaseRLEnv):
    """
    User can override any function in BaseRLEnv and gym.Env. Here the user
    Adds 5 actions.
    """

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

        # self.A_t, self.B_t = 0.000639, 0.00001954
        self.r_t_change = 0.

        self.returns_report = []

    def seed(self, seed=None):
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
        step_reward = self.calculate_reward(action)
        self.total_reward += step_reward

        trade_type = None
        if self.is_tradesignal(action): # exclude 3 case not trade  
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
                print("case not define")

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
