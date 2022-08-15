import logging
from typing import Any, Dict  # , Tuple

import numpy as np
# import numpy.typing as npt
# import pandas as pd
import torch as th
# from pandas import DataFrame
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from freqtrade.freqai.RL.Base3ActionRLEnv import Base3ActionRLEnv, Actions, Positions
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel
import gym
logger = logging.getLogger(__name__)


def make_env(env_id: str, rank: int, seed: int, train_df, price,
             reward_params, window_size) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:

        env = MyRLEnv(df=train_df, prices=price, window_size=window_size,
                      reward_kwargs=reward_params, id=env_id, seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


class ReinforcementLearningPPO_multiproc(BaseReinforcementLearningModel):
    """
    User created Reinforcement Learning Model prediction model.
    """

    def fit(self, data_dictionary: Dict[str, Any], pair: str = ''):

        agent_params = self.freqai_info['model_training_parameters']
        reward_params = self.freqai_info['model_reward_parameters']
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]
        eval_freq = agent_params.get("eval_cycles", 4) * len(test_df)
        total_timesteps = agent_params["train_cycles"] * len(train_df)

        # price data for model training and evaluation
        price = self.dd.historic_data[pair][f"{self.config['timeframe']}"].tail(len(train_df.index))
        price_test = self.dd.historic_data[pair][f"{self.config['timeframe']}"].tail(
            len(test_df.index))

        env_id = "CartPole-v1"
        num_cpu = 4
        train_env = SubprocVecEnv([make_env(env_id, i, 1, train_df, price, reward_params,
                                   self.CONV_WIDTH) for i in range(num_cpu)])

        eval_env = SubprocVecEnv([make_env(env_id, i, 1, test_df, price_test, reward_params,
                                  self.CONV_WIDTH) for i in range(num_cpu)])

        path = self.dk.data_path
        eval_callback = EvalCallback(eval_env, best_model_save_path=f"{path}/",
                                     log_path=f"{path}/ppo/logs/", eval_freq=int(eval_freq),
                                     deterministic=True, render=False)

        # model arch
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=[256, 256, 128])

        model = PPO('MlpPolicy', train_env, policy_kwargs=policy_kwargs,
                    tensorboard_log=f"{path}/ppo/tensorboard/", learning_rate=0.00025, gamma=0.9, verbose=1
                    )

        model.learn(
            total_timesteps=int(total_timesteps),
            callback=eval_callback
        )

        print('Training finished!')

        return model


class MyRLEnv(Base3ActionRLEnv):
    """
    User can override any function in BaseRLEnv and gym.Env
    """

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
