import logging
from typing import Any, Dict  # , Tuple

import numpy as np
# import numpy.typing as npt
# import pandas as pd
import torch as th
# from pandas import DataFrame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import SubprocVecEnv
from freqtrade.freqai.RL.BaseRLEnv import BaseRLEnv, Actions, Positions
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel


logger = logging.getLogger(__name__)


class ReinforcementLearningPPO(BaseReinforcementLearningModel):
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
                                     log_path=f"{path}/ppo/logs/", eval_freq=int(eval_freq),
                                     deterministic=True, render=False)

        # model arch
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=[256, 256, 128])

        model = PPO('MultiInputPolicy', train_env, policy_kwargs=policy_kwargs,
                    tensorboard_log=f"{path}/ppo/tensorboard/", learning_rate=0.00025, gamma=0.9
                    )

        model.learn(
            total_timesteps=int(total_timesteps),
            callback=eval_callback
        )

        print('Training finished!')

        return model


class MyRLEnv(BaseRLEnv):
    """
    User can override any function in BaseRLEnv and gym.Env
    """

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
