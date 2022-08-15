import logging
from typing import Any, Dict  # Optional
import torch as th
import numpy as np
import gym
from typing import Callable
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import DQN
from freqtrade.freqai.RL.Base3ActionRLEnv import Base3ActionRLEnv, Actions, Positions
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel
from freqtrade.freqai.RL.TDQNagent import TDQN
from stable_baselines3.common.buffers import ReplayBuffer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)

def make_env(env_id: str, rank: int, seed: int, train_df, price,
             reward_params, window_size, monitor=False) -> Callable:
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
        if monitor:
            env = Monitor(env, ".")
        return env
    set_random_seed(seed)
    return _init

class ReinforcementLearningTDQN_multiproc(BaseReinforcementLearningModel):
    """
    User created Reinforcement Learning Model prediction model.
    """

    def fit_rl(self, data_dictionary: Dict[str, Any], pair: str, dk: FreqaiDataKitchen):

        agent_params = self.freqai_info['model_training_parameters']
        reward_params = self.freqai_info['model_reward_parameters']
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]
        eval_freq = agent_params["eval_cycles"] * len(test_df)
        total_timesteps = agent_params["train_cycles"] * len(train_df)
        learning_rate = agent_params["learning_rate"]

        # price data for model training and evaluation
        price = self.dd.historic_data[pair][f"{self.config['timeframe']}"].tail(len(train_df.index))
        price_test = self.dd.historic_data[pair][f"{self.config['timeframe']}"].tail(
            len(test_df.index))

        env_id = "train_env"
        num_cpu = int(dk.thread_count / 2)
        train_env = SubprocVecEnv([make_env(env_id, i, 1, train_df, price, reward_params,
                                   self.CONV_WIDTH) for i in range(num_cpu)])

        eval_env_id = 'eval_env'
        eval_env = SubprocVecEnv([make_env(eval_env_id, i, 1, test_df, price_test, reward_params,
                                  self.CONV_WIDTH, monitor=True) for i in range(num_cpu)])

        path = dk.data_path
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=2)
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=2)
        eval_callback = EvalCallback(eval_env, best_model_save_path=f"{path}/",
                                     log_path=f"{path}/tdqn/logs/", eval_freq=int(eval_freq),
                                     deterministic=True, render=True, callback_after_eval=stop_train_callback, callback_on_new_best=callback_on_best, verbose=2)
        # model arch
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=[512, 512, 512])

        model = TDQN('TMultiInputPolicy', train_env,
                     policy_kwargs=policy_kwargs,
                     tensorboard_log=f"{path}/tdqn/tensorboard/",
                     learning_rate=learning_rate, gamma=0.9,
                     target_update_interval=5000, buffer_size=50000,
                     exploration_initial_eps=1, exploration_final_eps=0.1,
                     replay_buffer_class=ReplayBuffer
                     )

        model.learn(
            total_timesteps=int(total_timesteps),
            callback=eval_callback
        )

        best_model = DQN.load(dk.data_path / "best_model.zip")
        print('Training finished!')
        eval_env.close()

        return best_model


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

# User can inherit and customize 5 action environment
# class MyRLEnv(Base5ActionRLEnv):
#     """
#     User can override any function in BaseRLEnv and gym.Env. Here the user
#     Adds 5 actions.
#     """

#     def calculate_reward(self, action):

#         if self._last_trade_tick is None:
#             return 0.

#         # close long
#         if action == Actions.Long_sell.value and self._position == Positions.Long:
#             last_trade_price = self.add_buy_fee(self.prices.iloc[self._last_trade_tick].open)
#             current_price = self.add_sell_fee(self.prices.iloc[self._current_tick].open)
#             return float(np.log(current_price) - np.log(last_trade_price))

#         if action == Actions.Long_sell.value and self._position == Positions.Long:
#             if self.close_trade_profit[-1] > self.profit_aim * self.rr:
#                 last_trade_price = self.add_buy_fee(self.prices.iloc[self._last_trade_tick].open)
#                 current_price = self.add_sell_fee(self.prices.iloc[self._current_tick].open)
#                 return float((np.log(current_price) - np.log(last_trade_price)) * 2)

#         # close short
#         if action == Actions.Short_buy.value and self._position == Positions.Short:
#             last_trade_price = self.add_sell_fee(self.prices.iloc[self._last_trade_tick].open)
#             current_price = self.add_buy_fee(self.prices.iloc[self._current_tick].open)
#             return float(np.log(last_trade_price) - np.log(current_price))

#         if action == Actions.Short_buy.value and self._position == Positions.Short:
#             if self.close_trade_profit[-1] > self.profit_aim * self.rr:
#                 last_trade_price = self.add_sell_fee(self.prices.iloc[self._last_trade_tick].open)
#                 current_price = self.add_buy_fee(self.prices.iloc[self._current_tick].open)
#                 return float((np.log(last_trade_price) - np.log(current_price)) * 2)

#         return 0.
