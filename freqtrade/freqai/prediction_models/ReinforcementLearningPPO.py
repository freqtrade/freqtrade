import gc
import logging
from typing import Any, Dict  # , Tuple

import numpy as np
# import numpy.typing as npt
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base3ActionRLEnv import Actions, Base3ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel


logger = logging.getLogger(__name__)


class ReinforcementLearningPPO(BaseReinforcementLearningModel):
    """
    User created Reinforcement Learning Model prediction model.
    """

    def fit_rl(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen):

        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]
        eval_freq = self.freqai_info["rl_config"]["eval_cycles"] * len(test_df)
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

        path = dk.data_path
        eval_callback = EvalCallback(self.eval_env, best_model_save_path=f"{path}/",
                                     log_path=f"{path}/ppo/logs/", eval_freq=int(eval_freq),
                                     deterministic=True, render=False)

        # model arch
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=[256, 256, 128])

        model = PPO('MlpPolicy', self.train_env, policy_kwargs=policy_kwargs,
                    tensorboard_log=f"{path}/ppo/tensorboard/",
                    **self.freqai_info['model_training_parameters']
                    )

        model.learn(
            total_timesteps=int(total_timesteps),
            callback=eval_callback
        )

        del model
        best_model = PPO.load(dk.data_path / "best_model")

        print('Training finished!')
        gc.collect()

        return best_model

    def set_train_and_eval_environments(self, data_dictionary, prices_train, prices_test):
        """
        User overrides this as shown here if they are using a custom MyRLEnv
        """
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]

        # environments
        if not self.train_env:
            self.train_env = MyRLEnv(df=train_df, prices=prices_train, window_size=self.CONV_WIDTH,
                                     reward_kwargs=self.reward_params)
            self.eval_env = Monitor(MyRLEnv(df=test_df, prices=prices_test,
                                    window_size=self.CONV_WIDTH,
                                    reward_kwargs=self.reward_params), ".")
        else:
            self.train_env.reset_env(train_df, prices_train, self.CONV_WIDTH, self.reward_params)
            self.eval_env.reset_env(train_df, prices_train, self.CONV_WIDTH, self.reward_params)
            self.train_env.reset()
            self.eval_env.reset()


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
