import logging
from typing import Any, Dict

import torch as th
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel
from pathlib import Path
from pandas import DataFrame
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np

logger = logging.getLogger(__name__)


class ReinforcementLearner(BaseReinforcementLearningModel):
    """
    User created Reinforcement Learning Model prediction model.
    """

    def fit_rl(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen):

        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=[512, 512, 256])

        if dk.pair not in self.dd.model_dictionary or not self.continual_retraining:
            model = self.MODELCLASS(self.policy_type, self.train_env, policy_kwargs=policy_kwargs,
                                    tensorboard_log=Path(dk.data_path / "tensorboard"),
                                    **self.freqai_info['model_training_parameters']
                                    )
        else:
            logger.info('Continual training activated - starting training from previously '
                        'trained agent.')
            model = self.dd.model_dictionary[dk.pair]
            model.tensorboard_log = Path(dk.data_path / "tensorboard")
            model.set_env(self.train_env)

        model.learn(
            total_timesteps=int(total_timesteps),
            callback=self.eval_callback
        )

        if Path(dk.data_path / "best_model.zip").is_file():
            logger.info('Callback found a best model.')
            best_model = self.MODELCLASS.load(dk.data_path / "best_model")
            return best_model

        logger.info('Couldnt find best model, using final model instead.')

        return model

    def set_train_and_eval_environments(self, data_dictionary: Dict[str, DataFrame],
                                        prices_train: DataFrame, prices_test: DataFrame,
                                        dk: FreqaiDataKitchen):
        """
        User can override this if they are using a custom MyRLEnv
        """
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]
        eval_freq = self.freqai_info["rl_config"]["eval_cycles"] * len(test_df)

        self.train_env = MyRLEnv(df=train_df, prices=prices_train, window_size=self.CONV_WIDTH,
                                 reward_kwargs=self.reward_params, config=self.config)
        self.eval_env = Monitor(MyRLEnv(df=test_df, prices=prices_test,
                                window_size=self.CONV_WIDTH,
                                reward_kwargs=self.reward_params, config=self.config))
        self.eval_callback = EvalCallback(self.eval_env, deterministic=True,
                                          render=False, eval_freq=eval_freq,
                                          best_model_save_path=str(dk.data_path))


class MyRLEnv(Base5ActionRLEnv):
    """
    User can override any function in BaseRLEnv and gym.Env. Here the user
    sets a custom reward based on profit and trade duration.
    """

    def calculate_reward(self, action):

        # first, penalize if the action is not valid
        if not self._is_valid(action):
            return -15

        pnl = self.get_unrealized_profit()
        rew = np.sign(pnl) * (pnl + 1)
        factor = 100

        # reward agent for entering trades
        if action in (Actions.Long_enter.value, Actions.Short_enter.value):
            return 25
        # discourage agent from not entering trades
        if action == Actions.Neutral.value and self._position == Positions.Neutral:
            return -15

        max_trade_duration = self.rl_config.get('max_trade_duration_candles', 300)
        trade_duration = self._current_tick - self._last_trade_tick

        if trade_duration <= max_trade_duration:
            factor *= 1.5
        elif trade_duration > max_trade_duration:
            factor *= 0.5

        # discourage sitting in position
        if self._position in (Positions.Short, Positions.Long):
            return -50 * trade_duration / max_trade_duration

        # close long
        if action == Actions.Long_exit.value and self._position == Positions.Long:
            if pnl > self.profit_aim * self.rr:
                factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
            return float(rew * factor)

        # close short
        if action == Actions.Short_exit.value and self._position == Positions.Short:
            if pnl > self.profit_aim * self.rr:
                factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
            return float(rew * factor)

        return 0.
