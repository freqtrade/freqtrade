import logging
from typing import Any, Dict

import torch as th
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel
from pathlib import Path

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


class MyRLEnv(Base5ActionRLEnv):
    """
    User can override any function in BaseRLEnv and gym.Env. Here the user
    sets a custom reward based on profit and trade duration.
    """

    def calculate_reward(self, action):

        if self._last_trade_tick is None:
            return 0.

        pnl = self.get_unrealized_profit()
        max_trade_duration = self.rl_config.get('max_trade_duration_candles', 100)
        trade_duration = self._current_tick - self._last_trade_tick

        factor = 1
        if trade_duration <= max_trade_duration:
            factor *= 1.5
        elif trade_duration > max_trade_duration:
            factor *= 0.5

        # close long
        if action == Actions.Long_exit.value and self._position == Positions.Long:
            if self.close_trade_profit and self.close_trade_profit[-1] > self.profit_aim * self.rr:
                factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
            return float(pnl * factor)

        # close short
        if action == Actions.Short_exit.value and self._position == Positions.Short:
            factor = 1
            if self.close_trade_profit and self.close_trade_profit[-1] > self.profit_aim * self.rr:
                factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
            return float(pnl * factor)

        return 0.
