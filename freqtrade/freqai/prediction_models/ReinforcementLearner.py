import logging
from typing import Any, Dict  # , Tuple

# import numpy.typing as npt
import torch as th
import numpy as np
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
                             net_arch=[256, 256, 128])

        model = self.MODELCLASS(self.policy_type, self.train_env, policy_kwargs=policy_kwargs,
                                tensorboard_log=Path(dk.data_path / "tensorboard"),
                                **self.freqai_info['model_training_parameters']
                                )

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
    User can modify any part of the environment by overriding base
    functions
    """
    def calculate_reward(self, action):

        if self._last_trade_tick is None:
            return 0.

        # close long
        if action == Actions.Long_exit.value and self._position == Positions.Long:
            last_trade_price = self.add_entry_fee(self.prices.iloc[self._last_trade_tick].open)
            current_price = self.add_exit_fee(self.prices.iloc[self._current_tick].open)
            factor = 1
            if self.close_trade_profit and self.close_trade_profit[-1] > self.profit_aim * self.rr:
                factor = self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
            return float((np.log(current_price) - np.log(last_trade_price)) * factor)

        # close short
        if action == Actions.Short_exit.value and self._position == Positions.Short:
            last_trade_price = self.add_exit_fee(self.prices.iloc[self._last_trade_tick].open)
            current_price = self.add_entry_fee(self.prices.iloc[self._current_tick].open)
            factor = 1
            if self.close_trade_profit and self.close_trade_profit[-1] > self.profit_aim * self.rr:
                factor = self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
            return float(np.log(last_trade_price) - np.log(current_price) * factor)

        return 0.
