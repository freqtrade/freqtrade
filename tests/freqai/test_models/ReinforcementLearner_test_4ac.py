import logging

import numpy as np

from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base4ActionRLEnv import Actions, Base4ActionRLEnv, Positions


logger = logging.getLogger(__name__)


class ReinforcementLearner_test_4ac(ReinforcementLearner):
    """
    User created Reinforcement Learning Model prediction model.
    """

    class MyRLEnv(Base4ActionRLEnv):
        """
        User can override any function in BaseRLEnv and gym.Env. Here the user
        sets a custom reward based on profit and trade duration.

        Warning!
        This is function is a showcase of functionality designed to show as many possible
        environment control features as possible. It is also designed to run quickly
        on small computers. This is a benchmark, it is *not* for live production.
        """

        def calculate_reward(self, action: int) -> float:

            # first, penalize if the action is not valid
            if not self._is_valid(action):
                return -2

            pnl = self.get_unrealized_profit()
            rew = np.sign(pnl) * (pnl + 1)
            factor = 100.

            # reward agent for entering trades
            if (action in (Actions.Long_enter.value, Actions.Short_enter.value)
                    and self._position == Positions.Neutral):
                return 25
            # discourage agent from not entering trades
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return -1

            max_trade_duration = self.rl_config.get('max_trade_duration_candles', 300)
            trade_duration = self._current_tick - self._last_trade_tick  # type: ignore

            if trade_duration <= max_trade_duration:
                factor *= 1.5
            elif trade_duration > max_trade_duration:
                factor *= 0.5

            # discourage sitting in position
            if (self._position in (Positions.Short, Positions.Long) and
                    action == Actions.Neutral.value):
                return -1 * trade_duration / max_trade_duration

            # close long
            if action == Actions.Exit.value and self._position == Positions.Long:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
                return float(rew * factor)

            # close short
            if action == Actions.Exit.value and self._position == Positions.Short:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
                return float(rew * factor)

            return 0.
