from enum import Enum
from typing import Any, Dict, Type, Union

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from freqtrade.freqai.RL.BaseEnvironment import BaseActions, BaseEnvironment


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard and
    episodic summary reports.
    """
    def __init__(self, verbose=1, actions: Type[Enum] = BaseActions):
        super(TensorboardCallback, self).__init__(verbose)
        self.model: Any = None
        self.logger = None  # type: Any
        self.training_env: BaseEnvironment = None  # type: ignore
        self.actions: Type[Enum] = actions

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": self.model.learning_rate,
            # "gamma": self.model.gamma,
            # "gae_lambda": self.model.gae_lambda,
            # "batch_size": self.model.batch_size,
            # "n_steps": self.model.n_steps,
        }
        metric_dict: Dict[str, Union[float, int]] = {
            "eval/mean_reward": 0,
            "rollout/ep_rew_mean": 0,
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0,
            "train/explained_variance": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        custom_info = self.training_env.get_attr("custom_info")[0]
        self.logger.record("_state/position", self.locals["infos"][0]["position"])
        self.logger.record("_state/trade_duration", self.locals["infos"][0]["trade_duration"])
        self.logger.record("_state/current_profit_pct", self.locals["infos"]
                           [0]["current_profit_pct"])
        self.logger.record("_reward/total_profit", self.locals["infos"][0]["total_profit"])
        self.logger.record("_reward/total_reward", self.locals["infos"][0]["total_reward"])
        self.logger.record_mean("_reward/mean_trade_duration", self.locals["infos"]
                                [0]["trade_duration"])
        self.logger.record("_actions/action", self.locals["infos"][0]["action"])
        self.logger.record("_actions/_Invalid", custom_info["Invalid"])
        self.logger.record("_actions/_Unknown", custom_info["Unknown"])
        self.logger.record("_actions/Hold", custom_info["Hold"])
        for action in self.actions:
            self.logger.record(f"_actions/{action.name}", custom_info[action.name])
        return True
