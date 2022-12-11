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

        local_info = self.locals["infos"][0]
        tensorboard_metrics = self.training_env.get_attr("tensorboard_metrics")[0]

        for info in local_info:
            if info not in ["episode", "terminal_observation"]:
                self.logger.record(f"_info/{info}", local_info[info])

        for info in tensorboard_metrics:
            if info in [action.name for action in self.actions]:
                self.logger.record(f"_actions/{info}", tensorboard_metrics[info])
            else:
                self.logger.record(f"_custom/{info}", tensorboard_metrics[info])

        return True
