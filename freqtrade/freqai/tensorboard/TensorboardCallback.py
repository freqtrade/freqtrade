from enum import Enum
from typing import Any, Union

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from freqtrade.freqai.RL.BaseEnvironment import BaseActions


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard and
    episodic summary reports.
    """

    def __init__(self, verbose=1, actions: type[Enum] = BaseActions):
        super().__init__(verbose)
        self.model: Any = None
        self.actions: type[Enum] = actions

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": self.model.learning_rate,
            # "gamma": self.model.gamma,
            # "gae_lambda": self.model.gae_lambda,
            # "batch_size": self.model.batch_size,
            # "n_steps": self.model.n_steps,
        }
        metric_dict: dict[str, Union[float, int]] = {
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

        if hasattr(self.training_env, "envs"):
            tensorboard_metrics = self.training_env.envs[0].unwrapped.tensorboard_metrics

        else:
            # For RL-multiproc - usage of [0] might need to be evaluated
            tensorboard_metrics = self.training_env.get_attr("tensorboard_metrics")[0]

        for metric in local_info:
            if metric not in ["episode", "terminal_observation"]:
                self.logger.record(f"info/{metric}", local_info[metric])

        for category in tensorboard_metrics:
            for metric in tensorboard_metrics[category]:
                self.logger.record(f"{category}/{metric}", tensorboard_metrics[category][metric])

        return True
