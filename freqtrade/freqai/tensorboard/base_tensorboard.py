import logging
from pathlib import Path
from typing import Any

from xgboost.callback import TrainingCallback


logger = logging.getLogger(__name__)


class BaseTensorboardLogger:
    def __init__(self, logdir: Path, activate: bool = True):
        logger.warning("Tensorboard is not installed, no logs will be written."
                       "Ensure torch is installed, or use the torch/RL docker images")

    def log_scalar(self, tag: str, scalar_value: Any, step: int):
        return

    def close(self):
        return


class BaseTensorBoardCallback(TrainingCallback):

    def __init__(self, logdir: Path, activate: bool = True):
        logger.warning("Tensorboard is not installed, no logs will be written."
                       "Ensure torch is installed, or use the torch/RL docker images")

    def after_iteration(
        self, model, epoch: int, evals_log: TrainingCallback.EvalsLog
    ) -> bool:
        return False

    def after_training(self, model):
        return model
