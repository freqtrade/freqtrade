import logging
from pathlib import Path
from typing import Any

import xgboost as xgb


logger = logging.getLogger(__name__)


class BaseTensorboardLogger:
    def __init__(self, logdir: Path):
        logger.warning("Tensorboard is not installed, no logs will be written."
                       "Ensure torch is installed, or use the torch/RL docker images")

    def log_scaler(self, tag: str, scalar_value: Any, step: int):
        return

    def close(self):
        return


class BaseTensorBoardCallback(xgb.callback.TrainingCallback):

    def __init__(self, logdir: Path):
        logger.warning("Tensorboard is not installed, no logs will be written."
                       "Ensure torch is installed, or use the torch/RL docker images")

    def after_iteration(
        self, model, epoch: int, evals_log: xgb.callback.TrainingCallback.EvalsLog
    ) -> bool:
        return False

    def after_training(self, model):
        return model
