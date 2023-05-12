import logging
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter
from xgboost import callback

from freqtrade.freqai.tensorboard.base_tensorboard import (BaseTensorBoardCallback,
                                                           BaseTensorboardLogger)


logger = logging.getLogger(__name__)


class TensorboardLogger(BaseTensorboardLogger):
    def __init__(self, logdir: Path):
        self.writer: SummaryWriter = SummaryWriter(f"{str(logdir)}/tensorboard")

    def log_scalar(self, tag: str, scalar_value: Any, step: int):
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.flush()
        self.writer.close()


class TensorBoardCallback(BaseTensorBoardCallback):

    def __init__(self, logdir: Path):
        self.writer: SummaryWriter = SummaryWriter(f"{str(logdir)}/tensorboard")

    def after_iteration(
        self, model, epoch: int, evals_log: callback.TrainingCallback.EvalsLog
    ) -> bool:
        if not evals_log:
            return False

        for data, metric in evals_log.items():
            for metric_name, log in metric.items():
                score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                if data == "train":
                    self.writer.add_scalar("train_loss", score**2, epoch)
                else:
                    self.writer.add_scalar("valid_loss", score**2, epoch)

        return False

    def after_training(self, model):
        self.writer.flush()
        self.writer.close()

        return model
