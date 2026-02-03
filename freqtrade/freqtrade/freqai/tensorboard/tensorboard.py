import logging
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter
from xgboost import callback

from freqtrade.freqai.tensorboard.base_tensorboard import (
    BaseTensorBoardCallback,
    BaseTensorboardLogger,
)


logger = logging.getLogger(__name__)


class TensorboardLogger(BaseTensorboardLogger):
    def __init__(self, logdir: Path, activate: bool = True):
        self.activate = activate
        if self.activate:
            self.writer: SummaryWriter = SummaryWriter(f"{str(logdir)}/tensorboard")

    def log_scalar(self, tag: str, scalar_value: Any, step: int):
        if self.activate:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        if self.activate:
            self.writer.flush()
            self.writer.close()


class TensorBoardCallback(BaseTensorBoardCallback):
    def __init__(self, logdir: Path, activate: bool = True):
        self.activate = activate
        if self.activate:
            self.writer: SummaryWriter = SummaryWriter(f"{str(logdir)}/tensorboard")

    def after_iteration(
        self, model, epoch: int, evals_log: callback.TrainingCallback.EvalsLog
    ) -> bool:
        if not self.activate:
            return False
        if not evals_log:
            return False

        evals = ["validation", "train"]
        for metric, eval_ in zip(evals_log.items(), evals, strict=False):
            for metric_name, log in metric[1].items():
                score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                self.writer.add_scalar(f"{eval_}-{metric_name}", score, epoch)

        return False

    def after_training(self, model):
        if not self.activate:
            return model
        self.writer.flush()
        self.writer.close()

        return model
