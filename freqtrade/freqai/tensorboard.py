import logging
from pathlib import Path
from typing import Any

import xgboost as xgb


logger = logging.getLogger(__name__)


class BaseTensorboardLogger:
    def __init__(self, logdir: str = "tensorboard", id: str = "unique-id"):
        logger.warning("Tensorboard is not installed, no logs will be written."
                       "Use ensure torch is installed, or use the torch/RL docker images")

    def log_scaler(self, tag: str, scalar_value: Any, step: int):
        return

    def close(self):
        return


class BaseTensorBoardCallback(xgb.callback.TrainingCallback):

    def __init__(self, logdir: str = "tensorboard", id: str = "uniqu-id", test_size=1):
        logger.warning("Tensorboard is not installed, no logs will be written."
                       "Use ensure torch is installed, or use the torch/RL docker images")

    def after_iteration(
        self, model, epoch: int, evals_log: xgb.callback.TrainingCallback.EvalsLog
    ) -> bool:
        return False

    def after_training(self, model):
        return model


class TensorboardLogger(BaseTensorboardLogger):
    def __init__(self, logdir: Path = Path("tensorboard")):
        from torch.utils.tensorboard import SummaryWriter
        self.writer: SummaryWriter = SummaryWriter(f"{str(logdir)}/tensorboard")

    def log_scalar(self, tag: str, scalar_value: Any, step: int):
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.flush()
        self.writer.close()


class TensorBoardCallback(BaseTensorBoardCallback):

    def __init__(self, logdir: Path = Path("tensorboard")):
        from torch.utils.tensorboard import SummaryWriter
        self.writer: SummaryWriter = SummaryWriter(f"{str(logdir)}/tensorboard")

    def after_iteration(
        self, model, epoch: int, evals_log: xgb.callback.TrainingCallback.EvalsLog
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
