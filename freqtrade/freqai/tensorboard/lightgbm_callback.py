from __future__ import annotations

from freqtrade.freqai.tensorboard.tensorboard import TensorboardLogger


class LightGBMTensorboardCallback:
    def __init__(self, logdir, activate: bool) -> None:
        self.activate = activate
        self.logger = TensorboardLogger(logdir, activate)

    def __call__(self, env) -> None:
        if not self.activate:
            return

        evals = getattr(env, "evaluation_result_list", None)
        if not evals:
            return

        for data_name, metric_name, value, _ in evals:
            self.logger.log_scalar(f"{data_name}-{metric_name}", value, env.iteration)

        end_iteration = getattr(env, "end_iteration", None)
        if end_iteration is not None and env.iteration + 1 >= end_iteration:
            self.logger.close()
