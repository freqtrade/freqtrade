# ensure users can still use a non-torch freqai version
try:
    from freqtrade.freqai.tensorboard.lightgbm_callback import LightGBMTensorboardCallback
    from freqtrade.freqai.tensorboard.tensorboard import TensorBoardCallback, TensorboardLogger

    TBLogger = TensorboardLogger
    TBCallback = TensorBoardCallback
    LightGBMCallback = LightGBMTensorboardCallback
except ModuleNotFoundError:
    from freqtrade.freqai.tensorboard.base_tensorboard import (
        BaseTensorBoardCallback,
        BaseTensorboardLogger,
    )

    TBLogger = BaseTensorboardLogger  # type: ignore
    TBCallback = BaseTensorBoardCallback  # type: ignore
    LightGBMCallback = None  # type: ignore

__all__ = ("TBLogger", "TBCallback", "LightGBMCallback")
