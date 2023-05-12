# ensure users can still use a non-torch freqai version
try:
    from freqtrade.freqai.tensorboard.tensorboard import TensorBoardCallback, TensorboardLogger
    TBLogger = TensorboardLogger
    TBCallback = TensorBoardCallback
except ModuleNotFoundError:
    from freqtrade.freqai.tensorboard.base_tensorboard import (BaseTensorBoardCallback,
                                                               BaseTensorboardLogger)
    TBLogger = BaseTensorboardLogger
    TBCallback = BaseTensorBoardCallback

__all__ = (
    "TBLogger",
    "TBCallback"
)
