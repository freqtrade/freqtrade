# ensure users can still use a non-torch freqai version
try:
    from freqtrade.freqai.tensorboard import TensorBoardCallback, TensorboardLogger
    TBLogger = TensorboardLogger
    TBCallback = TensorBoardCallback
except ModuleNotFoundError:
    from freqtrade.freqai.tensorboard import BaseTensorBoardCallback, BaseTensorboardLogger
    TBLogger = BaseTensorboardLogger  # type: ignore
    TBCallback = BaseTensorBoardCallback  # type: ignore

__all__ = (
    "TBLogger",
    "TBCallback"
)
