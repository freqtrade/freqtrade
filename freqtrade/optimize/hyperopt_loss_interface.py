"""
IHyperOptLoss interface
This module defines the interface for the loss-function for hyperopt
"""

from abc import ABC, abstractmethod
from datetime import datetime

from pandas import DataFrame


class IHyperOptLoss(ABC):
    """
    Interface for freqtrade hyperopt Loss functions.
    Defines the custom loss function (`hyperopt_loss_function()` which is evaluated every epoch.)
    """
    ticker_interval: str

    @staticmethod
    @abstractmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime, *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for better results
        """
