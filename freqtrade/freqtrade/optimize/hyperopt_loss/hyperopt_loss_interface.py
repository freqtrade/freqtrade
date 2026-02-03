"""
IHyperOptLoss interface
This module defines the interface for the loss-function for hyperopt
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pandas import DataFrame

from freqtrade.constants import Config


class IHyperOptLoss(ABC):
    """
    Interface for freqtrade hyperopt Loss functions.
    Defines the custom loss function (`hyperopt_loss_function()` which is evaluated every epoch.)
    """

    timeframe: str

    @staticmethod
    @abstractmethod
    def hyperopt_loss_function(
        *,
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        processed: dict[str, DataFrame],
        backtest_stats: dict[str, Any],
        starting_balance: float,
        **kwargs,
    ) -> float:
        """
        Objective function, returns smaller number for better results
        """
