import logging
from datetime import datetime

from pandas import DataFrame

from freqtrade.exceptions import StrategyError


logger = logging.getLogger(__name__)


class StrategyResultValidator:
    def __init__(self, dataframe: DataFrame, warn_only: bool = False):
        self._warn_only = warn_only
        self._length: int = len(dataframe)
        self._close: float = dataframe["close"].iloc[-1]
        self._date: datetime = dataframe["date"].iloc[-1]

    def assert_df(self, dataframe: DataFrame):
        """
        Ensure dataframe (length, last candle) was not modified, and has all elements we need.
        Raises a StrategyError if the dataframe does not match the expected values.
        If warn_only is set, it will log a warning instead of raising an error.
        :param dataframe: DataFrame to validate
        :raises StrategyError: If the dataframe does not match the expected values.
        :logs Warning: If warn_only is set and the dataframe does not match the expected values.
        """
        message_template = "Dataframe returned from strategy has mismatching {}."
        message = ""
        if dataframe is None:
            message = "No dataframe returned (return statement missing?)."
        elif self._length != len(dataframe):
            message = message_template.format("length")
        elif self._close != dataframe["close"].iloc[-1]:
            message = message_template.format("last close price")
        elif self._date != dataframe["date"].iloc[-1]:
            message = message_template.format("last date")
        if message:
            if self._warn_only:
                logger.warning(message)
            else:
                raise StrategyError(message)
