# pragma pylint: disable=missing-docstring

import logging
from datetime import datetime
from typing import List, Dict, Tuple
import operator

import arrow
from pandas import DataFrame


from freqtrade.arguments import TimeRange
from freqtrade.optimize.default_hyperopt import DefaultHyperOpts  # noqa: F401

logger = logging.getLogger(__name__)


def get_timeframe(data: Dict[str, DataFrame]) -> Tuple[arrow.Arrow, arrow.Arrow]:
    """
    Get the maximum timeframe for the given backtest data
    :param data: dictionary with preprocessed backtesting data
    :return: tuple containing min_date, max_date
    """
    timeframe = [
        (arrow.get(frame['date'].min()), arrow.get(frame['date'].max()))
        for frame in data.values()
    ]
    return min(timeframe, key=operator.itemgetter(0))[0], \
        max(timeframe, key=operator.itemgetter(1))[1]


def validate_backtest_data(data: Dict[str, DataFrame], min_date: datetime,
                           max_date: datetime, ticker_interval_mins: int) -> bool:
    """
    Validates preprocessed backtesting data for missing values and shows warnings about it that.

    :param data: dictionary with preprocessed backtesting data
    :param min_date: start-date of the data
    :param max_date: end-date of the data
    :param ticker_interval_mins: ticker interval in minutes
    """
    # total difference in minutes / interval-minutes
    expected_frames = int((max_date - min_date).total_seconds() // 60 // ticker_interval_mins)
    found_missing = False
    for pair, df in data.items():
        dflen = len(df)
        if dflen < expected_frames:
            found_missing = True
            logger.warning("%s has missing frames: expected %s, got %s, that's %s missing values",
                           pair, expected_frames, dflen, expected_frames - dflen)
    return found_missing
