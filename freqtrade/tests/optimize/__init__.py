from typing import NamedTuple, List

import arrow
from pandas import DataFrame

from freqtrade.strategy.interface import SellType

ticker_start_time = arrow.get(2018, 10, 3)
ticker_interval_in_minute = 60


class BTrade(NamedTuple):
    """
    Minimalistic Trade result used for functional backtesting
    """
    sell_reason: SellType
    open_tick: int
    close_tick: int


class BTContainer(NamedTuple):
    """
    Minimal BacktestContainer defining Backtest inputs and results.
    """
    data: List[float]
    stop_loss: float
    roi: float
    trades: List[BTrade]
    profit_perc: float


def _get_frame_time_from_offset(offset):
    return ticker_start_time.shift(
        minutes=(offset * ticker_interval_in_minute)).datetime


def _build_backtest_dataframe(ticker_with_signals):
    columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'buy', 'sell']

    frame = DataFrame.from_records(ticker_with_signals, columns=columns)
    frame['date'] = frame['date'].apply(_get_frame_time_from_offset)
    # Ensure floats are in place
    for column in ['open', 'high', 'low', 'close', 'volume']:
        frame[column] = frame[column].astype('float64')
    return frame
