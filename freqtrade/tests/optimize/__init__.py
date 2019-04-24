from typing import NamedTuple, List

import arrow
from pandas import DataFrame

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy.interface import SellType

ticker_start_time = arrow.get(2018, 10, 3)
tests_ticker_interval = '1h'


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
    trailing_stop: bool = False


def _get_frame_time_from_offset(offset):
    return ticker_start_time.shift(minutes=(offset * timeframe_to_minutes(tests_ticker_interval))
                                   ).datetime


def _build_backtest_dataframe(ticker_with_signals):
    columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'buy', 'sell']

    frame = DataFrame.from_records(ticker_with_signals, columns=columns)
    frame['date'] = frame['date'].apply(_get_frame_time_from_offset)
    # Ensure floats are in place
    for column in ['open', 'high', 'low', 'close', 'volume']:
        frame[column] = frame[column].astype('float64')
    return frame
