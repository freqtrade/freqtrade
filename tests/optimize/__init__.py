from typing import Dict, List, NamedTuple, Optional

import arrow
from pandas import DataFrame

from freqtrade.enums import SellType
from freqtrade.exchange import timeframe_to_minutes


tests_start_time = arrow.get(2018, 10, 3)
tests_timeframe = '1h'


class BTrade(NamedTuple):
    """
    Minimalistic Trade result used for functional backtesting
    """
    sell_reason: SellType
    open_tick: int
    close_tick: int
    buy_tag: Optional[str] = ''


class BTContainer(NamedTuple):
    """
    Minimal BacktestContainer defining Backtest inputs and results.
    """
    data: List[List[float]]
    stop_loss: float
    roi: Dict[str, float]
    trades: List[BTrade]
    profit_perc: float
    trailing_stop: bool = False
    trailing_only_offset_is_reached: bool = False
    trailing_stop_positive: Optional[float] = None
    trailing_stop_positive_offset: float = 0.0
    use_sell_signal: bool = False
    use_custom_stoploss: bool = False


def _get_frame_time_from_offset(offset):
    minutes = offset * timeframe_to_minutes(tests_timeframe)
    return tests_start_time.shift(minutes=minutes).datetime


def _build_backtest_dataframe(data):
    columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'buy', 'sell', 'buy_tag']

    frame = DataFrame.from_records(data, columns=columns)
    frame['date'] = frame['date'].apply(_get_frame_time_from_offset)
    # Ensure floats are in place
    for column in ['open', 'high', 'low', 'close', 'volume']:
        frame[column] = frame[column].astype('float64')
    return frame
