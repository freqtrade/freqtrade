from datetime import datetime, timezone
from typing import Optional

import ccxt
from ccxt import ROUND_DOWN, ROUND_UP

from freqtrade.util.datetime_helpers import dt_from_ts, dt_ts


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Translates the timeframe interval value written in the human readable
    form ('1m', '5m', '1h', '1d', '1w', etc.) to the number
    of seconds for one timeframe interval.
    """
    return ccxt.Exchange.parse_timeframe(timeframe)


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Same as timeframe_to_seconds, but returns minutes.
    """
    return ccxt.Exchange.parse_timeframe(timeframe) // 60


def timeframe_to_msecs(timeframe: str) -> int:
    """
    Same as timeframe_to_seconds, but returns milliseconds.
    """
    return ccxt.Exchange.parse_timeframe(timeframe) * 1000


def timeframe_to_resample_freq(timeframe: str) -> str:
    """
    Translates the timeframe interval value written in the human readable
    form ('1m', '5m', '1h', '1d', '1w', etc.) to the resample frequency
    used by pandas ('1T', '5T', '1H', '1D', '1W', etc.)
    """
    if timeframe == "1y":
        return "1YS"
    timeframe_seconds = timeframe_to_seconds(timeframe)
    timeframe_minutes = timeframe_seconds // 60
    resample_interval = f"{timeframe_seconds}s"
    if 10000 < timeframe_minutes < 43200:
        resample_interval = "1W-MON"
    elif timeframe_minutes >= 43200 and timeframe_minutes < 525600:
        # Monthly candles need special treatment to stick to the 1st of the month
        resample_interval = f"{timeframe}S"
    elif timeframe_minutes > 43200:
        resample_interval = timeframe
    return resample_interval


def timeframe_to_prev_date(timeframe: str, date: Optional[datetime] = None) -> datetime:
    """
    Use Timeframe and determine the candle start date for this date.
    Does not round when given a candle start date.
    :param timeframe: timeframe in string format (e.g. "5m")
    :param date: date to use. Defaults to now(utc)
    :returns: date of previous candle (with utc timezone)
    """
    if not date:
        date = datetime.now(timezone.utc)

    new_timestamp = ccxt.Exchange.round_timeframe(timeframe, dt_ts(date), ROUND_DOWN) // 1000
    return dt_from_ts(new_timestamp)


def timeframe_to_next_date(timeframe: str, date: Optional[datetime] = None) -> datetime:
    """
    Use Timeframe and determine next candle.
    :param timeframe: timeframe in string format (e.g. "5m")
    :param date: date to use. Defaults to now(utc)
    :returns: date of next candle (with utc timezone)
    """
    if not date:
        date = datetime.now(timezone.utc)
    new_timestamp = ccxt.Exchange.round_timeframe(timeframe, dt_ts(date), ROUND_UP) // 1000
    return dt_from_ts(new_timestamp)
