import re
from datetime import UTC, datetime
from time import time

import humanize

from freqtrade.constants import DATETIME_PRINT_FORMAT


def dt_now() -> datetime:
    """Return the current datetime in UTC."""
    return datetime.now(UTC)


def dt_now_no_micro() -> datetime:
    """Return the current datetime in UTC without microseconds.
    Should not be used outside of tests.
    """
    return dt_now().replace(microsecond=0)


def dt_utc(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
) -> datetime:
    """Return a datetime in UTC."""
    return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=UTC)


def dt_ts(dt: datetime | None = None) -> int:
    """
    Return dt in ms as a timestamp in UTC.
    If dt is None, return the current datetime in UTC.
    """
    if dt:
        return int(dt.timestamp() * 1000)
    return int(time() * 1000)


def dt_ts_def(dt: datetime | None, default: int = 0) -> int:
    """
    Return dt in ms as a timestamp in UTC.
    If dt is None, return the given default.
    """
    if dt:
        return int(dt.timestamp() * 1000)
    return default


def dt_ts_none(dt: datetime | None) -> int | None:
    """
    Return dt in ms as a timestamp in UTC.
    If dt is None, return the given default.
    """
    if dt:
        return int(dt.timestamp() * 1000)
    return None


def dt_floor_day(dt: datetime) -> datetime:
    """Return the floor of the day for the given datetime."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def dt_from_ts(timestamp: float) -> datetime:
    """
    Return a datetime from a timestamp.
    :param timestamp: timestamp in seconds or milliseconds
    """
    if timestamp > 1e10:
        # Timezone in ms - convert to seconds
        timestamp /= 1000
    return datetime.fromtimestamp(timestamp, tz=UTC)


_SHORTEN_DATE_SUBS = [
    (re.compile("seconds?"), "sec"),
    (re.compile("minutes?"), "min"),
    (re.compile("hours?"), "h"),
    (re.compile("days?"), "d"),
    (re.compile("^an?"), "1"),
]


def shorten_date(_date: str) -> str:
    """
    Trim the date so it fits on small screens
    """
    new_date = _date
    for pattern, repl in _SHORTEN_DATE_SUBS:
        new_date = pattern.sub(repl, new_date)
    return new_date


def dt_humanize_delta(dt: datetime):
    """
    Return a humanized string for the given timedelta.
    """
    return humanize.naturaltime(dt)


def format_date(date: datetime | None, fallback: str = "") -> str:
    """
    Return a formatted date string.
    Returns an empty string if date is None.
    :param date: datetime to format
    :param fallback: value to return if date is None
    """
    if date:
        return date.strftime(DATETIME_PRINT_FORMAT)
    return fallback


def format_ms_time(date: float) -> str:
    """
    convert MS date to readable format.
    : epoch-string in ms
    """
    return dt_from_ts(date).strftime("%Y-%m-%dT%H:%M:%S")


def format_ms_time_det(date: float) -> str:
    """
    convert MS date to readable format - detailed.
    : epoch-string in ms
    """
    # return dt_from_ts(date).isoformat(timespec="milliseconds")
    return dt_from_ts(date).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
