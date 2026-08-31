"""
This module contains the argument manager class
"""

import logging
import re
from datetime import UTC, datetime
from typing import Self

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.exceptions import ConfigurationError
from freqtrade.util import dt_from_ts


logger = logging.getLogger(__name__)

# Supported formats for one side of a timerange.
# Datetime formats are interpreted as UTC - a format of None means "epoch"
# (seconds for 10 digits, milliseconds for 13 digits).
_TIMERANGE_FORMATS: list[tuple[str, str | None]] = [
    (r"\d{8}", "%Y%m%d"),
    (r"\d{8}T\d{4}", "%Y%m%dT%H%M"),
    (r"\d{8}T\d{6}", "%Y%m%dT%H%M%S"),
    (r"\d{10}", None),
    (r"\d{13}", None),
]


class TimeRange:
    """
    object defining timerange inputs.
    [start/stop]type defines if [start/stop]ts shall be used.
    if *type is None, don't use corresponding startvalue.
    """

    def __init__(
        self,
        starttype: str | None = None,
        stoptype: str | None = None,
        startts: int = 0,
        stopts: int = 0,
    ):
        self.starttype: str | None = starttype
        self.stoptype: str | None = stoptype
        self.startts: int = startts
        self.stopts: int = stopts

    def copy(self) -> Self:
        """Return an independent copy of this timerange."""
        return type(self)(
            starttype=self.starttype,
            stoptype=self.stoptype,
            startts=self.startts,
            stopts=self.stopts,
        )

    @property
    def startdt(self) -> datetime | None:
        if self.startts:
            return dt_from_ts(self.startts)
        return None

    @property
    def stopdt(self) -> datetime | None:
        if self.stopts:
            return dt_from_ts(self.stopts)
        return None

    @staticmethod
    def _format_dt(dt: datetime) -> str:
        """
        Format a datetime with the lowest precision that keeps all information.
        yyyymmdd for midnight, yyyymmddThhmm otherwise - yyyymmddThhmmss if seconds are set.
        """
        if dt.second:
            return dt.strftime("%Y%m%dT%H%M%S")
        if dt.hour or dt.minute:
            return dt.strftime("%Y%m%dT%H%M")
        return dt.strftime("%Y%m%d")

    @property
    def timerange_str(self) -> str:
        """
        Returns a string representation of the timerange as used by parse_timerange.
        Follows the format yyyymmdd-yyyymmdd - leaving out the parts that are not set.
        Timeranges that are not aligned to midnight use the yyyymmddThhmm[ss] format instead.
        """
        start = ""
        stop = ""
        if startdt := self.startdt:
            start = self._format_dt(startdt)
        if stopdt := self.stopdt:
            stop = self._format_dt(stopdt)
        return f"{start}-{stop}"

    @property
    def start_fmt(self) -> str:
        """
        Returns a string representation of the start date
        """
        val = "unbounded"
        if (startdt := self.startdt) is not None:
            val = startdt.strftime(DATETIME_PRINT_FORMAT)
        return val

    @property
    def stop_fmt(self) -> str:
        """
        Returns a string representation of the stop date
        """
        val = "unbounded"
        if (stopdt := self.stopdt) is not None:
            val = stopdt.strftime(DATETIME_PRINT_FORMAT)
        return val

    def __repr__(self) -> str:
        return f"TimeRange({self.timerange_str})"

    def __eq__(self, other):
        """Override the default Equals behavior"""
        return (
            self.starttype == other.starttype
            and self.stoptype == other.stoptype
            and self.startts == other.startts
            and self.stopts == other.stopts
        )

    def subtract_start(self, seconds: int) -> None:
        """
        Subtracts <seconds> from startts if startts is set.
        :param seconds: Seconds to subtract from starttime
        :return: None (Modifies the object in place)
        """
        if self.startts:
            self.startts = self.startts - seconds

    def adjust_start_if_necessary(
        self, timeframe_secs: int, startup_candles: int, min_date: datetime
    ) -> None:
        """
        Adjust startts by <startup_candles> candles.
        Applies only if no startup-candles have been available.
        :param timeframe_secs: Timeframe in seconds e.g. `timeframe_to_seconds('5m')`
        :param startup_candles: Number of candles to move start-date forward
        :param min_date: Minimum data date loaded. Key kriterium to decide if start-time
                         has to be moved
        :return: None (Modifies the object in place)
        """
        if not self.starttype or (startup_candles and min_date.timestamp() >= self.startts):
            # If no startts was defined, or backtest-data starts at the defined backtest-date
            logger.warning(
                "Moving start-date by %s candles to account for startup time.", startup_candles
            )
            self.startts = int(min_date.timestamp() + timeframe_secs * startup_candles)
            self.starttype = "date"

    @staticmethod
    def _parse_timerange_part(text: str) -> int:
        """
        Parse one side of a timerange to a timestamp in seconds.
        :param text: One side of the timerange - an empty string means "unbounded"
        :return: Timestamp in seconds - 0 if unbounded
        :raises ValueError: If the value doesn't match any supported format
        """
        if not text:
            return 0
        for rex, fmt in _TIMERANGE_FORMATS:
            if not re.fullmatch(rex, text):
                continue
            if fmt is None:
                # Epoch - in seconds (10 digits) or milliseconds (13 digits)
                return int(text) // 1000 if len(text) == 13 else int(text)
            return int(datetime.strptime(text, fmt).replace(tzinfo=UTC).timestamp())
        raise ValueError(f'Invalid timerange value "{text}"')

    @classmethod
    def parse_timerange(cls, text: str | None) -> Self:
        """
        Parse the value of the argument --timerange to determine what is the range desired.
        Both sides are parsed independently and may use any of the supported formats:
        yyyymmdd, yyyymmddThhmm, yyyymmddThhmmss, epoch (seconds) or epoch (milliseconds).
        :param text: value from --timerange
        :return: Start and End range period
        """
        if not text:
            return cls(None, None, 0, 0)
        parts = text.split("-")
        if len(parts) != 2 or not any(parts):
            raise ConfigurationError(f'Incorrect syntax for timerange "{text}"')
        starts, stops = parts
        try:
            start = cls._parse_timerange_part(starts)
            stop = cls._parse_timerange_part(stops)
        except ValueError as e:
            raise ConfigurationError(f'Incorrect syntax for timerange "{text}"') from e

        if start > stop > 0:
            raise ConfigurationError(f'Start date is after stop date for timerange "{text}"')
        return cls("date" if starts else None, "date" if stops else None, start, stop)
