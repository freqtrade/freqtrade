"""
This module contains the argument manager class
"""
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from typing_extensions import Self

from freqtrade.constants import DATETIME_PRINT_FORMAT
from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


class TimeRange:
    """
    object defining timerange inputs.
    [start/stop]type defines if [start/stop]ts shall be used.
    if *type is None, don't use corresponding startvalue.
    """

    def __init__(self, starttype: Optional[str] = None, stoptype: Optional[str] = None,
                 startts: int = 0, stopts: int = 0):

        self.starttype: Optional[str] = starttype
        self.stoptype: Optional[str] = stoptype
        self.startts: int = startts
        self.stopts: int = stopts

    @property
    def startdt(self) -> Optional[datetime]:
        if self.startts:
            return datetime.fromtimestamp(self.startts, tz=timezone.utc)
        return None

    @property
    def stopdt(self) -> Optional[datetime]:
        if self.stopts:
            return datetime.fromtimestamp(self.stopts, tz=timezone.utc)
        return None

    @property
    def timerange_str(self) -> str:
        """
        Returns a string representation of the timerange as used by parse_timerange.
        Follows the format yyyymmdd-yyyymmdd - leaving out the parts that are not set.
        """
        start = ''
        stop = ''
        if startdt := self.startdt:
            start = startdt.strftime('%Y%m%d')
        if stopdt := self.stopdt:
            stop = stopdt.strftime('%Y%m%d')
        return f"{start}-{stop}"

    @property
    def start_fmt(self) -> str:
        """
        Returns a string representation of the start date
        """
        val = 'unbounded'
        if (startdt := self.startdt) is not None:
            val = startdt.strftime(DATETIME_PRINT_FORMAT)
        return val

    @property
    def stop_fmt(self) -> str:
        """
        Returns a string representation of the stop date
        """
        val = 'unbounded'
        if (stopdt := self.stopdt) is not None:
            val = stopdt.strftime(DATETIME_PRINT_FORMAT)
        return val

    def __eq__(self, other):
        """Override the default Equals behavior"""
        return (self.starttype == other.starttype and self.stoptype == other.stoptype
                and self.startts == other.startts and self.stopts == other.stopts)

    def subtract_start(self, seconds: int) -> None:
        """
        Subtracts <seconds> from startts if startts is set.
        :param seconds: Seconds to subtract from starttime
        :return: None (Modifies the object in place)
        """
        if self.startts:
            self.startts = self.startts - seconds

    def adjust_start_if_necessary(self, timeframe_secs: int, startup_candles: int,
                                  min_date: datetime) -> None:
        """
        Adjust startts by <startup_candles> candles.
        Applies only if no startup-candles have been available.
        :param timeframe_secs: Timeframe in seconds e.g. `timeframe_to_seconds('5m')`
        :param startup_candles: Number of candles to move start-date forward
        :param min_date: Minimum data date loaded. Key kriterium to decide if start-time
                         has to be moved
        :return: None (Modifies the object in place)
        """
        if (not self.starttype or (startup_candles
                                   and min_date.timestamp() >= self.startts)):
            # If no startts was defined, or backtest-data starts at the defined backtest-date
            logger.warning("Moving start-date by %s candles to account for startup time.",
                           startup_candles)
            self.startts = int(min_date.timestamp() + timeframe_secs * startup_candles)
            self.starttype = 'date'

    @classmethod
    def parse_timerange(cls, text: Optional[str]) -> Self:
        """
        Parse the value of the argument --timerange to determine what is the range desired
        :param text: value from --timerange
        :return: Start and End range period
        """
        if not text:
            return cls(None, None, 0, 0)
        syntax = [(r'^-(\d{8})$', (None, 'date')),
                  (r'^(\d{8})-$', ('date', None)),
                  (r'^(\d{8})-(\d{8})$', ('date', 'date')),
                  (r'^-(\d{10})$', (None, 'date')),
                  (r'^(\d{10})-$', ('date', None)),
                  (r'^(\d{10})-(\d{10})$', ('date', 'date')),
                  (r'^-(\d{13})$', (None, 'date')),
                  (r'^(\d{13})-$', ('date', None)),
                  (r'^(\d{13})-(\d{13})$', ('date', 'date')),
                  ]
        for rex, stype in syntax:
            # Apply the regular expression to text
            match = re.match(rex, text)
            if match:  # Regex has matched
                rvals = match.groups()
                index = 0
                start: int = 0
                stop: int = 0
                if stype[0]:
                    starts = rvals[index]
                    if stype[0] == 'date' and len(starts) == 8:
                        start = int(datetime.strptime(starts, '%Y%m%d').replace(
                            tzinfo=timezone.utc).timestamp())
                    elif len(starts) == 13:
                        start = int(starts) // 1000
                    else:
                        start = int(starts)
                    index += 1
                if stype[1]:
                    stops = rvals[index]
                    if stype[1] == 'date' and len(stops) == 8:
                        stop = int(datetime.strptime(stops, '%Y%m%d').replace(
                            tzinfo=timezone.utc).timestamp())
                    elif len(stops) == 13:
                        stop = int(stops) // 1000
                    else:
                        stop = int(stops)
                if start > stop > 0:
                    raise OperationalException(
                        f'Start date is after stop date for timerange "{text}"')
                return cls(stype[0], stype[1], start, stop)
        raise OperationalException(f'Incorrect syntax for timerange "{text}"')
