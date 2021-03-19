"""
This module contains the argument manager class
"""
import logging
import re
from typing import Optional

import arrow

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
                                  min_date: arrow.Arrow) -> None:
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
                                   and min_date.int_timestamp >= self.startts)):
            # If no startts was defined, or backtest-data starts at the defined backtest-date
            logger.warning("Moving start-date by %s candles to account for startup time.",
                           startup_candles)
            self.startts = (min_date.int_timestamp + timeframe_secs * startup_candles)
            self.starttype = 'date'

    @staticmethod
    def parse_timerange(text: Optional[str]) -> 'TimeRange':
        """
        Parse the value of the argument --timerange to determine what is the range desired
        :param text: value from --timerange
        :return: Start and End range period
        """
        if text is None:
            return TimeRange(None, None, 0, 0)
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
                        start = arrow.get(starts, 'YYYYMMDD').int_timestamp
                    elif len(starts) == 13:
                        start = int(starts) // 1000
                    else:
                        start = int(starts)
                    index += 1
                if stype[1]:
                    stops = rvals[index]
                    if stype[1] == 'date' and len(stops) == 8:
                        stop = arrow.get(stops, 'YYYYMMDD').int_timestamp
                    elif len(stops) == 13:
                        stop = int(stops) // 1000
                    else:
                        stop = int(stops)
                if start > stop > 0:
                    raise OperationalException(
                        f'Start date is after stop date for timerange "{text}"')
                return TimeRange(stype[0], stype[1], start, stop)
        raise OperationalException(f'Incorrect syntax for timerange "{text}"')
