"""
This module contains the argument manager class
"""
import re
from typing import Optional

import arrow


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

    @staticmethod
    def parse_timerange(text: Optional[str]):
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
                        start = arrow.get(starts, 'YYYYMMDD').timestamp
                    elif len(starts) == 13:
                        start = int(starts) // 1000
                    else:
                        start = int(starts)
                    index += 1
                if stype[1]:
                    stops = rvals[index]
                    if stype[1] == 'date' and len(stops) == 8:
                        stop = arrow.get(stops, 'YYYYMMDD').timestamp
                    elif len(stops) == 13:
                        stop = int(stops) // 1000
                    else:
                        stop = int(stops)
                return TimeRange(stype[0], stype[1], start, stop)
        raise Exception('Incorrect syntax for timerange "%s"' % text)
