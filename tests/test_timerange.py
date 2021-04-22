# pragma pylint: disable=missing-docstring, C0103
import arrow
import pytest

from freqtrade.configuration import TimeRange
from freqtrade.exceptions import OperationalException


def test_parse_timerange_incorrect():

    assert TimeRange('date', None, 1274486400, 0) == TimeRange.parse_timerange('20100522-')
    assert TimeRange(None, 'date', 0, 1274486400) == TimeRange.parse_timerange('-20100522')
    timerange = TimeRange.parse_timerange('20100522-20150730')
    assert timerange == TimeRange('date', 'date', 1274486400, 1438214400)

    # Added test for unix timestamp - BTC genesis date
    assert TimeRange('date', None, 1231006505, 0) == TimeRange.parse_timerange('1231006505-')
    assert TimeRange(None, 'date', 0, 1233360000) == TimeRange.parse_timerange('-1233360000')
    timerange = TimeRange.parse_timerange('1231006505-1233360000')
    assert TimeRange('date', 'date', 1231006505, 1233360000) == timerange

    timerange = TimeRange.parse_timerange('1231006505000-1233360000000')
    assert TimeRange('date', 'date', 1231006505, 1233360000) == timerange

    timerange = TimeRange.parse_timerange('1231006505000-')
    assert TimeRange('date', None, 1231006505, 0) == timerange

    timerange = TimeRange.parse_timerange('-1231006505000')
    assert TimeRange(None, 'date', 0, 1231006505) == timerange

    with pytest.raises(OperationalException, match=r'Incorrect syntax.*'):
        TimeRange.parse_timerange('-')

    with pytest.raises(OperationalException,
                       match=r'Start date is after stop date for timerange.*'):
        TimeRange.parse_timerange('20100523-20100522')


def test_subtract_start():
    x = TimeRange('date', 'date', 1274486400, 1438214400)
    x.subtract_start(300)
    assert x.startts == 1274486400 - 300

    # Do nothing if no startdate exists
    x = TimeRange(None, 'date', 0, 1438214400)
    x.subtract_start(300)
    assert not x.startts

    x = TimeRange('date', None, 1274486400, 0)
    x.subtract_start(300)
    assert x.startts == 1274486400 - 300


def test_adjust_start_if_necessary():
    min_date = arrow.Arrow(2017, 11, 14, 21, 15, 00)

    x = TimeRange('date', 'date', 1510694100, 1510780500)
    # Adjust by 20 candles - min_date == startts
    x.adjust_start_if_necessary(300, 20, min_date)
    assert x.startts == 1510694100 + (20 * 300)

    x = TimeRange('date', 'date', 1510700100, 1510780500)
    # Do nothing, startup is set and different min_date
    x.adjust_start_if_necessary(300, 20, min_date)
    assert x.startts == 1510694100 + (20 * 300)

    x = TimeRange(None, 'date', 0, 1510780500)
    # Adjust by 20 candles = 20 * 5m
    x.adjust_start_if_necessary(300, 20, min_date)
    assert x.startts == 1510694100 + (20 * 300)
