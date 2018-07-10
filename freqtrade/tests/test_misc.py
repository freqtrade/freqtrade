# pragma pylint: disable=missing-docstring,C0103

"""
Unit test file for misc.py
"""

import datetime
from unittest.mock import MagicMock

from freqtrade.analyze import Analyze
from freqtrade.misc import (common_datearray, datesarray_to_datetimearray,
                            file_dump_json, format_ms_time, shorten_date)
from freqtrade.optimize.__init__ import load_tickerdata_file


def test_shorten_date() -> None:
    """
    Test shorten_date() function
    :return: None
    """
    str_data = '1 day, 2 hours, 3 minutes, 4 seconds ago'
    str_shorten_data = '1 d, 2 h, 3 min, 4 sec ago'
    assert shorten_date(str_data) == str_shorten_data


def test_datesarray_to_datetimearray(ticker_history):
    """
    Test datesarray_to_datetimearray() function
    :return: None
    """
    dataframes = Analyze.parse_ticker_dataframe(ticker_history)
    dates = datesarray_to_datetimearray(dataframes['date'])

    assert isinstance(dates[0], datetime.datetime)
    assert dates[0].year == 2017
    assert dates[0].month == 11
    assert dates[0].day == 26
    assert dates[0].hour == 8
    assert dates[0].minute == 50

    date_len = len(dates)
    assert date_len == 2


def test_common_datearray(default_conf) -> None:
    """
    Test common_datearray()
    :return: None
    """
    analyze = Analyze(default_conf)
    tick = load_tickerdata_file(None, 'UNITTEST/BTC', '1m')
    tickerlist = {'UNITTEST/BTC': tick}
    dataframes = analyze.tickerdata_to_dataframe(tickerlist)

    dates = common_datearray(dataframes)

    assert dates.size == dataframes['UNITTEST/BTC']['date'].size
    assert dates[0] == dataframes['UNITTEST/BTC']['date'][0]
    assert dates[-1] == dataframes['UNITTEST/BTC']['date'][-1]


def test_file_dump_json(mocker) -> None:
    """
    Test file_dump_json()
    :return: None
    """
    file_open = mocker.patch('freqtrade.misc.open', MagicMock())
    json_dump = mocker.patch('json.dump', MagicMock())
    file_dump_json('somefile', [1, 2, 3])
    assert file_open.call_count == 1
    assert json_dump.call_count == 1
    file_open = mocker.patch('freqtrade.misc.gzip.open', MagicMock())
    json_dump = mocker.patch('json.dump', MagicMock())
    file_dump_json('somefile', [1, 2, 3], True)
    assert file_open.call_count == 1
    assert json_dump.call_count == 1


def test_format_ms_time() -> None:
    """
    test format_ms_time()
    :return: None
    """
    # Date 2018-04-10 18:02:01
    date_in_epoch_ms = 1523383321000
    date = format_ms_time(date_in_epoch_ms)
    assert type(date) is str
    res = datetime.datetime(2018, 4, 10, 18, 2, 1, tzinfo=datetime.timezone.utc)
    assert date == res.astimezone(None).strftime('%Y-%m-%dT%H:%M:%S')
    res = datetime.datetime(2017, 12, 13, 8, 2, 1, tzinfo=datetime.timezone.utc)
    # Date 2017-12-13 08:02:01
    date_in_epoch_ms = 1513152121000
    assert format_ms_time(date_in_epoch_ms) == res.astimezone(None).strftime('%Y-%m-%dT%H:%M:%S')
