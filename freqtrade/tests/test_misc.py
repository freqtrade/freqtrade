# pragma pylint: disable=missing-docstring,C0103

"""
Unit test file for misc.py
"""

import datetime
from unittest.mock import MagicMock
from freqtrade.analyze import Analyze
from freqtrade.misc import (shorten_date, datesarray_to_datetimearray, file_dump_json)


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
    assert date_len == 3


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
