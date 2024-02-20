from datetime import datetime, timedelta, timezone

import pytest
import time_machine

from freqtrade.util import (dt_floor_day, dt_from_ts, dt_humanize, dt_now, dt_ts, dt_ts_def,
                            dt_ts_none, dt_utc, format_date, format_ms_time, shorten_date)


def test_dt_now():
    with time_machine.travel("2021-09-01 05:01:00 +00:00", tick=False) as t:
        now = datetime.now(timezone.utc)
        assert dt_now() == now
        assert dt_ts() == int(now.timestamp() * 1000)
        assert dt_ts(now) == int(now.timestamp() * 1000)

        t.shift(timedelta(hours=5))
        assert dt_now() >= now
        assert dt_now() == datetime.now(timezone.utc)
        assert dt_ts() == int(dt_now().timestamp() * 1000)
        # Test with different time than now
        assert dt_ts(now) == int(now.timestamp() * 1000)


def test_dt_ts_def():
    assert dt_ts_def(None) == 0
    assert dt_ts_def(None, 123) == 123
    assert dt_ts_def(datetime(2023, 5, 5, tzinfo=timezone.utc)) == 1683244800000
    assert dt_ts_def(datetime(2023, 5, 5, tzinfo=timezone.utc), 123) == 1683244800000


def test_dt_ts_none():
    assert dt_ts_none(None) is None
    assert dt_ts_none(None) is None
    assert dt_ts_none(datetime(2023, 5, 5, tzinfo=timezone.utc)) == 1683244800000
    assert dt_ts_none(datetime(2023, 5, 5, tzinfo=timezone.utc)) == 1683244800000


def test_dt_utc():
    assert dt_utc(2023, 5, 5) == datetime(2023, 5, 5, tzinfo=timezone.utc)
    assert dt_utc(2023, 5, 5, 0, 0, 0, 555500) == datetime(2023, 5, 5, 0, 0, 0, 555500,
                                                           tzinfo=timezone.utc)


@pytest.mark.parametrize('as_ms', [True, False])
def test_dt_from_ts(as_ms):
    multi = 1000 if as_ms else 1
    assert dt_from_ts(1683244800.0 * multi) == datetime(2023, 5, 5, tzinfo=timezone.utc)
    assert dt_from_ts(1683244800.5555 * multi) == datetime(2023, 5, 5, 0, 0, 0, 555500,
                                                           tzinfo=timezone.utc)
    # As int
    assert dt_from_ts(1683244800 * multi) == datetime(2023, 5, 5, tzinfo=timezone.utc)
    # As milliseconds
    assert dt_from_ts(1683244800 * multi) == datetime(2023, 5, 5, tzinfo=timezone.utc)
    assert dt_from_ts(1683242400 * multi) == datetime(2023, 5, 4, 23, 20, tzinfo=timezone.utc)


def test_dt_floor_day():
    now = datetime(2023, 9, 1, 5, 2, 3, 455555, tzinfo=timezone.utc)

    assert dt_floor_day(now) == datetime(2023, 9, 1, tzinfo=timezone.utc)


def test_shorten_date() -> None:
    str_data = '1 day, 2 hours, 3 minutes, 4 seconds ago'
    str_shorten_data = '1 d, 2 h, 3 min, 4 sec ago'
    assert shorten_date(str_data) == str_shorten_data


def test_dt_humanize() -> None:
    assert dt_humanize(dt_now()) == 'just now'
    assert dt_humanize(dt_now(), only_distance=True) == 'instantly'
    assert dt_humanize(dt_now() - timedelta(hours=16), only_distance=True) == '16 hours'


def test_format_ms_time() -> None:
    # Date 2018-04-10 18:02:01
    date_in_epoch_ms = 1523383321000
    date = format_ms_time(date_in_epoch_ms)
    assert isinstance(date, str)
    res = datetime(2018, 4, 10, 18, 2, 1, tzinfo=timezone.utc)
    assert date == res.astimezone(None).strftime('%Y-%m-%dT%H:%M:%S')
    res = datetime(2017, 12, 13, 8, 2, 1, tzinfo=timezone.utc)
    # Date 2017-12-13 08:02:01
    date_in_epoch_ms = 1513152121000
    assert format_ms_time(date_in_epoch_ms) == res.astimezone(None).strftime('%Y-%m-%dT%H:%M:%S')


def test_format_date() -> None:

    date = datetime(2023, 9, 1, 5, 2, 3, 455555, tzinfo=timezone.utc)
    assert format_date(date) == '2023-09-01 05:02:03'
    assert format_date(None) == ''

    date = datetime(2021, 9, 30, 22, 59, 3, 455555, tzinfo=timezone.utc)
    assert format_date(date) == '2021-09-30 22:59:03'
    assert format_date(None) == ''
