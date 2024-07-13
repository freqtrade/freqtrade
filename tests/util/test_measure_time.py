from unittest.mock import MagicMock

import time_machine

from freqtrade.util import MeasureTime


def test_measure_time():
    callback = MagicMock()
    with time_machine.travel("2021-09-01 05:00:00 +00:00", tick=False) as t:
        measure = MeasureTime(callback, 5, ttl=60)
        with measure:
            pass

        assert callback.call_count == 0

        with measure:
            t.shift(10)

        assert callback.call_count == 1
        callback.reset_mock()
        with measure:
            t.shift(10)
        assert callback.call_count == 0

        callback.reset_mock()
        # Shift past the ttl
        t.shift(45)

        with measure:
            t.shift(10)
        assert callback.call_count == 1
