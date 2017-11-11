# pragma pylint: disable=missing-docstring
import time

from freqtrade.misc import throttle


def test_throttle():

    def func():
        return 42

    start = time.time()
    result = throttle(func, 0.1)
    end = time.time()

    assert result == 42
    assert end - start > 0.1

    result = throttle(func, -1)
    assert result == 42
