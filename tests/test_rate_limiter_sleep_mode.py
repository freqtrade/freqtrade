import pytest
import os
import time
from unittest import mock
from adapters.ccxt_shim.rate_limiter import RateLimiter


@pytest.fixture
def rate_limiter_sleep():
    with mock.patch.dict(
        os.environ,
        {
            "FT_RATE_LIMIT_PER_MINUTE": "60",  # 1 token per second
            "FT_RATE_LIMIT_MODE": "sleep",
            "FT_RATE_LIMIT_DISABLE": "0",
        },
    ):
        yield RateLimiter()


def test_sleep_mode_delays(rate_limiter_sleep):
    # Drain 60 tokens
    for _ in range(60):
        rate_limiter_sleep.allow("test_drain")

    start_time = time.time()

    # 61st call should sleep for approx 1s (to get 1 token)
    # We cheat/check token logic: refill rate is 1/sec.
    # Cost 1. Needed 1. Time = 1s.
    rate_limiter_sleep.allow("test_sleep")

    elapsed = time.time() - start_time
    assert elapsed >= 0.9  # Allow slight timing jitter
    assert elapsed < 2.0  # Shouldn't sleep excessively
