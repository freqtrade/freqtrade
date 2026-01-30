import pytest
import os
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


class FakeClock:
    def __init__(self):
        self.time = 1000000.0

    def now(self):
        return self.time

    def sleep(self, duration):
        self.time += duration


def test_sleep_mode_delays():
    clock = FakeClock()

    with mock.patch.dict(
        os.environ,
        {
            "FT_RATE_LIMIT_PER_MINUTE": "60",  # 1 token per second
            "FT_RATE_LIMIT_MODE": "sleep",
            "FT_RATE_LIMIT_DISABLE": "0",
        },
    ):
        # Inject Fake Clock
        rl = RateLimiter(now_fn=clock.now, sleep_fn=clock.sleep)

        # Drain 60 tokens (Instant)
        # We need to ensure refill logic uses clock.now
        # _refill uses self._now()

        # Initial refill happens at init (1000000.0)

        for _ in range(60):
            rl.allow("test_drain")

        # Tokens should be 0.

        # 61st call should sleep.
        # Cost 1. Rate 1/sec. Sleep = 1.0s.

        rl.allow("test_sleep")

        # Clock should have advanced by 1.0s
        assert clock.time >= 1000000.99
        assert clock.time <= 1000001.1
