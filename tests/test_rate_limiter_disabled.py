import os
from unittest import mock
from adapters.ccxt_shim.rate_limiter import RateLimiter


class FakeClock:
    def __init__(self):
        self.time = 1000.0

    def now(self):
        return self.time

    def sleep(self, duration):
        self.time += duration


def test_rate_limiter_disabled():
    clock = FakeClock()
    # Set Env to Disable Rate Limiter
    with mock.patch.dict(os.environ, {"FT_RATE_LIMIT_DISABLE": "1"}):
        rl = RateLimiter(now_fn=clock.now, sleep_fn=clock.sleep)
        assert rl.enabled is False

        start = clock.now()
        # Should be able to call many times instantly without sleep or blocking
        for _ in range(200):
            rl.allow("test")
        duration = clock.now() - start

        # 200 checks should be near instant if disabled
        assert duration < 0.5
