import os
import pytest
import time
from unittest import mock
from adapters.ccxt_shim.rate_limiter import RateLimiter


def test_rate_limiter_disabled():
    # Set Env to Disable Rate Limiter
    with mock.patch.dict(os.environ, {"FT_RATE_LIMIT_DISABLE": "1"}):
        rl = RateLimiter()
        assert rl.enabled is False

        start = time.time()
        # Should be able to call many times instantly without sleep or blocking
        for _ in range(200):
            rl.allow("test")
        duration = time.time() - start

        # 200 checks should be near instant if disabled
        assert duration < 0.5
