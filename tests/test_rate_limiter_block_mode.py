import pytest
import os
from unittest import mock
from adapters.ccxt_shim.rate_limiter import RateLimiter
from freqtrade.exceptions import OperationalException


@pytest.fixture
def rate_limiter_block():
    with mock.patch.dict(
        os.environ,
        {
            "FT_RATE_LIMIT_PER_MINUTE": "60",  # 1 token per second
            "FT_RATE_LIMIT_MODE": "block",
            "FT_RATE_LIMIT_DISABLE": "0",
        },
    ):
        print(f"DEBUG: Created RateLimiter with env: {os.environ.get('FT_RATE_LIMIT_MODE')}")
        yield RateLimiter()


def test_block_mode_allow(rate_limiter_block):
    # Should start full
    assert rate_limiter_block.tokens == 60.0

    # Allow 1
    rate_limiter_block.allow("test_op")
    assert rate_limiter_block.tokens == 59.0


def test_block_mode_exceed_limit(rate_limiter_block):
    # Drain 60 tokens
    for _ in range(60):
        rate_limiter_block.allow("test_drain")

    # 61st call should raise
    with pytest.raises(OperationalException, match="rate_limit_block: test_fail"):
        rate_limiter_block.allow("test_fail")
