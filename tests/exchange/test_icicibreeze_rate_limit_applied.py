import pytest
import os
from unittest import mock
from freqtrade.exceptions import OperationalException
from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT


@pytest.fixture
def rate_limited_exchange():
    """
    Fixture for BreezeCCXT with strict rate limiting enabled via environment.
    """
    with mock.patch.dict(
        os.environ,
        {
            "BREEZE_MOCK": "1",
            "FT_RATE_LIMIT_PER_MINUTE": "10",
            "FT_RATE_LIMIT_MODE": "block",
            "FT_RATE_LIMIT_DISABLE": "0",
        },
    ):
        exposure = {"risk_guard": {"enabled": False}}  # Disable risk guard to isolate rate limit
        exchange = BreezeCCXT(config=exposure)
        yield exchange


def test_fetch_ticker_rate_limit(rate_limited_exchange):
    symbol = "RELIANCE/INR"

    # First 10 calls should pass
    for i in range(10):
        res = rate_limited_exchange.fetch_ticker(symbol)
        assert res["symbol"] == symbol

    # 11th call should block
    with pytest.raises(OperationalException, match="rate_limit_block: fetch_ticker"):
        rate_limited_exchange.fetch_ticker(symbol)


def test_fetch_markets_rate_limit(rate_limited_exchange):
    # Depending on how tests run, tokens might be shared if instance persists or class state
    # Fixture creates new instance each time, which creates new RateLimiter, so fresh bucket.

    # 10 calls pass
    for i in range(10):
        rate_limited_exchange.fetch_markets()

    # 11th blocks
    with pytest.raises(OperationalException, match="rate_limit_block: fetch_markets"):
        rate_limited_exchange.fetch_markets()


def test_create_order_rate_limit(rate_limited_exchange):
    symbol = "RELIANCE/INR"
    # To pass order creation checks, we need to ensure market hours/risk guard don't block first.
    # risk_guard disabled in fixture.
    # We mock market hours to allow.

    with mock.patch("adapters.ccxt_shim.market_hours.MarketHoursGuard.assert_can_create_order"):
        # P15 risk check calls fetch_ticker, which consumes a token.
        # We mock fetch_ticker to avoid double consumption and isolate create_order check.
        # We need to mock it on the INSTANCE, not the class, because the instance RateLimiter is what we rely on?
        # Use mock.patch.object
        with mock.patch.object(
            rate_limited_exchange,
            "fetch_ticker",
            return_value={"bid": 2490, "ask": 2510, "last": 2500},
        ):
            # 10 calls pass
            for i in range(10):
                rate_limited_exchange.create_order(symbol, "limit", "buy", 1, 2500)

            # 11th blocks
            with pytest.raises(OperationalException, match="rate_limit_block: create_order"):
                rate_limited_exchange.create_order(symbol, "limit", "buy", 1, 2500)
