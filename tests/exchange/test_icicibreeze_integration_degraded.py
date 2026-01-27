import pytest
import os
from unittest import mock
from freqtrade.exceptions import OperationalException
from adapters.ccxt_shim.breeze_ccxt import BreezeCCXT


@pytest.fixture
def degraded_exchange_forced():
    with mock.patch.dict(
        os.environ,
        {
            "BREEZE_MOCK": "1",
            "FT_DEGRADED_MODE": "1",
            "FT_DEGRADED_BLOCK_ENTRIES": "1",
            "FT_RATE_LIMIT_DISABLE": "1",  # Disable limit to isolate degraded
        },
    ):
        exposure = {"risk_guard": {"enabled": False}}
        exchange = BreezeCCXT(config=exposure)
        yield exchange


def test_degraded_integration_blocks_order(degraded_exchange_forced):
    symbol = "RELIANCE/INR"

    # Needs market hours mock
    with mock.patch("adapters.ccxt_shim.market_hours.MarketHoursGuard.assert_can_create_order"):
        with pytest.raises(OperationalException, match="degraded_block: buy"):
            degraded_exchange_forced.create_order(symbol, "limit", "buy", 1, 2500)
