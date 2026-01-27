import pytest
import os
from unittest import mock
from adapters.ccxt_shim.degraded_mode import DegradedModeGuard
from freqtrade.exceptions import OperationalException


@pytest.fixture
def degraded_guard_forced():
    with mock.patch.dict(os.environ, {"FT_DEGRADED_MODE": "1", "FT_DEGRADED_BLOCK_ENTRIES": "1"}):
        yield DegradedModeGuard()


def test_degraded_blocks_buy(degraded_guard_forced):
    # Buy (Entry) should block
    with pytest.raises(OperationalException, match="degraded_block: buy"):
        degraded_guard_forced.assert_can_order("buy", "RELIANCE/INR")


def test_degraded_allows_sell(degraded_guard_forced):
    # Sell (Exit) should allow
    degraded_guard_forced.assert_can_order("sell", "RELIANCE/INR")
    # No exception raised
