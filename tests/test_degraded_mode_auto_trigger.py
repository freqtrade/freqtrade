import pytest
import os
from unittest import mock
from adapters.ccxt_shim.degraded_mode import DegradedModeGuard
from freqtrade.exceptions import OperationalException


@pytest.fixture
def degraded_guard_auto():
    with mock.patch.dict(os.environ, {"FT_DEGRADED_MODE": "0", "FT_DEGRADED_BLOCK_ENTRIES": "1"}):
        yield DegradedModeGuard()


def test_degraded_auto_trigger(degraded_guard_auto):
    assert not degraded_guard_auto.is_degraded()

    # Record failures
    degraded_guard_auto.record_failure(Exception("Fail 1"))
    assert not degraded_guard_auto.is_degraded()

    degraded_guard_auto.record_failure(Exception("Fail 2"))
    assert not degraded_guard_auto.is_degraded()

    degraded_guard_auto.record_failure(Exception("Fail 3"))
    # Threshold reached
    assert degraded_guard_auto.is_degraded()

    # Should now block buy
    with pytest.raises(OperationalException, match="degraded_block: buy"):
        degraded_guard_auto.assert_can_order("buy", "RELIANCE/INR")

    # Should allow sell
    degraded_guard_auto.assert_can_order("sell", "RELIANCE/INR")
