import os
import pytest
from unittest import mock
from adapters.ccxt_shim.degraded_mode import DegradedModeGuard
from freqtrade.exceptions import OperationalException


def test_degraded_warn_only_allows_entry():
    # Force degradation but disable blocking via Env
    with mock.patch.dict(os.environ, {"FT_DEGRADED_MODE": "1", "FT_DEGRADED_BLOCK_ENTRIES": "0"}):
        guard = DegradedModeGuard()
        assert guard.is_degraded() is True
        assert guard.block_entries is False

        # Should NOT raise exception effectively allowing the order
        try:
            guard.assert_can_order("buy", "RELIANCE/INR")
        except OperationalException:
            pytest.fail("DegradedModeGuard raised OperationalException when block_entries=0")
