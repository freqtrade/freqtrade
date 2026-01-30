"""
Tests for GTT Hysteresis
"""

import pytest
from unittest.mock import MagicMock
from adapters.ccxt_shim.order_router import OrderRouter


def test_gtt_hysteresis_skip():
    router = OrderRouter(MagicMock())
    config = {"rearm_seconds": 20, "min_price_move_ticks": 2}

    # Initial state setup (mocking previous mod)
    router._mod_state = {"ord1": {"last_ts": 1000.0, "last_price": 100.0}}

    # Attempt mod at 1005 (diff 5s < 20s) with price 100.05 (move 0.05 / 0.05 = 1 tick < 2)
    # Should SKIP
    res = router.check_gtt_hysteresis("ord1", 100.05, 100.0, 1005.0, config)
    assert res["skip"] is True
    assert "SKIPPED_HYSTERESIS" in res["reason"]


def test_gtt_hysteresis_allow_time():
    router = OrderRouter(MagicMock())
    config = {"rearm_seconds": 20, "min_price_move_ticks": 2}

    router._mod_state = {"ord1": {"last_ts": 1000.0, "last_price": 100.0}}

    # Mod at 1025 (diff 25s > 20s) -> Allow
    res = router.check_gtt_hysteresis("ord1", 100.05, 100.0, 1025.0, config)
    assert res["skip"] is False


def test_gtt_hysteresis_allow_price():
    router = OrderRouter(MagicMock())
    config = {"rearm_seconds": 20, "min_price_move_ticks": 2}

    router._mod_state = {"ord1": {"last_ts": 1000.0, "last_price": 100.0}}

    # Mod at 1005 (diff 5s) but price 100.20 (move 0.20 / 0.05 = 4 ticks > 2) -> Allow
    res = router.check_gtt_hysteresis("ord1", 100.20, 100.0, 1005.0, config)
    assert res["skip"] is False
