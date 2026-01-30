"""
Tests for Sniper Cancel & ATR Buffer
"""

import pytest
from unittest.mock import MagicMock
from adapters.ccxt_shim.order_router import OrderRouter


# Sniper Tests
def test_sniper_cancel_trigger():
    router = OrderRouter(MagicMock())
    config = {"cancel_after_seconds": 3}

    open_ts = 1000.0
    now_ts = 1003.5  # 3.5s elapsed

    assert router.check_sniper_cancel(open_ts, now_ts, config) is True


def test_sniper_cancel_hold():
    router = OrderRouter(MagicMock())
    config = {"cancel_after_seconds": 3}

    open_ts = 1000.0
    now_ts = 1001.0  # 1s elapsed

    assert router.check_sniper_cancel(open_ts, now_ts, config) is False


# ATR Buffer Tests
def test_atr_limit_buffer_buy():
    router = OrderRouter(MagicMock())
    config = {"buffer_mult": 0.5}

    last_price = 100.0
    atr = 10.0
    # Buy: last + (atr * mult) = 100 + 5 = 105
    res = router.calculate_atr_limit_buffer(last_price, "buy", atr, config)
    assert res == 105.0


def test_atr_limit_buffer_sell():
    router = OrderRouter(MagicMock())
    config = {"buffer_mult": 0.5}

    last_price = 100.0
    atr = 10.0
    # Sell: last - (atr * mult) = 100 - 5 = 95
    res = router.calculate_atr_limit_buffer(last_price, "sell", atr, config)
    assert res == 95.0
