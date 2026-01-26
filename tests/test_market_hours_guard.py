"""Unit tests for MarketHoursGuard."""

import os
from datetime import datetime, time, timedelta, timezone
from unittest import mock

import pytest

from adapters.ccxt_shim.market_hours import IST_OFFSET, MARKET_CLOSE, MARKET_OPEN, MarketHoursGuard


@pytest.fixture
def guard():
    """Fixture to provide a clean MarketHoursGuard instance."""
    # Ensure env vars are clean
    with mock.patch.dict(os.environ, {}, clear=True):
        yield MarketHoursGuard()


def test_market_hours_guard_defaults(guard):
    """Test default state is determined by overrides (which are off)."""
    assert not guard._force_open
    assert not guard._force_closed


def test_is_market_open_weekdays():
    """Test weekday logic (Mon-Fri)."""
    guard = MarketHoursGuard()

    # Monday 10:00 AM IST (Open)
    # 2026-01-26 is Monday
    mon_open = datetime(2026, 1, 26, 10, 0, tzinfo=IST_OFFSET)
    assert guard.is_market_open(mon_open) is True

    # Saturday 10:00 AM IST (Closed)
    # 2026-01-24 was Saturday
    sat_closed = datetime(2026, 1, 24, 10, 0, tzinfo=IST_OFFSET)
    assert guard.is_market_open(sat_closed) is False

    # Sunday (Closed)
    sun_closed = datetime(2026, 1, 25, 10, 0, tzinfo=IST_OFFSET)
    assert guard.is_market_open(sun_closed) is False


def test_is_market_open_time_boundaries():
    """Test time boundaries (09:15 - 15:30 IST)."""
    guard = MarketHoursGuard()
    base_date = datetime(2026, 1, 27, 0, 0, tzinfo=IST_OFFSET)  # Tuesday

    # 09:14:59 (Closed)
    t1 = base_date.replace(hour=9, minute=14, second=59)
    assert guard.is_market_open(t1) is False

    # 09:15:00 (Open)
    t2 = base_date.replace(hour=9, minute=15, second=0)
    assert guard.is_market_open(t2) is True

    # 15:29:59 (Open)
    t3 = base_date.replace(hour=15, minute=29, second=59)
    assert guard.is_market_open(t3) is True

    # 15:30:00 (Closed)
    t4 = base_date.replace(hour=15, minute=30, second=0)
    assert guard.is_market_open(t4) is False


def test_overrides_force_open(guard):
    """Test FT_FORCE_MARKET_OPEN."""
    with mock.patch.dict(os.environ, {"FT_FORCE_MARKET_OPEN": "1"}):
        # Even on Sunday (Closed normally)
        sun_closed = datetime(2026, 1, 25, 10, 0, tzinfo=IST_OFFSET)
        assert guard.is_market_open(sun_closed) is True


def test_overrides_force_closed(guard):
    """Test FT_FORCE_MARKET_CLOSED."""
    with mock.patch.dict(os.environ, {"FT_FORCE_MARKET_CLOSED": "1"}):
        # Even on Monday 10am (Open normally)
        mon_open = datetime(2026, 1, 26, 10, 0, tzinfo=IST_OFFSET)
        assert guard.is_market_open(mon_open) is False


def test_assert_method_buy_blocked(guard):
    """Verify assert_can_create_order blocks buy when closed."""
    # Mock closed state
    with mock.patch.object(guard, "is_market_open", return_value=False):
        # Buy should fail
        with pytest.raises(Exception, match="market_closed: blocking entry order"):
            guard.assert_can_create_order("buy", "RELIANCE/INR")

        # Sell should pass (Exits allowed)
        guard.assert_can_create_order("sell", "RELIANCE/INR")


def test_assert_method_cancel_blocked(guard):
    """Verify assert_can_cancel_order blocks when closed."""
    with mock.patch.object(guard, "is_market_open", return_value=False):
        with pytest.raises(Exception, match="market_closed: blocking cancel order"):
            guard.assert_can_cancel_order("123", "RELIANCE/INR")


def test_assert_method_open(guard):
    """Verify everything allowed when open."""
    with mock.patch.object(guard, "is_market_open", return_value=True):
        guard.assert_can_create_order("buy", "RELIANCE/INR")
        guard.assert_can_create_order("sell", "RELIANCE/INR")
        guard.assert_can_cancel_order("123", "RELIANCE/INR")
