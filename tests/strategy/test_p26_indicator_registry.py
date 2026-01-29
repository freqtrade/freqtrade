"""
Tests for P26 Indicator Registry
"""

import pytest
from user_data.strategies.indicator_registry import IndicatorRegistry


def test_defaults_stock_opt():
    defaults = IndicatorRegistry.get_defaults("STOCK_OPT")
    assert defaults["timeframe"] == "5m"
    assert defaults["startup_candle_count"] == 50
    assert defaults["rsi_bull"] == 55
    assert defaults["stale_tolerance_seconds"] == 600


def test_startup_candle_count():
    assert IndicatorRegistry.get_startup_candle_count("STOCK_OPT") == 50
    # Auto inherits
    assert IndicatorRegistry.get_startup_candle_count("AUTO_OPT") == 50


def test_required_indicators():
    specs = IndicatorRegistry.get_required_indicators("STOCK_OPT")
    names = [s.name for s in specs]
    assert "ema_fast" in names
    assert "ema_slow" in names
    assert "rsi" in names

    # Check params
    ema_fast = next(s for s in specs if s.name == "ema_fast")
    assert ema_fast.params["period"] == 5
