import os
from unittest.mock import MagicMock, patch

import pytest
from adapters.ccxt_shim.risk_guard import RiskGuard


@pytest.fixture
def base_config():
    return {
        "risk_guard": {
            "enabled": True,
            "max_trades_per_day": 3,
            "intraday_entry_cutoff_ist": "15:00",
            "spread_guard": {"enabled": True, "max_spread_pct": 0.5},
            "allow_exits_when_blocked": True,
        }
    }


def test_risk_guard_disabled(base_config):
    base_config["risk_guard"]["enabled"] = False
    guard = RiskGuard(base_config)
    blocked, reason = guard.should_block_entry("RELIANCE/INR", "buy", {})
    assert not blocked


def test_max_trades_per_day(base_config):
    guard = RiskGuard(base_config)
    guard.record_trade_attempt("A", "buy")
    guard.record_trade_attempt("B", "buy")

    # 2/3 ok
    blocked, _ = guard.should_block_entry("C", "buy", {})
    assert not blocked

    guard.record_trade_attempt("C", "buy")
    # 3/3 ok (next one blocked)

    blocked, reason = guard.should_block_entry("D", "buy", {})
    assert blocked
    assert reason == "max_trades_per_day"


def test_intraday_cutoff(base_config):
    guard = RiskGuard(base_config)

    # Force time before cutoff
    with patch.dict(os.environ, {"FT_IST_NOW": "2026-01-26T14:59:00+05:30"}):
        blocked, _ = guard.should_block_entry("A", "buy", {})
        assert not blocked

    # Force time after cutoff
    with patch.dict(os.environ, {"FT_IST_NOW": "2026-01-26T15:01:00+05:30"}):
        blocked, reason = guard.should_block_entry("A", "buy", {})
        assert blocked
        assert reason == "intraday_cutoff"


def test_spread_guard(base_config):
    guard = RiskGuard(base_config)

    # Good spread
    # Mid=100.25, Spread=0.5, Pct=0.49 < 0.5
    surface_good = {"bid": 100.0, "ask": 100.5}
    blocked, _ = guard.should_block_entry("A", "buy", surface_good)
    assert not blocked

    # Bad spread
    # Mid=100.5, Spread=1.0, Pct=0.99 > 0.5
    surface_bad = {"bid": 100.0, "ask": 101.0}
    blocked, reason = guard.should_block_entry("A", "buy", surface_bad)
    assert blocked
    assert reason == "spread_guard"


def test_allow_exits(base_config):
    base_config["risk_guard"]["max_trades_per_day"] = 0
    guard = RiskGuard(base_config)

    # Entry blocked
    blocked, reason = guard.should_block_entry("A", "buy", {})
    assert blocked

    # Exit allowed
    blocked, _ = guard.should_block_entry("A", "sell", {})
    assert not blocked
