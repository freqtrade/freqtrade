"""
P27 Negative Test
This test is intended to FAIL.
It attempts to assert that a BAD snapshot (Low OI) is ALLOWED.
Since the production logic (SmartMoneyEngine) correctly sets allow_trade=False, this assertion fails.
The Acceptance Gate will observe this failure.
"""

import pytest
from user_data.strategies.smart_money_fr203 import SmartMoneyEngine, OptionChainSnapshot, StrikeRow


def make_snapshot(oi_pct=5.0) -> OptionChainSnapshot:
    # Low OI (5% < 10% Required)
    return OptionChainSnapshot(
        underlying="NIFTY",
        ts_utc="2025-01-01T00:00:00Z",
        strikes=[
            StrikeRow(
                strike=20000,
                right="CE",
                oi_change_pct=oi_pct,
                volume=150000,
                ltp_change_pct=2.0,
                iv_change_pct=0.1,
            ),
        ],
    )


def test_negative_control_bad_snapshot_should_fail():
    snap = make_snapshot(oi_pct=5.0)
    decision = SmartMoneyEngine.evaluate(snap)

    # We assert that trade IS allowed.
    # THIS ASSERTION IS WRONG BY DESIGN.
    # The production logic will return False, causing this assertion to fail (pytest exit code 1).
    assert decision.allow_trade is True, (
        "TEST FAILED: Logic correctly rejected the trade (allow_trade=False)"
    )
