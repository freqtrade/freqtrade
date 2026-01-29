"""
P27 Smart Money Tests
"""

import pytest
from user_data.strategies.smart_money_fr203 import (
    SmartMoneyEngine,
    OptionChainSnapshot,
    StrikeRow,
    SmartMoneyDecision,
)


def make_snapshot(oi_pct=15.0, volume=150000, ltp_pct=2.0) -> OptionChainSnapshot:
    return OptionChainSnapshot(
        underlying="NIFTY",
        ts_utc="2025-01-01T00:00:00Z",
        strikes=[
            StrikeRow(
                strike=20000,
                right="CE",
                oi_change_pct=oi_pct,
                volume=volume,
                ltp_change_pct=ltp_pct,
                iv_change_pct=0.1,
            ),
            StrikeRow(
                strike=20050,
                right="CE",
                oi_change_pct=oi_pct,
                volume=volume,
                ltp_change_pct=ltp_pct,
                iv_change_pct=0.1,
            ),
            StrikeRow(
                strike=20100,
                right="CE",
                oi_change_pct=oi_pct,
                volume=volume,
                ltp_change_pct=ltp_pct,
                iv_change_pct=0.1,
            ),
        ],
    )


def test_missing_snapshot():
    decision = SmartMoneyEngine.evaluate(None)
    assert decision.allow_trade is True
    assert "NO_SNAPSHOT_BYPASS" in decision.reasons


def test_snapshot_good():
    # 3 strikes, score = 3 * 20 = 60. Should Pass.
    snap = make_snapshot(oi_pct=15.0, volume=150000, ltp_pct=2.0)
    decision = SmartMoneyEngine.evaluate(snap)
    assert decision.bias_strength == 60
    assert decision.allow_trade is True


def test_snapshot_bad_low_oi():
    # High volume/LTP but low OI
    snap = make_snapshot(oi_pct=5.0)
    decision = SmartMoneyEngine.evaluate(snap)
    assert decision.bias_strength == 0  # No strong strikes
    assert decision.allow_trade is False
    assert "LOW_BIAS_STRENGTH" in decision.reasons or "NO_STRONG_STRIKES" in decision.reasons


def test_snapshot_bad_ltp_decay():
    # Negative LTP
    snap = make_snapshot(ltp_pct=-1.0)
    decision = SmartMoneyEngine.evaluate(snap)
    assert decision.allow_trade is False
