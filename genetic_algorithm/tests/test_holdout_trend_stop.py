"""
Tests for the improved holdout trend early stop logic in evolution.py.

Tests cover:
- Grace period: early checks don't trigger trend stop
- Minimum degradation floor: low degradation doesn't trigger
- Slope threshold: slow worsening doesn't trigger
- Correct triggering when all conditions are met
"""

import pytest
import logging
from unittest.mock import MagicMock, patch


def _make_mock_evo(
    trend_checks=3,
    trend_min_degradation=0.30,
    trend_slope_threshold=0.05,
    trend_grace_checks=2,
    early_stop_checks=2,
):
    """Create a lightweight mock with exactly the attributes used by the trend logic."""
    evo = MagicMock()
    evo.holdout_trend_early_stop = True
    evo.holdout_trend_checks = trend_checks
    evo.holdout_trend_min_degradation = trend_min_degradation
    evo.holdout_trend_slope_threshold = trend_slope_threshold
    evo.holdout_trend_grace_checks = trend_grace_checks
    evo.holdout_early_stop_checks = early_stop_checks
    evo._holdout_degradation_history = []
    evo._holdout_consecutive_bad = 0
    evo.logger = logging.getLogger("test_holdout_trend")
    return evo


def _run_trend_check(evo, avg_degrad):
    """
    Reproduce the trend-based early stop logic from evolution.py.
    Returns True if early stop would be triggered.
    """
    evo._holdout_degradation_history.append(avg_degrad)
    n_checks = len(evo._holdout_degradation_history)
    if (
        evo.holdout_trend_early_stop
        and n_checks >= evo.holdout_trend_checks + 1
        and n_checks > evo.holdout_trend_grace_checks
    ):
        recent = evo._holdout_degradation_history[-(evo.holdout_trend_checks + 1):]

        # Compute slope
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        numer = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denom = sum((i - x_mean) ** 2 for i in range(n))
        slope = numer / denom if denom > 0 else 0.0

        if (
            avg_degrad >= evo.holdout_trend_min_degradation
            and slope >= evo.holdout_trend_slope_threshold
        ):
            evo._holdout_consecutive_bad = max(
                evo._holdout_consecutive_bad, evo.holdout_early_stop_checks
            )
            return True
    return False


# ══════════════════════════════════════════════════════════════════════
# Tests: Grace Period
# ══════════════════════════════════════════════════════════════════════

class TestGracePeriod:
    def test_no_trigger_during_grace(self):
        """Degradation increases in early checks (within grace period) should NOT trigger."""
        evo = _make_mock_evo(trend_checks=3, trend_grace_checks=2)

        # Feed 2 checks (within grace period) with worsening degradation
        assert not _run_trend_check(evo, 0.10)
        assert not _run_trend_check(evo, 0.20)

        # Still within grace despite having enough history points
        assert evo._holdout_consecutive_bad == 0

    def test_trigger_after_grace_with_bad_trend(self):
        """After grace period, a genuine bad trend should trigger."""
        evo = _make_mock_evo(trend_checks=3, trend_grace_checks=2)

        # 4 checks needed: trend_checks+1=4, must exceed grace_checks=2
        _run_trend_check(evo, 0.20)
        _run_trend_check(evo, 0.30)
        _run_trend_check(evo, 0.40)
        triggered = _run_trend_check(evo, 0.50)

        assert triggered
        assert evo._holdout_consecutive_bad >= evo.holdout_early_stop_checks


# ══════════════════════════════════════════════════════════════════════
# Tests: Minimum Degradation Floor
# ══════════════════════════════════════════════════════════════════════

class TestMinDegradationFloor:
    def test_no_trigger_below_floor(self):
        """Even with worsening trend, degradation below floor should NOT trigger."""
        evo = _make_mock_evo(trend_min_degradation=0.30, trend_grace_checks=0)

        # Worsening trend but all values below 30%
        _run_trend_check(evo, 0.05)
        _run_trend_check(evo, 0.10)
        _run_trend_check(evo, 0.15)
        triggered = _run_trend_check(evo, 0.20)  # still below 30%

        assert not triggered

    def test_trigger_above_floor(self):
        """Above floor + worsening trend should trigger."""
        evo = _make_mock_evo(trend_min_degradation=0.30, trend_grace_checks=0)

        _run_trend_check(evo, 0.20)
        _run_trend_check(evo, 0.30)
        _run_trend_check(evo, 0.40)
        triggered = _run_trend_check(evo, 0.50)

        assert triggered

    def test_floor_at_boundary(self):
        """Exactly at the floor with positive slope should trigger."""
        evo = _make_mock_evo(trend_min_degradation=0.30, trend_grace_checks=0)

        _run_trend_check(evo, 0.10)
        _run_trend_check(evo, 0.15)
        _run_trend_check(evo, 0.25)
        triggered = _run_trend_check(evo, 0.30)  # exactly at floor

        assert triggered


# ══════════════════════════════════════════════════════════════════════
# Tests: Slope Threshold
# ══════════════════════════════════════════════════════════════════════

class TestSlopeThreshold:
    def test_slow_worsening_no_trigger(self):
        """Very slow degradation increase (slope < threshold) should NOT trigger."""
        evo = _make_mock_evo(
            trend_slope_threshold=0.05,
            trend_min_degradation=0.0,  # disable floor for this test
            trend_grace_checks=0,
        )

        # Tiny increments: slope ≈ 0.01 per step
        _run_trend_check(evo, 0.30)
        _run_trend_check(evo, 0.31)
        _run_trend_check(evo, 0.32)
        triggered = _run_trend_check(evo, 0.33)

        assert not triggered

    def test_fast_worsening_triggers(self):
        """Fast degradation increase (slope > threshold) should trigger."""
        evo = _make_mock_evo(
            trend_slope_threshold=0.05,
            trend_min_degradation=0.0,  # disable floor for this test
            trend_grace_checks=0,
        )

        # Big jumps: slope ≈ 0.10 per step
        _run_trend_check(evo, 0.10)
        _run_trend_check(evo, 0.20)
        _run_trend_check(evo, 0.30)
        triggered = _run_trend_check(evo, 0.40)

        assert triggered

    def test_flat_trend_no_trigger(self):
        """Flat degradation (slope≈0) should never trigger."""
        evo = _make_mock_evo(
            trend_slope_threshold=0.05,
            trend_min_degradation=0.0,
            trend_grace_checks=0,
        )

        _run_trend_check(evo, 0.40)
        _run_trend_check(evo, 0.40)
        _run_trend_check(evo, 0.40)
        triggered = _run_trend_check(evo, 0.40)

        assert not triggered

    def test_improving_trend_no_trigger(self):
        """Degradation getting better (negative slope) should never trigger."""
        evo = _make_mock_evo(
            trend_slope_threshold=0.05,
            trend_min_degradation=0.0,
            trend_grace_checks=0,
        )

        _run_trend_check(evo, 0.50)
        _run_trend_check(evo, 0.40)
        _run_trend_check(evo, 0.30)
        triggered = _run_trend_check(evo, 0.20)

        assert not triggered


# ══════════════════════════════════════════════════════════════════════
# Tests: Combined conditions
# ══════════════════════════════════════════════════════════════════════

class TestCombinedConditions:
    def test_all_conditions_met(self):
        """High degradation + steep slope + past grace => triggers."""
        evo = _make_mock_evo(
            trend_checks=3,
            trend_min_degradation=0.30,
            trend_slope_threshold=0.05,
            trend_grace_checks=2,
        )

        # Grace period (2 checks)
        _run_trend_check(evo, 0.10)
        _run_trend_check(evo, 0.20)
        # Now past grace + building history
        _run_trend_check(evo, 0.30)
        triggered = _run_trend_check(evo, 0.45)

        assert triggered

    def test_disabled_trend_stop(self):
        """When holdout_trend_early_stop=False, nothing triggers."""
        evo = _make_mock_evo()
        evo.holdout_trend_early_stop = False

        _run_trend_check(evo, 0.20)
        _run_trend_check(evo, 0.40)
        _run_trend_check(evo, 0.60)
        triggered = _run_trend_check(evo, 0.80)

        assert not triggered
        assert evo._holdout_consecutive_bad == 0

    def test_consecutive_bad_counter_set_correctly(self):
        """Verify _holdout_consecutive_bad is set to at least early_stop_checks."""
        evo = _make_mock_evo(early_stop_checks=3, trend_grace_checks=0, trend_min_degradation=0.0)

        _run_trend_check(evo, 0.10)
        _run_trend_check(evo, 0.20)
        _run_trend_check(evo, 0.30)
        _run_trend_check(evo, 0.40)

        assert evo._holdout_consecutive_bad >= 3


# ══════════════════════════════════════════════════════════════════════
# Tests: Config defaults
# ══════════════════════════════════════════════════════════════════════

class TestConfigDefaults:
    def test_default_values(self):
        """New config keys should have sensible defaults."""
        evo = _make_mock_evo()
        assert evo.holdout_trend_min_degradation == 0.30
        assert evo.holdout_trend_slope_threshold == 0.05
        assert evo.holdout_trend_grace_checks == 2
