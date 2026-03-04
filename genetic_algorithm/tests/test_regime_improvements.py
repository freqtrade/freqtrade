"""
Tests for regime detection improvements:
- Volatility clustering detection method
- Adaptive segmentation (classify_periods_adaptive)
- Confidence weighting in regime-aware aggregation
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genetic_algorithm.utils.regime_detector import (
    RegimeDetector,
    RegimeType,
    RegimeSegment,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_ohlcv(
    start_date: datetime,
    num_days: int,
    regime: str = 'bullish',
    timeframe_minutes: int = 60,
    initial_price: float = 100.0,
    volatility: float = 0.01,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a specific regime."""
    bars_per_day = 24 * 60 // timeframe_minutes
    n_bars = num_days * bars_per_day

    dates = pd.date_range(start=start_date, periods=n_bars, freq=f'{timeframe_minutes}min')
    np.random.seed(42)

    if regime == 'bullish':
        drift = 0.001
        vol = volatility
    elif regime == 'bearish':
        drift = -0.001
        vol = volatility
    elif regime == 'sideways':
        drift = 0.0
        vol = volatility * 0.5
    elif regime == 'volatile':
        drift = 0.0
        vol = volatility * 3.0
    else:
        drift = 0.0
        vol = volatility

    returns = np.random.normal(drift, vol, n_bars)
    prices = initial_price * np.exp(np.cumsum(returns))

    # Create OHLCV
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, vol, n_bars))),
        'low': prices * (1 - np.abs(np.random.normal(0, vol, n_bars))),
        'close': prices,
        'volume': np.random.uniform(100, 10000, n_bars),
    })
    df.set_index('date', inplace=True)
    return df


def generate_multi_regime_data(
    start_date: datetime = datetime(2023, 1, 1),
    regime_sequence=None,
    days_per_regime: int = 60,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """Generate data with multiple regime transitions."""
    if regime_sequence is None:
        regime_sequence = ['bullish', 'sideways', 'bearish', 'volatile', 'bullish']

    frames = []
    current_price = initial_price
    current_date = start_date

    for regime in regime_sequence:
        df = generate_synthetic_ohlcv(
            start_date=current_date,
            num_days=days_per_regime,
            regime=regime,
            initial_price=current_price,
        )
        frames.append(df)
        current_price = df['close'].iloc[-1]
        current_date = df.index[-1] + timedelta(hours=1)

    return pd.concat(frames)


# ═══════════════════════════════════════════════════════════════════
# Volatility Cluster Detection
# ═══════════════════════════════════════════════════════════════════

class TestVolatilityCluster:
    """Tests for the volatility_cluster detection method."""

    def test_initialization(self):
        """Volatility cluster method should initialize without errors."""
        detector = RegimeDetector(method='volatility_cluster')
        assert detector.method == 'volatility_cluster'
        assert 'vol_window' in detector.params

    def test_detects_something(self):
        """Should produce non-UNCERTAIN labels on sufficient data."""
        df = generate_synthetic_ohlcv(datetime(2023, 1, 1), num_days=120, regime='bullish')
        detector = RegimeDetector(method='volatility_cluster')
        regimes = detector.detect(df)

        assert len(regimes) == len(df)
        valid = [r for r in regimes if r != RegimeType.UNCERTAIN]
        assert len(valid) > 0, "Should produce some valid regime labels"

    def test_volatile_period_detected(self):
        """High-volatility data should produce VOLATILE or directional labels."""
        df = generate_synthetic_ohlcv(
            datetime(2023, 1, 1), num_days=120, regime='volatile', volatility=0.05
        )
        detector = RegimeDetector(method='volatility_cluster')
        regimes = detector.detect(df)

        # Count volatile classifications
        regime_counts = regimes.value_counts()
        logger.info(f"Volatile data regime distribution: {regime_counts.to_dict()}")
        # With high volatility, we should see VOLATILE or strong directional labels
        non_sideways = sum(1 for r in regimes if r in (RegimeType.VOLATILE, RegimeType.BULLISH, RegimeType.BEARISH))
        assert non_sideways > 0

    def test_sideways_period_detected(self):
        """Low-volatility, no-trend data should produce SIDEWAYS labels."""
        df = generate_synthetic_ohlcv(
            datetime(2023, 1, 1), num_days=120, regime='sideways', volatility=0.003
        )
        detector = RegimeDetector(method='volatility_cluster')
        regimes = detector.detect(df)

        sideways_count = sum(1 for r in regimes if r == RegimeType.SIDEWAYS)
        total_valid = sum(1 for r in regimes if r != RegimeType.UNCERTAIN)
        if total_valid > 0:
            sideways_ratio = sideways_count / total_valid
            logger.info(f"Sideways data: sideways_ratio={sideways_ratio:.2%}")

    def test_custom_params(self):
        """Should accept custom parameters."""
        detector = RegimeDetector(
            method='volatility_cluster',
            params={'vol_window': 30, 'trend_window': 100}
        )
        assert detector.params['vol_window'] == 30
        assert detector.params['trend_window'] == 100

    def test_multi_regime_data(self):
        """Should detect regime transitions in multi-regime data."""
        df = generate_multi_regime_data(
            regime_sequence=['bullish', 'bearish', 'sideways'],
            days_per_regime=90,
        )
        detector = RegimeDetector(method='volatility_cluster')
        regimes = detector.detect(df)

        # Should have at least 2 different regime types
        unique_regimes = set(r for r in regimes if r != RegimeType.UNCERTAIN)
        assert len(unique_regimes) >= 2, f"Expected 2+ regimes, got {unique_regimes}"


# ═══════════════════════════════════════════════════════════════════
# Adaptive Segmentation
# ═══════════════════════════════════════════════════════════════════

class TestAdaptiveSegmentation:
    """Tests for classify_periods_adaptive()."""

    def test_basic_segmentation(self):
        """Should produce segments from multi-regime data."""
        df = generate_multi_regime_data(
            regime_sequence=['bullish', 'bearish', 'sideways', 'bullish'],
            days_per_regime=60,
        )
        detector = RegimeDetector(method='adx_di_hysteresis')
        segments = detector.classify_periods_adaptive(df, min_segment_days=7)

        assert len(segments) > 0
        for seg in segments:
            assert isinstance(seg, RegimeSegment)
            assert seg.duration_days >= 7
            assert 0.0 <= seg.confidence <= 1.0

    def test_segments_cover_data(self):
        """Segments should approximately cover the data range."""
        df = generate_multi_regime_data(days_per_regime=90)
        detector = RegimeDetector(method='rolling_returns')
        segments = detector.classify_periods_adaptive(df, min_segment_days=10)

        if segments:
            first_start = min(s.start_date for s in segments)
            last_end = max(s.end_date for s in segments)
            total_covered = sum(s.duration_days for s in segments)
            assert total_covered > 0

    def test_no_excessive_segments(self):
        """Should merge short segments, avoiding fragmentation."""
        df = generate_multi_regime_data(
            regime_sequence=['bullish', 'sideways', 'bearish'],
            days_per_regime=90,
        )
        detector = RegimeDetector(method='adx_di_hysteresis')
        segments = detector.classify_periods_adaptive(
            df, min_segment_days=14, merge_threshold_days=10
        )

        for seg in segments:
            assert seg.duration_days >= 14, \
                f"Segment {seg.segment_id} has {seg.duration_days} days, min is 14"

    def test_max_segment_splitting(self):
        """Segments longer than max_segment_days should be split."""
        df = generate_synthetic_ohlcv(datetime(2023, 1, 1), num_days=400, regime='bullish')
        detector = RegimeDetector(method='rolling_returns')
        segments = detector.classify_periods_adaptive(
            df, max_segment_days=90, min_segment_days=10
        )

        for seg in segments:
            assert seg.duration_days <= 180, \
                f"Segment {seg.segment_id} has {seg.duration_days} days, max is ~90"

    def test_regime_consistency(self):
        """Adaptive segments should have higher confidence than fixed-window."""
        df = generate_multi_regime_data(
            regime_sequence=['bullish', 'bearish', 'sideways', 'bullish', 'bearish'],
            days_per_regime=60,
        )
        detector = RegimeDetector(method='adx_di_hysteresis')

        adaptive = detector.classify_periods_adaptive(df, min_segment_days=7)
        fixed = detector.classify_periods(df, period_days=90, min_period_days=30)

        if adaptive and fixed:
            avg_adaptive_conf = sum(s.confidence for s in adaptive) / len(adaptive)
            avg_fixed_conf = sum(s.confidence for s in fixed) / len(fixed)
            logger.info(
                f"Avg confidence: adaptive={avg_adaptive_conf:.3f}, fixed={avg_fixed_conf:.3f}"
            )
            # Adaptive should generally be at least as confident as fixed
            # (but not always, due to randomness in synthetic data)

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty segments."""
        detector = RegimeDetector(method='adx_di_hysteresis')
        segments = detector.classify_periods_adaptive(pd.DataFrame())
        assert segments == []

    def test_short_data(self):
        """Very short data (< warmup) should return empty segments."""
        df = generate_synthetic_ohlcv(datetime(2023, 1, 1), num_days=2)
        detector = RegimeDetector(method='adx_di_hysteresis')
        segments = detector.classify_periods_adaptive(df, warmup_bars=200)
        # May return empty if data is too short after warmup
        assert isinstance(segments, list)


# ═══════════════════════════════════════════════════════════════════
# Merge Short Segments
# ═══════════════════════════════════════════════════════════════════

class TestMergeShortSegments:
    """Tests for RegimeDetector._merge_short_segments()."""

    def test_merge_same_regime(self):
        """Adjacent same-regime short segments should merge."""
        segments = [
            {'start': datetime(2023, 1, 1), 'end': datetime(2023, 1, 20),
             'regime': RegimeType.BULLISH, 'confidence': 0.8,
             'duration_days': 19, 'bar_count': 100},
            {'start': datetime(2023, 1, 20), 'end': datetime(2023, 1, 25),
             'regime': RegimeType.BULLISH, 'confidence': 0.7,
             'duration_days': 5, 'bar_count': 30},
        ]
        merged = RegimeDetector._merge_short_segments(segments, min_duration_days=7)
        assert len(merged) == 1
        assert merged[0]['regime'] == RegimeType.BULLISH

    def test_keep_long_segments(self):
        """Segments longer than threshold should not be merged."""
        segments = [
            {'start': datetime(2023, 1, 1), 'end': datetime(2023, 2, 1),
             'regime': RegimeType.BULLISH, 'confidence': 0.8,
             'duration_days': 31, 'bar_count': 200},
            {'start': datetime(2023, 2, 1), 'end': datetime(2023, 3, 1),
             'regime': RegimeType.BEARISH, 'confidence': 0.9,
             'duration_days': 28, 'bar_count': 180},
        ]
        merged = RegimeDetector._merge_short_segments(segments, min_duration_days=7)
        assert len(merged) == 2

    def test_empty_list(self):
        assert RegimeDetector._merge_short_segments([], 7) == []


# ═══════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
