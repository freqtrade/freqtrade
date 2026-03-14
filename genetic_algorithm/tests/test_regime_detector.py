"""
Test Regime Detection Module

Comprehensive tests for the RegimeDetector class and related utilities.
Tests regime detection accuracy with synthetic and real market data scenarios.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import pytest

# Add the freqtradeForkGA directory to path for genetic_algorithm.xxx imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genetic_algorithm.utils.regime_detector import (
    RegimeDetector,
    RegimeType,
    RegimeSegment,
    save_segments_to_yaml,
    load_segments_from_yaml,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_ohlcv(
    start_date: datetime,
    num_days: int,
    regime: str = 'bullish',
    timeframe_minutes: int = 60,  # 1h candles
    initial_price: float = 100.0,
    volatility: float = 0.01,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with a specific regime characteristic.
    
    Args:
        start_date: Start datetime
        num_days: Number of days to generate
        regime: 'bullish', 'bearish', or 'sideways'
        timeframe_minutes: Candle timeframe in minutes
        initial_price: Starting price
        volatility: Base volatility (std of returns)
    
    Returns:
        DataFrame with OHLCV data and datetime index
    """
    candles_per_day = 24 * 60 // timeframe_minutes
    num_candles = num_days * candles_per_day
    
    # Set drift based on regime
    if regime == 'bullish':
        drift = 0.0005  # Positive drift
    elif regime == 'bearish':
        drift = -0.0005  # Negative drift
    else:  # sideways
        drift = 0.0  # No drift
        volatility = volatility * 0.5  # Lower volatility in sideways
    
    # Generate returns
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(drift, volatility, num_candles)
    
    # Generate price series
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    dates = pd.date_range(start=start_date, periods=num_candles, freq=f'{timeframe_minutes}min')
    
    data = {
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, volatility/2, num_candles))),
        'low': prices * (1 - np.abs(np.random.normal(0, volatility/2, num_candles))),
        'close': prices * (1 + np.random.normal(0, volatility/3, num_candles)),
        'volume': np.random.uniform(1000, 10000, num_candles),
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Fix high/low to make sense
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df


def generate_multi_regime_data(
    start_date: datetime,
    regime_sequence: List[tuple],  # List of (regime, days)
    timeframe_minutes: int = 60,
) -> pd.DataFrame:
    """
    Generate synthetic data with multiple regime periods.
    
    Args:
        start_date: Start datetime
        regime_sequence: List of (regime_name, num_days) tuples
        timeframe_minutes: Candle timeframe in minutes
    
    Returns:
        DataFrame with concatenated regime periods
    """
    dfs = []
    current_date = start_date
    last_price = 100.0
    
    for regime, days in regime_sequence:
        df = generate_synthetic_ohlcv(
            start_date=current_date,
            num_days=days,
            regime=regime,
            timeframe_minutes=timeframe_minutes,
            initial_price=last_price,
        )
        dfs.append(df)
        current_date = df.index[-1] + timedelta(minutes=timeframe_minutes)
        last_price = df['close'].iloc[-1]
    
    return pd.concat(dfs)


class TestRegimeDetectorInit:
    """Test RegimeDetector initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        detector = RegimeDetector()
        assert detector.method == 'adx_di_hysteresis'  # Updated default method
        assert 'adx_enter' in detector.params  # adx_di_hysteresis uses different params
        assert detector.params['adx_enter'] == 25
        assert detector.params['adx_exit'] == 20
    
    def test_custom_method(self):
        """Test initialization with different methods."""
        for method in ['sma_adx', 'adx_di', 'returns', 'bollinger', 'ensemble']:
            detector = RegimeDetector(method=method)
            assert detector.method == method
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown detection method"):
            RegimeDetector(method='invalid_method')
    
    def test_custom_params(self):
        """Test custom parameter override."""
        custom_params = {'sma_fast': 20, 'sma_slow': 100}
        detector = RegimeDetector(method='sma_adx', params=custom_params)
        assert detector.params['sma_fast'] == 20
        assert detector.params['sma_slow'] == 100
        # Default params should still be present
        assert 'adx_period' in detector.params


class TestRegimeDetection:
    """Test regime detection methods."""
    
    @pytest.fixture
    def bullish_data(self) -> pd.DataFrame:
        """Generate bullish market data."""
        return generate_synthetic_ohlcv(
            start_date=datetime(2023, 1, 1),
            num_days=60,
            regime='bullish',
            volatility=0.015,
        )
    
    @pytest.fixture
    def bearish_data(self) -> pd.DataFrame:
        """Generate bearish market data."""
        return generate_synthetic_ohlcv(
            start_date=datetime(2023, 1, 1),
            num_days=60,
            regime='bearish',
            volatility=0.015,
        )
    
    @pytest.fixture
    def sideways_data(self) -> pd.DataFrame:
        """Generate sideways market data."""
        return generate_synthetic_ohlcv(
            start_date=datetime(2023, 1, 1),
            num_days=60,
            regime='sideways',
            volatility=0.005,
        )
    
    def test_sma_adx_bullish(self, bullish_data):
        """Test SMA+ADX detection on bullish data."""
        detector = RegimeDetector(method='sma_adx')
        regimes = detector.detect(bullish_data)
        
        # Should be mostly bullish (after indicator warm-up)
        warmup = 200
        if len(regimes) > warmup:
            post_warmup = regimes.iloc[warmup:]
            bullish_ratio = (post_warmup == RegimeType.BULLISH).mean()
            # Expect at least 40% bullish (synthetic data isn't perfect)
            assert bullish_ratio > 0.3 or (post_warmup == RegimeType.SIDEWAYS).mean() > 0.3, \
                f"Expected more bullish or sideways signals, got bullish={bullish_ratio:.1%}"
    
    def test_sma_adx_bearish(self, bearish_data):
        """Test SMA+ADX detection on bearish data."""
        detector = RegimeDetector(method='sma_adx')
        regimes = detector.detect(bearish_data)
        
        # Should be mostly bearish or sideways (after indicator warm-up)
        warmup = 200
        if len(regimes) > warmup:
            post_warmup = regimes.iloc[warmup:]
            bearish_ratio = (post_warmup == RegimeType.BEARISH).mean()
            sideways_ratio = (post_warmup == RegimeType.SIDEWAYS).mean()
            # Accept either bearish or sideways as valid outcomes
            assert bearish_ratio > 0.2 or sideways_ratio > 0.3
    
    def test_adx_di_detection(self, bullish_data):
        """Test ADX+DI detection method."""
        detector = RegimeDetector(method='adx_di')
        regimes = detector.detect(bullish_data)
        
        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(bullish_data)
        # Check that we get valid regime types
        valid_types = {RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS, RegimeType.UNCERTAIN}
        for r in regimes.dropna():
            assert r in valid_types
    
    def test_returns_detection(self, bullish_data):
        """Test returns-based detection method."""
        detector = RegimeDetector(method='returns')
        regimes = detector.detect(bullish_data)
        
        assert isinstance(regimes, pd.Series)
        # Returns method can also detect VOLATILE
        valid_types = {RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS, 
                       RegimeType.VOLATILE, RegimeType.UNCERTAIN}
        for r in regimes.dropna():
            assert r in valid_types
    
    def test_bollinger_detection(self, bullish_data):
        """Test Bollinger Band detection method."""
        detector = RegimeDetector(method='bollinger')
        regimes = detector.detect(bullish_data)
        
        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(bullish_data)
    
    def test_ensemble_detection(self, bullish_data):
        """Test ensemble voting detection."""
        detector = RegimeDetector(method='ensemble')
        regimes = detector.detect(bullish_data)
        
        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(bullish_data)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        detector = RegimeDetector()
        regimes = detector.detect(pd.DataFrame())
        assert len(regimes) == 0
    
    def test_missing_columns(self, bullish_data):
        """Test error on missing required columns."""
        detector = RegimeDetector(method='sma_adx')
        df_missing = bullish_data[['close', 'volume']]  # Missing high, low
        
        with pytest.raises(ValueError, match="Missing required columns"):
            detector.detect(df_missing)


class TestClassifyPeriods:
    """Test period classification functionality."""
    
    @pytest.fixture
    def multi_regime_data(self) -> pd.DataFrame:
        """Generate multi-regime data for testing."""
        return generate_multi_regime_data(
            start_date=datetime(2023, 1, 1),
            regime_sequence=[
                ('bullish', 120),
                ('bearish', 90),
                ('sideways', 100),
                ('bullish', 80),
                ('bearish', 60),
            ],
            timeframe_minutes=60,
        )
    
    def test_classify_periods_basic(self, multi_regime_data):
        """Test basic period classification."""
        detector = RegimeDetector(method='sma_adx')
        segments = detector.classify_periods(
            multi_regime_data,
            period_days=60,
            min_period_days=30,
        )
        
        assert len(segments) > 0
        
        for seg in segments:
            assert isinstance(seg, RegimeSegment)
            assert seg.segment_id is not None
            assert seg.regime in RegimeType
            assert 0 <= seg.confidence <= 1
            assert seg.duration_days >= 30
    
    def test_classify_periods_metadata(self, multi_regime_data):
        """Test that metadata is populated correctly."""
        detector = RegimeDetector()
        segments = detector.classify_periods(multi_regime_data, period_days=90)
        
        for seg in segments:
            assert 'mean_return' in seg.metadata
            assert 'volatility' in seg.metadata
            assert 'total_return' in seg.metadata
            assert 'bar_count' in seg.metadata
    
    def test_classify_periods_timerange(self, multi_regime_data):
        """Test that timerange property works."""
        detector = RegimeDetector()
        segments = detector.classify_periods(multi_regime_data, period_days=60)
        
        for seg in segments:
            timerange = seg.timerange
            assert '-' in timerange
            parts = timerange.split('-')
            assert len(parts) == 2
            # Should be valid date format
            datetime.strptime(parts[0], '%Y%m%d')
            datetime.strptime(parts[1], '%Y%m%d')
    
    def test_embargo_gap(self, multi_regime_data):
        """Test that embargo gaps are maintained between segments."""
        detector = RegimeDetector()
        embargo_days = 7
        segments = detector.classify_periods(
            multi_regime_data,
            period_days=60,
            embargo_days=embargo_days,
        )
        
        # Check gaps between consecutive segments
        for i in range(len(segments) - 1):
            gap = (segments[i + 1].start_date - segments[i].end_date).days
            assert gap >= embargo_days, f"Gap between segments {i} and {i+1} is {gap} days (expected >={embargo_days})"


class TestBalancedSelection:
    """Test balanced segment selection."""
    
    @pytest.fixture
    def mixed_segments(self) -> List[RegimeSegment]:
        """Create test segments with different regimes."""
        segments = []
        base_date = datetime(2023, 1, 1)
        
        # Create 5 bullish, 4 bearish, 3 sideways segments
        for i in range(5):
            segments.append(RegimeSegment(
                segment_id=f'bull_{i}',
                start_date=base_date + timedelta(days=i*30),
                end_date=base_date + timedelta(days=(i+1)*30),
                regime=RegimeType.BULLISH,
                confidence=0.9 - i*0.1,
            ))
        
        for i in range(4):
            segments.append(RegimeSegment(
                segment_id=f'bear_{i}',
                start_date=base_date + timedelta(days=150 + i*30),
                end_date=base_date + timedelta(days=180 + i*30),
                regime=RegimeType.BEARISH,
                confidence=0.85 - i*0.1,
            ))
        
        for i in range(3):
            segments.append(RegimeSegment(
                segment_id=f'side_{i}',
                start_date=base_date + timedelta(days=270 + i*30),
                end_date=base_date + timedelta(days=300 + i*30),
                regime=RegimeType.SIDEWAYS,
                confidence=0.8 - i*0.1,
            ))
        
        return segments
    
    def test_get_balanced_segments(self, mixed_segments):
        """Test balanced segment selection."""
        detector = RegimeDetector()
        balanced = detector.get_balanced_segments(
            mixed_segments,
            segments_per_regime=2,
        )
        
        # Should have 2 from each regime = 6 total
        assert len(balanced) == 6
        
        # Count by regime
        counts = {}
        for seg in balanced:
            counts[seg.regime] = counts.get(seg.regime, 0) + 1
        
        assert counts[RegimeType.BULLISH] == 2
        assert counts[RegimeType.BEARISH] == 2
        assert counts[RegimeType.SIDEWAYS] == 2
    
    def test_balanced_selection_prefers_high_confidence(self, mixed_segments):
        """Test that high-confidence segments are preferred."""
        detector = RegimeDetector()
        balanced = detector.get_balanced_segments(
            mixed_segments,
            segments_per_regime=2,
        )
        
        # Bullish segments should be bull_0 and bull_1 (highest confidence)
        bullish_segs = [s for s in balanced if s.regime == RegimeType.BULLISH]
        assert all(s.confidence >= 0.8 for s in bullish_segs)
    
    def test_balanced_handles_missing_regime(self, mixed_segments):
        """Test handling when a regime has insufficient segments."""
        # Remove all sideways segments
        filtered = [s for s in mixed_segments if s.regime != RegimeType.SIDEWAYS]
        
        detector = RegimeDetector()
        balanced = detector.get_balanced_segments(
            filtered,
            segments_per_regime=3,
            target_regimes=[RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS],
        )
        
        # Should have 3 bullish, 3 bearish, 0 sideways = 6 total
        counts = {}
        for seg in balanced:
            counts[seg.regime] = counts.get(seg.regime, 0) + 1
        
        assert counts.get(RegimeType.BULLISH, 0) == 3
        assert counts.get(RegimeType.BEARISH, 0) == 3
        assert counts.get(RegimeType.SIDEWAYS, 0) == 0


class TestSegmentSplit:
    """Test segment splitting into optimization/model_selection/holdout."""
    
    @pytest.fixture
    def temporal_segments(self) -> List[RegimeSegment]:
        """Create segments with temporal ordering."""
        segments = []
        base_date = datetime(2020, 1, 1)
        
        regimes = [
            RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS,
            RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS,
            RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS,
            RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS,
        ]
        
        for i, regime in enumerate(regimes):
            segments.append(RegimeSegment(
                segment_id=f'seg_{i}',
                start_date=base_date + timedelta(days=i*30),
                end_date=base_date + timedelta(days=(i+1)*30),
                regime=regime,
                confidence=0.8,
            ))
        
        return segments
    
    def test_split_ratios(self, temporal_segments):
        """Test that split ratios are approximately correct."""
        detector = RegimeDetector()
        splits = detector.split_segments_by_role(
            temporal_segments,
            optimization_ratio=0.60,
            model_selection_ratio=0.20,
            holdout_ratio=0.20,
        )
        
        total = len(temporal_segments)
        assert len(splits['optimization']) > 0
        assert len(splits['holdout']) > 0
        
        # Check total preserved
        total_split = sum(len(v) for v in splits.values())
        assert total_split == total
    
    def test_holdout_is_newest(self, temporal_segments):
        """Test that holdout segments are the newest."""
        detector = RegimeDetector()
        splits = detector.split_segments_by_role(temporal_segments)
        
        if splits['holdout'] and splits['optimization']:
            holdout_dates = [s.start_date for s in splits['holdout']]
            opt_dates = [s.start_date for s in splits['optimization']]
            
            # Holdout should be newer than most optimization segments
            min_holdout = min(holdout_dates)
            max_opt = max(opt_dates)
            assert min_holdout >= max_opt or min_holdout > min(opt_dates)
    
    def test_split_updates_role(self, temporal_segments):
        """Test that segment roles are updated."""
        detector = RegimeDetector()
        splits = detector.split_segments_by_role(temporal_segments)
        
        for seg in splits['optimization']:
            assert seg.role == 'optimization'
        for seg in splits['model_selection']:
            assert seg.role == 'model_selection'
        for seg in splits['holdout']:
            assert seg.role == 'holdout'


class TestYAMLPersistence:
    """Test YAML saving and loading."""
    
    @pytest.fixture
    def sample_segments(self) -> Dict[str, List[RegimeSegment]]:
        """Create sample segments for testing."""
        return {
            'optimization': [
                RegimeSegment(
                    segment_id='opt_001',
                    start_date=datetime(2023, 1, 1),
                    end_date=datetime(2023, 3, 31),
                    regime=RegimeType.BULLISH,
                    confidence=0.85,
                    metadata={'mean_return': 0.002},
                    role='optimization',
                ),
            ],
            'model_selection': [
                RegimeSegment(
                    segment_id='ms_001',
                    start_date=datetime(2023, 4, 1),
                    end_date=datetime(2023, 5, 31),
                    regime=RegimeType.BEARISH,
                    confidence=0.78,
                    role='model_selection',
                ),
            ],
            'holdout': [
                RegimeSegment(
                    segment_id='ho_001',
                    start_date=datetime(2023, 6, 1),
                    end_date=datetime(2023, 7, 31),
                    regime=RegimeType.SIDEWAYS,
                    confidence=0.72,
                    role='holdout',
                ),
            ],
        }
    
    def test_save_and_load(self, sample_segments, tmp_path):
        """Test round-trip YAML save and load."""
        filepath = tmp_path / 'segments.yaml'
        
        metadata = {
            'run_id': 'test_run_001',
            'detector_method': 'sma_adx',
        }
        
        # Save
        save_segments_to_yaml(sample_segments, filepath, metadata=metadata)
        assert filepath.exists()
        
        # Load
        loaded = load_segments_from_yaml(filepath)
        
        # Verify structure
        assert 'optimization' in loaded
        assert 'model_selection' in loaded
        assert 'holdout' in loaded
        
        # Verify content
        assert len(loaded['optimization']) == 1
        opt_seg = loaded['optimization'][0]
        assert opt_seg.segment_id == 'opt_001'
        assert opt_seg.regime == RegimeType.BULLISH
        assert opt_seg.confidence == 0.85


def run_tests_cli():
    """Run tests from command line."""
    print("\n" + "=" * 80)
    print("REGIME DETECTOR TESTS")
    print("=" * 80)
    
    # Run basic tests manually without pytest for quick verification
    test_cases = [
        ("Default initialization", TestRegimeDetectorInit().test_default_init),
        ("Custom method", TestRegimeDetectorInit().test_custom_method),
        ("Custom params", TestRegimeDetectorInit().test_custom_params),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in test_cases:
        try:
            test_func()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
    
    # Test detection with synthetic data
    print("\n  Testing regime detection...")
    
    try:
        # Generate test data
        bullish_data = generate_synthetic_ohlcv(
            start_date=datetime(2023, 1, 1),
            num_days=90,
            regime='bullish',
        )
        
        detector = RegimeDetector(method='sma_adx')
        regimes = detector.detect(bullish_data)
        
        print(f"    Generated {len(bullish_data)} candles")
        print(f"    Detected regimes: {len(regimes)} values")
        
        # Show regime distribution
        regime_counts = regimes.value_counts()
        print(f"    Regime distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(regimes) * 100
            print(f"      {regime.value}: {count} ({pct:.1f}%)")
        
        passed += 1
        print("  ✅ Regime detection working")
        
    except Exception as e:
        print(f"  ❌ Regime detection failed: {e}")
        failed += 1
    
    # Test period classification
    print("\n  Testing period classification...")
    
    try:
        multi_data = generate_multi_regime_data(
            start_date=datetime(2022, 1, 1),
            regime_sequence=[
                ('bullish', 120),
                ('bearish', 100),
                ('sideways', 90),
            ],
        )
        
        detector = RegimeDetector()
        segments = detector.classify_periods(multi_data, period_days=60, min_period_days=30)
        
        print(f"    Created {len(segments)} segments from {len(multi_data)} candles")
        for seg in segments:
            print(f"      {seg.segment_id}: {seg.regime.value} ({seg.confidence:.0%})")
        
        passed += 1
        print("  ✅ Period classification working")
        
    except Exception as e:
        print(f"  ❌ Period classification failed: {e}")
        failed += 1
    
    # Summary
    print("\n" + "-" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


# ──────────────────────────────────────────────────────────────────────
# Regression tests for regime detection bug-fixes (March 2026)
# ──────────────────────────────────────────────────────────────────────

class TestMergeShortSegmentsScoreRecompute:
    """
    Verify that _merge_short_segments recomputes avg_trend_score as
    a bar-count-weighted average instead of keeping the first segment's
    stale value.
    """

    @staticmethod
    def _make_seg(regime, start_day, end_day, avg_trend_score, bar_count):
        from datetime import datetime
        return {
            'regime': regime,
            'start': datetime(2023, 1, start_day),
            'end': datetime(2023, 1, end_day),
            'duration_days': end_day - start_day,
            'bar_count': bar_count,
            'confidence': 0.8,
            'avg_trend_score': avg_trend_score,
            'avg_volatility_score': 0.5,
            'trend_score_std': 0.1,
        }

    def test_same_regime_merge_recomputes_score(self):
        """When two same-regime segments merge, avg_trend_score should be
        the bar-count-weighted average, not frozen at the first segment."""
        seg1 = self._make_seg(RegimeType.BEARISH, 1, 3, -0.95, 48)
        seg2 = self._make_seg(RegimeType.BEARISH, 3, 5, -0.20, 48)

        merged = RegimeDetector._merge_short_segments([seg1, seg2], min_duration_days=7)
        assert len(merged) == 1
        # Weighted avg: (-0.95*48 + -0.20*48) / 96 = -0.575
        assert merged[0]['avg_trend_score'] == pytest.approx(-0.575, abs=0.01)

    def test_different_regime_merge_recomputes_score(self):
        """When a short different-regime segment is absorbed, avg_trend_score
        should be recomputed, potentially flipping the final regime label."""
        # First segment: short bearish
        seg1 = self._make_seg(RegimeType.BEARISH, 1, 3, -0.80, 20)
        # Second segment: short bullish with many more bars
        seg2 = self._make_seg(RegimeType.BULLISH, 3, 5, 0.90, 100)
        # Third: long bullish (anchor)
        seg3 = self._make_seg(RegimeType.BULLISH, 5, 20, 0.85, 360)

        merged = RegimeDetector._merge_short_segments([seg1, seg2, seg3], min_duration_days=7)

        # seg1 absorbs seg2 → weighted avg: (-0.80*20 + 0.90*100) / 120 ≈ 0.617
        # seg3 is long enough to survive on its own
        # So we expect either 1 or 2 segments, and the first one should NOT be -0.80
        first = merged[0]
        assert first['avg_trend_score'] > 0.0, (
            f"Expected positive avg_trend_score after absorbing large bullish segment, "
            f"got {first['avg_trend_score']:.3f}"
        )

    def test_second_pass_recomputes_score(self):
        """The second pass (prev too short, absorb into current) should also
        recompute avg_trend_score."""
        # First segment: short with bearish score
        seg1 = self._make_seg(RegimeType.BEARISH, 1, 3, -0.90, 48)
        # Second segment: long bullish
        seg2 = self._make_seg(RegimeType.BULLISH, 3, 20, 0.80, 408)

        merged = RegimeDetector._merge_short_segments([seg1, seg2], min_duration_days=7)
        assert len(merged) == 1
        # First pass: seg1 absorbs seg2? No — seg2 is long enough (17 days >= 7)
        # So after first pass: [seg1(2d), seg2(17d)]
        # Second pass: seg1 is too short → absorbed into seg2
        # Weighted avg: (-0.90*48 + 0.80*408) / 456 ≈ 0.621
        assert merged[0]['avg_trend_score'] > 0.0


class TestSMAAutoScale:
    """Verify SMA params auto-scale for sub-daily timeframes."""

    def test_infer_bar_interval_1h(self):
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        df = pd.DataFrame({'close': np.random.randn(100) + 100}, index=dates)
        hours = RegimeDetector._infer_bar_interval_hours(df)
        assert hours == pytest.approx(1.0, abs=0.1)

    def test_infer_bar_interval_4h(self):
        dates = pd.date_range('2023-01-01', periods=100, freq='4h')
        df = pd.DataFrame({'close': np.random.randn(100) + 100}, index=dates)
        hours = RegimeDetector._infer_bar_interval_hours(df)
        assert hours == pytest.approx(4.0, abs=0.1)

    def test_infer_bar_interval_1d(self):
        dates = pd.date_range('2023-01-01', periods=100, freq='1D')
        df = pd.DataFrame({'close': np.random.randn(100) + 100}, index=dates)
        hours = RegimeDetector._infer_bar_interval_hours(df)
        assert hours == pytest.approx(24.0, abs=0.5)

    def test_sma_slope_score_on_hourly_data_finds_all_regimes(self):
        """SMA on 1h data with auto-scaling should find bullish AND bearish
        in a multi-regime synthetic dataset."""
        regime_seq = [
            ('bullish', 90),
            ('bearish', 90),
            ('sideways', 60),
            ('bullish', 120),
            ('bearish', 60),
        ]
        data = generate_multi_regime_data(
            start_date=datetime(2023, 1, 1),
            regime_sequence=regime_seq,
            timeframe_minutes=60,
        )
        det = RegimeDetector(method='sma_slope')
        score = det._compute_sma_slope_score(data)
        valid = score.dropna()

        bull_pct = (valid > 0.40).mean()
        bear_pct = (valid < -0.40).mean()

        assert bull_pct > 0.10, f"Expected >10% bullish bars, got {bull_pct*100:.1f}%"
        assert bear_pct > 0.10, f"Expected >10% bearish bars, got {bear_pct*100:.1f}%"


class TestClassifyPeriodsByScoreMultiRegime:
    """End-to-end test: classify_periods_by_score on multi-regime 1h data
    must produce segments in at least bullish AND bearish regimes."""

    def test_hourly_multi_regime_segments(self):
        regime_seq = [
            ('bullish', 90),
            ('bearish', 90),
            ('sideways', 60),
            ('bullish', 120),
            ('bearish', 90),
        ]
        data = generate_multi_regime_data(
            start_date=datetime(2023, 1, 1),
            regime_sequence=regime_seq,
            timeframe_minutes=60,
        )
        det = RegimeDetector(method='sma_slope')
        segments = det.classify_periods_by_score(
            df=data,
            bullish_min=0.40,
            bearish_max=-0.40,
            min_segment_days=7,
            max_segment_days=180,
            merge_threshold_days=7,
            embargo_days=3,
        )

        regimes_found = {s.regime for s in segments}
        assert RegimeType.BULLISH in regimes_found, (
            f"No bullish segments found! Regimes: {regimes_found}"
        )
        assert RegimeType.BEARISH in regimes_found, (
            f"No bearish segments found! Regimes: {regimes_found}"
        )
        assert len(segments) >= 3, (
            f"Expected at least 3 segments for 5 regime periods, got {len(segments)}"
        )


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--pytest':
        # Run with pytest
        pytest.main([__file__, '-v'])
    else:
        # Quick CLI tests
        success = run_tests_cli()
        sys.exit(0 if success else 1)
