"""
Tests for Regime-Aware Fitness Evaluator

Tests the integration of regime detection with the GA fitness evaluation pipeline.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from statistics import harmonic_mean

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from genetic_algorithm.evaluation.regime_aware import (
    RegimeAwareEvaluator,
    RegimeEvaluationResult,
    create_regime_aware_evaluator,
)
from genetic_algorithm.utils.regime_detector import (
    RegimeDetector,
    RegimeSegment,
    RegimeType,
)
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_config() -> Dict[str, Any]:
    """Create a minimal test configuration."""
    return {
        'backtesting': {
            'pairs': ['BTC/USDT'],
            'timerange': '20230101-20231231',
            'stake_amount': 100,
            'max_open_trades': 3,
            'fee': 0.001,
            'datadir': 'user_data/data/binance',
        },
        'fitness_weights': {
            'profit': 0.30,
            'sharpe_ratio': 0.20,
            'drawdown': 0.20,
            'win_rate': 0.15,
            'trade_frequency': 0.15,
        },
        'fitness_penalties': {
            'min_trades': 5,
            'max_drawdown': 0.30,
            'complexity_weight': 0.01,
        },
        'regime_aware': {
            'enabled': True,
            'method': 'sma_adx',
            'segments_per_regime': 2,
            'holdout_ratio': 0.20,
            'aggregation': 'harmonic_mean',
            'cvar_alpha': 0.20,
            'regime_weights': {
                'bullish': 1.0,
                'bearish': 1.0,
                'sideways': 1.0,
            },
        },
        'walk_forward': {
            'enabled': False,
        },
    }


def create_test_segments() -> Dict[str, List[RegimeSegment]]:
    """Create test regime segments."""
    base_date = datetime(2023, 1, 1)
    
    optimization_segments = [
        # Bullish segments
        RegimeSegment(
            segment_id='seg_001_bullish',
            start_date=base_date,
            end_date=base_date + timedelta(days=60),
            regime=RegimeType.BULLISH,
            confidence=0.85,
            role='optimization',
        ),
        RegimeSegment(
            segment_id='seg_002_bullish',
            start_date=base_date + timedelta(days=120),
            end_date=base_date + timedelta(days=180),
            regime=RegimeType.BULLISH,
            confidence=0.78,
            role='optimization',
        ),
        # Bearish segments
        RegimeSegment(
            segment_id='seg_003_bearish',
            start_date=base_date + timedelta(days=65),
            end_date=base_date + timedelta(days=115),
            regime=RegimeType.BEARISH,
            confidence=0.82,
            role='optimization',
        ),
        RegimeSegment(
            segment_id='seg_004_bearish',
            start_date=base_date + timedelta(days=185),
            end_date=base_date + timedelta(days=245),
            regime=RegimeType.BEARISH,
            confidence=0.75,
            role='optimization',
        ),
        # Sideways segments
        RegimeSegment(
            segment_id='seg_005_sideways',
            start_date=base_date + timedelta(days=250),
            end_date=base_date + timedelta(days=300),
            regime=RegimeType.SIDEWAYS,
            confidence=0.70,
            role='optimization',
        ),
    ]
    
    holdout_segments = [
        RegimeSegment(
            segment_id='seg_006_holdout',
            start_date=base_date + timedelta(days=305),
            end_date=base_date + timedelta(days=365),
            regime=RegimeType.BULLISH,
            confidence=0.80,
            role='holdout',
        ),
    ]
    
    return {
        'optimization': optimization_segments,
        'model_selection': [],
        'holdout': holdout_segments,
    }


def create_test_strategy_gene() -> StrategyGene:
    """Create a simple test strategy gene."""
    return StrategyGene(
        indicators=[
            IndicatorGene(
                indicator_type='RSI',
                params={'period': 14},
            ),
            IndicatorGene(
                indicator_type='EMA',
                params={'period': 20},
            ),
        ],
        conditions=[
            ConditionGene(
                indicator_key='RSI_0',
                operator='lt',
                threshold=30.0,
            ),
        ],
        exit_conditions=[
            ConditionGene(
                indicator_key='RSI_0',
                operator='gt',
                threshold=70.0,
            ),
        ],
        stoploss=-0.05,
        roi={0: 0.10, 60: 0.05, 120: 0.02},
        max_open_trades=3,
    )


class TestRegimeAwareEvaluator:
    """Test suite for RegimeAwareEvaluator."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def run_test(self, name: str, test_func):
        """Run a single test and track results."""
        try:
            test_func()
            self.passed += 1
            print(f"  ✅ {name}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  ❌ {name}: {e}")
        except Exception as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  💥 {name}: {type(e).__name__}: {e}")
    
    def test_initialization(self):
        """Test RegimeAwareEvaluator initialization."""
        config = create_test_config()
        segments = create_test_segments()
        
        evaluator = RegimeAwareEvaluator(config, segments)
        
        assert len(evaluator._optimization_segments) == 5, \
            f"Expected 5 optimization segments, got {len(evaluator._optimization_segments)}"
        assert len(evaluator._holdout_segments) == 1, \
            f"Expected 1 holdout segment, got {len(evaluator._holdout_segments)}"
        assert evaluator.aggregation_method == 'harmonic_mean', \
            f"Expected harmonic_mean aggregation, got {evaluator.aggregation_method}"
    
    def test_initialization_no_segments(self):
        """Test initialization without pre-computed segments."""
        config = create_test_config()
        
        evaluator = RegimeAwareEvaluator(config, segments={})
        
        assert len(evaluator._optimization_segments) == 0
        assert len(evaluator._holdout_segments) == 0
    
    def test_aggregation_methods(self):
        """Test different fitness aggregation methods."""
        # Test data: fitness scores from different segments
        fitness_values = [0.5, 0.8, 0.3, 0.6, 0.4]
        
        # Mean
        mean_result = sum(fitness_values) / len(fitness_values)
        assert abs(mean_result - 0.52) < 0.01, f"Mean should be ~0.52, got {mean_result}"
        
        # Min
        min_result = min(fitness_values)
        assert min_result == 0.3, f"Min should be 0.3, got {min_result}"
        
        # Harmonic mean (correct value is ~0.465)
        hm_result = harmonic_mean(fitness_values)
        assert abs(hm_result - 0.465) < 0.02, f"Harmonic mean should be ~0.465, got {hm_result}"
        
        # CVaR (bottom 20% = 1 value)
        sorted_values = sorted(fitness_values)
        n_worst = max(1, int(len(sorted_values) * 0.2))
        cvar_result = sum(sorted_values[:n_worst]) / n_worst
        assert cvar_result == 0.3, f"CVaR should be 0.3, got {cvar_result}"
    
    def test_regime_summary(self):
        """Test regime summary calculation."""
        config = create_test_config()
        segments = create_test_segments()
        
        evaluator = RegimeAwareEvaluator(config, segments)
        
        # Create mock results
        mock_results = [
            RegimeEvaluationResult(
                segment=segments['optimization'][0],  # BULLISH
                fitness=0.8,
                metrics={'profit': 10},
                success=True,
            ),
            RegimeEvaluationResult(
                segment=segments['optimization'][1],  # BULLISH
                fitness=0.7,
                metrics={'profit': 8},
                success=True,
            ),
            RegimeEvaluationResult(
                segment=segments['optimization'][2],  # BEARISH
                fitness=0.4,
                metrics={'profit': 2},
                success=True,
            ),
        ]
        
        summary = evaluator._get_regime_summary(mock_results)
        
        assert 'bullish' in summary, "Summary should contain bullish"
        assert 'bearish' in summary, "Summary should contain bearish"
        assert summary['bullish']['count'] == 2, f"Expected 2 bullish, got {summary['bullish']['count']}"
        assert abs(summary['bullish']['avg_fitness'] - 0.75) < 0.01, \
            f"Expected bullish avg ~0.75, got {summary['bullish']['avg_fitness']}"
    
    def test_cache_mechanism(self):
        """Test segment-level caching."""
        config = create_test_config()
        segments = create_test_segments()
        
        evaluator = RegimeAwareEvaluator(config, segments)
        
        # Initially empty cache
        stats = evaluator.get_cache_stats()
        assert stats['cache_size'] == 0, "Cache should be empty initially"
        
        # Test clear
        evaluator.clear_cache()
        stats = evaluator.get_cache_stats()
        assert stats['cache_hits'] == 0, "Cache hits should be 0 after clear"
    
    def test_segment_properties(self):
        """Test RegimeSegment properties."""
        segment = RegimeSegment(
            segment_id='test_seg',
            start_date=datetime(2023, 6, 1),
            end_date=datetime(2023, 8, 31),
            regime=RegimeType.BULLISH,
            confidence=0.85,
            role='optimization',
        )
        
        assert segment.timerange == '20230601-20230831', \
            f"Timerange format incorrect: {segment.timerange}"
        assert segment.duration_days == 91, \
            f"Duration should be 91 days, got {segment.duration_days}"
    
    def test_fallback_without_segments(self):
        """Test that evaluator falls back to standard evaluation without segments."""
        config = create_test_config()
        
        # Create evaluator with no segments
        evaluator = RegimeAwareEvaluator(config, segments={})
        
        # It should have no segments
        assert len(evaluator._optimization_segments) == 0
        assert len(evaluator._holdout_segments) == 0
    
    def test_holdout_protection(self):
        """Test that holdout segments are kept separate."""
        config = create_test_config()
        segments = create_test_segments()
        
        evaluator = RegimeAwareEvaluator(config, segments)
        
        # Optimization segments should not include holdout
        opt_ids = {s.segment_id for s in evaluator._optimization_segments}
        holdout_ids = {s.segment_id for s in evaluator._holdout_segments}
        
        assert len(opt_ids & holdout_ids) == 0, "Optimization and holdout should not overlap"
    
    def run_all(self):
        """Run all tests."""
        print("\n" + "=" * 70)
        print("REGIME-AWARE EVALUATOR TESTS")
        print("=" * 70)
        
        self.run_test("Initialization with segments", self.test_initialization)
        self.run_test("Initialization without segments", self.test_initialization_no_segments)
        self.run_test("Aggregation methods", self.test_aggregation_methods)
        self.run_test("Regime summary calculation", self.test_regime_summary)
        self.run_test("Cache mechanism", self.test_cache_mechanism)
        self.run_test("Segment properties", self.test_segment_properties)
        self.run_test("Fallback without segments", self.test_fallback_without_segments)
        self.run_test("Holdout protection", self.test_holdout_protection)
        
        print("-" * 70)
        if self.failed == 0:
            print(f"RESULTS: {self.passed} passed, {self.failed} failed ✅")
        else:
            print(f"RESULTS: {self.passed} passed, {self.failed} failed ❌")
            print("\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print("=" * 70)
        
        return self.failed == 0


class TestIntegrationWithRegimeDetector:
    """Integration tests connecting RegimeDetector with RegimeAwareEvaluator."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def run_test(self, name: str, test_func):
        """Run a single test and track results."""
        try:
            test_func()
            self.passed += 1
            print(f"  ✅ {name}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  ❌ {name}: {e}")
        except Exception as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  💥 {name}: {type(e).__name__}: {e}")
    
    def _create_synthetic_data(self, days: int = 365) -> pd.DataFrame:
        """Create synthetic OHLCV data with regime patterns."""
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Create price with different regime phases
        prices = []
        price = 100.0
        
        for i in range(days):
            if i < 90:  # Bullish phase
                trend = 0.002
                vol = 0.01
            elif i < 180:  # Bearish phase
                trend = -0.0015
                vol = 0.015
            elif i < 270:  # Sideways phase
                trend = 0.0001
                vol = 0.008
            else:  # Mixed
                trend = 0.001
                vol = 0.012
            
            change = np.random.normal(trend, vol)
            price *= (1 + change)
            prices.append(price)
        
        prices = np.array(prices)
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, days),
        }, index=dates)
        
        return df
    
    def test_full_pipeline(self):
        """Test complete pipeline: data → regime detection → segmentation → evaluator."""
        # Create synthetic data
        df = self._create_synthetic_data(days=400)
        
        # Detect regimes
        detector = RegimeDetector(method='sma_adx')
        segments = detector.classify_periods(
            df, period_days=60, min_period_days=40, embargo_days=3
        )
        
        assert len(segments) > 0, "Should create at least some segments"
        
        # Get balanced segments
        balanced = detector.get_balanced_segments(segments, segments_per_regime=2)
        
        # Split for evaluation
        splits = detector.split_segments_by_role(
            balanced,
            optimization_ratio=0.70,
            model_selection_ratio=0.0,
            holdout_ratio=0.30,
        )
        
        # Create evaluator
        config = create_test_config()
        evaluator = RegimeAwareEvaluator(config, splits)
        
        assert len(evaluator._optimization_segments) > 0 or len(evaluator._holdout_segments) > 0, \
            "Should have some segments for evaluation"
    
    def test_regime_distribution(self):
        """Test that regime detection produces diverse regimes."""
        df = self._create_synthetic_data(days=365)
        
        detector = RegimeDetector(method='sma_adx')
        regime_series = detector.detect(df)
        
        # Check we get diverse regimes
        unique_regimes = regime_series.value_counts()
        
        assert len(unique_regimes) >= 2, \
            f"Should detect at least 2 different regimes, got {len(unique_regimes)}"
    
    def test_segment_timeranges(self):
        """Test that segment timeranges are valid FreqTrade format."""
        df = self._create_synthetic_data(days=200)
        
        detector = RegimeDetector(method='returns')
        segments = detector.classify_periods(df, period_days=50, min_period_days=30)
        
        for segment in segments:
            timerange = segment.timerange
            assert '-' in timerange, f"Timerange should contain '-': {timerange}"
            
            start, end = timerange.split('-')
            assert len(start) == 8, f"Start date should be YYYYMMDD: {start}"
            assert len(end) == 8, f"End date should be YYYYMMDD: {end}"
    
    def run_all(self):
        """Run all integration tests."""
        print("\n" + "=" * 70)
        print("REGIME DETECTION → EVALUATION INTEGRATION TESTS")
        print("=" * 70)
        
        self.run_test("Full pipeline", self.test_full_pipeline)
        self.run_test("Regime distribution", self.test_regime_distribution)
        self.run_test("Segment timeranges", self.test_segment_timeranges)
        
        print("-" * 70)
        if self.failed == 0:
            print(f"RESULTS: {self.passed} passed, {self.failed} failed ✅")
        else:
            print(f"RESULTS: {self.passed} passed, {self.failed} failed ❌")
            print("\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print("=" * 70)
        
        return self.failed == 0


def run_demo():
    """Run a demonstration of the regime-aware evaluation pipeline."""
    print("\n" + "=" * 70)
    print("REGIME-AWARE EVALUATION DEMO")
    print("=" * 70)
    
    # Step 1: Create synthetic market data
    print("\n📊 Step 1: Creating synthetic market data with regime patterns...")
    
    np.random.seed(42)
    days = 365
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Create price with distinct regime phases
    prices = []
    price = 100.0
    regime_labels = []
    
    for i in range(days):
        if i < 100:  # Bullish
            trend, vol, label = 0.003, 0.01, 'BULL'
        elif i < 180:  # Bearish
            trend, vol, label = -0.002, 0.012, 'BEAR'
        elif i < 280:  # Sideways
            trend, vol, label = 0.0001, 0.006, 'SIDE'
        else:  # Bullish again
            trend, vol, label = 0.0025, 0.011, 'BULL'
        
        price *= (1 + np.random.normal(trend, vol))
        prices.append(price)
        regime_labels.append(label)
    
    prices = np.array(prices)
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.003, 0.003, days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.008, days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.008, days))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, days),
    }, index=dates)
    
    total_return = (prices[-1] / prices[0] - 1) * 100
    print(f"   Data: {days} days, total return: {total_return:.1f}%")
    print(f"   Synthetic regimes: BULL(0-100d), BEAR(100-180d), SIDE(180-280d), BULL(280-365d)")
    
    # Step 2: Detect regimes
    print("\n🔍 Step 2: Detecting market regimes using SMA/ADX method...")
    
    detector = RegimeDetector(method='sma_adx')
    regime_series = detector.detect(df)
    
    regime_counts = regime_series.value_counts()
    print(f"   Detected regime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(regime_series) * 100
        print(f"   - {regime.value:10s}: {count:3d} days ({pct:.1f}%)")
    
    # Step 3: Classify into periods
    print("\n📁 Step 3: Classifying data into regime segments...")
    
    segments = detector.classify_periods(
        df, 
        period_days=60, 
        min_period_days=40,
        embargo_days=3,
        warmup_bars=50
    )
    
    print(f"   Created {len(segments)} segments:")
    for seg in segments:
        print(f"   - {seg.segment_id}: {seg.regime.value:8s} ({seg.confidence:.1%} confidence) "
              f"[{seg.start_date.strftime('%Y-%m-%d')} to {seg.end_date.strftime('%Y-%m-%d')}]")
    
    # Step 4: Get balanced segments
    print("\n⚖️ Step 4: Selecting balanced segments (2 per regime)...")
    
    balanced = detector.get_balanced_segments(segments, segments_per_regime=2)
    
    regime_dist = {}
    for seg in balanced:
        regime_dist[seg.regime.value] = regime_dist.get(seg.regime.value, 0) + 1
    print(f"   Balanced distribution: {regime_dist}")
    
    # Step 5: Split for train/holdout
    print("\n🔀 Step 5: Splitting into optimization (70%) and holdout (30%)...")
    
    splits = detector.split_segments_by_role(
        balanced,
        optimization_ratio=0.70,
        model_selection_ratio=0.0,
        holdout_ratio=0.30,
    )
    
    print(f"   Optimization: {len(splits['optimization'])} segments")
    for seg in splits['optimization']:
        print(f"      - {seg.segment_id}: {seg.regime.value}")
    
    print(f"   Holdout: {len(splits['holdout'])} segments")
    for seg in splits['holdout']:
        print(f"      - {seg.segment_id}: {seg.regime.value}")
    
    # Step 6: Create evaluator
    print("\n🧪 Step 6: Creating RegimeAwareEvaluator...")
    
    config = create_test_config()
    evaluator = RegimeAwareEvaluator(config, splits)
    
    print(f"   Evaluator created with:")
    print(f"   - Optimization segments: {len(evaluator._optimization_segments)}")
    print(f"   - Holdout segments: {len(evaluator._holdout_segments)}")
    print(f"   - Aggregation method: {evaluator.aggregation_method}")
    print(f"   - Regime weights: {evaluator.regime_weights}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE - Pipeline ready for use!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Enable regime_aware in ga_config.yaml")
    print("2. Run GA evolution - strategies will be evaluated across regime segments")
    print("3. Final strategies will be validated on holdout segments")
    print("=" * 70)


if __name__ == '__main__':
    # Run unit tests
    unit_tests = TestRegimeAwareEvaluator()
    unit_success = unit_tests.run_all()
    
    # Run integration tests
    integration_tests = TestIntegrationWithRegimeDetector()
    integration_success = integration_tests.run_all()
    
    # Run demo
    run_demo()
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)
    total_passed = unit_tests.passed + integration_tests.passed
    total_failed = unit_tests.failed + integration_tests.failed
    
    if total_failed == 0:
        print(f"All tests passed: {total_passed} tests ✅")
    else:
        print(f"Tests: {total_passed} passed, {total_failed} failed ❌")
    print("=" * 70)
    
    sys.exit(0 if total_failed == 0 else 1)
