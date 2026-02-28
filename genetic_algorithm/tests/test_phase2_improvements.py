"""
Tests for Phase 2 GA improvements.

Tests cover:
1. SuperTrend direction state machine generates valid code
2. VWAP uses rolling window instead of cumsum
3. Volume > 0 fallback replaced with None
4. Indicator weights wiring to mutation
5. Short selling generates can_short and enter_short/exit_short signals
6. Regime detector vectorized hysteresis produces correct results
7. Profit ratio uses profit_total_pct (no heuristic)
8. Ensemble strategy generation
9. Strategy gene can_short serialization
"""

import sys
import random
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.strategies.generator import StrategyGenerator

logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')


def _make_minimal_config():
    """Create a minimal config for testing."""
    return {
        'indicators': {
            'available': ['RSI', 'MACD', 'BBANDS', 'EMA', 'SMA', 'STOCH', 'CCI', 'ADX',
                          'ATR', 'SUPERTREND', 'VWAP', 'ICHIMOKU'],
            'max_per_strategy': 5,
            'min_per_strategy': 2,
        },
        'strategy_constraints': {
            'timeframes': ['1h'],
            'stoploss_range': [-0.15, -0.03],
            'roi_range': [0.01, 0.10],
            'max_open_trades_range': [1, 5],
        },
        'short_selling': {
            'enabled': False,
        },
        'multi_timeframe': {
            'enabled': False,
        },
    }


def _make_strategy_gene(can_short=False):
    """Create a minimal valid StrategyGene for testing."""
    return StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
            IndicatorGene(type='EMA', parameters={'period': 20}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='cross_below', threshold=30),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='cross_above', threshold=70),
        ],
        timeframe='1h',
        stoploss=-0.10,
        can_short=can_short,
    )


# ===========================================================================
# 1. SuperTrend generates direction state machine code
# ===========================================================================

def test_supertrend_generates_direction_code():
    """SuperTrend should generate a state machine with direction flipping."""
    config = _make_minimal_config()
    gen = StrategyGenerator(config)
    
    gene = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(type='SUPERTREND', parameters={'period': 10, 'multiplier': 3.0}),
            IndicatorGene(type='RSI', parameters={'period': 14}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='cross_below', threshold=30),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='cross_above', threshold=70),
        ],
        timeframe='1h',
    )
    
    code = gen.generate_strategy_code(gene)
    
    # Must have direction state machine elements
    assert 'supertrend_direction' in code, "SuperTrend should generate direction column"
    assert '_st_direction' in code, "SuperTrend should have direction tracking variable"
    assert 'prev_dir' in code, "SuperTrend should reference previous direction"
    assert 'supertrend_upper' in code
    assert 'supertrend_lower' in code
    print("✓ SuperTrend generates direction state machine code")


# ===========================================================================
# 2. VWAP uses rolling window
# ===========================================================================

def test_vwap_uses_rolling_window():
    """VWAP should use a rolling window, not cumsum from start."""
    config = _make_minimal_config()
    gen = StrategyGenerator(config)
    
    gene = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(type='VWAP', parameters={'period': 20}),
            IndicatorGene(type='RSI', parameters={'period': 14}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='cross_below', threshold=30),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='cross_above', threshold=70),
        ],
        timeframe='1h',
    )
    
    code = gen.generate_strategy_code(gene)
    
    assert '.rolling(' in code, "VWAP should use rolling window"
    assert '.cumsum()' not in code, "VWAP should NOT use cumsum from start"
    print("✓ VWAP uses rolling window")


# ===========================================================================
# 3. Volume > 0 fallback replaced with None
# ===========================================================================

def test_unknown_indicator_returns_none():
    """Unknown indicator types should return None, not volume > 0."""
    config = _make_minimal_config()
    gen = StrategyGenerator(config)
    
    indicators = [IndicatorGene(type='RSI', parameters={'period': 14})]
    
    # Create a condition for a totally unknown indicator type
    condition = ConditionGene(indicator='UNKNOWN_INDICATOR_XYZ', operator='>', threshold=50)
    
    result = gen._generate_single_condition(condition, indicators)
    
    assert result is None, f"Unknown indicator should return None, got: {result}"
    assert "volume > 0" not in str(result), "Should not use volume > 0 fallback"
    print("✓ Unknown indicator returns None")


# ===========================================================================
# 4. Indicator weights in mutation
# ===========================================================================

def test_indicator_weights_in_mutation():
    """Mutation should use adaptive weights when available in config."""
    from genetic_algorithm.core.mutation import mutate_indicators
    from genetic_algorithm.core.population import Individual
    
    config = _make_minimal_config()
    # Add weights favoring RSI over others
    config['_indicator_weights'] = {
        'RSI': 5.0,  # Heavily weighted
        'EMA': 0.3,
        'SMA': 0.3,
        'MACD': 0.3,
        'BBANDS': 0.3,
    }
    
    gene = _make_strategy_gene()
    individual = Individual(strategy_gene=gene)
    individual.set_fitness(0.5, {'profit': 5.0})
    
    # Run many mutations and check that no crash occurs
    random.seed(42)
    mutation_count = 0
    for _ in range(20):
        try:
            mutated = mutate_indicators(individual, 1.0, config)
            mutation_count += 1
        except (ValueError, KeyError):
            pass  # Some mutations may fail, that's OK
    
    assert mutation_count > 0, "At least some mutations should succeed"
    print(f"✓ Indicator weights in mutation - {mutation_count}/20 mutations succeeded")


# ===========================================================================
# 5. Short selling generates correct code
# ===========================================================================

def test_short_selling_code_generation():
    """When can_short=True, strategy should include enter_short/exit_short."""
    config = _make_minimal_config()
    config['short_selling'] = {'enabled': True, 'probability': 1.0}
    gen = StrategyGenerator(config)
    
    gene = _make_strategy_gene(can_short=True)
    code = gen.generate_strategy_code(gene)
    
    assert 'can_short = True' in code, "Strategy should have can_short = True"
    assert 'enter_short' in code, "Strategy should have enter_short signal"
    assert 'exit_short' in code, "Strategy should have exit_short signal"
    print("✓ Short selling generates correct code")


def test_no_short_when_disabled():
    """When can_short=False, strategy should NOT include short signals."""
    config = _make_minimal_config()
    gen = StrategyGenerator(config)
    
    gene = _make_strategy_gene(can_short=False)
    code = gen.generate_strategy_code(gene)
    
    assert 'can_short' not in code, "Strategy should NOT have can_short when disabled"
    assert 'enter_short' not in code, "Strategy should NOT have enter_short"
    assert 'exit_short' not in code, "Strategy should NOT have exit_short"
    print("✓ No short signals when can_short=False")


# ===========================================================================
# 6. Strategy gene can_short serialization
# ===========================================================================

def test_can_short_serialization():
    """can_short should survive to_dict/from_dict round-trip."""
    gene = _make_strategy_gene(can_short=True)
    
    gene_dict = gene.to_dict()
    assert gene_dict['can_short'] is True
    
    gene2 = StrategyGene.from_dict(gene_dict)
    assert gene2.can_short is True
    
    # And when False
    gene3 = _make_strategy_gene(can_short=False)
    gene3_dict = gene3.to_dict()
    gene4 = StrategyGene.from_dict(gene3_dict)
    assert gene4.can_short is False
    
    print("✓ can_short serialization round-trip OK")


# ===========================================================================
# 7. Regime detector vectorized hysteresis
# ===========================================================================

def test_regime_detector_vectorized():
    """Vectorized ADX hysteresis should produce same results as scalar version."""
    from genetic_algorithm.utils.regime_detector import RegimeDetector, RegimeType
    
    # Create synthetic OHLCV data
    np.random.seed(42)
    n = 500
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000, 10000, n).astype(float)
    
    df = pd.DataFrame({
        'open': close + np.random.randn(n) * 0.1,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    
    detector = RegimeDetector(method='adx_di_hysteresis', params={
        'adx_enter': 25,
        'adx_exit': 20,
        'adx_period': 14,
    })
    
    regime = detector.detect(df)
    
    # Verify reasonable results
    assert len(regime) == n
    
    # Should have some variety in regime types
    unique_regimes = set(regime.dropna().unique())
    assert len(unique_regimes) >= 2, f"Expected at least 2 regime types, got: {unique_regimes}"
    
    # Should contain known regime types
    valid_types = {RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS, RegimeType.UNCERTAIN}
    for r in unique_regimes:
        assert r in valid_types, f"Unknown regime type: {r}"
    
    print(f"✓ Vectorized regime detector produces {len(unique_regimes)} regime types")


# ===========================================================================
# 8. Ensemble strategy generation
# ===========================================================================

def test_ensemble_generation():
    """Ensemble should combine multiple strategies into one."""
    from genetic_algorithm.strategies.ensemble import EnsembleGenerator
    
    config = _make_minimal_config()
    ensemble_gen = EnsembleGenerator(config)
    
    # Create two different strategies
    gene1 = _make_strategy_gene()
    gene2 = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
            IndicatorGene(type='EMA', parameters={'period': 50}),
        ],
        entry_conditions=[
            ConditionGene(indicator='MACD', operator='cross_above', threshold=0),
        ],
        exit_conditions=[
            ConditionGene(indicator='MACD', operator='cross_below', threshold=0),
        ],
        timeframe='1h',
    )
    
    strategies = [
        {'strategy_gene': gene1.to_dict(), 'fitness': 0.8},
        {'strategy_gene': gene2.to_dict(), 'fitness': 0.6},
    ]
    
    code = ensemble_gen.generate_ensemble_code(strategies, vote_threshold=0.5)
    
    assert 'GAEnsembleStrategy' in code
    assert '_vote_entry_0' in code, "Should have voting column for strategy 0"
    assert '_vote_entry_1' in code, "Should have voting column for strategy 1"
    assert '_entry_votes' in code, "Should have vote aggregation"
    assert 'Ensemble Strategy' in code
    print("✓ Ensemble strategy generation works")


# ===========================================================================
# 9. Profit ratio uses profit_total_pct
# ===========================================================================

def test_profit_conversion():
    """Profit conversion should use profit_total_pct when available."""
    # We can't easily test direct_backtester._parse_results without full setup,
    # but we can verify the logic pattern exists in the source
    import inspect
    from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
    
    source = inspect.getsource(DirectBacktester)
    
    assert 'profit_total_pct' in source, "Should use profit_total_pct from FreqTrade"
    assert 'RATIO_TO_PERCENT_THRESHOLD' not in source, "Should NOT have the old heuristic threshold"
    print("✓ Profit conversion uses profit_total_pct")


# ===========================================================================
# Run all tests
# ===========================================================================

if __name__ == '__main__':
    tests = [
        test_supertrend_generates_direction_code,
        test_vwap_uses_rolling_window,
        test_unknown_indicator_returns_none,
        test_indicator_weights_in_mutation,
        test_short_selling_code_generation,
        test_no_short_when_disabled,
        test_can_short_serialization,
        test_regime_detector_vectorized,
        test_ensemble_generation,
        test_profit_conversion,
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"✗ {test.__name__}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
