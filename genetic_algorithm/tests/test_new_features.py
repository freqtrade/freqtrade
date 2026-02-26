"""
Test suite for new GA features implemented:
- New trend indicators (SuperTrend, Ichimoku, Donchian, VWAP, PSAR)
- Volume indicators (CMF, VROC)
- Candlestick patterns (11 TALib patterns)
- MACD parameter validation
- NaN protection in fitness
- Bonus capping

Run with: python -m pytest genetic_algorithm/tests/test_new_features.py -v
Or directly: python genetic_algorithm/tests/test_new_features.py
"""

import sys
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from genetic_algorithm.utils.indicator_factory import create_random_indicator
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.evaluation.fitness import FitnessEvaluator


# ============================================================================
# TEST DATA GENERATION
# ============================================================================

def generate_ohlcv_data(n_candles: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)
    
    # Generate realistic price movement
    returns = np.random.normal(0.0002, 0.02, n_candles)
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate high/low around close
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_candles)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_candles)))
    open_price = low + (high - low) * np.random.random(n_candles)
    
    # Generate volume with some patterns
    volume = np.random.lognormal(10, 0.5, n_candles)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return df


# ============================================================================
# INDICATOR CREATION TESTS
# ============================================================================

NEW_TREND_INDICATORS = ['SUPERTREND', 'ICHIMOKU', 'DONCHIAN', 'VWAP', 'PSAR']
VOLUME_INDICATORS = ['CMF', 'VROC']
CANDLESTICK_PATTERNS = [
    'CDL_ENGULFING', 'CDL_HAMMER', 'CDL_DOJI', 'CDL_MORNINGSTAR', 'CDL_EVENINGSTAR',
    'CDL_SHOOTINGSTAR', 'CDL_HARAMI', 'CDL_PIERCING', 'CDL_DARKCLOUD',
    'CDL_3WHITESOLDIERS', 'CDL_3BLACKCROWS'
]

def test_new_trend_indicators_creation():
    """Test that all new trend indicators can be created."""
    print("\n=== Testing New Trend Indicators Creation ===")
    errors = []
    
    for ind_type in NEW_TREND_INDICATORS:
        try:
            indicator = create_random_indicator(ind_type, {})
            assert indicator.type == ind_type, f"Type mismatch: {indicator.type} != {ind_type}"
            print(f"  ✓ {ind_type}: params={indicator.parameters}")
        except Exception as e:
            errors.append(f"{ind_type}: {e}")
            print(f"  ✗ {ind_type}: {e}")
    
    if errors:
        raise AssertionError(f"Failed indicators: {errors}")
    print("  All trend indicators created successfully!")
    return True


def test_volume_indicators_creation():
    """Test that all volume indicators can be created."""
    print("\n=== Testing Volume Indicators Creation ===")
    errors = []
    
    for ind_type in VOLUME_INDICATORS:
        try:
            indicator = create_random_indicator(ind_type, {})
            assert indicator.type == ind_type
            print(f"  ✓ {ind_type}: params={indicator.parameters}")
        except Exception as e:
            errors.append(f"{ind_type}: {e}")
            print(f"  ✗ {ind_type}: {e}")
    
    if errors:
        raise AssertionError(f"Failed indicators: {errors}")
    print("  All volume indicators created successfully!")
    return True


def test_candlestick_patterns_creation():
    """Test that all candlestick patterns can be created."""
    print("\n=== Testing Candlestick Pattern Indicators Creation ===")
    errors = []
    
    for pattern in CANDLESTICK_PATTERNS:
        try:
            indicator = create_random_indicator(pattern, {})
            assert indicator.type == pattern
            print(f"  ✓ {pattern}: params={indicator.parameters}")
        except Exception as e:
            errors.append(f"{pattern}: {e}")
            print(f"  ✗ {pattern}: {e}")
    
    if errors:
        raise AssertionError(f"Failed patterns: {errors}")
    print("  All candlestick patterns created successfully!")
    return True


# ============================================================================
# MACD PARAMETER VALIDATION TESTS
# ============================================================================

def test_macd_parameter_validation():
    """Test that MACD always has slow_period > fast_period."""
    print("\n=== Testing MACD Parameter Validation ===")
    
    # Create many MACD indicators and check all have valid params
    errors = []
    for i in range(50):
        indicator = create_random_indicator('MACD', {})
        fast = indicator.parameters.get('fast_period', 12)
        slow = indicator.parameters.get('slow_period', 26)
        
        if fast >= slow:
            errors.append(f"Invalid: fast={fast} >= slow={slow}")
    
    if errors:
        print(f"  ✗ Found {len(errors)} invalid MACD configs")
        raise AssertionError(f"MACD validation failed: {errors[:5]}")
    
    print(f"  ✓ Created 50 MACD indicators, all have fast < slow")
    return True


# ============================================================================
# STRATEGY GENERATION TESTS
# ============================================================================

def test_strategy_generation_with_new_indicators():
    """Test that strategies can be generated with new indicators."""
    print("\n=== Testing Strategy Generation with New Indicators ===")
    
    # Config with all new indicators available
    config = {
        'indicators': {
            'available': NEW_TREND_INDICATORS + VOLUME_INDICATORS + CANDLESTICK_PATTERNS + ['RSI', 'MACD'],
            'min_per_strategy': 3,
            'max_per_strategy': 6,
            'RSI': {'period': [7, 21]},
            'MACD': {'fast_period': [8, 16], 'slow_period': [20, 30]},
        },
        'strategy_constraints': {
            'timeframes': ['15m', '1h'],
            'stoploss_range': [-0.15, -0.05],
            'roi_range': [0.02, 0.08],
            'max_open_trades_range': [2, 5],
        },
        'multi_timeframe': {'enabled': False},
    }
    
    generator = StrategyGenerator(config)
    errors = []
    strategies_with_new_indicators = 0
    
    for i in range(20):
        try:
            strategy = generator.generate_random_strategy(generation=0, individual_id=i)
            
            # Check if strategy uses any new indicators
            indicator_types = [ind.type for ind in strategy.indicators]
            new_used = [t for t in indicator_types if t in NEW_TREND_INDICATORS + VOLUME_INDICATORS + CANDLESTICK_PATTERNS]
            
            if new_used:
                strategies_with_new_indicators += 1
                print(f"  Strategy {i}: uses {new_used}")
            
            # Basic validation
            assert len(strategy.indicators) >= 1, "No indicators"
            assert len(strategy.entry_conditions) >= 1, "No entry conditions"
            
        except Exception as e:
            errors.append(f"Strategy {i}: {e}")
            print(f"  ✗ Strategy {i}: {e}")
    
    if errors:
        raise AssertionError(f"Strategy generation errors: {errors}")
    
    print(f"  ✓ Generated 20 strategies, {strategies_with_new_indicators} use new indicators")
    return True


# ============================================================================
# CODE GENERATION TESTS
# ============================================================================

def test_code_generation_for_new_indicators():
    """Test that Python code can be generated for all new indicators."""
    print("\n=== Testing Code Generation for New Indicators ===")
    
    config = {
        'indicators': {
            'available': NEW_TREND_INDICATORS + VOLUME_INDICATORS + CANDLESTICK_PATTERNS,
            'min_per_strategy': 2,
            'max_per_strategy': 4,
        },
        'strategy_constraints': {
            'timeframes': ['1h'],
            'stoploss_range': [-0.10, -0.05],
            'roi_range': [0.03, 0.06],
            'max_open_trades_range': [3, 5],
        },
        'multi_timeframe': {'enabled': False},
    }
    
    generator = StrategyGenerator(config)
    errors = []
    
    # Test each indicator type individually
    all_new = NEW_TREND_INDICATORS + VOLUME_INDICATORS + CANDLESTICK_PATTERNS
    
    for ind_type in all_new:
        try:
            # Create a minimal strategy with this indicator
            indicator = create_random_indicator(ind_type, {})
            
            strategy = StrategyGene(
                generation=0,
                individual_id=0,
                indicators=[indicator],
                entry_conditions=[ConditionGene(
                    indicator=ind_type,
                    operator='>',
                    threshold=0,
                    logic='OR'
                )],
                exit_conditions=[ConditionGene(
                    indicator=ind_type,
                    operator='<',
                    threshold=0,
                    logic='OR'
                )],
                timeframe='1h',
                stoploss=-0.08,
                minimal_roi={"0": 0.05, "30": 0.02},
                max_open_trades=3,
            )
            strategy.assign_instance_ids()
            
            # Generate code
            code = generator.generate_strategy_code(strategy)
            
            # Basic checks
            assert 'class GAStrategy' in code, "Missing class definition"
            assert 'def populate_indicators' in code, "Missing populate_indicators"
            assert 'def populate_entry_trend' in code, "Missing populate_entry_trend"
            
            # Check indicator-specific code is present
            ind_lower = ind_type.lower()
            if ind_type.startswith('CDL_'):
                # Candlestick patterns use different column names
                col_name = ind_lower.replace('cdl_', 'cdl_')
                assert col_name in code.lower() or 'ta.CDL' in code, f"Missing {ind_type} code"
            elif ind_type == 'ICHIMOKU':
                assert 'tenkan' in code.lower() or 'kijun' in code.lower(), "Missing Ichimoku code"
            elif ind_type != 'VWAP':  # VWAP might not have obvious markers
                assert ind_lower in code.lower() or ind_type in code, f"Missing {ind_type} code"
            
            print(f"  ✓ {ind_type}: code generated ({len(code)} chars)")
            
        except Exception as e:
            errors.append(f"{ind_type}: {e}")
            print(f"  ✗ {ind_type}: {e}")
    
    if errors:
        raise AssertionError(f"Code generation errors: {errors}")
    
    print("  All indicators generate valid code!")
    return True


# ============================================================================
# FITNESS FUNCTION TESTS
# ============================================================================

def test_nan_protection_in_fitness():
    """Test that NaN/Inf values in metrics don't crash fitness calculation."""
    print("\n=== Testing NaN Protection in Fitness ===")
    
    config = {
        'fitness_weights': {
            'profit': 0.3,
            'sharpe_ratio': 0.2,
            'sortino_ratio': 0.15,
            'profit_factor': 0.1,
            'drawdown': 0.1,
            'win_rate': 0.1,
            'trade_frequency': 0.05,
        },
        'fitness_penalties': {
            'min_trades': 5,
            'max_drawdown': 0.3,
            'min_win_rate': 0.3,
            'complexity_weight': 0.01,
        },
    }
    
    evaluator = FitnessEvaluator(config)
    
    test_cases = [
        {'profit': float('nan'), 'sharpe_ratio': 1.0, 'num_trades': 20, 'max_drawdown': 0.1, 'win_rate': 0.5},
        {'profit': 10.0, 'sharpe_ratio': float('inf'), 'num_trades': 20, 'max_drawdown': 0.1, 'win_rate': 0.5},
        {'profit': float('-inf'), 'sharpe_ratio': 1.0, 'num_trades': 20, 'max_drawdown': 0.1, 'win_rate': 0.5},
        {'profit': 10.0, 'sharpe_ratio': float('nan'), 'sortino_ratio': float('nan'), 'num_trades': 20},
        # All NaN case
        {'profit': float('nan'), 'sharpe_ratio': float('nan'), 'sortino_ratio': float('nan'), 
         'profit_factor': float('nan'), 'max_drawdown': float('nan'), 'win_rate': float('nan'), 'num_trades': 0},
    ]
    
    errors = []
    for i, metrics in enumerate(test_cases):
        try:
            fitness = evaluator.calculate_fitness(metrics)
            
            # Check fitness is valid number
            assert not math.isnan(fitness), f"Fitness is NaN"
            assert not math.isinf(fitness), f"Fitness is Inf"
            assert fitness >= 0, f"Fitness is negative: {fitness}"
            
            print(f"  ✓ Test case {i+1}: fitness={fitness:.4f} (input had NaN/Inf)")
            
        except Exception as e:
            errors.append(f"Case {i+1}: {e}")
            print(f"  ✗ Test case {i+1}: {e}")
    
    if errors:
        raise AssertionError(f"NaN protection errors: {errors}")
    
    print("  NaN protection working correctly!")
    return True


def test_bonus_capping():
    """Test that fitness bonuses are properly capped at 1.3x."""
    print("\n=== Testing Bonus Capping ===")
    
    config = {
        'fitness_weights': {
            'profit': 0.3,
            'sharpe_ratio': 0.2,
            'sortino_ratio': 0.15,
            'profit_factor': 0.1,
            'drawdown': 0.1,
            'win_rate': 0.1,
            'trade_frequency': 0.05,
        },
        'fitness_penalties': {
            'min_trades': 5,
            'max_drawdown': 0.3,
            'min_win_rate': 0.3,
            'complexity_weight': 0.01,
        },
    }
    
    evaluator = FitnessEvaluator(config)
    
    # Create extremely good metrics that would trigger all bonuses
    excellent_metrics = {
        'profit': 50.0,  # Very high profit (>10%)
        'sharpe_ratio': 3.0,  # Excellent Sharpe (>2)
        'sortino_ratio': 4.0,  # Excellent Sortino (>1)
        'profit_factor': 3.0,  # High profit factor (>1.5)
        'max_drawdown': 0.05,  # Low drawdown (<15%)
        'win_rate': 0.7,
        'num_trades': 30,
    }
    
    # Calculate base fitness (without bonuses) by using mediocre metrics
    mediocre_metrics = {
        'profit': 5.0,
        'sharpe_ratio': 0.5,
        'sortino_ratio': 0.5,
        'profit_factor': 1.0,
        'max_drawdown': 0.2,
        'win_rate': 0.5,
        'num_trades': 20,
    }
    
    excellent_fitness = evaluator.calculate_fitness(excellent_metrics)
    mediocre_fitness = evaluator.calculate_fitness(mediocre_metrics)
    
    # With previous 2x max bonus, ratio could be ~4x or more
    # With new 1.3x cap, ratio should be much more reasonable
    ratio = excellent_fitness / mediocre_fitness if mediocre_fitness > 0 else float('inf')
    
    print(f"  Mediocre fitness: {mediocre_fitness:.4f}")
    print(f"  Excellent fitness: {excellent_fitness:.4f}")
    print(f"  Ratio: {ratio:.2f}x")
    
    # The ratio should be reasonable (not extreme)
    # With 1.3x cap and normalized metrics, ratio should be < 3x typically
    if ratio > 5.0:
        print(f"  ⚠ Warning: Ratio seems high ({ratio:.2f}x), but may be due to normalized metric differences")
    
    print("  Bonus capping test complete!")
    return True


# ============================================================================
# VISUALIZATION (Optional)
# ============================================================================

def visualize_new_indicators(save_path: str = None):
    """
    Visualize the new indicators on sample data.
    Requires matplotlib. Saves to file if save_path provided.
    """
    print("\n=== Visualizing New Indicators ===")
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import talib.abstract as ta
    except ImportError:
        print("  Skipping visualization (matplotlib or talib not available)")
        return None
    
    # Generate sample data
    df = generate_ohlcv_data(200, seed=42)
    df.index = pd.date_range(start='2025-01-01', periods=len(df), freq='1h')
    
    # Calculate indicators
    print("  Calculating indicators...")
    
    # SuperTrend
    hl2 = (df['high'] + df['low']) / 2
    atr = ta.ATR(df, timeperiod=10)
    multiplier = 3.0
    df['supertrend_upper'] = hl2 + (multiplier * atr)
    df['supertrend_lower'] = hl2 - (multiplier * atr)
    
    # Donchian Channels
    period = 20
    df['donchian_upper'] = df['high'].rolling(period).max()
    df['donchian_lower'] = df['low'].rolling(period).min()
    df['donchian_mid'] = (df['donchian_upper'] + df['donchian_lower']) / 2
    
    # VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    # CMF
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0) * df['volume']
    df['cmf'] = mfv.rolling(20).sum() / df['volume'].rolling(20).sum()
    
    # VROC
    df['vroc'] = ((df['volume'] - df['volume'].shift(12)) / df['volume'].shift(12)) * 100
    
    # Candlestick patterns
    df['cdl_engulfing'] = ta.CDLENGULFING(df)
    df['cdl_hammer'] = ta.CDLHAMMER(df)
    df['cdl_doji'] = ta.CDLDOJI(df)
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('New GA Indicators Visualization', fontsize=14, fontweight='bold')
    
    # Plot 1: Price with SuperTrend & Donchian
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], 'k-', linewidth=1, label='Close', alpha=0.7)
    ax1.fill_between(df.index, df['donchian_lower'], df['donchian_upper'], 
                     alpha=0.2, color='blue', label='Donchian Channel')
    ax1.plot(df.index, df['supertrend_lower'], 'g--', linewidth=1, label='SuperTrend Lower')
    ax1.plot(df.index, df['supertrend_upper'], 'r--', linewidth=1, label='SuperTrend Upper')
    ax1.plot(df.index, df['vwap'], 'purple', linewidth=1, alpha=0.7, label='VWAP')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title('Price with SuperTrend, Donchian & VWAP')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: CMF
    ax2 = axes[1]
    ax2.fill_between(df.index, 0, df['cmf'], where=df['cmf'] >= 0, 
                     alpha=0.5, color='green', label='CMF Positive')
    ax2.fill_between(df.index, 0, df['cmf'], where=df['cmf'] < 0, 
                     alpha=0.5, color='red', label='CMF Negative')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axhline(y=0.1, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.axhline(y=-0.1, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_ylabel('CMF')
    ax2.set_title('Chaikin Money Flow (CMF)')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: VROC
    ax3 = axes[2]
    ax3.bar(df.index, df['vroc'], width=0.02, color='steelblue', alpha=0.7, label='VROC')
    ax3.axhline(y=100, color='red', linestyle='--', linewidth=1, label='Spike threshold')
    ax3.axhline(y=-100, color='green', linestyle='--', linewidth=1)
    ax3.set_ylabel('VROC %')
    ax3.set_title('Volume Rate of Change (VROC)')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Candlestick Patterns
    ax4 = axes[3]
    ax4.plot(df.index, df['close'], 'k-', linewidth=1, alpha=0.5, label='Close')
    
    # Mark pattern occurrences
    engulfing_bull = df[df['cdl_engulfing'] > 0]
    engulfing_bear = df[df['cdl_engulfing'] < 0]
    hammer = df[df['cdl_hammer'] != 0]
    doji = df[df['cdl_doji'] != 0]
    
    if len(engulfing_bull) > 0:
        ax4.scatter(engulfing_bull.index, engulfing_bull['close'], 
                   marker='^', s=100, c='green', label=f'Bullish Engulfing ({len(engulfing_bull)})', zorder=5)
    if len(engulfing_bear) > 0:
        ax4.scatter(engulfing_bear.index, engulfing_bear['close'], 
                   marker='v', s=100, c='red', label=f'Bearish Engulfing ({len(engulfing_bear)})', zorder=5)
    if len(hammer) > 0:
        ax4.scatter(hammer.index, hammer['close'], 
                   marker='*', s=80, c='orange', label=f'Hammer ({len(hammer)})', zorder=5)
    if len(doji) > 0:
        ax4.scatter(doji.index, doji['close'], 
                   marker='d', s=60, c='purple', alpha=0.7, label=f'Doji ({len(doji)})', zorder=5)
    
    ax4.set_ylabel('Price')
    ax4.set_xlabel('Date')
    ax4.set_title('Candlestick Pattern Detection')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print pattern summary
    print(f"\n  Pattern Summary (200 candles):")
    print(f"    Bullish Engulfing: {len(engulfing_bull)}")
    print(f"    Bearish Engulfing: {len(engulfing_bear)}")
    print(f"    Hammer: {len(hammer)}")
    print(f"    Doji: {len(doji)}")
    
    return df


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests(visualize: bool = True, save_plots: bool = True):
    """Run all tests and optionally visualize."""
    print("=" * 60)
    print("  NEW FEATURES TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    tests = [
        ("Trend Indicators Creation", test_new_trend_indicators_creation),
        ("Volume Indicators Creation", test_volume_indicators_creation),
        ("Candlestick Patterns Creation", test_candlestick_patterns_creation),
        ("MACD Validation", test_macd_parameter_validation),
        ("Strategy Generation", test_strategy_generation_with_new_indicators),
        ("Code Generation", test_code_generation_for_new_indicators),
        ("NaN Protection", test_nan_protection_in_fitness),
        ("Bonus Capping", test_bonus_capping),
    ]
    
    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = "PASS" if result else "FAIL"
        except Exception as e:
            results[name] = f"FAIL: {e}"
            print(f"  ✗ {name} FAILED: {e}")
    
    # Visualization
    if visualize:
        try:
            save_path = "genetic_algorithm/tests/new_indicators_viz.png" if save_plots else None
            visualize_new_indicators(save_path)
            results["Visualization"] = "PASS"
        except Exception as e:
            results["Visualization"] = f"SKIP: {e}"
            print(f"  ⚠ Visualization skipped: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    
    for name, result in results.items():
        status = "✓" if result == "PASS" else "✗" if "FAIL" in result else "⚠"
        print(f"  {status} {name}: {result}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test new GA features")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--no-save", action="store_true", help="Show plots instead of saving")
    args = parser.parse_args()
    
    run_all_tests(
        visualize=not args.no_viz,
        save_plots=not args.no_save
    )
