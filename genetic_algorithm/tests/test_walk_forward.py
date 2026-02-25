"""
Test Walk-Forward Optimization Implementation

This script tests the walk-forward evaluation functionality.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.evaluation.fitness import FitnessEvaluator
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.utils.timerange import create_walk_forward_windows

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_window_creation():
    """Test that walk-forward windows are created correctly."""
    print("\n" + "="*80)
    print("TEST 1: Walk-Forward Window Creation")
    print("="*80)
    
    # Test with a 90-day period
    timerange = "20230101-20230331"
    
    # Rolling windows
    print("\nRolling Windows (60 train, 15 val, 15 step):")
    windows = create_walk_forward_windows(
        timerange=timerange,
        train_days=60,
        validation_days=15,
        step_days=15,
        mode='rolling'
    )
    
    for window in windows:
        print(f"  {window}")
    
    assert len(windows) > 0, "Should create at least one window"
    assert windows[0].window_index == 0, "First window should have index 0"
    
    # Anchored windows
    print("\nAnchored Windows (30 train, 10 val, 10 step):")
    windows = create_walk_forward_windows(
        timerange=timerange,
        train_days=30,
        validation_days=10,
        step_days=10,
        mode='anchored'
    )
    
    for window in windows:
        print(f"  {window}")
    
    assert len(windows) > 0, "Should create at least one window"
    
    print("\n✅ Window creation tests passed!")


def test_fitness_evaluator_config():
    """Test that FitnessEvaluator correctly detects walk-forward config."""
    print("\n" + "="*80)
    print("TEST 2: FitnessEvaluator Walk-Forward Configuration")
    print("="*80)
    
    # Config without walk-forward
    config1 = {
        'fitness_weights': {
            'profit': 0.3,
            'sharpe_ratio': 0.2,
            'sortino_ratio': 0.1,
            'profit_factor': 0.1,
            'drawdown': 0.1,
            'win_rate': 0.1,
            'trade_frequency': 0.1
        },
        'fitness_penalties': {},
        'backtesting': {
            'timerange': '20230101-20230331',
            'pairs': ['BTC/USDT'],
        },
        'walk_forward': {
            'enabled': False
        }
    }
    
    evaluator1 = FitnessEvaluator(config1)
    assert not evaluator1.walk_forward_config.get('enabled'), "Walk-forward should be disabled"
    print("✅ Walk-forward disabled config works")
    
    # Config with walk-forward enabled
    config2 = {
        **config1,
        'walk_forward': {
            'enabled': True,
            'train_days': 60,
            'validation_days': 15,
            'step_days': 15,
            'mode': 'rolling',
            'aggregation': 'mean',
            'min_train_trades': 10
        }
    }
    
    evaluator2 = FitnessEvaluator(config2)
    assert evaluator2.walk_forward_config.get('enabled'), "Walk-forward should be enabled"
    print("✅ Walk-forward enabled config works")
    
    print("\n✅ FitnessEvaluator config tests passed!")


def test_mock_strategy_evaluation():
    """Test that we can create a mock strategy for evaluation."""
    print("\n" + "="*80)
    print("TEST 3: Mock Strategy Creation")
    print("="*80)
    
    # Create a simple strategy gene
    strategy_gene = StrategyGene(
        indicators=[
            IndicatorGene(
                type='RSI',
                parameters={'period': 14},
                weight=1.0,
                instance_id='RSI_0'
            ),
            IndicatorGene(
                type='EMA',
                parameters={'period': 20},
                weight=1.0,
                instance_id='EMA_0'
            )
        ],
        entry_conditions=[
            ConditionGene(
                indicator='RSI_0',
                operator='<',
                threshold=30.0,
                logic='AND'
            )
        ],
        exit_conditions=[
            ConditionGene(
                indicator='RSI_0',
                operator='>',
                threshold=70.0,
                logic='AND'
            )
        ],
        timeframe='5m',
        stoploss=-0.05,
        minimal_roi={0: 0.10, 60: 0.05, 120: 0.02},
        generation=0,
        individual_id=0
    )
    
    complexity = strategy_gene.calculate_complexity()
    print(f"Strategy complexity: {complexity}")
    print(f"Number of indicators: {len(strategy_gene.indicators)}")
    print(f"Number of entry conditions: {len(strategy_gene.entry_conditions)}")
    print(f"Number of exit conditions: {len(strategy_gene.exit_conditions)}")
    
    assert complexity > 0, "Complexity should be positive"
    assert len(strategy_gene.indicators) == 2, "Should have 2 indicators"
    
    print("\n✅ Mock strategy creation tests passed!")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("WALK-FORWARD OPTIMIZATION TEST SUITE")
    print("="*80)
    
    tests = [
        ("Window Creation", test_window_creation),
        ("FitnessEvaluator Config", test_fitness_evaluator_config),
        ("Mock Strategy Creation", test_mock_strategy_evaluation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name}: FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*80)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
