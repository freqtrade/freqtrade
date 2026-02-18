"""
Test complexity penalty implementation.

This test verifies that:
1. StrategyGene.calculate_complexity() correctly counts indicators and conditions
2. FitnessEvaluator applies complexity penalty when configured
3. More complex strategies receive lower fitness (all else equal)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.evaluation.fitness import FitnessEvaluator


def test_calculate_complexity_simple():
    """Test complexity calculation for simple strategy."""
    print("\nTest: Calculate complexity for simple strategy")
    
    # Create a simple strategy with 2 indicators, 1 entry condition, 0 exit conditions
    strategy = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
            IndicatorGene(type='MACD', parameters={'fast': 12, 'slow': 26, 'signal': 9}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30),
        ],
        exit_conditions=[]
    )
    
    # Complexity = 2 indicators + 1 entry condition + 0 exit conditions = 3
    complexity = strategy.calculate_complexity()
    expected = 3
    assert complexity == expected, f"Expected complexity {expected}, got {complexity}"
    print(f"✓ Simple strategy complexity: {complexity}")


def test_calculate_complexity_complex():
    """Test complexity calculation for complex strategy."""
    print("\nTest: Calculate complexity for complex strategy")
    
    # Create a complex strategy with 4 indicators, 3 entry conditions, 2 exit conditions
    strategy = StrategyGene(
        generation=0,
        individual_id=2,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
            IndicatorGene(type='MACD', parameters={'fast': 12, 'slow': 26, 'signal': 9}),
            IndicatorGene(type='BBANDS', parameters={'period': 20, 'std_dev': 2.0}),
            IndicatorGene(type='EMA', parameters={'period': 50}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30),
            ConditionGene(indicator='MACD', operator='>', threshold=0),
            ConditionGene(indicator='BBANDS', operator='<', threshold=0.5),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='>', threshold=70),
            ConditionGene(indicator='EMA', operator='cross_above', threshold=0),
        ]
    )
    
    # Complexity = 4 indicators + 3 entry conditions + 2 exit conditions = 9
    complexity = strategy.calculate_complexity()
    expected = 9
    assert complexity == expected, f"Expected complexity {expected}, got {complexity}"
    print(f"✓ Complex strategy complexity: {complexity}")


def test_complexity_penalty_applied():
    """Test that complexity penalty is applied to fitness."""
    print("\nTest: Complexity penalty is applied to fitness")
    
    # Create two strategies with identical metrics but different complexity
    simple_strategy = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30),
        ],
        exit_conditions=[]
    )
    
    complex_strategy = StrategyGene(
        generation=0,
        individual_id=2,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
            IndicatorGene(type='MACD', parameters={'fast': 12, 'slow': 26, 'signal': 9}),
            IndicatorGene(type='BBANDS', parameters={'period': 20, 'std_dev': 2.0}),
            IndicatorGene(type='EMA', parameters={'period': 50}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30),
            ConditionGene(indicator='MACD', operator='>', threshold=0),
            ConditionGene(indicator='BBANDS', operator='<', threshold=0.5),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='>', threshold=70),
            ConditionGene(indicator='EMA', operator='cross_above', threshold=0),
        ]
    )
    
    # Verify complexity calculation
    simple_complexity = simple_strategy.calculate_complexity()
    complex_complexity = complex_strategy.calculate_complexity()
    print(f"  Simple strategy complexity: {simple_complexity}")
    print(f"  Complex strategy complexity: {complex_complexity}")
    assert simple_complexity == 2, f"Expected simple complexity 2, got {simple_complexity}"
    assert complex_complexity == 9, f"Expected complex complexity 9, got {complex_complexity}"
    
    # Create fitness evaluator with complexity penalty enabled
    config = {
        'fitness_weights': {
            'profit': 0.25,
            'sharpe_ratio': 0.15,
            'sortino_ratio': 0.15,
            'profit_factor': 0.10,
            'drawdown': 0.15,
            'win_rate': 0.10,
            'trade_frequency': 0.10,
        },
        'fitness_penalties': {
            'min_trades': 5,
            'max_drawdown': 0.30,
            'min_win_rate': 0.30,
            'complexity_weight': 0.01,  # Enable complexity penalty
        },
        'backtesting': {},
        'indicators': {
            'available': ['RSI', 'MACD', 'BBANDS', 'EMA'],
        }
    }
    
    evaluator = FitnessEvaluator(config)
    
    # Identical metrics for both strategies
    metrics = {
        'profit': 15.0,
        'sharpe_ratio': 1.5,
        'sortino_ratio': 1.8,
        'profit_factor': 2.0,
        'max_drawdown': 0.15,
        'win_rate': 0.55,
        'num_trades': 20,
    }
    
    # Calculate fitness for both strategies
    fitness_simple = evaluator.calculate_fitness(metrics, simple_strategy)
    fitness_complex = evaluator.calculate_fitness(metrics, complex_strategy)
    
    print(f"  Simple strategy fitness: {fitness_simple:.4f}")
    print(f"  Complex strategy fitness: {fitness_complex:.4f}")
    
    # Simple strategy should have higher fitness due to lower complexity penalty
    assert fitness_simple > fitness_complex, \
        f"Simple strategy fitness ({fitness_simple:.4f}) should be higher than complex strategy ({fitness_complex:.4f})"
    
    # Verify the penalty difference is approximately correct
    penalty_diff = fitness_simple - fitness_complex
    expected_penalty_diff = 0.01 * (complex_complexity - simple_complexity)
    print(f"  Penalty difference: {penalty_diff:.4f} (expected: {expected_penalty_diff:.4f})")
    assert abs(penalty_diff - expected_penalty_diff) < 0.001, \
        f"Penalty difference ({penalty_diff:.4f}) should be approximately {expected_penalty_diff:.4f}"
    
    print("✓ Complexity penalty correctly applied")


def test_complexity_penalty_disabled():
    """Test that complexity penalty can be disabled."""
    print("\nTest: Complexity penalty can be disabled")
    
    strategy = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
            IndicatorGene(type='MACD', parameters={'fast': 12, 'slow': 26, 'signal': 9}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30),
            ConditionGene(indicator='MACD', operator='>', threshold=0),
        ],
        exit_conditions=[]
    )
    
    # Create fitness evaluator with complexity penalty disabled
    config = {
        'fitness_weights': {
            'profit': 0.25,
            'sharpe_ratio': 0.15,
            'sortino_ratio': 0.15,
            'profit_factor': 0.10,
            'drawdown': 0.15,
            'win_rate': 0.10,
            'trade_frequency': 0.10,
        },
        'fitness_penalties': {
            'min_trades': 5,
            'max_drawdown': 0.30,
            'min_win_rate': 0.30,
            'complexity_weight': 0.0,  # Disabled
        },
        'backtesting': {},
        'indicators': {
            'available': ['RSI', 'MACD'],
        }
    }
    
    evaluator = FitnessEvaluator(config)
    
    metrics = {
        'profit': 15.0,
        'sharpe_ratio': 1.5,
        'sortino_ratio': 1.8,
        'profit_factor': 2.0,
        'max_drawdown': 0.15,
        'win_rate': 0.55,
        'num_trades': 20,
    }
    
    # Calculate fitness with and without strategy_gene
    fitness_with_gene = evaluator.calculate_fitness(metrics, strategy)
    fitness_without_gene = evaluator.calculate_fitness(metrics, None)
    
    print(f"  Fitness with gene: {fitness_with_gene:.4f}")
    print(f"  Fitness without gene: {fitness_without_gene:.4f}")
    
    # Both should be equal when complexity_weight is 0
    assert abs(fitness_with_gene - fitness_without_gene) < 0.0001, \
        "Fitness should be the same with complexity_weight=0"
    
    print("✓ Complexity penalty correctly disabled")


def test_complexity_penalty_without_gene():
    """Test that fitness calculation works without strategy_gene (backward compatibility)."""
    print("\nTest: Fitness calculation works without strategy_gene (backward compatibility)")
    
    config = {
        'fitness_weights': {
            'profit': 0.25,
            'sharpe_ratio': 0.15,
            'sortino_ratio': 0.15,
            'profit_factor': 0.10,
            'drawdown': 0.15,
            'win_rate': 0.10,
            'trade_frequency': 0.10,
        },
        'fitness_penalties': {
            'min_trades': 5,
            'max_drawdown': 0.30,
            'min_win_rate': 0.30,
            'complexity_weight': 0.01,
        },
        'backtesting': {},
        'indicators': {
            'available': ['RSI'],
        }
    }
    
    evaluator = FitnessEvaluator(config)
    
    metrics = {
        'profit': 15.0,
        'sharpe_ratio': 1.5,
        'sortino_ratio': 1.8,
        'profit_factor': 2.0,
        'max_drawdown': 0.15,
        'win_rate': 0.55,
        'num_trades': 20,
    }
    
    # Should work without raising an exception
    fitness = evaluator.calculate_fitness(metrics, None)
    print(f"  Fitness without gene: {fitness:.4f}")
    assert fitness > 0, "Fitness should be positive"
    print("✓ Backward compatibility maintained")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("TEST: Complexity Penalty Implementation")
    print("="*80)
    
    try:
        test_calculate_complexity_simple()
        test_calculate_complexity_complex()
        test_complexity_penalty_applied()
        test_complexity_penalty_disabled()
        test_complexity_penalty_without_gene()
        
        print("\n" + "="*80)
        print("All tests passed! ✓")
        print("="*80 + "\n")
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

