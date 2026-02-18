#!/usr/bin/env python3
"""
Test Instance-Based Indicator Encoding

Tests the new instance-based encoding where indicators have unique instance IDs
and conditions reference specific indicator instances.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.core.crossover import single_point_crossover, uniform_crossover
from genetic_algorithm.core.mutation import mutate_indicators, mutate_conditions


def test_indicator_instance_id_assignment():
    """Test that indicators get unique instance IDs."""
    # Create a strategy with two RSI indicators
    strategy = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 7}),
            IndicatorGene(type='RSI', parameters={'period': 21}),
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30),
        ],
    )
    
    # Assign instance IDs
    strategy.assign_instance_ids()
    
    # Check that indicators have unique instance IDs
    assert strategy.indicators[0].instance_id == 'RSI_0'
    assert strategy.indicators[1].instance_id == 'RSI_1'
    assert strategy.indicators[2].instance_id == 'MACD_0'
    
    # Check that condition references the first RSI instance
    assert strategy.entry_conditions[0].indicator == 'RSI_0'


def test_instance_id_in_serialization():
    """Test that instance IDs are preserved in to_dict/from_dict."""
    strategy = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, instance_id='MACD_0'),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI_0', operator='<', threshold=30),
        ],
    )
    
    # Serialize and deserialize
    data = strategy.to_dict()
    restored = StrategyGene.from_dict(data)
    
    # Check that instance IDs are preserved
    assert restored.indicators[0].instance_id == 'RSI_0'
    assert restored.indicators[1].instance_id == 'MACD_0'
    assert restored.entry_conditions[0].indicator == 'RSI_0'


def test_multiple_instances_same_type():
    """Test strategies with multiple instances of the same indicator type."""
    strategy = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(type='EMA', parameters={'period': 9}),
            IndicatorGene(type='EMA', parameters={'period': 21}),
            IndicatorGene(type='EMA', parameters={'period': 50}),
        ],
        entry_conditions=[
            ConditionGene(indicator='EMA', operator='cross_above', threshold=0),
        ],
    )
    
    # Assign instance IDs
    strategy.assign_instance_ids()
    
    # All EMAs should have unique instance IDs
    assert strategy.indicators[0].instance_id == 'EMA_0'
    assert strategy.indicators[1].instance_id == 'EMA_1'
    assert strategy.indicators[2].instance_id == 'EMA_2'
    
    # Condition should reference the first EMA (default behavior)
    assert strategy.entry_conditions[0].indicator == 'EMA_0'


def test_strategy_generator_assigns_instance_ids():
    """Test that StrategyGenerator assigns instance IDs automatically."""
    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'EMA'],
            'min_per_strategy': 2,
            'max_per_strategy': 3,
            'RSI': {'period': [7, 21], 'buy_threshold': [20, 40]},
            'MACD': {'fast_period': [8, 21], 'slow_period': [21, 50], 'signal_period': [5, 14]},
            'EMA': {'period': [10, 50]},
        },
        'strategy_constraints': {
            'timeframes': ['5m'],
            'stoploss_range': [-0.10, -0.05],
            'roi_range': [0.01, 0.05],
        }
    }
    
    generator = StrategyGenerator(config)
    strategy = generator.generate_random_strategy(generation=0, individual_id=0)
    
    # All indicators should have instance IDs
    for indicator in strategy.indicators:
        assert indicator.instance_id is not None
        assert indicator.instance_id.startswith(indicator.type + '_')


def test_crossover_reassigns_instance_ids():
    """Test that crossover operations reassign instance IDs to avoid conflicts."""
    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'EMA'],
            'RSI': {'period': [7, 21]},
            'MACD': {'fast_period': [8, 21], 'slow_period': [21, 50], 'signal_period': [5, 14]},
            'EMA': {'period': [10, 50]},
        }
    }
    
    # Create two parent strategies
    parent1_gene = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 7}, instance_id='RSI_0'),
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, instance_id='MACD_0'),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI_0', operator='<', threshold=30),
        ],
    )
    
    parent2_gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 21}, instance_id='RSI_0'),
            IndicatorGene(type='EMA', parameters={'period': 50}, instance_id='EMA_0'),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI_0', operator='>', threshold=70),
        ],
    )
    
    parent1 = Individual(strategy_gene=parent1_gene)
    parent2 = Individual(strategy_gene=parent2_gene)
    
    # Perform crossover
    child1, child2 = single_point_crossover(parent1, parent2, generation=1, ind_id=0, config=config)
    
    # Check that children have proper instance IDs
    for indicator in child1.strategy_gene.indicators:
        assert indicator.instance_id is not None
    
    for indicator in child2.strategy_gene.indicators:
        assert indicator.instance_id is not None
    
    # Check that conditions reference valid instance IDs
    all_instance_ids = {ind.instance_id for ind in child1.strategy_gene.indicators}
    for condition in child1.strategy_gene.entry_conditions:
        # Condition should reference either an instance_id or a type
        assert condition.indicator in all_instance_ids or any(
            ind.type == condition.indicator for ind in child1.strategy_gene.indicators
        )


def test_mutation_maintains_instance_ids():
    """Test that mutation operations maintain proper instance IDs."""
    config = {
        'indicators': {
            'available': ['RSI', 'MACD', 'EMA', 'BBANDS'],
            'min_per_strategy': 2,
            'max_per_strategy': 5,
            'RSI': {'period': [7, 21], 'buy_threshold': [20, 40], 'sell_threshold': [60, 80]},
            'MACD': {'fast_period': [8, 21], 'slow_period': [21, 50], 'signal_period': [5, 14]},
            'EMA': {'period': [10, 50]},
            'BBANDS': {'period': [15, 30], 'std_dev': [1.5, 3.0]},
        }
    }
    
    # Create a strategy
    strategy_gene = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, instance_id='MACD_0'),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI_0', operator='<', threshold=30),
        ],
    )
    
    individual = Individual(strategy_gene=strategy_gene)
    
    # Mutate indicators
    mutated = mutate_indicators(individual, mutation_rate=1.0, config=config)
    
    # All indicators should have instance IDs
    for indicator in mutated.strategy_gene.indicators:
        assert indicator.instance_id is not None
        assert indicator.instance_id.startswith(indicator.type + '_')
    
    # Conditions should reference valid indicators
    all_refs = {ind.instance_id for ind in mutated.strategy_gene.indicators}
    all_refs.update({ind.type for ind in mutated.strategy_gene.indicators})
    for condition in mutated.strategy_gene.entry_conditions:
        assert condition.indicator in all_refs


def test_get_missing_indicators_with_instance_ids():
    """Test get_missing_indicators works with instance IDs."""
    strategy = StrategyGene(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}, instance_id='RSI_0'),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI_0', operator='<', threshold=30),
            ConditionGene(indicator='MACD_0', operator='cross_above', threshold=0),  # Missing!
        ],
    )
    
    missing = strategy.get_missing_indicators()
    assert 'MACD_0' in missing
    assert 'RSI_0' not in missing


if __name__ == '__main__':
    print("Running Instance-Based Encoding Tests...\n")
    
    tests = [
        test_indicator_instance_id_assignment,
        test_instance_id_in_serialization,
        test_multiple_instances_same_type,
        test_strategy_generator_assigns_instance_ids,
        test_crossover_reassigns_instance_ids,
        test_mutation_maintains_instance_ids,
        test_get_missing_indicators_with_instance_ids,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"Running {test.__name__}...", end=" ")
            test()
            print("✓ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Tests: {passed} passed, {failed} failed out of {len(tests)} total")
    print(f"{'='*60}")
    
    sys.exit(0 if failed == 0 else 1)
