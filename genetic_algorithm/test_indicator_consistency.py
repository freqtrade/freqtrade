#!/usr/bin/env python3
"""
Test script to verify that all indicators referenced in conditions
are calculated in populate_indicators().

This tests the fix for the critical bug where crossover/mutation
could create strategies referencing indicators that aren't calculated.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.crossover import crossover
from genetic_algorithm.core.mutation import mutate_indicators, mutate_conditions
from genetic_algorithm.strategies.generator import StrategyGenerator
import yaml


def test_strategy_consistency(strategy_gene):
    """
    Verify that all indicators referenced in conditions are in the indicators list.
    
    Returns:
        True if consistent, False otherwise
    """
    # Get all indicator types present
    present_types = {ind.type for ind in strategy_gene.indicators}
    
    # Get all indicator types referenced in conditions
    referenced_types = set()
    for cond in strategy_gene.entry_conditions + strategy_gene.exit_conditions:
        referenced_types.add(cond.indicator)
    
    # Check for missing indicators
    missing_types = referenced_types - present_types
    
    if missing_types:
        print(f"  ✗ Missing indicators: {missing_types}")
        print(f"    Present: {present_types}")
        print(f"    Referenced: {referenced_types}")
        return False
    
    print(f"  ✓ All referenced indicators are present")
    return True


def test_generated_code_has_all_indicators(strategy_gene, generator):
    """
    Verify that generated code calculates all indicators used in conditions.
    
    Returns:
        True if all indicators are calculated, False otherwise
    """
    code = generator.generate_strategy_code(strategy_gene)
    
    # Extract indicators from entry/exit conditions
    referenced_indicators = set()
    for cond in strategy_gene.entry_conditions + strategy_gene.exit_conditions:
        referenced_indicators.add(cond.indicator)
    
    # Check that each indicator is calculated in the code
    missing_in_code = []
    for ind_type in referenced_indicators:
        # Check for indicator calculation patterns
        found = False
        
        if ind_type == 'RSI':
            found = 'ta.RSI(dataframe' in code
        elif ind_type == 'MACD':
            found = 'ta.MACD(dataframe' in code and "dataframe['macd']" in code
        elif ind_type == 'BBANDS':
            found = 'ta.BBANDS(dataframe' in code and "dataframe['bb_" in code
        elif ind_type == 'EMA':
            found = 'ta.EMA(dataframe' in code
        elif ind_type == 'SMA':
            found = 'ta.SMA(dataframe' in code
        elif ind_type == 'STOCH':
            found = 'ta.STOCH(dataframe' in code and "dataframe['slowk']" in code
        elif ind_type == 'ATR':
            found = 'ta.ATR(dataframe' in code
        elif ind_type == 'ADX':
            found = 'ta.ADX(dataframe' in code
        elif ind_type == 'CCI':
            found = 'ta.CCI(dataframe' in code
        
        if not found:
            missing_in_code.append(ind_type)
    
    if missing_in_code:
        print(f"  ✗ Indicators missing in generated code: {missing_in_code}")
        print("\n--- Generated Code ---")
        print(code)
        return False
    
    print(f"  ✓ All indicators are calculated in generated code")
    return True


def test_crossover_consistency(config):
    """Test that crossover maintains indicator-condition consistency."""
    print("\n=== Testing Crossover Consistency ===")
    
    # Create parent 1: RSI + MACD
    parent1_gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0),
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, weight=0.8),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND'),
            ConditionGene(indicator='MACD', operator='cross_above', threshold=0, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='>', threshold=70, logic='OR'),
        ],
    )
    
    # Create parent 2: BBANDS + ADX
    parent2_gene = StrategyGene(
        generation=0,
        individual_id=2,
        indicators=[
            IndicatorGene(type='BBANDS', parameters={'period': 20, 'std_dev': 2.0}, weight=0.9),
            IndicatorGene(type='ADX', parameters={'period': 14}, weight=0.7),
        ],
        entry_conditions=[
            ConditionGene(indicator='BBANDS', operator='cross_below', threshold=0, logic='AND'),
            ConditionGene(indicator='ADX', operator='>', threshold=25, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='BBANDS', operator='cross_above', threshold=0, logic='OR'),
        ],
    )
    
    parent1 = Individual(strategy_gene=parent1_gene)
    parent2 = Individual(strategy_gene=parent2_gene)
    
    print("Parent 1 indicators:", [ind.type for ind in parent1_gene.indicators])
    print("Parent 1 conditions:", [cond.indicator for cond in parent1_gene.entry_conditions + parent1_gene.exit_conditions])
    print("Parent 2 indicators:", [ind.type for ind in parent2_gene.indicators])
    print("Parent 2 conditions:", [cond.indicator for cond in parent2_gene.entry_conditions + parent2_gene.exit_conditions])
    
    # Test different crossover methods
    for method in ['single_point', 'uniform', 'component']:
        print(f"\n--- Testing {method} crossover ---")
        child1, child2 = crossover(
            parent1, parent2,
            generation=1,
            ind_id=100,
            method=method,
            config=config
        )
        
        print(f"Child 1 indicators: {[ind.type for ind in child1.strategy_gene.indicators]}")
        print(f"Child 1 conditions: {[cond.indicator for cond in child1.strategy_gene.entry_conditions + child1.strategy_gene.exit_conditions]}")
        
        if not test_strategy_consistency(child1.strategy_gene):
            return False
        
        print(f"Child 2 indicators: {[ind.type for ind in child2.strategy_gene.indicators]}")
        print(f"Child 2 conditions: {[cond.indicator for cond in child2.strategy_gene.entry_conditions + child2.strategy_gene.exit_conditions]}")
        
        if not test_strategy_consistency(child2.strategy_gene):
            return False
    
    return True


def test_mutation_consistency(config):
    """Test that mutation maintains indicator-condition consistency."""
    print("\n=== Testing Mutation Consistency ===")
    
    # Create initial strategy
    gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0),
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, weight=0.8),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='MACD', operator='cross_below', threshold=0, logic='OR'),
        ],
    )
    
    individual = Individual(strategy_gene=gene)
    
    print("Original indicators:", [ind.type for ind in gene.indicators])
    print("Original conditions:", [cond.indicator for cond in gene.entry_conditions + gene.exit_conditions])
    
    # Test indicator mutation
    print("\n--- Testing indicator mutation ---")
    mutated = mutate_indicators(individual, mutation_rate=1.0, config=config)
    
    print(f"Mutated indicators: {[ind.type for ind in mutated.strategy_gene.indicators]}")
    print(f"Mutated conditions: {[cond.indicator for cond in mutated.strategy_gene.entry_conditions + mutated.strategy_gene.exit_conditions]}")
    
    if not test_strategy_consistency(mutated.strategy_gene):
        return False
    
    # Test condition mutation
    print("\n--- Testing condition mutation ---")
    mutated2 = mutate_conditions(individual, mutation_rate=1.0, config=config)
    
    print(f"Mutated indicators: {[ind.type for ind in mutated2.strategy_gene.indicators]}")
    print(f"Mutated conditions: {[cond.indicator for cond in mutated2.strategy_gene.entry_conditions + mutated2.strategy_gene.exit_conditions]}")
    
    if not test_strategy_consistency(mutated2.strategy_gene):
        return False
    
    return True


def test_random_generation_consistency(config):
    """Test that randomly generated strategies are consistent."""
    print("\n=== Testing Random Strategy Generation ===")
    
    generator = StrategyGenerator(config)
    
    # Generate multiple random strategies
    for i in range(10):
        strategy_gene = generator.generate_random_strategy(generation=0, individual_id=i)
        
        print(f"\nStrategy {i}:")
        print(f"  Indicators: {[ind.type for ind in strategy_gene.indicators]}")
        print(f"  Entry conditions: {[cond.indicator for cond in strategy_gene.entry_conditions]}")
        print(f"  Exit conditions: {[cond.indicator for cond in strategy_gene.exit_conditions]}")
        
        if not test_strategy_consistency(strategy_gene):
            return False
        
        if not test_generated_code_has_all_indicators(strategy_gene, generator):
            return False
    
    return True


def main():
    """Run all consistency tests."""
    print("=" * 70)
    print("Indicator-Condition Consistency Tests")
    print("=" * 70)
    
    # Load configuration
    config_path = Path(__file__).parent / 'config' / 'ga_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run tests
    try:
        if not test_random_generation_consistency(config):
            raise AssertionError("Random generation consistency test failed")
        
        if not test_crossover_consistency(config):
            raise AssertionError("Crossover consistency test failed")
        
        if not test_mutation_consistency(config):
            raise AssertionError("Mutation consistency test failed")
        
        print("\n" + "=" * 70)
        print("✓ All indicator-condition consistency tests passed!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
