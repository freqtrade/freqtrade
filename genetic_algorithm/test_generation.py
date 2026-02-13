#!/usr/bin/env python3
"""
Test script to verify genetic algorithm components work correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.population import Population
from genetic_algorithm.strategies.generator import StrategyGenerator
import yaml


def test_strategy_gene():
    """Test creating a StrategyGene."""
    print("\n=== Testing StrategyGene ===")
    
    indicators = [
        IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0),
        IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, weight=0.8),
    ]
    
    entry_conditions = [
        ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND'),
        ConditionGene(indicator='MACD', operator='cross_above', threshold=0, logic='AND'),
    ]
    
    exit_conditions = [
        ConditionGene(indicator='RSI', operator='>', threshold=70, logic='OR'),
    ]
    
    gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        timeframe='5m',
        stoploss=-0.10,
    )
    
    print(f"Created StrategyGene: Gen {gene.generation}, ID {gene.individual_id}")
    print(f"Indicators: {len(gene.indicators)}")
    print(f"Entry conditions: {len(gene.entry_conditions)}")
    print(f"Exit conditions: {len(gene.exit_conditions)}")
    
    return gene


def test_strategy_code_generation(config):
    """Test strategy code generation."""
    print("\n=== Testing Strategy Code Generation ===")
    
    generator = StrategyGenerator(config)
    
    # Generate a random strategy
    strategy_gene = generator.generate_random_strategy(generation=0, individual_id=42)
    
    print(f"Generated random strategy with:")
    print(f"  - {len(strategy_gene.indicators)} indicators")
    print(f"  - {len(strategy_gene.entry_conditions)} entry conditions")
    print(f"  - {len(strategy_gene.exit_conditions)} exit conditions")
    print(f"  - Timeframe: {strategy_gene.timeframe}")
    print(f"  - Stoploss: {strategy_gene.stoploss:.2%}")
    
    # Generate Python code
    code = generator.generate_strategy_code(strategy_gene)
    
    print("\nGenerated strategy code (first 500 chars):")
    print(code[:500])
    print("...")
    
    # Check that code is syntactically valid Python
    try:
        compile(code, '<string>', 'exec')
        print("✓ Generated code is syntactically valid Python")
    except SyntaxError as e:
        print(f"✗ Syntax error in generated code: {e}")
        return False
    
    return True


def test_population():
    """Test population management."""
    print("\n=== Testing Population ===")
    
    population = Population(size=5, generation=0)
    
    # Create some individuals with dummy fitness
    for i in range(5):
        indicators = [
            IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0),
        ]
        entry_conditions = [
            ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND'),
        ]
        
        gene = StrategyGene(
            generation=0,
            individual_id=i,
            indicators=indicators,
            entry_conditions=entry_conditions,
            exit_conditions=[],
        )
        
        individual = Individual(strategy_gene=gene)
        individual.fitness = i * 0.1  # Dummy fitness
        individual.evaluated = True
        
        population.add_individual(individual)
    
    print(f"Created population with {len(population)} individuals")
    
    # Get statistics
    stats = population.get_stats()
    print(f"Population stats:")
    print(f"  - Best fitness: {stats.best_fitness:.3f}")
    print(f"  - Avg fitness: {stats.avg_fitness:.3f}")
    print(f"  - Worst fitness: {stats.worst_fitness:.3f}")
    print(f"  - Diversity: {stats.diversity_score:.3f}")
    
    # Get best individuals
    best = population.get_best(2)
    print(f"Top 2 individuals have fitness: {[ind.fitness for ind in best]}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Genetic Algorithm Component Tests")
    print("=" * 60)
    
    # Load configuration
    config_path = Path(__file__).parent / 'config' / 'ga_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run tests
    try:
        test_strategy_gene()
        test_strategy_code_generation(config)
        test_population()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
