"""
Test error handling in mutation and evolution processes.

This test validates that the genetic algorithm continues running even when
individual mutations or operations fail.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.mutation import mutate, mutate_conditions, mutate_adaptive_per_gene
from genetic_algorithm.strategies.generator import StrategyGenerator
import yaml


def setup_logging():
    """Set up logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config():
    """Load configuration."""
    config_path = Path(__file__).parent / "config" / "ga_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_fitness_none_handling():
    """Test that mutation handles None fitness without crashing."""
    config = load_config()
    generator = StrategyGenerator(config)
    
    # Create an individual with None fitness (unevaluated)
    strategy_gene = generator.generate_strategy()
    individual = Individual(strategy_gene=strategy_gene, fitness=None)
    
    # This should not crash even with None fitness
    mutated = mutate_adaptive_per_gene(individual, 0.5, config)
    
    # Verify we got a result
    assert mutated is not None
    print("✓ Successfully handled None fitness in mutation")


def test_entry_conditions_preservation():
    """Test that mutation never removes all entry conditions."""
    config = load_config()
    generator = StrategyGenerator(config)
    
    # Create a strategy with exactly 1 entry condition
    strategy_gene = generator.generate_strategy()
    
    # Ensure we start with exactly 1 entry condition by removing extras
    while len(strategy_gene.entry_conditions) > 1:
        strategy_gene.entry_conditions.pop()
    
    assert len(strategy_gene.entry_conditions) == 1, "Should start with 1 entry condition"
    
    individual = Individual(strategy_gene=strategy_gene, fitness=0.5)
    
    # Try mutation multiple times - should never violate constraint
    for i in range(50):
        try:
            mutated = mutate_conditions(individual, 0.8, config)
            # Check that entry conditions are never empty
            assert len(mutated.strategy_gene.entry_conditions) >= 1, \
                f"Iteration {i}: Entry conditions became empty!"
        except ValueError as e:
            if "at least one entry condition" in str(e):
                raise AssertionError(f"Mutation violated entry condition constraint: {e}")
            raise
    
    print("✓ Successfully preserved entry conditions through 50 mutations")


def test_mutation_error_recovery():
    """Test that mutation continues even if one mutation method fails."""
    config = load_config()
    generator = StrategyGenerator(config)
    
    strategy_gene = generator.generate_strategy()
    individual = Individual(strategy_gene=strategy_gene, fitness=None)
    
    # Call mutate with high mutation rate to trigger multiple methods
    # Should not crash even if one method fails
    mutated = mutate(individual, 1.0, config)
    
    # Verify we got a result (could be original or mutated)
    assert mutated is not None
    assert mutated.strategy_gene is not None
    print("✓ Successfully recovered from mutation errors")


def test_multiple_mutations_with_none_fitness():
    """Test multiple mutation operations with None fitness."""
    config = load_config()
    generator = StrategyGenerator(config)
    
    strategy_gene = generator.generate_strategy()
    individual = Individual(strategy_gene=strategy_gene, fitness=None)
    
    # Apply multiple mutations
    for _ in range(10):
        individual = mutate(individual, 0.5, config)
        # Should not crash
        assert individual is not None
    
    print("✓ Successfully applied 10 mutations with None fitness")


def test_zero_and_negative_fitness():
    """Test that mutation handles zero and negative fitness values."""
    config = load_config()
    generator = StrategyGenerator(config)
    
    strategy_gene = generator.generate_strategy()
    
    # Test with zero fitness
    individual_zero = Individual(strategy_gene=strategy_gene.copy(), fitness=0.0)
    mutated_zero = mutate_adaptive_per_gene(individual_zero, 0.5, config)
    assert mutated_zero is not None
    print("✓ Successfully handled zero fitness")
    
    # Test with negative fitness
    individual_neg = Individual(strategy_gene=strategy_gene.copy(), fitness=-0.5)
    mutated_neg = mutate_adaptive_per_gene(individual_neg, 0.5, config)
    assert mutated_neg is not None
    print("✓ Successfully handled negative fitness")


if __name__ == "__main__":
    setup_logging()
    
    print("\n" + "="*60)
    print("Testing Error Handling in Genetic Algorithm")
    print("="*60 + "\n")
    
    test_fitness_none_handling()
    test_entry_conditions_preservation()
    test_mutation_error_recovery()
    test_multiple_mutations_with_none_fitness()
    test_zero_and_negative_fitness()
    
    print("\n" + "="*60)
    print("All error handling tests passed!")
    print("="*60)
