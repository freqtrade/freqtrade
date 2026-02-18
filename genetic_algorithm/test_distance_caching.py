#!/usr/bin/env python3
"""
Tests for distance matrix caching optimization.

Verifies that:
1. Distance matrix can be computed once and reused
2. Results are identical with and without caching
3. Functions work with backward compatibility (no distance_matrix provided)
"""

import sys
import random
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.population import (
    Population, 
    apply_fitness_sharing, 
    calculate_genetic_diversity,
    calculate_pairwise_distances
)
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


def create_test_strategy_gene(generation: int, individual_id: int, variation: int = 0) -> StrategyGene:
    """Create a test strategy gene with some variation."""
    indicators = [
        IndicatorGene(type='rsi', parameters={'period': 14 + variation}),
        IndicatorGene(type='macd', parameters={'fast': 12, 'slow': 26, 'signal': 9}),
    ]
    
    entry_conditions = [
        ConditionGene(
            indicator='rsi',
            operator='<',
            threshold=30.0 + variation,
            logic='AND'
        )
    ]
    
    exit_conditions = [
        ConditionGene(
            indicator='rsi',
            operator='>',
            threshold=70.0 - variation,
            logic='AND'
        )
    ]
    
    return StrategyGene(
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        timeframe='5m',
        stoploss=-0.05 - (variation * 0.01),
        generation=generation,
        individual_id=individual_id
    )


def test_distance_matrix_computation():
    """Test that distance matrix is computed correctly."""
    print("\n=== Test: Distance Matrix Computation ===")
    
    # Create test population
    pop = Population(size=5, generation=0)
    for i in range(5):
        gene = create_test_strategy_gene(generation=0, individual_id=i, variation=i)
        ind = Individual(strategy_gene=gene)
        ind.set_fitness(100.0 - i * 5, {'profit': 10.0})
        pop.add_individual(ind)
    
    # Compute distance matrix
    distances = calculate_pairwise_distances(list(pop.individuals))
    
    # Verify structure
    assert len(distances) == 5, f"Expected 5x5 matrix, got {len(distances)}x{len(distances[0])}"
    assert len(distances[0]) == 5, f"Expected 5x5 matrix, got {len(distances)}x{len(distances[0])}"
    
    # Verify symmetry
    for i in range(5):
        for j in range(5):
            assert distances[i][j] == distances[j][i], f"Matrix not symmetric at [{i}][{j}]"
    
    # Verify diagonal is zero
    for i in range(5):
        assert distances[i][i] == 0.0, f"Diagonal element [{i}][{i}] should be 0"
    
    # Verify distances are positive
    for i in range(5):
        for j in range(i + 1, 5):
            assert distances[i][j] >= 0.0, f"Distance [{i}][{j}] should be non-negative"
    
    print("✅ Distance matrix computed correctly")
    return distances


def test_fitness_sharing_with_cache():
    """Test that fitness sharing produces identical results with and without cache."""
    print("\n=== Test: Fitness Sharing with Cached Distances ===")
    
    # Create two identical populations
    pop1 = Population(size=5, generation=0)
    pop2 = Population(size=5, generation=0)
    
    for i in range(5):
        gene = create_test_strategy_gene(generation=0, individual_id=i, variation=i)
        ind1 = Individual(strategy_gene=gene)
        ind1.set_fitness(100.0 - i * 5, {'profit': 10.0})
        pop1.add_individual(ind1)
        
        # Create identical individual for pop2
        gene2 = create_test_strategy_gene(generation=0, individual_id=i, variation=i)
        ind2 = Individual(strategy_gene=gene2)
        ind2.set_fitness(100.0 - i * 5, {'profit': 10.0})
        pop2.add_individual(ind2)
    
    # Apply fitness sharing without cache (backward compatibility)
    apply_fitness_sharing(pop1, sigma_share=0.5)
    
    # Apply fitness sharing with cache
    distances = calculate_pairwise_distances(list(pop2.individuals))
    apply_fitness_sharing(pop2, sigma_share=0.5, distance_matrix=distances)
    
    # Compare results
    for i, (ind1, ind2) in enumerate(zip(pop1.individuals, pop2.individuals)):
        assert abs(ind1.fitness - ind2.fitness) < 1e-6, \
            f"Individual {i}: shared fitness differs (without cache: {ind1.fitness}, with cache: {ind2.fitness})"
        assert abs(ind1.raw_fitness - ind2.raw_fitness) < 1e-6, \
            f"Individual {i}: raw fitness differs"
    
    print("✅ Fitness sharing produces identical results with and without cache")


def test_genetic_diversity_with_cache():
    """Test that genetic diversity calculation produces identical results with and without cache."""
    print("\n=== Test: Genetic Diversity with Cached Distances ===")
    
    # Create two identical populations
    pop1 = Population(size=5, generation=0)
    pop2 = Population(size=5, generation=0)
    
    for i in range(5):
        gene = create_test_strategy_gene(generation=0, individual_id=i, variation=i)
        ind1 = Individual(strategy_gene=gene)
        ind1.set_fitness(100.0 - i * 5, {'profit': 10.0})
        pop1.add_individual(ind1)
        
        # Create identical individual for pop2
        gene2 = create_test_strategy_gene(generation=0, individual_id=i, variation=i)
        ind2 = Individual(strategy_gene=gene2)
        ind2.set_fitness(100.0 - i * 5, {'profit': 10.0})
        pop2.add_individual(ind2)
    
    # Calculate diversity without cache (backward compatibility)
    diversity1 = calculate_genetic_diversity(pop1)
    
    # Calculate diversity with cache
    distances = calculate_pairwise_distances(list(pop2.individuals))
    diversity2 = calculate_genetic_diversity(pop2, distance_matrix=distances)
    
    # Compare results
    assert abs(diversity1 - diversity2) < 1e-6, \
        f"Genetic diversity differs (without cache: {diversity1}, with cache: {diversity2})"
    
    print(f"✅ Genetic diversity produces identical results: {diversity1:.6f}")


def test_get_stats_with_cache():
    """Test that get_stats produces identical results with and without cache."""
    print("\n=== Test: get_stats() with Cached Distances ===")
    
    # Create two identical populations
    pop1 = Population(size=5, generation=0)
    pop2 = Population(size=5, generation=0)
    
    for i in range(5):
        gene = create_test_strategy_gene(generation=0, individual_id=i, variation=i)
        ind1 = Individual(strategy_gene=gene)
        ind1.set_fitness(100.0 - i * 5, {'profit': 10.0})
        pop1.add_individual(ind1)
        
        # Create identical individual for pop2
        gene2 = create_test_strategy_gene(generation=0, individual_id=i, variation=i)
        ind2 = Individual(strategy_gene=gene2)
        ind2.set_fitness(100.0 - i * 5, {'profit': 10.0})
        pop2.add_individual(ind2)
    
    # Get stats without cache (backward compatibility)
    stats1 = pop1.get_stats()
    
    # Get stats with cache
    distances = calculate_pairwise_distances(list(pop2.individuals))
    stats2 = pop2.get_stats(distance_matrix=distances)
    
    # Compare results
    assert abs(stats1.best_fitness - stats2.best_fitness) < 1e-6, "Best fitness differs"
    assert abs(stats1.avg_fitness - stats2.avg_fitness) < 1e-6, "Avg fitness differs"
    assert abs(stats1.genetic_diversity - stats2.genetic_diversity) < 1e-6, \
        f"Genetic diversity differs (without cache: {stats1.genetic_diversity}, with cache: {stats2.genetic_diversity})"
    
    print(f"✅ get_stats() produces identical results")
    print(f"   - Best fitness: {stats1.best_fitness:.4f}")
    print(f"   - Avg fitness: {stats1.avg_fitness:.4f}")
    print(f"   - Genetic diversity: {stats1.genetic_diversity:.6f}")


def test_combined_usage():
    """Test the combined usage pattern as used in evolution.py"""
    print("\n=== Test: Combined Usage Pattern (as in evolution.py) ===")
    
    # Create population
    pop = Population(size=5, generation=0)
    for i in range(5):
        gene = create_test_strategy_gene(generation=0, individual_id=i, variation=i)
        ind = Individual(strategy_gene=gene)
        ind.set_fitness(100.0 - i * 5, {'profit': 10.0})
        pop.add_individual(ind)
    
    # Simulate the evolution.py pattern:
    # 1. Compute distances once
    distances = calculate_pairwise_distances(list(pop.individuals))
    
    # 2. Apply fitness sharing with cached distances
    apply_fitness_sharing(pop, sigma_share=0.5, distance_matrix=distances)
    
    # 3. Get stats with cached distances
    stats = pop.get_stats(distance_matrix=distances)
    
    # Verify results are valid
    assert stats.best_fitness is not None, "Best fitness should be set"
    assert stats.genetic_diversity is not None, "Genetic diversity should be set"
    assert stats.genetic_diversity >= 0.0, "Genetic diversity should be non-negative"
    
    # Verify fitness sharing was applied
    for ind in pop.individuals:
        assert ind.fitness is not None, "Fitness should be set"
        assert ind.raw_fitness is not None, "Raw fitness should be set"
        # Shared fitness should be different from raw fitness for similar strategies
        # (but we can't guarantee this for all cases, so just check they exist)
    
    print(f"✅ Combined usage pattern works correctly")
    print(f"   - Distance matrix computed once")
    print(f"   - Reused for fitness sharing and stats calculation")
    print(f"   - Genetic diversity: {stats.genetic_diversity:.6f}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("DISTANCE CACHING TEST SUITE")
    print("="*60)
    
    try:
        test_distance_matrix_computation()
        test_fitness_sharing_with_cache()
        test_genetic_diversity_with_cache()
        test_get_stats_with_cache()
        test_combined_usage()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
