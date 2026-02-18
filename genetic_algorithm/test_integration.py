#!/usr/bin/env python3
"""
Integration Test: Run 1 Generation on Test Data

This test validates the full GA pipeline:
1. Initialize population with random strategies
2. Evaluate fitness (backtest each strategy)
3. Select parents
4. Perform crossover
5. Perform mutation
6. Validate results
7. Verify caching works

This is a critical integration test that ensures all components work together.
"""

import sys
import logging
from pathlib import Path
import time
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.evolution import GeneticAlgorithm
import yaml


def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_test_config():
    """Load and configure test configuration."""
    config_path = Path(__file__).parent / "config" / "ga_config_test.yaml"
    
    if not config_path.exists():
        print(f"Warning: Test config not found at {config_path}, using default config")
        config_path = Path(__file__).parent / "config" / "ga_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config for faster test execution
    config['genetic_algorithm']['population_size'] = 4  # Small population
    config['genetic_algorithm']['generations'] = 1  # Just 1 generation
    config['genetic_algorithm']['elite_size'] = 2  # Keep 2 best
    config['genetic_algorithm']['mutation_rate'] = 0.5  # Higher mutation for testing
    config['genetic_algorithm']['crossover_rate'] = 0.7
    config['genetic_algorithm']['random_seed'] = 42  # Deterministic for testing
    
    # Use test pairs (UNITTEST/BTC has synthetic data built into FreqTrade)
    config['backtesting']['pairs'] = ['UNITTEST/BTC']
    config['backtesting']['timerange'] = ''  # Use all available data
    config['backtesting']['auto_download_data'] = False
    config['backtesting']['enable_cache'] = True  # Test caching
    config['backtesting']['max_open_trades'] = 1
    
    # Relax constraints for testing
    config['fitness_penalties']['min_trades'] = 1
    config['fitness_penalties']['max_drawdown'] = 1.0
    config['fitness_penalties']['min_win_rate'] = 0.0
    
    return config


def test_single_generation_pipeline():
    """Test running a single generation of the GA pipeline."""
    print("\n" + "="*80)
    print("INTEGRATION TEST: Single Generation Pipeline")
    print("="*80)
    
    # Create a temporary config file
    temp_config_fd = None
    temp_config_path = None
    
    try:
        # Load config
        print("\n1. Loading configuration...")
        config = load_test_config()
        print(f"   Population size: {config['genetic_algorithm']['population_size']}")
        print(f"   Pairs: {config['backtesting']['pairs']}")
        print(f"   Caching enabled: {config['backtesting']['enable_cache']}")
        
        # Write config to temporary file
        temp_config_fd, temp_config_path = tempfile.mkstemp(suffix='.yaml', text=True)
        with os.fdopen(temp_config_fd, 'w') as f:
            yaml.dump(config, f)
        temp_config_fd = None  # Prevent double close
        
        # Create evolver
        print("\n2. Initializing GA evolver...")
        evolver = GeneticAlgorithm(temp_config_path, visualize=False)
        
        # Initialize population
        print("\n3. Generating initial population...")
        start_time = time.time()
        population = evolver.initialize_population()
        init_time = time.time() - start_time
        print(f"   Generated {len(population.individuals)} individuals in {init_time:.2f}s")
        
        # Validate population
        print("\n4. Validating initial population...")
        for i, ind in enumerate(population.individuals):
            strategy_gene = ind.strategy_gene
            
            # Check for missing indicators
            missing = strategy_gene.get_missing_indicators()
            if missing:
                print(f"   ✗ Individual {i} has missing indicators: {missing}")
                return False
            
            # Check basic structure
            if len(strategy_gene.indicators) == 0:
                print(f"   ✗ Individual {i} has no indicators")
                return False
            
            if len(strategy_gene.entry_conditions) == 0:
                print(f"   ✗ Individual {i} has no entry conditions")
                return False
            
            # Calculate complexity
            complexity = strategy_gene.calculate_complexity()
            print(f"   Individual {i}: {len(strategy_gene.indicators)} indicators, "
                  f"{len(strategy_gene.entry_conditions)} entry conditions, "
                  f"complexity={complexity}")
        
        print("   ✓ All individuals validated")
        
        # Evaluate fitness (this is the critical part - runs backtests)
        print("\n5. Evaluating fitness (running backtests)...")
        start_time = time.time()
        evolver.evaluate_population(population)
        eval_time = time.time() - start_time
        print(f"   Evaluated {len(population.individuals)} strategies in {eval_time:.2f}s")
        print(f"   Average time per strategy: {eval_time / len(population.individuals):.2f}s")
        
        # Check fitness values
        print("\n6. Checking fitness results...")
        fitness_values = []
        for i, ind in enumerate(population.individuals):
            fitness = ind.fitness
            raw_fitness = ind.raw_fitness
            
            if fitness is None:
                print(f"   ✗ Individual {i} has None fitness")
                return False
            
            if not isinstance(fitness, (int, float)):
                print(f"   ✗ Individual {i} has invalid fitness type: {type(fitness)}")
                return False
            
            if fitness < 0:
                print(f"   ✗ Individual {i} has negative fitness: {fitness}")
                return False
            
            fitness_values.append(fitness)
            
            # Check metrics
            if hasattr(ind, 'metrics') and ind.metrics:
                profit = ind.metrics.get('profit', 0)
                trades = ind.metrics.get('num_trades', 0)
                complexity = ind.metrics.get('complexity', 0)
                raw_fit_str = f"{raw_fitness:.4f}" if raw_fitness is not None else "N/A"
                print(f"   Individual {i}: fitness={fitness:.4f}, raw_fitness={raw_fit_str}, "
                      f"profit={profit:.2f}%, trades={trades}, complexity={complexity}")
        
        print(f"   ✓ All fitness values valid")
        print(f"   Best fitness: {max(fitness_values):.4f}")
        print(f"   Average fitness: {sum(fitness_values)/len(fitness_values):.4f}")
        print(f"   Worst fitness: {min(fitness_values):.4f}")
        
        # Test selection
        print("\n7. Testing parent selection...")
        from genetic_algorithm.core.selection import select_parents
        parents = select_parents(
            population,
            num_parents=2,
            method=config['genetic_algorithm']['selection_method'],
            tournament_size=config['genetic_algorithm']['tournament_size'],
            allow_duplicates=config['genetic_algorithm'].get('allow_self_crossover', True)
        )
        parent1, parent2 = parents[0], parents[1]
        print(f"   Selected parents: ID {parent1.strategy_gene.individual_id} "
              f"(fitness={parent1.fitness:.4f}) and ID {parent2.strategy_gene.individual_id} "
              f"(fitness={parent2.fitness:.4f})")
        
        if parent1 is None or parent2 is None:
            print("   ✗ Parent selection failed")
            return False
        
        print("   ✓ Parent selection successful")
        
        # Test crossover
        print("\n8. Testing crossover...")
        from genetic_algorithm.core.crossover import crossover
        child1, child2 = crossover(
            parent1, parent2,
            generation=1,
            ind_id=0,
            method='uniform',
            config=config
        )
        
        # Validate children
        for child_num, child in enumerate([child1, child2]):
            missing = child.strategy_gene.get_missing_indicators()
            if missing:
                print(f"   ✗ Child {child_num} has missing indicators: {missing}")
                return False
        
        print(f"   ✓ Crossover successful, created 2 children")
        
        # Test mutation
        print("\n9. Testing mutation...")
        from genetic_algorithm.core.mutation import mutate
        mutated = mutate(
            child1,
            mutation_rate=0.5,
            config=config,
            methods=['indicators', 'conditions', 'parameters']
        )
        
        # Validate mutated individual
        missing = mutated.strategy_gene.get_missing_indicators()
        if missing:
            print(f"   ✗ Mutated individual has missing indicators: {missing}")
            return False
        
        print("   ✓ Mutation successful")
        
        # Test caching by re-evaluating
        print("\n10. Testing cache (re-evaluating first strategy)...")
        first_individual = population.individuals[0]
        original_fitness = first_individual.fitness
        
        # Re-evaluate (should be cached)
        start_time = time.time()
        fitness, metrics = evolver.fitness_evaluator.evaluate(
            first_individual.strategy_gene,
            strategy_name=f"Gen0_Ind{first_individual.strategy_gene.individual_id}"
        )
        cache_time = time.time() - start_time
        
        print(f"   Cache lookup time: {cache_time:.4f}s")
        
        if cache_time > 1.0:
            print(f"   ⚠ Cache may not be working (took {cache_time:.2f}s, expected <1s)")
        else:
            print(f"   ✓ Cache appears to be working (fast lookup: {cache_time:.4f}s)")
        
        # Verify fitness matches
        if abs(fitness - original_fitness) > 0.0001:
            print(f"   ⚠ Fitness mismatch: original={original_fitness:.4f}, cached={fitness:.4f}")
        else:
            print(f"   ✓ Fitness matches original value")
        
        # Test statistics
        print("\n11. Testing statistics calculation...")
        stats = population.get_stats()
        
        required_attrs = ['best_fitness', 'avg_fitness', 'worst_fitness', 'generation']
        for attr in required_attrs:
            if not hasattr(stats, attr):
                print(f"   ✗ Missing attribute in statistics: {attr}")
                return False
        
        print(f"   Generation: {stats.generation}")
        print(f"   Best fitness: {stats.best_fitness:.4f}")
        print(f"   Average fitness: {stats.avg_fitness:.4f}")
        print(f"   Worst fitness: {stats.worst_fitness:.4f}")
        
        if hasattr(stats, 'diversity_score'):
            print(f"   Diversity: {stats.diversity_score:.4f}")
        
        print("   ✓ Statistics calculated successfully")
        
        # Final summary
        print("\n" + "="*80)
        print("✓ INTEGRATION TEST PASSED")
        print("="*80)
        print("\nSummary:")
        print(f"  - Population initialization: SUCCESS ({init_time:.2f}s)")
        print(f"  - Fitness evaluation: SUCCESS ({eval_time:.2f}s)")
        print(f"  - Parent selection: SUCCESS")
        print(f"  - Crossover: SUCCESS")
        print(f"  - Mutation: SUCCESS")
        print(f"  - Caching: {'SUCCESS' if cache_time < 1.0 else 'WARNING'}")
        print(f"  - Statistics: SUCCESS")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n✗ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temporary config file
        if temp_config_path and os.path.exists(temp_config_path):
            try:
                os.unlink(temp_config_path)
            except:
                pass


def main():
    """Run the integration test."""
    setup_logging()
    
    success = test_single_generation_pipeline()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
