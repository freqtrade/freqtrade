#!/usr/bin/env python3
"""
Final validation script - Simulates a complete GA workflow to ensure
the fix works end-to-end.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.crossover import crossover
from genetic_algorithm.core.mutation import mutate
from genetic_algorithm.strategies.generator import StrategyGenerator
import yaml
import random


def simulate_ga_workflow(config, num_generations=2, population_size=4):
    """
    Simulate a simplified GA workflow: generate population, crossover, mutate.
    
    This tests that the fix works in a realistic scenario.
    """
    print("=" * 70)
    print("Simulating Genetic Algorithm Workflow")
    print("=" * 70)
    
    generator = StrategyGenerator(config)
    
    # Initialize population
    print(f"\n--- Generation 0: Initial Population ({population_size} individuals) ---")
    population = []
    for i in range(population_size):
        strategy_gene = generator.generate_random_strategy(generation=0, individual_id=i)
        individual = Individual(strategy_gene=strategy_gene)
        population.append(individual)
        
        # Validate
        missing = strategy_gene.get_missing_indicators()
        if missing:
            print(f"✗ Individual {i} has missing indicators: {missing}")
            return False
        
        print(f"Individual {i}: {len(strategy_gene.indicators)} indicators, "
              f"{len(strategy_gene.entry_conditions)} entry, "
              f"{len(strategy_gene.exit_conditions)} exit conditions")
    
    # Simulate evolution
    for gen in range(1, num_generations + 1):
        print(f"\n--- Generation {gen}: Crossover and Mutation ---")
        new_population = []
        
        # Crossover
        for i in range(0, len(population) - 1, 2):
            parent1 = population[i]
            parent2 = population[i + 1]
            
            # Perform crossover
            method = random.choice(['single_point', 'uniform', 'component'])
            child1, child2 = crossover(
                parent1, parent2,
                generation=gen,
                ind_id=len(new_population),
                method=method,
                config=config
            )
            
            # Validate children
            for child_num, child in enumerate([child1, child2]):
                missing = child.strategy_gene.get_missing_indicators()
                if missing:
                    print(f"✗ Child {len(new_population) + child_num} from {method} "
                          f"crossover has missing indicators: {missing}")
                    print(f"  Indicators: {[ind.type for ind in child.strategy_gene.indicators]}")
                    print(f"  Conditions: {[c.indicator for c in child.strategy_gene.entry_conditions + child.strategy_gene.exit_conditions]}")
                    return False
            
            new_population.extend([child1, child2])
        
        # Mutation
        for i in range(len(new_population)):
            if random.random() < 0.5:  # 50% mutation rate
                mutated = mutate(
                    new_population[i],
                    mutation_rate=0.3,
                    config=config,
                    methods=['indicators', 'conditions']
                )
                
                # Validate mutated individual
                missing = mutated.strategy_gene.get_missing_indicators()
                if missing:
                    print(f"✗ Mutated individual {i} has missing indicators: {missing}")
                    print(f"  Indicators: {[ind.type for ind in mutated.strategy_gene.indicators]}")
                    print(f"  Conditions: {[c.indicator for c in mutated.strategy_gene.entry_conditions + mutated.strategy_gene.exit_conditions]}")
                    return False
                
                new_population[i] = mutated
        
        population = new_population
        print(f"Generated {len(population)} offspring, all consistent")
        
        # Validate all can generate code without errors
        for i, individual in enumerate(population):
            try:
                code = generator.generate_strategy_code(individual.strategy_gene)
                # Check that code compiles
                compile(code, f'<strategy_{i}>', 'exec')
            except Exception as e:
                print(f"✗ Failed to generate/compile code for individual {i}: {e}")
                return False
    
    print("\n" + "=" * 70)
    print("✓ Simulated GA workflow completed successfully!")
    print(f"  - {num_generations} generations")
    print(f"  - {population_size} individuals per generation")
    print(f"  - All strategies maintained indicator-condition consistency")
    print(f"  - All generated code compiled successfully")
    print("=" * 70)
    
    return True


def test_edge_cases(config):
    """Test specific edge cases that could cause problems."""
    print("\n" + "=" * 70)
    print("Testing Edge Cases")
    print("=" * 70)
    
    generator = StrategyGenerator(config)
    
    # Edge case 1: Strategy with many indicators, few conditions
    print("\n--- Edge Case 1: Many indicators, few conditions ---")
    gene = StrategyGene(
        generation=0,
        individual_id=1,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0),
            IndicatorGene(type='MACD', parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9}, weight=0.8),
            IndicatorGene(type='BBANDS', parameters={'period': 20, 'std_dev': 2.0}, weight=0.9),
            IndicatorGene(type='EMA', parameters={'period': 20}, weight=0.7),
            IndicatorGene(type='ADX', parameters={'period': 14}, weight=0.6),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='MACD', operator='cross_below', threshold=0, logic='OR'),
        ],
    )
    
    individual = Individual(strategy_gene=gene)
    
    # Try crossover with another strategy
    gene2 = StrategyGene(
        generation=0,
        individual_id=2,
        indicators=[
            IndicatorGene(type='CCI', parameters={'period': 20}, weight=1.0),
            IndicatorGene(type='STOCH', parameters={'k_period': 14, 'd_period': 3}, weight=0.8),
        ],
        entry_conditions=[
            ConditionGene(indicator='CCI', operator='<', threshold=-100, logic='AND'),
            ConditionGene(indicator='STOCH', operator='<', threshold=20, logic='OR'),
        ],
        exit_conditions=[
            ConditionGene(indicator='STOCH', operator='>', threshold=80, logic='OR'),
        ],
    )
    
    individual2 = Individual(strategy_gene=gene2)
    
    child1, child2 = crossover(individual, individual2, generation=1, ind_id=100, config=config)
    
    missing1 = child1.strategy_gene.get_missing_indicators()
    missing2 = child2.strategy_gene.get_missing_indicators()
    
    if missing1 or missing2:
        print(f"✗ Edge case 1 failed: Child1 missing {missing1}, Child2 missing {missing2}")
        return False
    
    print("✓ Edge case 1 passed")
    
    # Edge case 2: After multiple mutations
    print("\n--- Edge Case 2: Multiple sequential mutations ---")
    test_ind = Individual(strategy_gene=gene.copy())
    
    for i in range(10):
        test_ind = mutate(test_ind, mutation_rate=1.0, config=config)
        missing = test_ind.strategy_gene.get_missing_indicators()
        if missing:
            print(f"✗ Edge case 2 failed after {i+1} mutations: missing {missing}")
            return False
    
    print("✓ Edge case 2 passed (10 sequential mutations)")
    
    # Edge case 3: Generate code and verify all indicators
    print("\n--- Edge Case 3: Verify generated code ---")
    code = generator.generate_strategy_code(test_ind.strategy_gene)
    
    # Check that all condition indicators have calculations
    for cond in test_ind.strategy_gene.entry_conditions + test_ind.strategy_gene.exit_conditions:
        ind_type = cond.indicator
        
        # Build expected patterns based on indicator type
        patterns = {
            'RSI': 'ta.RSI(',
            'MACD': 'ta.MACD(',
            'BBANDS': 'ta.BBANDS(',
            'EMA': 'ta.EMA(',
            'SMA': 'ta.SMA(',
            'STOCH': 'ta.STOCH(',
            'ATR': 'ta.ATR(',
            'ADX': 'ta.ADX(',
            'CCI': 'ta.CCI(',
        }
        
        if ind_type in patterns:
            if patterns[ind_type] not in code:
                print(f"✗ Edge case 3 failed: {ind_type} calculation not found in code")
                return False
    
    print("✓ Edge case 3 passed (all indicators calculated in code)")
    
    return True


def main():
    """Run validation tests."""
    # Load configuration
    config_path = Path(__file__).parent / 'config' / 'ga_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # Run workflow simulation
        if not simulate_ga_workflow(config, num_generations=3, population_size=6):
            return 1
        
        # Test edge cases
        if not test_edge_cases(config):
            return 1
        
        print("\n" + "=" * 70)
        print("✓✓✓ ALL VALIDATION TESTS PASSED ✓✓✓")
        print("The fix successfully prevents indicator-condition mismatches!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
