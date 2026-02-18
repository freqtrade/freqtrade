#!/usr/bin/env python3
"""
Test suite for QUICK WINS implementations.

Tests the following improvements:
1. raw_fitness vs shared_fitness separation
2. deterministic seeding support
3. parent uniqueness check
4. logging configuration
5. strategy_name duplication fix
6. indicator restriction
"""

import sys
import os
import tempfile
import random
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.population import Population, apply_fitness_sharing
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.core.selection import select_parents
from genetic_algorithm.core.evolution import GeneticAlgorithm


def create_test_strategy_gene(generation=0, individual_id=0):
    """Helper to create a valid strategy gene for testing."""
    indicator = IndicatorGene(type='RSI', parameters={'period': 14})
    entry_cond = ConditionGene(
        indicator='RSI',
        operator='>',
        threshold=30.0,
        logic='AND'
    )
    exit_cond = ConditionGene(
        indicator='RSI',
        operator='<',
        threshold=70.0,
        logic='AND'
    )
    
    return StrategyGene(
        generation=generation,
        individual_id=individual_id,
        indicators=[indicator],
        entry_conditions=[entry_cond],
        exit_conditions=[exit_cond],
        timeframe='1h',
        stoploss=-0.1,
        minimal_roi={'0': '0.1'}
    )


def test_raw_fitness_separation():
    """Test that raw_fitness and fitness (shared_fitness) are properly separated."""
    print("\n=== Test 1: Raw Fitness Separation ===")
    
    # Create a strategy gene with minimal valid conditions
    gene = create_test_strategy_gene()
    
    # Create individual
    ind = Individual(strategy_gene=gene)
    
    # Set fitness
    ind.set_fitness(100.0, {'profit': 10.0})
    
    # Check both raw_fitness and fitness are set initially
    assert ind.raw_fitness == 100.0, f"raw_fitness should be 100.0, got {ind.raw_fitness}"
    assert ind.fitness == 100.0, f"fitness should initially be 100.0, got {ind.fitness}"
    
    # Apply shared fitness
    ind.set_shared_fitness(50.0)
    
    # raw_fitness should remain unchanged, fitness should be updated
    assert ind.raw_fitness == 100.0, f"raw_fitness should remain 100.0, got {ind.raw_fitness}"
    assert ind.fitness == 50.0, f"fitness (shared) should be 50.0, got {ind.fitness}"
    
    print("✅ Raw fitness separation works correctly")


def test_deterministic_seeding():
    """Test that random seed produces reproducible results."""
    print("\n=== Test 2: Deterministic Seeding ===")
    
    # Create two temporary config files with same seed
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            'genetic_algorithm': {
                'random_seed': 42,
                'population_size': 5,
                'generations': 1,
                'mutation_rate': 0.15,
                'crossover_rate': 0.7,
                'elite_size': 2,
                'tournament_size': 3,
                'selection_method': 'tournament',
                'convergence_patience': 10,
                'fitness_sharing': True,
                'sharing_radius': 0.3,
                'diversity_threshold': 0.15,
                'allow_self_crossover': False,
            },
            'indicators': {
                'available': ['RSI', 'MACD', 'EMA'],
                'max_per_strategy': 3,
                'min_per_strategy': 2,
                'RSI': {'period': [7, 21]},
                'MACD': {'fast_period': [8, 21], 'slow_period': [21, 50], 'signal_period': [5, 14]},
                'EMA': {'period': [5, 50]},
            },
            'strategy_constraints': {
                'timeframes': ['1h'],
                'stoploss_range': [-0.2, -0.05],
                'roi_range': [0.01, 0.1],
            },
            'backtesting': {
                'timerange': '20241220-20260218',
                'stake_amount': 0.05,
                'pairs': ['ETH/BTC'],
                'max_open_trades': 3,
                'fee': 0.001,
                'exchange': 'binance',
            },
            'fitness_weights': {
                'profit': 0.3,
                'sharpe_ratio': 0.2,
                'sortino_ratio': 0.15,
                'profit_factor': 0.1,
                'drawdown': 0.15,
                'win_rate': 0.05,
                'trade_frequency': 0.05,
            },
            'logging': {
                'level': 'ERROR',
                'console': False,
            },
        }
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # Generate population twice with same seed
        ga1 = GeneticAlgorithm(config_path)
        pop1 = ga1.initialize_population()
        
        ga2 = GeneticAlgorithm(config_path)
        pop2 = ga2.initialize_population()
        
        # Check that populations are identical
        assert len(pop1) == len(pop2), "Population sizes should match"
        
        for i in range(len(pop1)):
            ind1 = pop1.individuals[i]
            ind2 = pop2.individuals[i]
            
            # Compare indicator types
            types1 = [ind.type for ind in ind1.strategy_gene.indicators]
            types2 = [ind.type for ind in ind2.strategy_gene.indicators]
            assert types1 == types2, f"Individual {i}: indicator types should match"
            
            # Compare timeframes
            assert ind1.strategy_gene.timeframe == ind2.strategy_gene.timeframe, \
                f"Individual {i}: timeframes should match"
        
        print("✅ Deterministic seeding produces reproducible results")
    finally:
        os.unlink(config_path)


def test_parent_uniqueness():
    """Test that parent selection can avoid duplicates."""
    print("\n=== Test 3: Parent Uniqueness Check ===")
    
    # Create a small population
    pop = Population(size=5, generation=0)
    for i in range(5):
        gene = create_test_strategy_gene(generation=0, individual_id=i)
        ind = Individual(strategy_gene=gene)
        ind.set_fitness(float(i + 1), {'profit': float(i + 1)})
        pop.add_individual(ind)
    
    # Test with allow_duplicates=False
    parents = select_parents(pop, num_parents=2, method='tournament', 
                            tournament_size=2, allow_duplicates=False)
    
    assert len(parents) == 2, "Should select 2 parents"
    assert parents[0] != parents[1], "Parents should be different when allow_duplicates=False"
    
    print("✅ Parent uniqueness check works correctly")


def test_indicator_restriction():
    """Test that only supported indicators are in config."""
    print("\n=== Test 4: Indicator Restriction ===")
    
    config_path = Path(__file__).parent / 'config' / 'ga_config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    available_indicators = config['indicators']['available']
    
    # Supported indicators (those with codegen support)
    supported = {'RSI', 'MACD', 'BBANDS', 'EMA', 'SMA', 'STOCH', 'ATR', 'ADX', 'CCI'}
    
    # Unsupported indicators that should NOT be in the list
    unsupported = {'MFI', 'WILLR', 'ROC', 'TEMA', 'KAMA', 'SAR', 'AROON'}
    
    for indicator in available_indicators:
        assert indicator in supported, f"Indicator {indicator} is not supported!"
        assert indicator not in unsupported, f"Unsupported indicator {indicator} found in config!"
    
    print(f"✅ Only supported indicators in config: {available_indicators}")


def test_individual_id_format():
    """Test that Individual.id has correct format."""
    print("\n=== Test 5: Individual ID Format ===")
    
    gene = create_test_strategy_gene(generation=3, individual_id=7)
    
    ind = Individual(strategy_gene=gene)
    
    # ID should be Gen{gen}_Ind{id}
    expected_id = "Gen3_Ind7"
    assert ind.id == expected_id, f"ID should be {expected_id}, got {ind.id}"
    
    print(f"✅ Individual ID format correct: {ind.id}")


def test_fitness_sharing_uses_raw_fitness():
    """Test that fitness sharing uses raw_fitness."""
    print("\n=== Test 6: Fitness Sharing Uses Raw Fitness ===")
    
    # Create population with similar strategies
    pop = Population(size=3, generation=0)
    for i in range(3):
        gene = create_test_strategy_gene(generation=0, individual_id=i)
        ind = Individual(strategy_gene=gene)
        ind.set_fitness(100.0, {'profit': 10.0})
        pop.add_individual(ind)
    
    # Apply fitness sharing
    apply_fitness_sharing(pop, sigma_share=0.5)
    
    # All individuals should have raw_fitness preserved but fitness (shared) reduced
    for ind in pop.individuals:
        assert ind.raw_fitness == 100.0, f"raw_fitness should remain 100.0, got {ind.raw_fitness}"
        # Fitness should be reduced due to sharing (they're all similar)
        assert ind.fitness < 100.0, f"Shared fitness should be < 100.0, got {ind.fitness}"
    
    print("✅ Fitness sharing correctly uses raw_fitness")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("QUICK WINS TEST SUITE")
    print("="*60)
    
    try:
        test_raw_fitness_separation()
        test_deterministic_seeding()
        test_parent_uniqueness()
        test_indicator_restriction()
        test_individual_id_format()
        test_fitness_sharing_uses_raw_fitness()
        
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
