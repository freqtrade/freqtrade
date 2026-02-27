"""
Quick integration test - runs a mini GA cycle with new indicators.
Tests: strategy generation -> code generation -> backtest simulation -> fitness calculation

Run: python genetic_algorithm/tests/test_integration_quick.py
"""

import sys
import random
import copy
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.evaluation.fitness import FitnessEvaluator
from genetic_algorithm.utils.indicator_factory import create_random_indicator


def test_full_ga_cycle():
    """Run a complete GA cycle with all new features."""
    print("=" * 60)
    print("  INTEGRATION TEST: GA CYCLE WITH NEW FEATURES")
    print("=" * 60)
    
    # Config with all indicators
    config = {
        'indicators': {
            'available': [
                'RSI', 'MACD', 'BBANDS', 'EMA', 'SMA',
                'SUPERTREND', 'ICHIMOKU', 'DONCHIAN', 'VWAP', 'PSAR',
                'CMF', 'VROC',
                'CDL_ENGULFING', 'CDL_HAMMER', 'CDL_DOJI', 'CDL_MORNINGSTAR',
                'CDL_EVENINGSTAR', 'CDL_SHOOTINGSTAR', 'CDL_HARAMI',
                'CDL_PIERCING', 'CDL_DARKCLOUD', 'CDL_3WHITESOLDIERS', 'CDL_3BLACKCROWS'
            ],
            'min_per_strategy': 3,
            'max_per_strategy': 5,
        },
        'strategy_constraints': {
            'timeframes': ['15m', '1h'],
            'stoploss_range': [-0.15, -0.05],
            'roi_range': [0.02, 0.08],
            'max_open_trades_range': [2, 5],
        },
        'multi_timeframe': {'enabled': False},
        'mutation': {
            'rate': 0.3,
            'add_indicator_prob': 0.2,
            'remove_indicator_prob': 0.15,
            'modify_param_prob': 0.4,
            'modify_condition_prob': 0.3,
        },
        'crossover': {
            'rate': 0.8,
            'indicator_mix_prob': 0.5,
        },
        'fitness_weights': {
            'profit': 0.3,
            'sharpe_ratio': 0.2,
            'sortino_ratio': 0.15,
            'profit_factor': 0.1,
            'drawdown': 0.1,
            'win_rate': 0.1,
            'trade_frequency': 0.05,
        },
        'fitness_penalties': {
            'min_trades': 5,
            'max_drawdown': 0.3,
            'min_win_rate': 0.3,
            'complexity_weight': 0.01,
        },
    }
    
    # Initialize components
    generator = StrategyGenerator(config)
    fitness_eval = FitnessEvaluator(config)
    
    errors = []
    
    # Step 1: Generate initial population
    print("\n[1/5] Generating initial population (10 strategies)...")
    population = []
    for i in range(10):
        try:
            strategy = generator.generate_random_strategy(generation=0, individual_id=i)
            population.append(strategy)
            
            # Show indicator types
            ind_types = [ind.type for ind in strategy.indicators]
            new_inds = [t for t in ind_types if t not in ['RSI', 'MACD', 'BBANDS', 'EMA', 'SMA']]
            if new_inds:
                print(f"  Strategy {i}: {ind_types} (new: {new_inds})")
        except Exception as e:
            errors.append(f"Generation {i}: {e}")
            print(f"  ✗ Strategy {i}: {e}")
    
    if errors:
        print(f"  FAILED: {len(errors)} generation errors")
        return False
    print(f"  ✓ Generated {len(population)} strategies")
    
    # Step 2: Generate code for all
    print("\n[2/5] Generating Python code for all strategies...")
    codes = []
    for i, strategy in enumerate(population):
        try:
            code = generator.generate_strategy_code(strategy)
            codes.append(code)
            
            # Validate code compiles
            compile(code, f'strategy_{i}.py', 'exec')
        except Exception as e:
            errors.append(f"Code gen {i}: {e}")
            print(f"  ✗ Strategy {i} code: {e}")
    
    if errors:
        print(f"  FAILED: {len(errors)} code generation errors")
        return False
    print(f"  ✓ Generated and validated {len(codes)} code files")
    
    # Step 3: Simulate backtest results and calculate fitness
    print("\n[3/5] Calculating fitness scores...")
    fitness_scores = []
    for i, strategy in enumerate(population):
        try:
            # Simulate realistic backtest metrics
            fake_metrics = {
                'profit': random.uniform(-10, 30),
                'sharpe_ratio': random.uniform(-0.5, 2.5),
                'sortino_ratio': random.uniform(-1, 3),
                'profit_factor': random.uniform(0.5, 2),
                'max_drawdown': random.uniform(0.05, 0.25),
                'win_rate': random.uniform(0.35, 0.65),
                'num_trades': random.randint(10, 100),
            }
            
            # Test with some edge cases
            if i == 5:  # Test NaN
                fake_metrics['sharpe_ratio'] = float('nan')
            if i == 8:  # Test negative infinity
                fake_metrics['profit'] = float('-inf')
            
            fitness = fitness_eval.calculate_fitness(fake_metrics)
            fitness_scores.append(fitness)
            
            # Validate fitness
            import math
            assert not math.isnan(fitness), "Fitness is NaN"
            assert not math.isinf(fitness), "Fitness is Inf"
            assert fitness >= 0, f"Negative fitness: {fitness}"
            
        except Exception as e:
            errors.append(f"Fitness {i}: {e}")
            print(f"  ✗ Strategy {i} fitness: {e}")
    
    if errors:
        print(f"  FAILED: {len(errors)} fitness errors")
        return False
    
    avg_fitness = sum(fitness_scores) / len(fitness_scores)
    max_fitness = max(fitness_scores)
    print(f"  ✓ Calculated {len(fitness_scores)} fitness scores")
    print(f"    Avg: {avg_fitness:.4f}, Max: {max_fitness:.4f}")
    
    # Step 4: Simple mutation simulation (modify parameters)
    print("\n[4/5] Testing parameter mutation on strategies...")
    mutated = []
    for i, strategy in enumerate(population):
        try:
            # Create a copy and modify some parameters
            mutated_strategy = copy.deepcopy(strategy)
            
            # Mutate a random indicator's parameters
            if mutated_strategy.indicators:
                ind = random.choice(mutated_strategy.indicators)
                if 'period' in ind.parameters:
                    ind.parameters['period'] = random.randint(5, 30)
                elif 'multiplier' in ind.parameters:
                    ind.parameters['multiplier'] = random.uniform(1.5, 4.0)
            
            # Mutate stoploss
            mutated_strategy.stoploss = random.uniform(-0.15, -0.05)
            
            mutated.append(mutated_strategy)
            
            # Validate mutated strategy code compiles
            code = generator.generate_strategy_code(mutated_strategy)
            compile(code, f'mutated_{i}.py', 'exec')
            
        except Exception as e:
            errors.append(f"Mutation {i}: {e}")
            print(f"  ✗ Mutation {i}: {e}")
    
    if errors:
        print(f"  FAILED: {len(errors)} mutation errors")
        return False
    print(f"  ✓ Mutated {len(mutated)} strategies")
    
    # Step 5: Simple crossover simulation (swap indicators)
    print("\n[5/5] Testing crossover operations...")
    offspring = []
    for i in range(5):
        try:
            p1, p2 = random.sample(population, 2)
            
            # Create children by swapping indicators
            child1 = copy.deepcopy(p1)
            child2 = copy.deepcopy(p2)
            
            # Swap some indicators between parents
            if p1.indicators and p2.indicators:
                swap_idx = random.randint(0, min(len(p1.indicators), len(p2.indicators)) - 1)
                # Swap one indicator
                child1.indicators = list(child1.indicators)
                child2.indicators = list(child2.indicators)
                if swap_idx < len(child1.indicators) and swap_idx < len(child2.indicators):
                    child1.indicators[swap_idx], child2.indicators[swap_idx] = \
                        copy.deepcopy(child2.indicators[swap_idx]), copy.deepcopy(child1.indicators[swap_idx])
            
            offspring.extend([child1, child2])
            
            # Validate offspring code compiles
            for child in [child1, child2]:
                code = generator.generate_strategy_code(child)
                compile(code, f'child_{i}.py', 'exec')
            
        except Exception as e:
            errors.append(f"Crossover {i}: {e}")
            print(f"  ✗ Crossover {i}: {e}")
    
    if errors:
        print(f"  FAILED: {len(errors)} crossover errors")
        return False
    print(f"  ✓ Created {len(offspring)} offspring")
    
    # Summary
    print("\n" + "=" * 60)
    print("  INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"  ✓ Generated: {len(population)} strategies")
    print(f"  ✓ Code gen:  {len(codes)} validated")
    print(f"  ✓ Fitness:   {len(fitness_scores)} calculated (avg={avg_fitness:.4f})")
    print(f"  ✓ Mutations: {len(mutated)} performed")
    print(f"  ✓ Crossover: {len(offspring)} offspring")
    print("=" * 60)
    print("  ALL INTEGRATION TESTS PASSED!")
    print("=" * 60)
    
    return True


def test_strategy_code_sample():
    """Generate and print a sample strategy with new indicators."""
    print("\n" + "=" * 60)
    print("  SAMPLE STRATEGY CODE")
    print("=" * 60)
    
    config = {
        'indicators': {
            'available': ['SUPERTREND', 'ICHIMOKU', 'CMF', 'CDL_ENGULFING', 'CDL_HAMMER'],
            'min_per_strategy': 3,
            'max_per_strategy': 4,
        },
        'strategy_constraints': {
            'timeframes': ['1h'],
            'stoploss_range': [-0.08, -0.05],
            'roi_range': [0.03, 0.06],
            'max_open_trades_range': [3, 4],
        },
        'multi_timeframe': {'enabled': False},
    }
    
    generator = StrategyGenerator(config)
    random.seed(42)  # Reproducible
    
    strategy = generator.generate_random_strategy(generation=0, individual_id=0)
    code = generator.generate_strategy_code(strategy)
    
    print(f"\nIndicators: {[ind.type for ind in strategy.indicators]}")
    print(f"Entry conditions: {len(strategy.entry_conditions)}")
    print(f"Exit conditions: {len(strategy.exit_conditions)}")
    print(f"\n--- Generated Code ({len(code)} chars) ---\n")
    print(code)
    print("\n--- End Code ---")
    
    return code


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Show sample strategy code")
    args = parser.parse_args()
    
    success = test_full_ga_cycle()
    
    if args.sample:
        test_strategy_code_sample()
    
    sys.exit(0 if success else 1)
