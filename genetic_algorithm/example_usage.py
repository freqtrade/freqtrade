#!/usr/bin/env python3
"""
Example script demonstrating how to use the Genetic Algorithm for strategy evolution.

This script shows a simple evolution run with dummy fitness evaluation.
For real usage, you would integrate with FreqTrade backtesting.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.evolution import GeneticAlgorithm
from genetic_algorithm.strategies.generator import StrategyGenerator


def main():
    """Run a simple GA evolution example."""
    print("=" * 70)
    print("Genetic Algorithm for FreqTrade Strategy Evolution - Example")
    print("=" * 70)
    print()
    print("This example demonstrates the GA framework with dummy fitness values.")
    print("For real use, integrate with FreqTrade backtesting in fitness evaluator.")
    print()
    
    # Initialize GA with default configuration
    config_path = "genetic_algorithm/config/ga_config.yaml"
    
    try:
        ga = GeneticAlgorithm(config_path)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        print("Please ensure you're running this from the repository root.")
        return 1
    
    print(f"Initialized GA with:")
    print(f"  - Population size: {ga.population_size}")
    print(f"  - Generations: {ga.generations}")
    print(f"  - Mutation rate: {ga.mutation_rate}")
    print(f"  - Crossover rate: {ga.crossover_rate}")
    print(f"  - Elite size: {ga.elite_size}")
    print()
    
    # For this example, we'll just create a population and show some strategies
    print("Creating initial population...")
    population = ga.initialize_population()
    
    print(f"✓ Created {len(population)} strategies")
    print()
    
    # Show a few example strategies
    print("Example strategies generated:")
    print("-" * 70)
    
    for i in range(min(3, len(population))):
        individual = population[i]
        gene = individual.strategy_gene
        
        print(f"\nStrategy {i+1} (Gen {gene.generation}, ID {gene.individual_id}):")
        print(f"  Timeframe: {gene.timeframe}")
        print(f"  Stop Loss: {gene.stoploss:.2%}")
        print(f"  Trailing Stop: {gene.trailing_stop}")
        print(f"  Indicators ({len(gene.indicators)}):")
        
        for ind in gene.indicators:
            params_str = ', '.join(f"{k}={v}" for k, v in ind.parameters.items())
            print(f"    - {ind.type}: {params_str} (weight={ind.weight:.2f})")
        
        print(f"  Entry Conditions ({len(gene.entry_conditions)}):")
        for cond in gene.entry_conditions:
            print(f"    - {cond.indicator} {cond.operator} {cond.threshold} ({cond.logic})")
        
        print(f"  Exit Conditions ({len(gene.exit_conditions)}):")
        if gene.exit_conditions:
            for cond in gene.exit_conditions:
                print(f"    - {cond.indicator} {cond.operator} {cond.threshold} ({cond.logic})")
        else:
            print(f"    - (using default ROI/stoploss)")
    
    print()
    print("-" * 70)
    
    # Generate code for one strategy
    print("\nGenerating Python code for Strategy 1...")
    strategy_code = ga.strategy_generator.generate_strategy_code(population[0].strategy_gene)
    
    # Save to file
    output_dir = Path("genetic_algorithm/examples")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "example_strategy.py"
    
    with open(output_file, 'w') as f:
        f.write(strategy_code)
    
    print(f"✓ Saved example strategy to: {output_file}")
    print()
    
    print("=" * 70)
    print("Example complete!")
    print()
    print("Next steps:")
    print("  1. Review the generated strategy in genetic_algorithm/examples/")
    print("  2. Integrate FreqTrade backtesting in evaluation/fitness.py")
    print("  3. Run full evolution with: ga.evolve()")
    print("  4. Review results and top strategies")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
