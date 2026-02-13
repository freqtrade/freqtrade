#!/usr/bin/env python3
"""
Quick Demo of the GA Runner

This is a minimal demonstration of run_ga.py with very small parameters
to quickly show how the GA works without waiting too long.

This demo uses:
- Population size: 5 (instead of 50)
- Generations: 2 (instead of 20)
- Top strategies: 3 (instead of 5)

Expected runtime: 2-5 minutes
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.evolution import GeneticAlgorithm
from genetic_algorithm.strategies.generator import StrategyGenerator
import yaml
import tempfile


def main():
    """Run quick demo."""
    print("\n" + "=" * 80)
    print(" " * 25 + "GA RUNNER - QUICK DEMO")
    print("=" * 80)
    print()
    print("This is a quick demonstration of the Genetic Algorithm runner.")
    print("It uses minimal parameters for a fast demo:")
    print("  - Population: 5 individuals")
    print("  - Generations: 2")
    print("  - Expected time: 2-5 minutes")
    print()
    print("=" * 80)
    print()
    
    # Load and modify config
    config_path = Path(__file__).parent / "config" / "ga_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set minimal parameters
    config['genetic_algorithm']['population_size'] = 5
    config['genetic_algorithm']['generations'] = 2
    config['genetic_algorithm']['elite_size'] = 2
    
    print("Configuration:")
    print(f"  Population Size: {config['genetic_algorithm']['population_size']}")
    print(f"  Generations: {config['genetic_algorithm']['generations']}")
    print(f"  Elite Size: {config['genetic_algorithm']['elite_size']}")
    print()
    print("=" * 80)
    print()
    print("Starting evolution...")
    print()
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_config:
        yaml.dump(config, tmp_config)
        tmp_config_path = tmp_config.name
    
    try:
        # Initialize and run GA
        ga = GeneticAlgorithm(tmp_config_path)
        top_individuals = ga.evolve()
        
        # Get top 3 strategies
        top_strategies = top_individuals[:3]
        
        # Display results
        print("\n" + "=" * 80)
        print("DEMO COMPLETE - TOP 3 STRATEGIES")
        print("=" * 80)
        print()
        
        for rank, individual in enumerate(top_strategies, 1):
            gene = individual.strategy_gene
            metrics = individual.metrics
            
            print(f"Rank {rank}: Gen{gene.generation}_Ind{gene.individual_id}")
            print(f"  Fitness: {individual.fitness:.4f}")
            print(f"  Profit: {metrics.get('profit', 0):.2f}%")
            print(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"  Trades: {metrics.get('num_trades', 0)}")
            print()
        
        print("=" * 80)
        print("Demo successful! The full GA runner (run_ga.py) works the same way")
        print("but with larger population and more generations.")
        print("=" * 80)
        print()
        
    finally:
        # Clean up temporary config
        Path(tmp_config_path).unlink()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
