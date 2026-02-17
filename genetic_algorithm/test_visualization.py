#!/usr/bin/env python3
"""
Test script for visualization functionality.

Tests the GAVisualizer with mock data without running a full GA evolution.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.visualization import GAVisualizer
from genetic_algorithm.core.population import Population, PopulationStats
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
import time


def create_mock_individual(gen: int, ind_id: int, fitness: float, 
                          profit: float, sharpe: float, win_rate: float, 
                          drawdown: float) -> Individual:
    """Create a mock individual with specific metrics."""
    indicators = [
        IndicatorGene(type='RSI', parameters={'period': 14}, weight=1.0),
    ]
    
    entry_conditions = [
        ConditionGene(indicator='RSI', operator='<', threshold=30, logic='AND'),
    ]
    
    gene = StrategyGene(
        generation=gen,
        individual_id=ind_id,
        indicators=indicators,
        entry_conditions=entry_conditions,
        exit_conditions=[],
        timeframe='5m',
        stoploss=-0.10,
    )
    
    individual = Individual(strategy_gene=gene)
    individual.set_fitness(fitness, {
        'profit': profit,
        'sharpe_ratio': sharpe,
        'max_drawdown': drawdown,
        'win_rate': win_rate,
        'num_trades': 25,
        'profit_factor': 1.5
    })
    
    return individual


def test_visualization():
    """Test the visualization with mock data."""
    print("\n" + "=" * 80)
    print("Testing GA Visualization")
    print("=" * 80 + "\n")
    
    # Create visualizer
    visualizer = GAVisualizer(
        enabled=True,
        interactive=True,
        save_plots=True
    )
    
    print("Visualizer created. Simulating 10 generations...")
    print("Close the plot window when done viewing.\n")
    
    # Simulate evolution over 10 generations
    for gen in range(10):
        print(f"Simulating generation {gen}...")
        
        # Create a mock population
        population = Population(size=20, generation=gen)
        
        # Add individuals with varying fitness
        # Fitness improves over generations
        base_fitness = 0.3 + (gen * 0.05)
        base_profit = 5.0 + (gen * 2.0)
        base_sharpe = 0.5 + (gen * 0.15)
        
        for i in range(20):
            # Add some variance
            import random
            variance = random.uniform(-0.1, 0.1)
            
            fitness = max(0, base_fitness + variance)
            profit = max(0, base_profit + (variance * 10))
            sharpe = max(0, base_sharpe + (variance * 0.5))
            win_rate = 0.45 + (gen * 0.02) + (variance * 0.05)
            win_rate = max(0.3, min(0.7, win_rate))  # Clamp to realistic range
            drawdown = 0.15 - (gen * 0.01) + abs(variance * 0.05)
            drawdown = max(0.05, min(0.3, drawdown))
            
            individual = create_mock_individual(
                gen=gen,
                ind_id=i,
                fitness=fitness,
                profit=profit,
                sharpe=sharpe,
                win_rate=win_rate,
                drawdown=drawdown
            )
            population.add_individual(individual)
        
        # Get statistics
        stats = population.get_stats()
        
        # Update visualization
        visualizer.update(gen, stats, population)
        
        # Small delay to see the animation
        time.sleep(0.5)
    
    print("\nSimulation complete!")
    print("The visualization should show:")
    print("  - Fitness increasing over generations")
    print("  - Performance metrics improving")
    print("  - Population diversity changing")
    print("  - Fitness distribution spreading")
    print("\nClose the plot window to finish...\n")
    
    # Close visualization (this will save the plot and wait for user to close window)
    visualizer.close()
    
    print("\n" + "=" * 80)
    print("Visualization test complete!")
    print("=" * 80 + "\n")


def test_non_interactive():
    """Test non-interactive mode (save plots only)."""
    print("\n" + "=" * 80)
    print("Testing Non-Interactive Visualization (save plots only)")
    print("=" * 80 + "\n")
    
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    visualizer = GAVisualizer(
        enabled=True,
        interactive=False,
        save_plots=True
    )
    
    print("Generating 5 generations of mock data...")
    
    for gen in range(5):
        population = Population(size=10, generation=gen)
        
        base_fitness = 0.4 + (gen * 0.08)
        for i in range(10):
            import random
            fitness = base_fitness + random.uniform(-0.05, 0.05)
            individual = create_mock_individual(
                gen=gen,
                ind_id=i,
                fitness=max(0, fitness),
                profit=10.0 + (gen * 3.0),
                sharpe=1.0 + (gen * 0.2),
                win_rate=0.5 + (gen * 0.03),
                drawdown=0.2 - (gen * 0.02)
            )
            population.add_individual(individual)
        
        stats = population.get_stats()
        visualizer.update(gen, stats, population)
    
    visualizer.close()
    
    print("\n" + "=" * 80)
    print("Non-interactive visualization test complete!")
    print("Check genetic_algorithm/output/plots/ for the saved plot")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GA visualization')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Test non-interactive mode')
    args = parser.parse_args()
    
    if args.non_interactive:
        test_non_interactive()
    else:
        test_visualization()
