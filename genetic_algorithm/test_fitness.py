"""
Test fitness evaluation with generated strategies.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.evaluation.fitness import FitnessEvaluator
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
        config = yaml.safe_load(f)
    # Disable caching for testing
    config['backtesting']['enable_cache'] = False
    return config


def test_fitness_evaluation():
    """Test fitness evaluation with generated strategies."""
    print("\n" + "="*80)
    print("TEST: Fitness Evaluation with Generated Strategies")
    print("="*80)
    
    config = load_config()
    
    # Initialize components
    generator = StrategyGenerator(config)
    evaluator = FitnessEvaluator(config)
    
    # Generate a few strategies and evaluate them
    num_strategies = 3
    results = []
    
    for i in range(num_strategies):
        print(f"\n{'-'*80}")
        print(f"Strategy {i+1}/{num_strategies}")
        print(f"{'-'*80}")
        
        # Generate random strategy
        strategy_gene = generator.generate_random_strategy(generation=0, individual_id=i)
        
        print(f"  Indicators: {[ind.type for ind in strategy_gene.indicators]}")
        print(f"  Entry conditions: {len(strategy_gene.entry_conditions)}")
        print(f"  Exit conditions: {len(strategy_gene.exit_conditions)}")
        print(f"  Timeframe: {strategy_gene.timeframe}")
        print(f"  Stoploss: {strategy_gene.stoploss:.2%}")
        
        # Evaluate fitness
        print(f"\n  Running fitness evaluation...")
        fitness, metrics = evaluator.evaluate(strategy_gene)
        
        # Print results
        print(f"\n  Results:")
        print(f"    Fitness Score: {fitness:.4f}")
        print(f"    Profit %: {metrics.get('profit', 0):.2f}%")
        print(f"    Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"    Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"    Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"    Total Trades: {metrics.get('num_trades', 0)}")
        
        results.append({
            'fitness': fitness,
            'metrics': metrics,
            'strategy_gene': strategy_gene
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    # Sort by fitness
    results.sort(key=lambda x: x['fitness'], reverse=True)
    
    print(f"\nStrategies ranked by fitness:")
    for i, result in enumerate(results):
        gene = result['strategy_gene']
        print(f"  {i+1}. Gen{gene.generation}_Ind{gene.individual_id}: "
              f"fitness={result['fitness']:.4f}, "
              f"profit={result['metrics'].get('profit', 0):.2f}%, "
              f"trades={result['metrics'].get('num_trades', 0)}")
    
    print(f"\nBest strategy:")
    best = results[0]
    best_gene = best['strategy_gene']
    print(f"  Generation: {best_gene.generation}, Individual: {best_gene.individual_id}")
    print(f"  Fitness: {best['fitness']:.4f}")
    print(f"  Indicators: {[ind.type for ind in best_gene.indicators]}")
    
    return len(results) > 0 and all(r['fitness'] >= 0 for r in results)


def main():
    """Run test."""
    setup_logging()
    
    print("\n" + "="*80)
    print("FITNESS EVALUATION TEST")
    print("="*80)
    
    success = test_fitness_evaluation()
    
    print("\n" + "="*80)
    print(f"Test {'PASSED' if success else 'FAILED'}")
    print("="*80)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
