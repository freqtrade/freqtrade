#!/usr/bin/env python3
"""
Genetic Algorithm Runner - Main Entry Point

This script provides a pre-configured "run button" to start the Genetic Algorithm
for evolving trading strategies. It runs the complete evolution process and outputs
the top 5 most successful strategies at the end.

Configuration can be adjusted in the USER CONFIGURATION section below.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.core.evolution import GeneticAlgorithm
from genetic_algorithm.strategies.generator import StrategyGenerator
import yaml


# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Basic GA Parameters (override config file)
POPULATION_SIZE = 50          # Number of strategies per generation (default: 100)
GENERATIONS = 20              # Number of generations to evolve (default: 50)
MUTATION_RATE = 0.15          # Probability of mutation (0.0-1.0, default: 0.15)
CROSSOVER_RATE = 0.7          # Probability of crossover (0.0-1.0, default: 0.7)
ELITE_SIZE = 5                # Number of top strategies to preserve (default: 10)

# Number of top strategies to display and save at the end
TOP_STRATEGIES_COUNT = 5

# Output configuration
SAVE_STRATEGIES = True        # Save top strategies to files
OUTPUT_DIR = Path("genetic_algorithm/output")  # Directory for output files
LOG_DIR = Path("genetic_algorithm/logs")       # Directory for log files

# Configuration file path
CONFIG_FILE = Path("genetic_algorithm/config/ga_config.yaml")

# ============================================================================


def setup_logging():
    """Set up logging for the GA run."""
    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = LOG_DIR / f'ga_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )


def load_and_update_config(config_path) -> dict:
    """
    Load configuration from YAML file and update with user parameters.
    
    Args:
        config_path: Path to configuration file (string or Path object)
        
    Returns:
        Updated configuration dictionary
    """
    config_path = Path(config_path) if not isinstance(config_path, Path) else config_path
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update GA parameters with user configuration
    config['genetic_algorithm']['population_size'] = POPULATION_SIZE
    config['genetic_algorithm']['generations'] = GENERATIONS
    config['genetic_algorithm']['mutation_rate'] = MUTATION_RATE
    config['genetic_algorithm']['crossover_rate'] = CROSSOVER_RATE
    config['genetic_algorithm']['elite_size'] = ELITE_SIZE
    
    return config


def print_banner():
    """Print startup banner."""
    print("\n" + "=" * 80)
    print(" " * 20 + "GENETIC ALGORITHM - STRATEGY EVOLUTION")
    print("=" * 80)
    print()
    print("This script will evolve trading strategies using a Genetic Algorithm.")
    print("The best performing strategies will be identified and saved.")
    print()


def print_configuration(config: dict):
    """Print current configuration."""
    ga_config = config['genetic_algorithm']
    
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print()
    print("Genetic Algorithm Parameters:")
    print(f"  Population Size:    {ga_config['population_size']}")
    print(f"  Generations:        {ga_config['generations']}")
    print(f"  Mutation Rate:      {ga_config['mutation_rate']:.2%}")
    print(f"  Crossover Rate:     {ga_config['crossover_rate']:.2%}")
    print(f"  Elite Size:         {ga_config['elite_size']}")
    print(f"  Selection Method:   {ga_config['selection_method']}")
    print()
    
    backtest_config = config['backtesting']
    print("Backtesting Configuration:")
    print(f"  Trading Pairs:      {', '.join(backtest_config['pairs'])}")
    print(f"  Stake Amount:       {backtest_config['stake_amount']}")
    print(f"  Max Open Trades:    {backtest_config['max_open_trades']}")
    print(f"  Fee:                {backtest_config['fee']:.3%}")
    print()
    
    fitness_weights = config['fitness_weights']
    print("Fitness Weights:")
    print(f"  Profit:             {fitness_weights['profit']:.2%}")
    print(f"  Sharpe Ratio:       {fitness_weights['sharpe_ratio']:.2%}")
    print(f"  Drawdown:           {fitness_weights['drawdown']:.2%}")
    print(f"  Win Rate:           {fitness_weights['win_rate']:.2%}")
    print(f"  Trade Frequency:    {fitness_weights['trade_frequency']:.2%}")
    print()
    print("=" * 80)
    print()


def print_top_strategies(top_strategies: list, strategy_generator: StrategyGenerator):
    """
    Print detailed information about top strategies.
    
    Args:
        top_strategies: List of top Individual objects
        strategy_generator: StrategyGenerator instance for code generation
    """
    print("\n" + "=" * 80)
    print(f"TOP {len(top_strategies)} STRATEGIES")
    print("=" * 80)
    print()
    
    for rank, individual in enumerate(top_strategies, 1):
        gene = individual.strategy_gene
        metrics = individual.metrics
        
        print(f"RANK {rank}: Strategy Gen{gene.generation}_Ind{gene.individual_id}")
        print("-" * 80)
        print(f"  Fitness Score:      {individual.fitness:.4f}")
        print()
        print("  Performance Metrics:")
        print(f"    Profit:           {metrics.get('profit', 0):.2f}%")
        print(f"    Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"    Max Drawdown:     {metrics.get('max_drawdown', 0):.2%}")
        print(f"    Win Rate:         {metrics.get('win_rate', 0):.2%}")
        print(f"    Total Trades:     {metrics.get('num_trades', 0)}")
        print(f"    Profit Factor:    {metrics.get('profit_factor', 0):.2f}")
        print()
        print("  Strategy Parameters:")
        print(f"    Timeframe:        {gene.timeframe}")
        print(f"    Stop Loss:        {gene.stoploss:.2%}")
        print(f"    Trailing Stop:    {gene.trailing_stop}")
        print(f"    ROI:              {gene.minimal_roi}")
        print()
        print(f"  Indicators ({len(gene.indicators)}):")
        for ind in gene.indicators:
            params = ', '.join(f"{k}={v}" for k, v in ind.parameters.items())
            print(f"    • {ind.type}: {params} (weight={ind.weight:.2f})")
        print()
        print(f"  Entry Conditions ({len(gene.entry_conditions)}):")
        for cond in gene.entry_conditions:
            print(f"    • {cond.indicator} {cond.operator} {cond.threshold} ({cond.logic})")
        print()
        
        if gene.exit_conditions:
            print(f"  Exit Conditions ({len(gene.exit_conditions)}):")
            for cond in gene.exit_conditions:
                print(f"    • {cond.indicator} {cond.operator} {cond.threshold} ({cond.logic})")
        else:
            print("  Exit Conditions: Using default ROI/stoploss")
        print()
        
        if rank < len(top_strategies):
            print()


def save_top_strategies(top_strategies: list, strategy_generator: StrategyGenerator, output_dir: Path):
    """
    Save top strategies to Python files.
    
    Args:
        top_strategies: List of top Individual objects
        strategy_generator: StrategyGenerator instance
        output_dir: Directory to save strategies
    """
    if not SAVE_STRATEGIES:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SAVING STRATEGIES")
    print("=" * 80)
    print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for rank, individual in enumerate(top_strategies, 1):
        gene = individual.strategy_gene
        
        # Generate strategy code
        strategy_code = strategy_generator.generate_strategy_code(gene)
        
        # Create filename
        filename = f"strategy_rank{rank}_gen{gene.generation}_ind{gene.individual_id}_{timestamp}.py"
        filepath = output_dir / filename
        
        # Save to file
        with open(filepath, 'w') as f:
            f.write(strategy_code)
        
        print(f"  ✓ Saved Rank {rank}: {filename}")
        print(f"      Fitness: {individual.fitness:.4f}, Profit: {individual.metrics.get('profit', 0):.2f}%")
    
    print()
    print(f"All strategies saved to: {output_dir.absolute()}")
    print()


def save_summary_report(top_strategies: list, output_dir: Path, config: dict):
    """
    Save a summary report of the GA run.
    
    Args:
        top_strategies: List of top Individual objects
        output_dir: Directory to save report
        config: Configuration dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"ga_summary_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GENETIC ALGORITHM RUN SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        ga_config = config['genetic_algorithm']
        f.write("Configuration:\n")
        f.write(f"  Population Size: {ga_config['population_size']}\n")
        f.write(f"  Generations: {ga_config['generations']}\n")
        f.write(f"  Mutation Rate: {ga_config['mutation_rate']:.2%}\n")
        f.write(f"  Crossover Rate: {ga_config['crossover_rate']:.2%}\n")
        f.write(f"  Elite Size: {ga_config['elite_size']}\n\n")
        
        f.write(f"Top {len(top_strategies)} Strategies:\n")
        f.write("-" * 80 + "\n\n")
        
        for rank, individual in enumerate(top_strategies, 1):
            gene = individual.strategy_gene
            metrics = individual.metrics
            
            f.write(f"Rank {rank}: Gen{gene.generation}_Ind{gene.individual_id}\n")
            f.write(f"  Fitness: {individual.fitness:.4f}\n")
            f.write(f"  Profit: {metrics.get('profit', 0):.2f}%\n")
            f.write(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n")
            f.write(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n")
            f.write(f"  Win Rate: {metrics.get('win_rate', 0):.2%}\n")
            f.write(f"  Total Trades: {metrics.get('num_trades', 0)}\n\n")
    
    print(f"Summary report saved to: {report_path.absolute()}")


def main():
    """Main entry point for GA runner."""
    # Print banner
    print_banner()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        config = load_and_update_config(CONFIG_FILE)
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at {CONFIG_FILE.absolute()}")
        print("Please ensure you're running this from the correct directory.")
        return 1
    
    # Print configuration
    print_configuration(config)
    
    # Confirm start
    print("Press Enter to start evolution, or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return 0
    
    print("\n" + "=" * 80)
    print("STARTING EVOLUTION")
    print("=" * 80)
    print()
    
    # Initialize and run GA
    try:
        # Create temporary config file with updated parameters
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_config:
            yaml.dump(config, tmp_config)
            tmp_config_path = tmp_config.name
        
        # Initialize GA with updated config
        ga = GeneticAlgorithm(tmp_config_path)
        
        # Run evolution
        logger.info("Starting evolution process...")
        top_individuals = ga.evolve()
        
        # Get top N strategies
        top_strategies = top_individuals[:TOP_STRATEGIES_COUNT]
        
        # Clean up temporary config
        Path(tmp_config_path).unlink()
        
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("EVOLUTION INTERRUPTED BY USER")
        print("=" * 80)
        return 0
    except Exception as e:
        print("\n\n" + "=" * 80)
        print("ERROR DURING EVOLUTION")
        print("=" * 80)
        print(f"\n{e}")
        logger.exception("Evolution failed")
        return 1
    
    # Print results
    print("\n\n" + "=" * 80)
    print("EVOLUTION COMPLETE")
    print("=" * 80)
    print()
    
    total_generations = ga.current_generation + 1
    print(f"✓ Completed {total_generations} generations")
    print(f"✓ Evaluated {ga.population_size * total_generations} strategies")
    print()
    
    # Display top strategies
    print_top_strategies(top_strategies, ga.strategy_generator)
    
    # Save strategies
    save_top_strategies(top_strategies, ga.strategy_generator, OUTPUT_DIR)
    
    # Save summary report
    save_summary_report(top_strategies, OUTPUT_DIR, config)
    
    # Final summary
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Review the top strategies in the output above")
    print(f"2. Check saved strategy files in: {OUTPUT_DIR.absolute()}")
    print("3. Copy promising strategies to: user_data/strategies/")
    print("4. Backtest with full data: freqtrade backtesting --strategy <StrategyName>")
    print("5. Test in dry-run mode: freqtrade trade --dry-run --strategy <StrategyName>")
    print("6. Deploy to live trading when confident")
    print()
    print("=" * 80)
    print("✓ GA RUN COMPLETE!")
    print("=" * 80)
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
