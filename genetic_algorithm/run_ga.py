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
import argparse
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

# Number of top strategies to display and save at the end
TOP_STRATEGIES_COUNT = 5

# Output configuration
SAVE_STRATEGIES = True        # Save top strategies to files
OUTPUT_DIR = Path("genetic_algorithm/output")  # Directory for output files
LOG_DIR = Path("genetic_algorithm/logs")       # Directory for log files

# Configuration file path
CONFIG_FILE = Path("genetic_algorithm/config/ga_config.yaml")

# Timestamp format for files and logs
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# ============================================================================


def setup_logging():
    """Set up logging for the GA run."""
    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = LOG_DIR / f'ga_run_{datetime.now().strftime(TIMESTAMP_FORMAT)}.log'
    
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
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file (string or Path object)
        
    Returns:
        Configuration dictionary loaded from file
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # All configuration is now read from the config file
    # No more hardcoded overrides - edit the config file to change parameters
    
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
    print(f"\n{'='*80}\nTOP {len(top_strategies)} STRATEGIES\n{'='*80}\n")
    
    for rank, individual in enumerate(top_strategies, 1):
        gene = individual.strategy_gene
        metrics = individual.metrics
        
        # Header and fitness
        print(f"RANK {rank}: Strategy Gen{gene.generation}_Ind{gene.individual_id}")
        print(f"{'-'*80}\n  Fitness Score:      {individual.fitness:.4f}\n")
        
        # Performance metrics (consolidated print statements)
        print(f"  Performance Metrics:\n"
              f"    Profit:           {metrics.get('profit', 0):.2f}%\n"
              f"    Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.2f}\n"
              f"    Max Drawdown:     {metrics.get('max_drawdown', 0):.2%}\n"
              f"    Win Rate:         {metrics.get('win_rate', 0):.2%}\n"
              f"    Total Trades:     {metrics.get('num_trades', 0)}\n"
              f"    Profit Factor:    {metrics.get('profit_factor', 0):.2f}\n")
        
        # Strategy parameters (consolidated)
        print(f"  Strategy Parameters:\n"
              f"    Timeframe:        {gene.timeframe}\n"
              f"    Stop Loss:        {gene.stoploss:.2%}\n"
              f"    Trailing Stop:    {gene.trailing_stop}\n"
              f"    ROI:              {gene.minimal_roi}\n")
        
        # Indicators
        print(f"  Indicators ({len(gene.indicators)}):")
        for ind in gene.indicators:
            params = ', '.join(f"{k}={v}" for k, v in ind.parameters.items())
            print(f"    • {ind.type}: {params} (weight={ind.weight:.2f})")
        
        # Entry conditions
        print(f"\n  Entry Conditions ({len(gene.entry_conditions)}):")
        for cond in gene.entry_conditions:
            print(f"    • {cond.indicator} {cond.operator} {cond.threshold} ({cond.logic})")
        
        # Exit conditions
        print()
        if gene.exit_conditions:
            print(f"  Exit Conditions ({len(gene.exit_conditions)}):")
            for cond in gene.exit_conditions:
                print(f"    • {cond.indicator} {cond.operator} {cond.threshold} ({cond.logic})")
        else:
            print("  Exit Conditions: Using default ROI/stoploss")
        
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
    
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    
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
    
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
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


def validate_config(config: dict) -> bool:
    """
    Validate GA configuration and warn about issues.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if config is valid, False otherwise
    """
    backtest_cfg = config.get('backtesting', {})
    pairs = backtest_cfg.get('pairs', [])
    timerange = backtest_cfg.get('timerange', '')
    auto_download = backtest_cfg.get('auto_download_data', True)
    exchange = backtest_cfg.get('exchange', 'binance')
    
    issues = []
    warnings = []
    info = []
    
    # Critical issues
    if not pairs:
        issues.append("❌ No pairs configured in 'backtesting.pairs'")
    
    # Warnings for test data
    if any('UNITTEST' in p for p in pairs):
        warnings.append("⚠️  WARNING: Using UNITTEST pairs (test data from 2018)")
        warnings.append("   For real strategy development:")
        warnings.append("   1. Edit config file and set: backtesting.pairs = ['BTC/USDT']")
        warnings.append("   2. Set timerange: backtesting.timerange = '20250120-20250219'")
        if auto_download:
            warnings.append("   3. GA will auto-download data when it starts (auto_download_data: true)")
        else:
            warnings.append("   3. Download data: freqtrade download-data --pairs BTC/USDT --timeframes 1h --days 90")
    
    if not timerange and not any('UNITTEST' in p for p in pairs):
        warnings.append("⚠️  WARNING: No timerange specified - will use all available data")
        warnings.append("   Consider setting: backtesting.timerange = '20250120-20250219'")
    
    # Info about auto-download
    if auto_download:
        info.append("ℹ️  Auto-download enabled: Missing data will be downloaded automatically")
        info.append(f"   Exchange: {exchange}")
    else:
        info.append("ℹ️  Auto-download disabled: You must manually download data before running")
        info.append(f"   Use: freqtrade download-data --exchange {exchange} --pairs {' '.join(pairs)} --timeframes 1h --days 90")
    
    # Display results
    if info:
        print("\n" + "="*80)
        for msg in info:
            print(msg)
        print("="*80)
    
    if warnings:
        print("\n" + "="*80)
        for warning in warnings:
            print(warning)
        print("="*80 + "\n")
    
    if issues:
        print("\n" + "="*80)
        print("❌ CONFIG VALIDATION FAILED:")
        for issue in issues:
            print(issue)
        print("="*80 + "\n")
        return False
    
    # Summary
    print("✓ Config validation passed")
    print(f"  Pairs: {pairs}")
    print(f"  Timerange: {timerange if timerange else 'ALL AVAILABLE DATA'}")
    print(f"  Population: {config.get('genetic_algorithm', {}).get('population_size')}")
    print(f"  Generations: {config.get('genetic_algorithm', {}).get('generations')}")
    print()
    
    return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='FreqTrade Genetic Algorithm - Strategy Evolution System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python genetic_algorithm/run_ga.py
  
  # Use custom config
  python genetic_algorithm/run_ga.py --config my_config.yaml
  
  # Use example config for real data
  python genetic_algorithm/run_ga.py --config genetic_algorithm/config/ga_config_example.yaml
  
  # Validate config without running
  python genetic_algorithm/run_ga.py --config my_config.yaml --validate-only
  
  # Run with visualization
  python genetic_algorithm/run_ga.py --visualize
  python genetic_algorithm/run_ga.py --visualize --no-interactive
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='genetic_algorithm/config/ga_config.yaml',
        help='Path to GA configuration file (default: genetic_algorithm/config/ga_config.yaml)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate config and exit (useful for checking before long runs)'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Enable live visualization of evolution progress'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Disable interactive plotting (save plots only, no live display)'
    )
    
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt and start evolution immediately'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume evolution from the latest checkpoint (if available)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for GA runner."""
    # Parse arguments
    args = parse_arguments()
    
    # Load config from specified file
    config_file = Path(args.config)
    
    # Check if config file exists
    if not config_file.exists():
        print(f"❌ Error: Config file not found: {config_file}")
        print(f"\n📁 Available config files:")
        config_dir = Path("genetic_algorithm/config")
        if config_dir.exists():
            for cfg_file in sorted(config_dir.glob("*.yaml")):
                print(f"   - {cfg_file}")
        print(f"\n💡 Tip: Create your own config by copying ga_config_example.yaml")
        return 1
    
    print(f"📂 Loading configuration from: {config_file}")
    
    # Print banner
    print_banner()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        config = load_and_update_config(config_file)
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at {config_file.absolute()}")
        print("Please ensure you're running this from the correct directory.")
        return 1
    
    # Validate config
    if not validate_config(config):
        print("❌ Config validation failed. Please fix the issues above.")
        return 1
    
    if args.validate_only:
        print("✅ Config validation passed!")
        return 0
    
    # Print configuration
    print_configuration(config)
    
    # Print visualization info
    if args.visualize:
        print("Visualization: ENABLED")
        if args.no_interactive:
            print("  Mode: Static (plots saved to file)")
        else:
            print("  Mode: Live Interactive")
        print()
    else:
        print("Visualization: DISABLED")
        print("  (Use --visualize flag to enable)")
        print()
    
    # Confirm start
    if not args.yes:
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
        ga = GeneticAlgorithm(
            tmp_config_path, 
            visualize=args.visualize,
            interactive=not args.no_interactive
        )
        
        # Run evolution
        logger.info("Starting evolution process...")
        top_individuals = ga.evolve(resume=args.resume)
        
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
    total_strategies = ga.population_size * total_generations
    print(f"✓ Completed {total_generations} generations")
    print(f"✓ Created {total_strategies} strategies")
    print()
    
    # === Out-of-Sample Holdout Validation ===
    holdout_config = config.get('holdout_validation', {})
    holdout_enabled = holdout_config.get('enabled', False)
    
    if holdout_enabled and top_strategies:
        holdout_pct = holdout_config.get('holdout_pct', 0.15)
        original_timerange = config.get('backtesting', {}).get('timerange', '')
        
        if original_timerange:
            from genetic_algorithm.evaluation.fitness import FitnessEvaluator
            
            evolution_tr, holdout_tr = FitnessEvaluator.split_timerange_for_holdout(
                original_timerange, holdout_pct
            )
            
            print("=" * 80)
            print("OUT-OF-SAMPLE HOLDOUT VALIDATION")
            print("=" * 80)
            print(f"  Holdout period: {holdout_tr} ({holdout_pct:.0%} of data)")
            print(f"  Evaluating top {len(top_strategies)} strategies on unseen data...")
            print()
            
            holdout_evaluator = FitnessEvaluator(config)
            
            for rank, individual in enumerate(top_strategies, 1):
                gene = individual.strategy_gene
                holdout_fitness, holdout_metrics = holdout_evaluator.evaluate_holdout(
                    gene, holdout_tr
                )
                
                # Store holdout results in individual metrics
                individual.metrics['holdout_fitness'] = holdout_fitness
                individual.metrics['holdout_profit'] = holdout_metrics.get('profit', 0)
                individual.metrics['holdout_sharpe'] = holdout_metrics.get('sharpe_ratio', 0)
                individual.metrics['holdout_drawdown'] = holdout_metrics.get('max_drawdown', 0)
                individual.metrics['holdout_trades'] = holdout_metrics.get('num_trades', 0)
                
                # Calculate degradation from evolution fitness to holdout fitness
                evo_fitness = individual.fitness
                if evo_fitness > 0:
                    degradation = (evo_fitness - holdout_fitness) / evo_fitness
                else:
                    degradation = 0
                individual.metrics['holdout_degradation'] = degradation
                
                status = "✓" if degradation < 0.3 else "⚠️"
                print(f"  {status} Rank {rank}: "
                      f"Evo fitness={evo_fitness:.4f} → Holdout fitness={holdout_fitness:.4f} "
                      f"(degradation={degradation:.1%})")
                print(f"      Holdout: profit={holdout_metrics.get('profit', 0):.2f}%, "
                      f"trades={holdout_metrics.get('num_trades', 0)}, "
                      f"drawdown={holdout_metrics.get('max_drawdown', 0):.1%}")
            
            print()
            print("  Legend: ✓ = <30% degradation (robust), ⚠️  = ≥30% degradation (potential overfit)")
            print()
        else:
            print("⚠️  Holdout validation skipped: no timerange configured")
            print()
    
    # === Monte-Carlo Robustness Validation ===
    mc_config = config.get('monte_carlo', {})
    mc_enabled = mc_config.get('enabled', False)

    if mc_enabled and top_strategies:
        from genetic_algorithm.evaluation.monte_carlo import run_monte_carlo
        from genetic_algorithm.evaluation.direct_backtester import DirectBacktester

        print("=" * 80)
        print("MONTE-CARLO ROBUSTNESS ANALYSIS")
        print("=" * 80)
        num_perms = mc_config.get('num_permutations', 100)
        print(f"  Running {num_perms} permutations per strategy...")
        print()

        mc_backtester = DirectBacktester(config)

        for rank, individual in enumerate(top_strategies, 1):
            gene = individual.strategy_gene
            strategy_code = ga.strategy_generator.generate_strategy_code(gene)
            strategy_name = f"GAStrategy_Gen{gene.generation}_Ind{gene.individual_id}"

            bt_result = mc_backtester.backtest_strategy_with_trades(
                strategy_code, strategy_name,
                strategy_max_open_trades=gene.max_open_trades
            )

            if bt_result.success and bt_result.trades:
                mc_result = run_monte_carlo(bt_result.trades, mc_config)
                individual.metrics['mc_robustness'] = mc_result.robustness_score
                individual.metrics['mc_mean_profit'] = mc_result.mean_profit
                individual.metrics['mc_profit_p5'] = mc_result.profit_p5
                individual.metrics['mc_profit_std'] = mc_result.profit_std

                status = "✓" if mc_result.robustness_score >= 0.8 else "⚠️"
                print(f"  {status} Rank {rank}: robustness={mc_result.robustness_score:.1%}, "
                      f"mean_profit={mc_result.mean_profit:.2f}%, "
                      f"p5={mc_result.profit_p5:.2f}%, p95={mc_result.profit_p95:.2f}%")
            else:
                individual.metrics['mc_robustness'] = 0.0
                print(f"  ⚠️  Rank {rank}: backtest failed or no trades — skipped")

        print()
        print("  Legend: ✓ = ≥80% permutations profitable (robust), ⚠️ = <80% (fragile)")
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
