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


def setup_logging(monitor_active: bool = False):
    """Set up logging for the GA run.
    
    Args:
        monitor_active: If True, suppress console StreamHandler on the root
            logger because the terminal monitor captures logs via its own handler.
    """
    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    log_file = LOG_DIR / f'ga_run_{datetime.now().strftime(TIMESTAMP_FORMAT)}.log'
    
    handlers = [logging.FileHandler(log_file)]
    if not monitor_active:
        handlers.insert(0, logging.StreamHandler())
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,  # Override any prior basicConfig calls
    )
    
    # When monitor is active, ensure NO StreamHandlers leak to the console.
    # This covers cases where libraries add their own handlers to the root logger.
    if monitor_active:
        root = logging.getLogger()
        for h in list(root.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                root.removeHandler(h)


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
        fitness_str = f"{individual.fitness:.4f}" if individual.fitness is not None else "N/A"
        print(f"RANK {rank}: Strategy Gen{gene.generation}_Ind{gene.individual_id}")
        print(f"{'-'*80}\n  Fitness Score:      {fitness_str}\n")
        
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


def save_summary_report(top_strategies: list, output_dir: Path, config: dict,
                        generation_stats=None, generation_holdout_history=None):
    """
    Save a summary report of the GA run, including holdout and Monte Carlo metrics.
    
    Args:
        top_strategies: List of top Individual objects
        output_dir: Directory to save report
        config: Configuration dictionary
        generation_stats: Optional list of PopulationStats for fitness history
        generation_holdout_history: Optional list of GenerationHoldoutStats
    """
    from genetic_algorithm.utils.overfit_analysis import (
        classify_overfitting, OverfitThresholds, OverfitAssessment,
        generate_detailed_results, save_detailed_results, print_overfit_summary,
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    report_path = output_dir / f"ga_summary_{timestamp}.txt"
    
    thresholds = OverfitThresholds.from_config(config)
    
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
        
        assessments = []
        for rank, individual in enumerate(top_strategies, 1):
            gene = individual.strategy_gene
            metrics = individual.metrics
            
            f.write(f"Rank {rank}: Gen{gene.generation}_Ind{gene.individual_id}\n")
            f.write(f"  Raw Fitness: {(individual.raw_fitness if individual.raw_fitness is not None else 0):.4f}\n")
            f.write(f"  Shared Fitness: {individual.fitness:.4f}\n")
            f.write(f"  Profit: {metrics.get('profit', 0):.2f}%\n")
            f.write(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n")
            f.write(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n")
            f.write(f"  Win Rate: {metrics.get('win_rate', 0):.2%}\n")
            f.write(f"  Total Trades: {metrics.get('num_trades', 0)}\n")
            
            # Holdout metrics (if available)
            holdout_fitness = metrics.get('holdout_fitness')
            if holdout_fitness is not None:
                f.write(f"  Holdout Fitness: {holdout_fitness:.4f}\n")
                f.write(f"  Holdout Profit: {metrics.get('holdout_profit', 0):.2f}%\n")
                f.write(f"  Holdout Drawdown: {metrics.get('holdout_drawdown', 0):.2%}\n")
                f.write(f"  Holdout Degradation: {metrics.get('holdout_degradation', 0):.1%}\n")
            
            # Monte Carlo metrics (if available)
            mc_robustness = metrics.get('mc_robustness')
            if mc_robustness is not None:
                f.write(f"  MC Robustness: {mc_robustness:.1%}\n")
                f.write(f"  MC Mean Profit: {metrics.get('mc_mean_profit', 0):.2f}%\n")
                f.write(f"  MC Profit P5: {metrics.get('mc_profit_p5', 0):.2f}%\n")
            
            # Walk-forward gap (if available)
            train_val_gap = metrics.get('train_val_gap')
            if train_val_gap is not None:
                f.write(f"  WF Train-Val Gap: {train_val_gap:.1%}\n")
            
            # CPCV / PBO metrics (if available)
            cpcv_pbo = metrics.get('cpcv_pbo')
            if cpcv_pbo is not None:
                f.write(f"  CPCV PBO: {cpcv_pbo:.3f}\n")
                f.write(f"  CPCV Penalty: {metrics.get('cpcv_penalty', 1.0):.3f}\n")
                cpcv_mean_oos = metrics.get('cpcv_mean_oos')
                if cpcv_mean_oos is not None:
                    f.write(f"  CPCV Mean OOS: {cpcv_mean_oos:.3f}\n")
            
            # Overfitting classification
            assessment = classify_overfitting(
                metrics=metrics, fitness=individual.fitness,
                thresholds=thresholds, strategy_gene=gene,
            )
            assessment.individual_id = getattr(individual, 'id', f'rank_{rank}')
            assessments.append(assessment)
            
            if assessment.overall_label != "UNKNOWN":
                f.write(f"  Overfit Risk: {assessment.overall_label} "
                        f"(composite={assessment.composite_score:.3f})\n")
            
            f.write("\n")
        
        # Summary section
        if assessments:
            f.write("=" * 80 + "\n")
            f.write("OVERFITTING ANALYSIS SUMMARY\n")
            f.write("-" * 80 + "\n")
            labels = [a.overall_label for a in assessments]
            composites = [a.composite_score for a in assessments if a.composite_score is not None]
            f.write(f"  SAFE: {labels.count('SAFE')}  WARNING: {labels.count('WARNING')}  "
                    f"OVERFIT: {labels.count('OVERFIT')}  UNKNOWN: {labels.count('UNKNOWN')}\n")
            if composites:
                f.write(f"  Avg composite score: {sum(composites)/len(composites):.3f}\n")
            f.write("\n")
    
    print(f"Summary report saved to: {report_path.absolute()}")
    
    # Print overfitting summary to console
    if assessments:
        print_overfit_summary(assessments)
    
    # Save detailed JSON results
    try:
        report = generate_detailed_results(
            top_strategies=top_strategies,
            config=config,
            generation_holdout_history=generation_holdout_history,
            generation_stats=generation_stats,
            thresholds=thresholds,
        )
        json_path = save_detailed_results(report, output_dir)
        print(f"Detailed JSON results saved to: {json_path.absolute()}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to save detailed results JSON: {e}")


def validate_config(config: dict) -> bool:
    """
    Validate GA configuration and warn about issues.
    
    Checks for:
    - Missing required sections and keys
    - Wrong key names (common typos like max_workers vs num_workers)
    - Fitness weights summing to ~1.0
    - Logical inconsistencies (elite_size > population_size, etc.)
    - Stale or suspicious data configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if config is valid, False otherwise
    """
    backtest_cfg = config.get('backtesting', {})
    ga_cfg = config.get('genetic_algorithm', {})
    pairs = backtest_cfg.get('pairs', [])
    timerange = backtest_cfg.get('timerange', '')
    auto_download = backtest_cfg.get('auto_download_data', True)
    exchange = backtest_cfg.get('exchange', 'binance')
    
    issues = []
    warnings = []
    info = []
    
    # ── Critical issues ──
    if not pairs:
        issues.append("❌ No pairs configured in 'backtesting.pairs'")
    
    if not ga_cfg:
        issues.append("❌ Missing 'genetic_algorithm' section")
    
    pop_size = ga_cfg.get('population_size', 0)
    elite_size = ga_cfg.get('elite_size', 0)
    if pop_size and elite_size and elite_size >= pop_size:
        issues.append(f"❌ elite_size ({elite_size}) >= population_size ({pop_size})")
    
    # ── Wrong key names (common mistakes) ──
    parallel_cfg = config.get('parallel_evaluation', {})
    wrong_keys = {
        'max_workers': 'num_workers',
        'timeout_per_eval': 'backtest_timeout',
        'chunk_size': None,  # Unused, just warn
    }
    for wrong_key, correct_key in wrong_keys.items():
        if wrong_key in parallel_cfg:
            if correct_key:
                warnings.append(f"⚠️  parallel_evaluation.{wrong_key} is not read by code — "
                               f"did you mean '{correct_key}'?")
            else:
                warnings.append(f"⚠️  parallel_evaluation.{wrong_key} is not used by the code")
    
    # ── Fitness weights validation ──
    weights = config.get('fitness_weights', {})
    if weights:
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            warnings.append(f"⚠️  fitness_weights sum to {weight_sum:.3f} (expected ~1.0)")
    
    # ── GA parameter sanity ──
    mutation_rate = ga_cfg.get('mutation_rate', 0)
    max_mutation = ga_cfg.get('max_mutation_rate', 0.5)
    if mutation_rate and max_mutation and mutation_rate > max_mutation:
        warnings.append(f"⚠️  mutation_rate ({mutation_rate}) > max_mutation_rate ({max_mutation})")
    
    tournament_size = ga_cfg.get('tournament_size', 3)
    if pop_size and tournament_size and tournament_size > pop_size // 2:
        warnings.append(f"⚠️  tournament_size ({tournament_size}) is very large relative to population ({pop_size})")
    
    random_immigrants = ga_cfg.get('random_immigrants', 0)
    if pop_size and random_immigrants and random_immigrants > pop_size * 0.5:
        warnings.append(f"⚠️  random_immigrants ({random_immigrants}) is >50% of population ({pop_size}) — "
                       f"this replaces most evolved strategies each generation")
    
    # ── Adaptive mutation without max_mutation_rate ──
    if ga_cfg.get('adaptive_mutation', False) and 'max_mutation_rate' not in ga_cfg:
        info.append("ℹ️  adaptive_mutation enabled but max_mutation_rate not set (defaulting to 0.5)")
    
    # ── Walk-forward + parallel warning ──
    wf_cfg = config.get('walk_forward', {})
    if wf_cfg.get('enabled', False) and parallel_cfg.get('enabled', False):
        info.append("ℹ️  Walk-forward + parallel: WF runs post-hoc on elites (not inside workers)")
    
    # ── Warnings for test data ──
    if any('UNITTEST' in p for p in pairs):
        warnings.append("⚠️  Using UNITTEST pairs (test data from 2018)")
        warnings.append("   For real strategy development:")
        warnings.append("   1. Edit config and set: backtesting.pairs = ['BTC/USDT']")
        warnings.append("   2. Set timerange: backtesting.timerange = '20250120-20260228'")
        if auto_download:
            warnings.append("   3. GA will auto-download data (auto_download_data: true)")
        else:
            warnings.append("   3. Download data: freqtrade download-data --pairs BTC/USDT --timeframes 1h --days 90")
    
    if not timerange and not any('UNITTEST' in p for p in pairs):
        warnings.append("⚠️  No timerange specified — will use all available data")
    
    # ── Info ──
    if auto_download:
        info.append("ℹ️  Auto-download enabled: Missing data will be downloaded automatically")
        info.append(f"   Exchange: {exchange}")
    else:
        info.append("ℹ️  Auto-download disabled: You must manually download data before running")
        info.append(f"   Use: freqtrade download-data --exchange {exchange} --pairs {' '.join(pairs)} --timeframes 1h --days 90")
    
    # ── Display results ──
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
    print(f"  Population: {pop_size}")
    print(f"  Generations: {ga_cfg.get('generations')}")
    if weights:
        print(f"  Fitness weights sum: {sum(weights.values()):.3f}")
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
  
  # Use a preset config
  python genetic_algorithm/run_ga.py --config genetic_algorithm/config/ga_config_fast.yaml
  
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
        '--resume', '-r',
        type=str,
        default=None,
        help='Resume evolution from a checkpoint file (e.g., genetic_algorithm/data/checkpoints/checkpoint_gen17_*.json)'
    )
    
    parser.add_argument(
        '--seed',
        type=str,
        default=None,
        help='Seed population from a checkpoint or strategy JSON file. Injects top strategies as immigrants.'
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
        '--no-monitor',
        action='store_true',
        help='Disable the terminal monitor (use classic scrolling log output)'
    )
    
    parser.add_argument(
        '--monitor-mode',
        choices=['simple', 'detailed', 'logs'],
        default=None,
        help='Set the initial terminal monitor view mode (default: from config)'
    )

    # Web dashboard flags
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Start evolution with the web dashboard (accessible at http://localhost:8501)'
    )
    parser.add_argument(
        '--dashboard-only',
        action='store_true',
        help='Start only the web dashboard server (no immediate evolution run)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port for the web dashboard (default: 8501)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for GA runner."""
    # Parse arguments
    args = parse_arguments()

    # ── Dashboard-only mode ─────────────────────────────────────
    if args.dashboard_only:
        return _start_dashboard_only(args)

    # ── Dashboard mode (evolution + dashboard) ──────────────────
    if args.dashboard:
        return _start_with_dashboard(args)

    # ── Classic terminal mode ───────────────────────────────────
    
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
        print(f"\n💡 Tip: Use one of the available configs or copy ga_config_fast.yaml as a starting point")
        return 1
    
    print(f"📂 Loading configuration from: {config_file}")
    
    # Print banner
    print_banner()
    
    # Load configuration first (before logging setup) to check monitor config
    try:
        config = load_and_update_config(config_file)
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at {config_file.absolute()}")
        print("Please ensure you're running this from the correct directory.")
        return 1
    
    # Apply CLI overrides for terminal monitor
    if args.no_monitor:
        config.setdefault('terminal_monitor', {})['enabled'] = False
    if args.monitor_mode:
        config.setdefault('terminal_monitor', {})['default_mode'] = args.monitor_mode
    
    # Determine if the monitor will be active (for logging setup)
    # Default to True (matching create_monitor) so log suppression
    # activates even when the terminal_monitor section is missing.
    monitor_cfg = config.get('terminal_monitor', {})
    monitor_will_be_active = monitor_cfg.get('enabled', True)
    if monitor_will_be_active:
        try:
            import rich  # noqa: F401
        except ImportError:
            monitor_will_be_active = False
    
    # Set up logging (suppress console handler when monitor is active)
    setup_logging(monitor_active=monitor_will_be_active)
    logger = logging.getLogger(__name__)
    
    # Validate config
    if not validate_config(config):
        print("❌ Config validation failed. Please fix the issues above.")
        return 1
    
    # Run richer config validation (walk-forward bounds, weight polarity, etc.)
    try:
        from genetic_algorithm.utils.config_validator import validate_and_log
        validate_and_log(config)
    except Exception as e:
        logger.debug(f"Extended config validation skipped: {e}")
    
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
        
        # ── Island Model branch ──
        island_cfg = config.get('island_model', {})
        if island_cfg.get('enabled', False):
            from genetic_algorithm.core.island_model import IslandModelEvolution
            
            print("  ┌───────────────────────────────────────┐")
            print("  │   ISLAND MODEL EVOLUTION              │")
            n_islands = len(island_cfg.get('islands', []))
            n_gens = config.get('genetic_algorithm', {}).get('generations', 25)
            print(f"  │   Islands: {n_islands}  Generations: {n_gens:<13}│")
            print("  └───────────────────────────────────────┘")
            print()
            
            island_evo = IslandModelEvolution(
                config_path=tmp_config_path,
                visualize=args.visualize,
                interactive=not args.no_interactive,
            )
            
            logger.info("Starting island model evolution...")
            results = island_evo.evolve()
            
            # Collect top strategies across all islands — pool ALL individuals
            # then rank globally to avoid missing strong candidates
            top_strategies = []
            for island_name, individuals in results.items():
                top_strategies.extend(individuals)
            top_strategies.sort(
                key=lambda x: x.raw_fitness if x.raw_fitness else 0,
                reverse=True,
            )
            top_strategies = top_strategies[:TOP_STRATEGIES_COUNT]
            
            # Use island_evo for reporting
            ga = None  # No single GA instance
            
            # Get a strategy_generator from the first island GA for saving
            first_island_name = list(island_evo.islands.keys())[0]
            strategy_generator = island_evo.islands[first_island_name].strategy_generator
            
            # Clean up temporary config
            Path(tmp_config_path).unlink(missing_ok=True)
        else:
            # ── Standard single-population GA ──
            ga = GeneticAlgorithm(
                tmp_config_path, 
                visualize=args.visualize,
                interactive=not args.no_interactive
            )
        
            # Handle --seed: load strategies from file and queue as immigrants
            if args.seed:
                seed_path = Path(args.seed)
                if not seed_path.exists():
                    print(f"❌ Error: Seed file not found: {seed_path}")
                    return 1
                seed_individuals = ga.load_seed_strategies(str(seed_path))
                ga.inject_immigrants(seed_individuals)
                print(f"  ✓ Loaded {len(seed_individuals)} seed strategies from {seed_path}")
            
            # Run evolution (with optional resume)
            logger.info("Starting evolution process...")
            resume_path = args.resume if args.resume else None
            
            if resume_path:
                resume_file = Path(resume_path)
                if not resume_file.exists():
                    print(f"❌ Error: Checkpoint file not found: {resume_file}")
                    return 1
                print(f"  ✓ Resuming from checkpoint: {resume_file}")
            
            top_individuals = ga.evolve(resume_from=resume_path)
            
            # Get top N strategies
            top_strategies = top_individuals[:TOP_STRATEGIES_COUNT]
            strategy_generator = ga.strategy_generator
            
            # Clean up temporary config
            Path(tmp_config_path).unlink(missing_ok=True)
        
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("EVOLUTION INTERRUPTED BY USER")
        print("=" * 80)
        return 0
    except Exception as e:
        import traceback
        print("\n\n" + "=" * 80)
        print("ERROR DURING EVOLUTION")
        print("=" * 80)
        print(f"\n{e}")
        traceback.print_exc()
        logger.exception("Evolution failed")
        return 1
    
    # Print results
    print("\n\n" + "=" * 80)
    print("EVOLUTION COMPLETE")
    print("=" * 80)
    print()
    
    if ga is not None:
        total_generations = ga.current_generation + 1
        total_strategies = ga.population_size * total_generations
        print(f"✓ Completed {total_generations} generations")
        print(f"✓ Created {total_strategies} strategies")
    else:
        # Island model — info is already printed by IslandModelEvolution
        print(f"✓ Island model evolution finished")
        print(f"✓ Top strategies collected: {len(top_strategies)}")
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
                if evo_fitness is not None and holdout_fitness is not None and evo_fitness > 0:
                    degradation = (evo_fitness - holdout_fitness) / evo_fitness
                else:
                    degradation = 0
                individual.metrics['holdout_degradation'] = degradation

                evo_fitness_str = f"{evo_fitness:.4f}" if evo_fitness is not None else "N/A"
                holdout_fitness_str = f"{holdout_fitness:.4f}" if holdout_fitness is not None else "N/A"
                status = "✓" if degradation < 0.3 else "⚠️"
                print(f"  {status} Rank {rank}: "
                      f"Evo fitness={evo_fitness_str} → Holdout fitness={holdout_fitness_str} "
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
    
    # === CPCV / PBO Validation ===
    cpcv_config = config.get('cpcv', {})
    cpcv_enabled = cpcv_config.get('enabled', False)

    if cpcv_enabled and top_strategies:
        if ga is None:
            print("\n  \u26a0\ufe0f  CPCV skipped: not supported in island model mode (yet)\n")
        else:
            from genetic_algorithm.evaluation.cpcv import CPCVValidator
            from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
            import numpy as np

            print("=" * 80)
            print("COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)")
            print("=" * 80)
            n_groups = cpcv_config.get('n_groups', 6)
            pbo_threshold = cpcv_config.get('pbo_threshold', 0.5)
            print(f"  Groups: {n_groups}, PBO threshold: {pbo_threshold}")
            print(f"  Evaluating top {len(top_strategies)} strategies...")
            print()

            cpcv_validator = CPCVValidator(config)
            cpcv_backtester = DirectBacktester(config)

            # Compute the time blocks for splitting backtest results
            timerange = config.get('backtesting', {}).get('timerange', '')
            strategy_block_results = {}

            for rank, individual in enumerate(top_strategies, 1):
                gene = individual.strategy_gene
                strategy_name = f"GAStrategy_Gen{gene.generation}_Ind{gene.individual_id}"
                strategy_code = strategy_generator.generate_strategy_code(gene)
                strategy_id = f"rank_{rank}"

                # Parse timerange to create block boundaries
                if timerange and '-' in timerange:
                    from datetime import datetime as dt, timedelta
                    start_str, end_str = timerange.split('-')
                    start_date = dt.strptime(start_str, '%Y%m%d')
                    end_date = dt.strptime(end_str, '%Y%m%d')
                    total_days = (end_date - start_date).days
                    block_days = total_days // n_groups

                    block_profits = []
                    for block_idx in range(n_groups):
                        block_start = start_date + timedelta(days=block_idx * block_days)
                        block_end = (start_date + timedelta(days=(block_idx + 1) * block_days)
                                     if block_idx < n_groups - 1 else end_date)
                        block_tr = f"{block_start.strftime('%Y%m%d')}-{block_end.strftime('%Y%m%d')}"

                        # Run backtest on this block
                        bt_result = cpcv_backtester.backtest_strategy(
                            strategy_code, strategy_name,
                            timerange_override=block_tr,
                            strategy_max_open_trades=gene.max_open_trades,
                        )
                        block_profit = bt_result.profit_percent if bt_result.success else 0.0
                        block_profits.append(block_profit)

                    strategy_block_results[strategy_id] = np.array(block_profits)
                    logger.info(f"  Rank {rank}: block profits = {[f'{p:.2f}' for p in block_profits]}")
                else:
                    logger.warning(f"  Rank {rank}: skipped — no timerange configured")

            # Run CPCV validation
            if len(strategy_block_results) >= 2:
                cpcv_result = cpcv_validator.validate_strategies(
                    strategy_block_results, timerange=timerange,
                )

                pbo = cpcv_result.get('pbo', 0.0)
                penalty = cpcv_result.get('penalty', 1.0)
                skipped = cpcv_result.get('skipped', False)

                if not skipped:
                    status = "✓" if pbo < pbo_threshold else "⚠️"
                    print(f"  {status} PBO = {pbo:.3f} (threshold: {pbo_threshold})")
                    print(f"      Penalty multiplier: {penalty:.3f}")
                    print(f"      Paths evaluated: {cpcv_result.get('n_paths', 0)}")

                    per_strategy_oos = cpcv_result.get('per_strategy_oos', {})
                    for sid, oos_info in per_strategy_oos.items():
                        print(f"      {sid}: mean_oos={oos_info['mean_oos']:.3f}, "
                              f"std_oos={oos_info['std_oos']:.3f}")

                    # Store PBO in individual metrics
                    for rank, individual in enumerate(top_strategies, 1):
                        individual.metrics['cpcv_pbo'] = pbo
                        individual.metrics['cpcv_penalty'] = penalty
                        oos_key = f"rank_{rank}"
                        if oos_key in per_strategy_oos:
                            individual.metrics['cpcv_mean_oos'] = per_strategy_oos[oos_key]['mean_oos']
                else:
                    print(f"  ⚠️  CPCV skipped: {cpcv_result.get('reason', 'unknown')}")
            else:
                print("  ⚠️  CPCV skipped: need >= 2 strategies with block results")

            print()
            print("  Legend: ✓ = PBO below threshold (low overfit risk), ⚠️ = PBO above threshold")
            print()

    # === Monte-Carlo Robustness Validation ===
    mc_config = config.get('monte_carlo', {})
    mc_enabled = mc_config.get('enabled', False)

    if mc_enabled and top_strategies:
        if ga is None:
            print("\n  \u26a0\ufe0f  Monte-Carlo skipped: not supported in island model mode (yet)\n")
        else:
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
                strategy_code = strategy_generator.generate_strategy_code(gene)
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
    print_top_strategies(top_strategies, strategy_generator)
    
    # Save strategies
    save_top_strategies(top_strategies, strategy_generator, OUTPUT_DIR)
    
    # Save summary report (with holdout/MC metrics and overfitting analysis)
    # Collect generation_stats from island GAs when in island mode
    _gen_stats = getattr(ga, 'generation_stats', None) if ga is not None else None
    _gen_holdout = getattr(ga, 'generation_holdout_history', None) if ga is not None else None
    if _gen_stats is None and ga is None and 'island_evo' in dir():
        # Aggregate generation_stats from all island GAs
        _gen_stats = []
        for _iname, _iga in island_evo.islands.items():
            for s in getattr(_iga, 'generation_stats', []):
                _gen_stats.append(s)
    save_summary_report(
        top_strategies, OUTPUT_DIR, config,
        generation_stats=_gen_stats,
        generation_holdout_history=_gen_holdout,
    )
    
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


def _start_dashboard_only(args):
    """Start the web dashboard server without an immediate evolution run."""
    print()
    print("=" * 80)
    print(" " * 15 + "GA EVOLUTION DASHBOARD — STANDALONE MODE")
    print("=" * 80)
    print()
    print(f"  Starting dashboard at http://127.0.0.1:{args.port}")
    print(f"  API docs at http://127.0.0.1:{args.port}/docs")
    print()
    print("  Start evolution runs from the dashboard UI or via the API.")
    print("  Press Ctrl+C to stop the server.")
    print()

    try:
        from genetic_algorithm.web.config import WebConfig
        from genetic_algorithm.web.server import start_server

        web_config = WebConfig(port=args.port)
        setup_logging(monitor_active=False)
        start_server(web_config=web_config)
    except ImportError as e:
        print(f"❌ Dashboard dependencies not installed: {e}")
        print("   Install with: pip install fastapi uvicorn")
        return 1
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")
    return 0


def _start_with_dashboard(args):
    """Start evolution AND web dashboard simultaneously."""
    import threading

    config_file = Path(args.config)
    if not config_file.exists():
        print(f"❌ Error: Config file not found: {config_file}")
        return 1

    config = load_and_update_config(config_file)
    config['_config_name'] = config_file.stem

    # Disable terminal monitor when dashboard is active
    config.setdefault('terminal_monitor', {})['enabled'] = False

    setup_logging(monitor_active=False)
    logger = logging.getLogger(__name__)

    if not validate_config(config):
        print("❌ Config validation failed.")
        return 1

    try:
        from genetic_algorithm.web.config import WebConfig
        from genetic_algorithm.web.run_manager import RunManager
        from genetic_algorithm.web.server import create_app

        web_config = WebConfig(port=args.port)
        run_manager = RunManager()
        app = create_app(web_config=web_config, run_manager=run_manager)

        print()
        print("=" * 80)
        print(" " * 15 + "GA EVOLUTION DASHBOARD — RUNNING")
        print("=" * 80)
        print()
        print(f"  Dashboard: http://127.0.0.1:{args.port}")
        print(f"  API docs:  http://127.0.0.1:{args.port}/docs")
        print()

        # Start evolution as a managed run
        handle = run_manager.start_run(
            config=config,
            resume_from=args.resume if args.resume else None,
        )
        print(f"  Evolution started: run_id={handle.run_id}")
        print()
        print("  Press Ctrl+C to stop server and evolution.")
        print()

        import uvicorn
        uvicorn.run(
            app,
            host=web_config.host,
            port=web_config.port,
            log_level=web_config.log_level,
        )

    except ImportError as e:
        print(f"❌ Dashboard dependencies not installed: {e}")
        print("   Install with: pip install fastapi uvicorn")
        return 1
    except KeyboardInterrupt:
        print("\n\nStopping dashboard and evolution...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
