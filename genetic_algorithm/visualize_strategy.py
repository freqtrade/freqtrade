#!/usr/bin/env python3
"""
Visualize Trades for a Strategy

This script allows you to visualize the trades made by a specific strategy
on candlestick charts with entry/exit markers.

Usage:
    python visualize_strategy.py <strategy_file> [--config <config_path>] [--output <output_dir>]

Examples:
    # Visualize a specific strategy
    python visualize_strategy.py output/strategy_rank1_gen5_ind0_20260221_175820.py
    
    # With custom config
    python visualize_strategy.py output/my_strategy.py --config config/ga_config.yaml
    
    # With custom output directory
    python visualize_strategy.py output/my_strategy.py --output ./my_plots
"""

import argparse
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Visualize strategy trades on candlestick charts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'strategy_file',
        type=str,
        help='Path to the strategy Python file to visualize'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='genetic_algorithm/config/ga_config.yaml',
        help='Path to GA config file (default: genetic_algorithm/config/ga_config.yaml)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='genetic_algorithm/output/trade_plots',
        help='Output directory for trade charts (default: genetic_algorithm/output/trade_plots)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger('visualize_strategy')
    
    # Validate strategy file exists
    strategy_path = Path(args.strategy_file)
    if not strategy_path.exists():
        logger.error(f"Strategy file not found: {strategy_path}")
        sys.exit(1)
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    logger.info(f"Visualizing strategy: {strategy_path}")
    logger.info(f"Using config: {config_path}")
    
    try:
        import yaml
        from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
        from genetic_algorithm.visualization.trade_visualizer import TradeVisualizer
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Read strategy code
        with open(strategy_path, 'r') as f:
            strategy_code = f.read()
        
        strategy_name = strategy_path.stem
        
        # Initialize backtester and run backtest with trade collection
        logger.info("Running backtest to collect trade data...")
        backtester = DirectBacktester(config)
        result = backtester.backtest_strategy_with_trades(strategy_code, strategy_name)
        
        if not result.success:
            logger.error(f"Backtest failed: {result.error_message}")
            sys.exit(1)
        
        logger.info(f"Backtest completed: {result.total_trades} trades, {result.profit_percent:.2f}% profit")
        
        if result.total_trades == 0:
            logger.warning("No trades to visualize - strategy may be too restrictive")
            sys.exit(0)
        
        # Initialize visualizer and generate charts
        output_dir = Path(args.output)
        visualizer = TradeVisualizer(output_dir=output_dir, enabled=True)
        
        logger.info("Generating trade charts...")
        saved_files = visualizer.visualize_strategy_from_backtest(
            strategy_name=strategy_name,
            backtest_result=result,
            generation=0,
            individual_idx=0
        )
        
        if saved_files:
            logger.info(f"Successfully generated {len(saved_files)} chart(s):")
            for filepath in saved_files:
                logger.info(f"  - {filepath}")
        else:
            logger.warning("No charts were generated")
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Make sure matplotlib and pandas are installed: pip install matplotlib pandas")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
