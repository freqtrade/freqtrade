"""
Test direct backtester with mocked exchange.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.evaluation.direct_backtester import DirectBacktester
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
        return yaml.safe_load(f)


def test_direct_backtest():
    """Test the direct backtester."""
    print("\n" + "="*80)
    print("TEST: Direct Backtester with Mocked Exchange")
    print("="*80)
    
    config = load_config()
    
    # Create a simple test strategy
    strategy_code = '''
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas as pd

class TestDirectStrategy(IStrategy):
    """Simple test strategy."""
    
    INTERFACE_VERSION = 3
    timeframe = '5m'
    stoploss = -0.10
    minimal_roi = {"0": 0.10}
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add indicators."""
        # Simple SMA
        dataframe['sma'] = dataframe['close'].rolling(window=10).mean()
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Buy signal: price crosses above SMA."""
        dataframe.loc[
            (dataframe['close'] > dataframe['sma']),
            'enter_long'
        ] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Sell signal: price crosses below SMA."""
        dataframe.loc[
            (dataframe['close'] < dataframe['sma']),
            'exit_long'
        ] = 1
        return dataframe
'''
    
    # Initialize backtester with cache disabled
    test_config = load_config()
    test_config['backtesting']['enable_cache'] = False  # Disable cache for testing
    backtester = DirectBacktester(test_config)
    
    # Run backtest
    print("\nRunning direct backtest...")
    result = backtester.backtest_strategy(strategy_code, "TestDirectStrategy")
    
    # Print results
    print(f"\nBacktest Result:")
    print(f"  Success: {result.success}")
    print(f"  Strategy: {result.strategy_name}")
    
    if result.success:
        print(f"  ✅ Backtest completed successfully!")
        print(f"  Total Profit: {result.total_profit:.4f}")
        print(f"  Profit %: {result.profit_percent:.2f}%")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate:.2%}")
        print(f"  Max Drawdown: {result.max_drawdown:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Execution Time: {result.execution_time:.2f}s")
    else:
        print(f"  ❌ Backtest failed")
        print(f"  Error: {result.error_message}")
    
    return result.success


def main():
    """Run test."""
    setup_logging()
    
    print("\n" + "="*80)
    print("DIRECT BACKTESTING TEST")
    print("="*80)
    
    success = test_direct_backtest()
    
    print("\n" + "="*80)
    print(f"Test {'PASSED' if success else 'FAILED'}")
    print("="*80)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
