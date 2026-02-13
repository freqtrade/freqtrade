"""
Test script for backtesting integration.

Tests the FreqTrade backtesting wrapper and fitness evaluation.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.evaluation.backtester import FreqTradeBacktester
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
        return yaml.safe_load(f)


def test_backtester():
    """Test the FreqTrade backtester."""
    print("\n" + "="*80)
    print("TEST 1: FreqTrade Backtester")
    print("="*80)
    
    config = load_config()
    
    # Create a simple test strategy
    strategy_code = '''
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class TestStrategy(IStrategy):
    """Simple test strategy."""
    
    INTERFACE_VERSION = 3
    timeframe = '5m'
    stoploss = -0.10
    minimal_roi = {"0": 0.10}
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add indicators."""
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Buy signal: always buy (for testing)."""
        dataframe.loc[:, 'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Sell signal: use ROI."""
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
'''
    
    # Initialize backtester
    backtester = FreqTradeBacktester(config)
    
    # Run backtest
    print("\nRunning backtest for test strategy...")
    result = backtester.backtest_strategy(strategy_code, "TestStrategy")
    
    # Print results
    print(f"\nBacktest Result:")
    print(f"  Success: {result.success}")
    print(f"  Strategy: {result.strategy_name}")
    
    if result.success:
        print(f"  Total Profit: {result.total_profit:.2f}")
        print(f"  Profit %: {result.profit_percent:.2f}%")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate:.2%}")
        print(f"  Max Drawdown: {result.max_drawdown:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Execution Time: {result.execution_time:.2f}s")
    else:
        print(f"  Error: {result.error_message}")
    
    return result.success


def test_strategy_generation_and_backtest():
    """Test strategy generation and backtesting."""
    print("\n" + "="*80)
    print("TEST 2: Strategy Generation + Backtesting")
    print("="*80)
    
    config = load_config()
    
    # Initialize components
    generator = StrategyGenerator(config)
    backtester = FreqTradeBacktester(config)
    
    # Generate a random strategy
    print("\nGenerating random strategy...")
    strategy_gene = generator.generate_random_strategy(generation=0, individual_id=0)
    
    print(f"  Indicators: {[ind.name for ind in strategy_gene.indicators]}")
    print(f"  Entry conditions: {len(strategy_gene.entry_conditions)}")
    print(f"  Exit conditions: {len(strategy_gene.exit_conditions)}")
    print(f"  Timeframe: {strategy_gene.timeframe}")
    
    # Generate code
    strategy_name = "RandomStrategy_Test"
    strategy_code = generator.generate_strategy_code(strategy_gene)
    print(f"\nGenerated {len(strategy_code)} characters of strategy code")
    
    # Run backtest
    print("\nRunning backtest for generated strategy...")
    result = backtester.backtest_strategy(strategy_code, strategy_name)
    
    # Print results
    print(f"\nBacktest Result:")
    print(f"  Success: {result.success}")
    
    if result.success:
        print(f"  Profit %: {result.profit_percent:.2f}%")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate:.2%}")
    else:
        print(f"  Error: {result.error_message}")
    
    return result.success


def test_fitness_evaluation():
    """Test fitness evaluation with real backtesting."""
    print("\n" + "="*80)
    print("TEST 3: Fitness Evaluation (End-to-End)")
    print("="*80)
    
    config = load_config()
    
    # Initialize components
    generator = StrategyGenerator(config)
    evaluator = FitnessEvaluator(config)
    
    # Generate a strategy
    print("\nGenerating strategy...")
    strategy_gene = generator.generate_random_strategy(generation=0, individual_id=1)
    
    # Evaluate fitness
    print("\nEvaluating fitness (includes backtesting)...")
    fitness, metrics = evaluator.evaluate(strategy_gene, strategy_name="FitnessTest_Strategy")
    
    # Print results
    print(f"\nFitness Evaluation Result:")
    print(f"  Fitness Score: {fitness:.4f}")
    print(f"  Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")
    
    return fitness > 0


def test_caching():
    """Test result caching."""
    print("\n" + "="*80)
    print("TEST 4: Result Caching")
    print("="*80)
    
    config = load_config()
    
    # Simple test strategy
    strategy_code = '''
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class CacheTestStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    stoploss = -0.10
    minimal_roi = {"0": 0.10}
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
'''
    
    backtester = FreqTradeBacktester(config)
    
    # First run
    print("\nFirst backtest (should compute)...")
    import time
    start = time.time()
    result1 = backtester.backtest_strategy(strategy_code, "CacheTestStrategy")
    time1 = time.time() - start
    print(f"  Time: {time1:.2f}s, Success: {result1.success}")
    
    # Second run (should use cache)
    print("\nSecond backtest (should use cache)...")
    start = time.time()
    result2 = backtester.backtest_strategy(strategy_code, "CacheTestStrategy")
    time2 = time.time() - start
    print(f"  Time: {time2:.2f}s, Success: {result2.success}")
    
    if result1.success and result2.success:
        print(f"\n  Cache speedup: {time1/max(time2, 0.01):.1f}x faster")
        print(f"  Results match: {result1.profit_percent == result2.profit_percent}")
    
    return result1.success and result2.success


def main():
    """Run all tests."""
    setup_logging()
    
    print("\n" + "="*80)
    print("BACKTESTING INTEGRATION TEST SUITE")
    print("="*80)
    
    tests = [
        ("Basic Backtester", test_backtester),
        ("Strategy Generation + Backtest", test_strategy_generation_and_backtest),
        ("Fitness Evaluation", test_fitness_evaluation),
        ("Result Caching", test_caching),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
