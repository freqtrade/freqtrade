"""
Direct Backtesting Integration Module

Uses FreqTrade Python API directly with mocked exchange to avoid network calls.
"""

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, PropertyMock, patch
from dataclasses import dataclass

from genetic_algorithm.evaluation.backtester import BacktestResult, BacktestCache

logger = logging.getLogger(__name__)


class DirectBacktester:
    """
    Direct backtester that uses FreqTrade's Python API with mocked exchange.
    
    This avoids network calls and allows offline backtesting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize direct backtester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backtest_config = config.get('backtesting', {})
        self.freqtrade_root = Path(__file__).parent.parent.parent
        self.strategy_dir = self.freqtrade_root / "user_data" / "strategies" / "ga_generated"
        self.strategy_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache if enabled
        self.cache = None
        if self.backtest_config.get('enable_cache', True):
            self.cache = BacktestCache()
        
        logger.info(f"Initialized DirectBacktester")
        logger.info(f"Strategy directory: {self.strategy_dir}")
    
    def backtest_strategy(self, 
                         strategy_code: str, 
                         strategy_name: str,
                         max_retries: int = 2) -> BacktestResult:
        """
        Run backtest for a strategy using direct Python API.
        
        Args:
            strategy_code: Python code for strategy
            strategy_name: Name of the strategy
            max_retries: Maximum number of retries on failure
            
        Returns:
            BacktestResult object
        """
        start_time = time.time()
        
        # Check cache first
        if self.cache:
            cached_result = self.cache.get(strategy_code, self.backtest_config)
            if cached_result:
                logger.info(f"Using cached result for {strategy_name}")
                return cached_result
        
        # Try multiple times in case of transient errors
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry {attempt}/{max_retries} for {strategy_name}")
                    time.sleep(1)
                
                result = self._run_backtest_direct(strategy_code, strategy_name)
                result.execution_time = time.time() - start_time
                
                # Cache successful result
                if result.success and self.cache:
                    self.cache.put(strategy_code, self.backtest_config, result)
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Backtest attempt {attempt + 1} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # All retries failed
        execution_time = time.time() - start_time
        return BacktestResult(
            success=False,
            strategy_name=strategy_name,
            error_message=f"Failed after {max_retries + 1} attempts: {str(last_error)}",
            execution_time=execution_time
        )
    
    def _run_backtest_direct(self, strategy_code: str, strategy_name: str) -> BacktestResult:
        """
        Run backtest using FreqTrade Python API with mocked exchange.
        
        Args:
            strategy_code: Python code for strategy
            strategy_name: Name of the strategy
            
        Returns:
            BacktestResult object
        """
        # Write strategy to file
        strategy_file = self.strategy_dir / f"{strategy_name}.py"
        try:
            with open(strategy_file, 'w') as f:
                f.write(strategy_code)
            logger.debug(f"Wrote strategy file: {strategy_file}")
        except Exception as e:
            logger.error(f"Failed to write strategy file: {e}")
            return BacktestResult(
                success=False,
                strategy_name=strategy_name,
                error_message=f"Failed to write strategy file: {e}"
            )
        
        try:
            # Import FreqTrade modules
            from freqtrade.configuration import Configuration
            from freqtrade.optimize.backtesting import Backtesting
            from freqtrade.exchange.exchange import Exchange
            
            # Create configuration
            config_dict = self._create_backtest_config(strategy_name)
            
            # Mock the exchange to avoid network calls
            with patch.object(Exchange, '_load_async_markets', return_value={}), \
                 patch.object(Exchange, 'markets', PropertyMock(return_value=self._get_mock_markets())), \
                 patch.object(Exchange, 'validate_config', MagicMock()), \
                 patch.object(Exchange, 'validate_timeframes', MagicMock()), \
                 patch.object(Exchange, '_init_ccxt', MagicMock()), \
                 patch.object(Exchange, 'get_fee', return_value=0.001), \
                 patch.object(Exchange, 'precisionMode', PropertyMock(return_value=2)), \
                 patch.object(Exchange, 'precision_mode_price', PropertyMock(return_value=2)), \
                 patch.object(Exchange, 'timeframes', PropertyMock(return_value=["1m", "5m", "15m", "1h", "1d"])):
                
                # Initialize backtesting
                backtesting = Backtesting(config_dict)
                
                # Run backtest
                backtesting.start()
                
                # Get results from the backtest results
                # The results are stored in backtesting.results which is a dictionary
                if not backtesting.results or 'strategy' not in backtesting.results:
                    logger.warning("No results available from backtest")
                    return BacktestResult(
                        success=True,
                        strategy_name=strategy_name,
                        total_trades=0,
                        error_message="No trades generated"
                    )
                
                # Parse results from the strategy results
                strategy_results = backtesting.results['strategy'].get(strategy_name, {})
                result = self._parse_stats(strategy_results, strategy_name)
                return result
                
        except Exception as e:
            logger.error(f"Backtest execution error: {e}")
            import traceback
            traceback.print_exc()
            return BacktestResult(
                success=False,
                strategy_name=strategy_name,
                error_message=f"Execution error: {str(e)}"
            )
    
    def _create_backtest_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        Create FreqTrade config for backtesting.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Configuration dictionary
        """
        # Use test data pairs that exist in tests/testdata
        pairs = self.backtest_config.get('pairs', ['UNITTEST/BTC'])
        stake_amount = self.backtest_config.get('stake_amount', 0.05)  # Match test config
        max_open_trades = self.backtest_config.get('max_open_trades', 3)
        fee = self.backtest_config.get('fee', 0.0025)  # Match test default
        timerange = self.backtest_config.get('timerange', '')
        
        # Use test data directory
        datadir = self.freqtrade_root / "tests" / "testdata"
        exportdir = self.freqtrade_root / "user_data" / "backtest_results"
        exportdir.mkdir(parents=True, exist_ok=True)
        
        config = {
            "strategy": strategy_name,
            "strategy_path": str(self.strategy_dir),
            "user_data_dir": self.freqtrade_root / "user_data",  # Path object
            "datadir": datadir,  # Path object
            "exportdirectory": exportdir,  # Path object for storing results
            "runmode": "backtest",  # Required for FreqTrade
            "stake_currency": "BTC",
            "stake_amount": stake_amount,
            "dry_run_wallet": 1000,
            "max_open_trades": max_open_trades,
            "timeframe": "5m",
            "exchange": {
                "name": "gate",
                "pair_whitelist": pairs,
                "ccxt_config": {},
                "ccxt_async_config": {},
            },
            "pairlists": [{"method": "StaticPairList"}],
            "fee": fee,
            "trading_mode": "spot",
            "margin_mode": "",
            "dry_run": True,
            "timerange": timerange if timerange else None,
        }
        
        # Store original config reference (required for backtest storage)
        config["original_config"] = config.copy()
        
        return config
    
    def _get_mock_markets(self) -> Dict[str, Any]:
        """
        Get mock markets data for offline backtesting.
        
        Returns:
            Mock markets dictionary
        """
        # Create mock market data for test pairs that exist in testdata
        mock_markets = {}
        
        test_pairs = [
            "UNITTEST/BTC", "ETH/BTC", "LTC/BTC", "XRP/BTC", "ADA/BTC",
            "DASH/BTC", "ETC/BTC", "XLM/BTC", "XMR/BTC", "NXT/BTC",
            "ZEC/BTC", "TRX/BTC"
        ]
        
        for pair in test_pairs:
            base, quote = pair.split("/")
            mock_markets[pair] = {
                "id": pair.replace("/", "_"),
                "symbol": pair,
                "base": base,
                "quote": quote,
                "active": True,
                "spot": True,
                "precision": {
                    "amount": 8,
                    "price": 8
                },
                "limits": {
                    "amount": {"min": 0.001, "max": 10000},
                    "price": {"min": 0.00000001, "max": 100000},
                    "cost": {"min": 0.001, "max": None}
                },
                "info": {}
            }
        
        return mock_markets
    
    def _parse_stats(self, stats: Dict[str, Any], strategy_name: str) -> BacktestResult:
        """
        Parse backtest statistics into BacktestResult.
        
        Args:
            stats: Statistics dictionary from backtesting
            strategy_name: Name of strategy
            
        Returns:
            BacktestResult object
        """
        # Extract metrics from stats
        return BacktestResult(
            success=True,
            strategy_name=strategy_name,
            total_profit=stats.get('profit_total_abs', 0.0),
            profit_percent=stats.get('profit_total', 0.0),
            total_trades=stats.get('total_trades', 0),
            wins=stats.get('wins', 0),
            losses=stats.get('losses', 0),
            win_rate=stats.get('winrate', 0.0),
            max_drawdown=stats.get('max_drawdown', 0.0),
            max_drawdown_abs=stats.get('max_drawdown_abs', 0.0),
            sharpe_ratio=stats.get('sharpe', 0.0),
            sortino_ratio=stats.get('sortino', 0.0),
            profit_factor=stats.get('profit_factor', 0.0),
            avg_profit=stats.get('profit_mean', 0.0),
            median_profit=stats.get('profit_median', 0.0),
            avg_duration=stats.get('duration_avg', ""),
        )
