"""
Backtesting Integration Module

Provides integration with FreqTrade backtesting engine for strategy evaluation.
Handles strategy testing, result parsing, and error management.
"""

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """
    Container for backtest results.
    """
    success: bool
    strategy_name: str
    
    # Performance metrics
    total_profit: float = 0.0
    profit_percent: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_abs: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    
    # Trade metrics
    avg_profit: float = 0.0
    median_profit: float = 0.0
    avg_duration: str = ""
    
    # Additional info
    error_message: Optional[str] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'strategy_name': self.strategy_name,
            'total_profit': self.total_profit,
            'profit_percent': self.profit_percent,
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.win_rate,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_abs': self.max_drawdown_abs,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'profit_factor': self.profit_factor,
            'avg_profit': self.avg_profit,
            'median_profit': self.median_profit,
            'avg_duration': self.avg_duration,
            'error_message': self.error_message,
            'execution_time': self.execution_time,
        }


class BacktestCache:
    """
    Simple cache for backtest results to avoid re-testing identical strategies.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir or Path("genetic_algorithm/data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, BacktestResult] = {}
        
    def _get_cache_key(self, strategy_code: str, config: Dict[str, Any]) -> str:
        """
        Generate cache key from strategy code and config.
        
        Args:
            strategy_code: Strategy Python code
            config: Backtest configuration
            
        Returns:
            Cache key (hash)
        """
        # Create deterministic string from strategy and config
        cache_input = f"{strategy_code}_{json.dumps(config, sort_keys=True)}"
        return hashlib.sha256(cache_input.encode()).hexdigest()
    
    def get(self, strategy_code: str, config: Dict[str, Any]) -> Optional[BacktestResult]:
        """
        Get cached result if available.
        
        Args:
            strategy_code: Strategy code
            config: Backtest configuration
            
        Returns:
            Cached result or None
        """
        cache_key = self._get_cache_key(strategy_code, config)
        
        # Check memory cache
        if cache_key in self.cache:
            logger.debug(f"Cache hit (memory): {cache_key[:8]}...")
            return self.cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    result = BacktestResult(**data)
                    self.cache[cache_key] = result
                    logger.debug(f"Cache hit (disk): {cache_key[:8]}...")
                    return result
            except Exception as e:
                logger.warning(f"Failed to load cache file: {e}")
        
        return None
    
    def put(self, strategy_code: str, config: Dict[str, Any], result: BacktestResult):
        """
        Store result in cache.
        
        Args:
            strategy_code: Strategy code
            config: Backtest configuration
            result: Backtest result
        """
        cache_key = self._get_cache_key(strategy_code, config)
        
        # Store in memory
        self.cache[cache_key] = result
        
        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result.to_dict(), f)
            logger.debug(f"Cached result: {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to write cache file: {e}")


class FreqTradeBacktester:
    """
    Wrapper for FreqTrade backtesting functionality.
    
    Handles:
    - Strategy file creation
    - FreqTrade command execution
    - Result parsing
    - Error handling
    - Result caching
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backtester.
        
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
        
        logger.info(f"Initialized FreqTradeBacktester")
        logger.info(f"Strategy directory: {self.strategy_dir}")
    
    def backtest_strategy(self, 
                         strategy_code: str, 
                         strategy_name: str,
                         max_retries: int = 2) -> BacktestResult:
        """
        Run backtest for a strategy.
        
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
                    time.sleep(1)  # Brief delay before retry
                
                result = self._run_backtest(strategy_code, strategy_name)
                result.execution_time = time.time() - start_time
                
                # Cache successful result
                if result.success and self.cache:
                    self.cache.put(strategy_code, self.backtest_config, result)
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Backtest attempt {attempt + 1} failed: {e}")
        
        # All retries failed
        execution_time = time.time() - start_time
        return BacktestResult(
            success=False,
            strategy_name=strategy_name,
            error_message=f"Failed after {max_retries + 1} attempts: {str(last_error)}",
            execution_time=execution_time
        )
    
    def _run_backtest(self, strategy_code: str, strategy_name: str) -> BacktestResult:
        """
        Internal method to run single backtest attempt.
        
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
        
        # Create temporary config file for backtesting
        config_data = self._create_backtest_config(strategy_name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = Path(f.name)
        
        try:
            # Run backtesting command
            result = self._execute_backtest(config_file, strategy_name)
            return result
            
        finally:
            # Cleanup temporary config file
            try:
                config_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp config: {e}")
    
    def _create_backtest_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        Create FreqTrade config for backtesting.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Configuration dictionary
        """
        pairs = self.backtest_config.get('pairs', ['BTC/USDT'])
        stake_amount = self.backtest_config.get('stake_amount', 100)
        max_open_trades = self.backtest_config.get('max_open_trades', 3)
        fee = self.backtest_config.get('fee', 0.001)
        
        config = {
            "strategy": strategy_name,
            "strategy_path": str(self.strategy_dir),
            "stake_currency": "USDT",
            "stake_amount": stake_amount,
            "dry_run_wallet": 1000,
            "max_open_trades": max_open_trades,
            "exchange": {
                "name": "binance",
                "pair_whitelist": pairs,
                "ccxt_config": {},
                "ccxt_async_config": {},
            },
            "pairlists": [{"method": "StaticPairList"}],  # Required by FreqTrade
            "fee": fee,
            "trading_mode": "spot",
            "margin_mode": "",
            "dry_run": True,  # Ensure we don't try to connect to exchange
        }
        
        return config
    
    def _execute_backtest(self, config_file: Path, strategy_name: str) -> BacktestResult:
        """
        Execute FreqTrade backtest command.
        
        Args:
            config_file: Path to config file
            strategy_name: Name of strategy
            
        Returns:
            BacktestResult object
        """
        timeout = self.backtest_config.get('timeout', 300)
        timerange = self.backtest_config.get('timerange', '')
        
        # Build command
        cmd = [
            'python', '-m', 'freqtrade', 'backtesting',
            '--config', str(config_file),
            '--strategy', strategy_name,
            '--export', 'trades',
            '--export-filename', f'backtest_{strategy_name}.json',
            '--breakdown', 'day',  # Add breakdown for more detail
        ]
        
        if timerange:
            cmd.extend(['--timerange', timerange])
        
        # Use test data for now (can be configured later)
        datadir = self.freqtrade_root / "tests" / "testdata"
        if datadir.exists():
            cmd.extend(['--datadir', str(datadir)])
        
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        try:
            # Execute command
            process = subprocess.run(
                cmd,
                cwd=str(self.freqtrade_root),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**subprocess.os.environ, 'FREQTRADE_NO_EXCHANGE_CHECK': '1'}  # Disable exchange check
            )
            
            if process.returncode != 0:
                error_msg = process.stderr or process.stdout
                logger.error(f"Backtest failed: {error_msg}")
                return BacktestResult(
                    success=False,
                    strategy_name=strategy_name,
                    error_message=f"Backtest command failed: {error_msg[:500]}"
                )
            
            # Parse results
            result = self._parse_backtest_output(process.stdout, strategy_name)
            return result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Backtest timed out after {timeout}s")
            return BacktestResult(
                success=False,
                strategy_name=strategy_name,
                error_message=f"Backtest timed out after {timeout}s"
            )
        except Exception as e:
            logger.error(f"Backtest execution error: {e}")
            return BacktestResult(
                success=False,
                strategy_name=strategy_name,
                error_message=f"Execution error: {str(e)}"
            )
    
    def _parse_backtest_output(self, output: str, strategy_name: str) -> BacktestResult:
        """
        Parse backtest output and extract metrics.
        
        Args:
            output: Command output text
            strategy_name: Name of strategy
            
        Returns:
            BacktestResult object
        """
        # Try to find and parse the JSON results file
        results_file = self.freqtrade_root / "user_data" / "backtest_results" / f"backtest_{strategy_name}.json"
        
        # If JSON file exists, parse it
        if results_file.exists():
            try:
                return self._parse_json_results(results_file, strategy_name)
            except Exception as e:
                logger.warning(f"Failed to parse JSON results: {e}")
        
        # Fallback: parse text output
        return self._parse_text_output(output, strategy_name)
    
    def _parse_json_results(self, results_file: Path, strategy_name: str) -> BacktestResult:
        """
        Parse JSON backtest results file.
        
        Args:
            results_file: Path to results JSON file
            strategy_name: Name of strategy
            
        Returns:
            BacktestResult object
        """
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Navigate to strategy results
        strategy_data = None
        if 'strategy' in data and strategy_name in data['strategy']:
            strategy_data = data['strategy'][strategy_name]
        elif 'strategy' in data and len(data['strategy']) > 0:
            # Take first strategy if name doesn't match
            strategy_data = list(data['strategy'].values())[0]
        else:
            raise ValueError("No strategy data found in results")
        
        # Extract metrics
        return BacktestResult(
            success=True,
            strategy_name=strategy_name,
            total_profit=strategy_data.get('profit_total_abs', 0.0),
            profit_percent=strategy_data.get('profit_total', 0.0),
            total_trades=strategy_data.get('total_trades', 0),
            wins=strategy_data.get('wins', 0),
            losses=strategy_data.get('losses', 0),
            win_rate=strategy_data.get('winrate', 0.0),
            max_drawdown=strategy_data.get('max_drawdown', 0.0),
            max_drawdown_abs=strategy_data.get('max_drawdown_abs', 0.0),
            sharpe_ratio=strategy_data.get('sharpe', 0.0),
            sortino_ratio=strategy_data.get('sortino', 0.0),
            profit_factor=strategy_data.get('profit_factor', 0.0),
            avg_profit=strategy_data.get('profit_mean', 0.0),
            median_profit=strategy_data.get('profit_median', 0.0),
            avg_duration=strategy_data.get('duration_avg', ""),
        )
    
    def _parse_text_output(self, output: str, strategy_name: str) -> BacktestResult:
        """
        Parse text output as fallback.
        
        Args:
            output: Command output text
            strategy_name: Name of strategy
            
        Returns:
            BacktestResult object with basic metrics
        """
        # Basic parsing of key metrics from text output
        # This is a simplified version - the JSON parsing is preferred
        
        result = BacktestResult(
            success=True,
            strategy_name=strategy_name
        )
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            
            # Try to extract some basic metrics
            if 'Total trades' in line or 'Trades' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.isdigit():
                            result.total_trades = int(part)
                            break
                except:
                    pass
            
            elif 'Win  %' in line or 'Winrate' in line:
                try:
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            result.win_rate = float(part.replace('%', '')) / 100
                            break
                except:
                    pass
            
            elif 'Profit' in line and '%' in line:
                try:
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            result.profit_percent = float(part.replace('%', ''))
                            break
                except:
                    pass
        
        logger.info(f"Parsed text output - Trades: {result.total_trades}, "
                   f"Win rate: {result.win_rate:.2%}, Profit: {result.profit_percent:.2f}%")
        
        return result
    
    def cleanup_strategy_file(self, strategy_name: str):
        """
        Remove strategy file after testing.
        
        Args:
            strategy_name: Name of strategy to cleanup
        """
        strategy_file = self.strategy_dir / f"{strategy_name}.py"
        try:
            if strategy_file.exists():
                strategy_file.unlink()
                logger.debug(f"Cleaned up strategy file: {strategy_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup strategy file: {e}")
