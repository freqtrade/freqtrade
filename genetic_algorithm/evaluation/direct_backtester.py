"""
Direct Backtesting Integration Module

Uses FreqTrade Python API directly with mocked exchange to avoid network calls.
"""

import json
import logging
import tempfile
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, PropertyMock, patch
from dataclasses import dataclass

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
        
        # Store in memory cache
        self.cache[cache_key] = result
        
        # Store in disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result.to_dict(), f)
            logger.debug(f"Cached result: {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to save cache file: {e}")



class DirectBacktester:
    """
    Direct backtester that uses FreqTrade's Python API with mocked exchange.
    
    This avoids network calls and allows offline backtesting.
    """
    
    # Default starting balances by stake currency type
    DEFAULT_BTC_BALANCE = 10  # 10 BTC (reasonable starting balance for BTC-denominated strategies)
    DEFAULT_STABLECOIN_BALANCE = 10000  # $10k for stablecoin-denominated strategies
    DEFAULT_EXCHANGE = 'binance'  # Default exchange for real pairs
    
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
        
        # DEBUG: Log what we loaded
        logger.info("=" * 80)
        logger.info("DirectBacktester initialized with config:")
        logger.info(f"  Pairs: {self.backtest_config.get('pairs')}")
        logger.info(f"  Timerange: {self.backtest_config.get('timerange')}")
        logger.info(f"  Stake amount: {self.backtest_config.get('stake_amount')}")
        logger.info(f"  Max open trades: {self.backtest_config.get('max_open_trades')}")
        logger.info(f"  Fee: {self.backtest_config.get('fee')}")
        logger.info(f"  Strategy directory: {self.strategy_dir}")
        logger.info("=" * 80)
        
        # Validate and auto-download data if enabled
        self._validate_and_download_data()
    
    def _validate_data_exists(self) -> Dict[str, list]:
        """
        Check if required data files exist for backtesting.
        
        Returns:
            Dictionary with 'missing' list of (pair, timeframe) tuples that are missing
        """
        pairs = self.backtest_config.get('pairs', [])
        timeframes = self.config.get('strategy_constraints', {}).get('timeframes', ['5m'])
        
        # Skip validation for test pairs
        if any('UNITTEST' in p for p in pairs):
            logger.debug("Using test pairs (UNITTEST), skipping data validation")
            return {'missing': []}
        
        # Determine exchange and data directory
        exchange = self.backtest_config.get('exchange', self.DEFAULT_EXCHANGE)
        datadir = self.freqtrade_root / "user_data" / "data" / exchange
        
        missing = []
        
        for pair in pairs:
            for timeframe in timeframes:
                # Convert pair format for filename (BTC/USDT -> BTC_USDT)
                pair_filename = pair.replace('/', '_')
                
                # Check for data file in various formats FreqTrade uses
                data_file_json = datadir / f"{pair_filename}-{timeframe}.json"
                data_file_feather = datadir / f"{pair_filename}-{timeframe}.feather"
                data_file_parquet = datadir / f"{pair_filename}-{timeframe}.parquet"
                
                if not (data_file_json.exists() or data_file_feather.exists() or data_file_parquet.exists()):
                    missing.append((pair, timeframe))
                    logger.debug(f"Missing data: {pair} {timeframe}")
        
        if missing:
            logger.info(f"Found {len(missing)} missing data file(s)")
        else:
            logger.info("All required data files exist")
        
        return {'missing': missing}
    
    def _auto_download_data(self, missing_data: list) -> bool:
        """
        Automatically download missing data files.
        
        Args:
            missing_data: List of (pair, timeframe) tuples to download
            
        Returns:
            True if download successful, False otherwise
        """
        if not missing_data:
            return True
        
        try:
            from freqtrade.resolvers import ExchangeResolver
            from freqtrade.data.history import refresh_backtest_ohlcv_data
            
            # Get unique pairs and timeframes
            pairs = list(set(p for p, _ in missing_data))
            timeframes = list(set(tf for _, tf in missing_data))
            
            exchange_name = self.backtest_config.get('exchange', self.DEFAULT_EXCHANGE)
            datadir = self.freqtrade_root / "user_data" / "data" / exchange_name
            datadir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Auto-downloading missing data for {len(pairs)} pair(s) and {len(timeframes)} timeframe(s)...")
            logger.info(f"  Pairs: {pairs}")
            logger.info(f"  Timeframes: {timeframes}")
            logger.info(f"  Exchange: {exchange_name}")
            
            # Create exchange configuration
            exchange_config = {
                'exchange': {
                    'name': exchange_name,
                    'key': '',
                    'secret': '',
                    'ccxt_config': {},
                    'ccxt_async_config': {},
                },
                'datadir': datadir,
                'user_data_dir': self.freqtrade_root / "user_data",
                'trading_mode': 'spot',
                'margin_mode': '',
                'stake_currency': 'USDT',
                'dry_run': True,
            }
            
            # Initialize exchange
            exchange = ExchangeResolver.load_exchange(exchange_config)
            
            # Download data
            refresh_backtest_ohlcv_data(
                exchange=exchange,
                pairs=pairs,
                timeframes=timeframes,
                datadir=datadir,
                timerange=None,  # Download all available
                erase=False
            )
            
            logger.info("✓ Data download completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to auto-download data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _validate_and_download_data(self):
        """
        Validate data exists and auto-download if enabled and missing.
        """
        # Check if auto-download is enabled
        auto_download = self.backtest_config.get('auto_download_data', True)
        
        # Validate data exists
        validation_result = self._validate_data_exists()
        missing = validation_result['missing']
        
        if not missing:
            logger.debug("Data validation passed - all required files exist")
            return
        
        # If data is missing
        if auto_download:
            logger.info(f"Missing {len(missing)} data file(s), attempting auto-download...")
            success = self._auto_download_data(missing)
            
            if not success:
                logger.warning("Auto-download failed, but continuing with existing data")
        else:
            # Auto-download disabled, show helpful error
            pairs = list(set(p for p, _ in missing))
            timeframes = list(set(tf for _, tf in missing))
            
            logger.warning("=" * 80)
            logger.warning("❌ Missing data files detected:")
            for pair, timeframe in missing:
                logger.warning(f"   • {pair} {timeframe}")
            logger.warning("")
            logger.warning("To fix this:")
            logger.warning("1. Enable auto-download in config: set 'backtesting.auto_download_data: true'")
            logger.warning("2. Or manually download:")
            logger.warning(f"   freqtrade download-data --pairs {' '.join(pairs)} "
                         f"--timeframes {' '.join(timeframes)} --days 90")
            logger.warning("=" * 80)
    
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
                logger.debug(f"Backtest results structure: {backtesting.results.keys() if backtesting.results else 'None'}")
                
                if not backtesting.results or 'strategy' not in backtesting.results:
                    logger.warning(f"No results available from backtest for {strategy_name}")
                    return BacktestResult(
                        success=True,
                        strategy_name=strategy_name,
                        total_trades=0,
                        error_message="No trades generated - strategy may be too restrictive"
                    )
                
                # Parse results from the strategy results
                strategy_results = backtesting.results['strategy'].get(strategy_name, {})
                logger.debug(f"Strategy results keys: {strategy_results.keys() if strategy_results else 'None'}")
                
                if not strategy_results:
                    logger.warning(f"Empty strategy results for {strategy_name}")
                    return BacktestResult(
                        success=True,
                        strategy_name=strategy_name,
                        total_trades=0,
                        error_message="No trades generated - check strategy conditions"
                    )
                
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
        Create FreqTrade config for backtesting from GA config.
        
        This is called automatically for each strategy backtest.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Configuration dictionary
        """
        # Read from GA config (stored in self.backtest_config)
        ga_cfg = self.backtest_config
        
        # Extract values from GA config
        pairs = ga_cfg.get('pairs', ['UNITTEST/BTC'])
        timerange = ga_cfg.get('timerange', '')
        stake_amount = ga_cfg.get('stake_amount', 0.05)
        max_open_trades = ga_cfg.get('max_open_trades', 3)
        fee = ga_cfg.get('fee', 0.001)
        
        # Determine stake currency from pairs
        # Extract quote currency from pairs (format: BASE/QUOTE)
        stake_currency = 'BTC'  # Default
        if pairs:
            # Use the quote currency from the first pair
            # All pairs should use the same quote currency for consistent backtesting
            first_pair = pairs[0]
            if '/' in first_pair:
                stake_currency = first_pair.split('/')[1]
        
        # Set reasonable starting balance based on stake currency
        if stake_currency == 'BTC':
            starting_balance = self.DEFAULT_BTC_BALANCE
        elif stake_currency in ('USDT', 'USD', 'USDC', 'BUSD'):
            starting_balance = self.DEFAULT_STABLECOIN_BALANCE
        else:
            starting_balance = self.DEFAULT_STABLECOIN_BALANCE  # Default to stablecoin balance for other currencies
            logger.warning(f"Unknown stake currency '{stake_currency}', using default balance of {self.DEFAULT_STABLECOIN_BALANCE}")
        
        # Determine exchange and data directory from pairs
        # Check if exchange is specified in GA config, otherwise use default
        exchange_name = self.backtest_config.get('exchange', self.DEFAULT_EXCHANGE)
        
        # For test pairs (UNITTEST/BTC), use test data directory
        if any('UNITTEST' in p for p in pairs):
            datadir = self.freqtrade_root / "tests" / "testdata"
        else:
            # Use user_data directory for real pairs
            datadir = self.freqtrade_root / "user_data" / "data" / exchange_name
        
        exportdir = self.freqtrade_root / "user_data" / "backtest_results"
        exportdir.mkdir(parents=True, exist_ok=True)
        
        # Build FreqTrade config
        config = {
            "strategy": strategy_name,
            "strategy_path": str(self.strategy_dir),
            "user_data_dir": self.freqtrade_root / "user_data",  # Path object
            "datadir": datadir,  # Path object - uses calculated directory based on pairs/exchange
            "exportdirectory": exportdir,  # Path object for storing results
            "runmode": "backtest",  # Required for FreqTrade
            
            # Critical config values from GA config
            "stake_currency": stake_currency,  # Calculated from pairs
            "stake_amount": stake_amount,  # From GA config
            "dry_run_wallet": starting_balance,  # Calculated based on stake currency
            "max_open_trades": max_open_trades,  # From GA config
            "fee": fee,  # From GA config
            
            # Timeframe will be overridden by strategy
            "timeframe": "5m",
            "timerange": timerange if timerange else None,  # From GA config
            
            # Exchange configuration
            "exchange": {
                "name": exchange_name,
                "pair_whitelist": pairs,  # From GA config
                "ccxt_config": {},
                "ccxt_async_config": {},
            },
            "pairlists": [{"method": "StaticPairList"}],
            
            "trading_mode": "spot",
            "margin_mode": "",
            "dry_run": True,
        }
        
        # Store original config reference (required for backtest storage)
        config["original_config"] = config.copy()
        
        # Log what we're using for debugging
        logger.info(f"Auto-generated backtest config for {strategy_name}:")
        logger.info(f"  Pairs: {pairs}")
        logger.info(f"  Timerange: {timerange}")
        logger.info(f"  Starting balance: {starting_balance} {stake_currency}")
        logger.info(f"  Stake amount: {stake_amount} {stake_currency}")
        logger.info(f"  Data directory: {datadir}")
        logger.info(f"  Max open trades: {max_open_trades}")
        logger.info(f"  Fee: {fee}")
        
        return config
    
    def _get_mock_markets(self) -> Dict[str, Any]:
        """
        Get mock markets data for offline backtesting.
        
        Dynamically builds market definitions from configured pairs to support
        both test pairs (UNITTEST/BTC) and real pairs (BTC/USDT, ETH/USDT, etc.).
        
        Returns:
            Mock markets dictionary
        """
        mock_markets = {}
        
        # Get pairs from config
        config_pairs = self.backtest_config.get('pairs', [])
        
        # Include common test pairs for backward compatibility
        test_pairs = [
            "UNITTEST/BTC", "ETH/BTC", "LTC/BTC", "XRP/BTC", "ADA/BTC",
            "DASH/BTC", "ETC/BTC", "XLM/BTC", "XMR/BTC", "NXT/BTC",
            "ZEC/BTC", "TRX/BTC"
        ]
        
        # Combine config pairs with test pairs (config pairs take precedence)
        all_pairs = list(set(config_pairs + test_pairs))
        
        for pair in all_pairs:
            if '/' not in pair:
                logger.warning(f"Invalid pair format: {pair} (expected BASE/QUOTE)")
                continue
                
            base, quote = pair.split("/", 1)  # Split only on first '/' to handle edge cases
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
        
        logger.debug(f"Created mock markets for {len(mock_markets)} pairs: {list(mock_markets.keys())}")
        
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
        # Log the raw stats for debugging profit issues
        logger.debug(f"Raw backtest stats for {strategy_name}: {stats}")
        
        # Extract metrics from stats - handle both percentage and absolute values
        profit_total = stats.get('profit_total', 0.0)
        profit_total_abs = stats.get('profit_total_abs', 0.0)
        
        # Convert profit_total to percentage if it's not already
        # FreqTrade typically returns profit_total as a ratio (e.g., 0.05 for 5%)
        # Heuristic: if absolute value < 10, assume it's a ratio; otherwise it's already a percentage
        # Note: This assumes profits won't exceed 1000% (ratio of 10), which is reasonable for backtests
        RATIO_TO_PERCENT_THRESHOLD = 10
        profit_percent = profit_total * 100 if abs(profit_total) < RATIO_TO_PERCENT_THRESHOLD else profit_total
        
        total_trades = stats.get('total_trades', 0)
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        
        # Calculate win rate if not provided
        win_rate = stats.get('winrate', 0.0)
        if win_rate == 0.0 and total_trades > 0:
            win_rate = wins / total_trades
        
        # Log with 4 decimal places to see small profits
        logger.info(f"Parsed {strategy_name}: profit={profit_percent:.4f}%, trades={total_trades}, win_rate={win_rate:.2%}")
        
        # Extract metrics from stats
        return BacktestResult(
            success=True,
            strategy_name=strategy_name,
            total_profit=profit_total_abs,
            profit_percent=profit_percent,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            max_drawdown=stats.get('max_drawdown', 0.0),
            max_drawdown_abs=stats.get('max_drawdown_abs', 0.0),
            sharpe_ratio=stats.get('sharpe', 0.0),
            sortino_ratio=stats.get('sortino', 0.0),
            profit_factor=stats.get('profit_factor', 0.0),
            avg_profit=stats.get('profit_mean', 0.0),
            median_profit=stats.get('profit_median', 0.0),
            avg_duration=stats.get('duration_avg', ""),
        )
