"""
Direct Backtesting Integration Module

Uses FreqTrade Python API directly with mocked exchange to avoid network calls.
"""

import json
import logging
import os
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
    
    # Trade visualization data (optional, only populated when requested)
    trades: Optional[list] = None  # List of trade dicts for visualization
    ohlcv_data: Optional[Dict[str, Any]] = None  # Dict of pair_timeframe -> DataFrame
    
    # Per-pair performance breakdown
    per_pair_profit: Optional[Dict[str, float]] = None  # pair -> profit percentage
    
    # Monthly return breakdown for stability analysis
    monthly_profits: Optional[list] = None  # List of monthly profit percentages
    
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
        
        # Log backtester initialization summary
        logger.info("[INIT] DirectBacktester initialized")
        logger.info(f"  Pairs: {self.backtest_config.get('pairs')}")
        logger.info(f"  Timerange: {self.backtest_config.get('timerange')}")
        logger.info(f"  Stake amount: {self.backtest_config.get('stake_amount')}")
        logger.info(f"  Max open trades: {self.backtest_config.get('max_open_trades')}")
        logger.info(f"  Fee: {self.backtest_config.get('fee')}")
        
        # Validate and auto-download data if enabled
        self._validate_and_download_data()
    
    def get_available_data_range(self) -> Optional[str]:
        """
        Detect the actual available data range from disk for the configured pairs.
        
        Checks all configured timeframes and uses the widest (union) range across
        them, then intersects with the configured timerange to produce an effective
        timerange that can be used for walk-forward window creation.
        
        Returns:
            Effective timerange string (YYYYMMDD-YYYYMMDD), or None if no data found.
        """
        pairs = self.backtest_config.get('pairs', [])
        timeframes = list(self.config.get('strategy_constraints', {}).get('timeframes', ['5m']))
        if not timeframes:
            timeframes = ['5m']
        
        # Include multi-timeframe timeframes when enabled (consistent with
        # _validate_data_exists) so that higher-TF data with wider history
        # contributes to the effective range instead of being ignored.
        multi_tf_config = self.config.get('multi_timeframe', {})
        if multi_tf_config.get('enabled', False):
            multi_tf_available = multi_tf_config.get('available', [])
            for tf in multi_tf_available:
                if tf not in timeframes:
                    timeframes.append(tf)
        
        # Skip for test pairs
        if any('UNITTEST' in p for p in pairs):
            return self.backtest_config.get('timerange', '')
        
        exchange_name = self.backtest_config.get('exchange', self.DEFAULT_EXCHANGE)
        datadir = self.freqtrade_root / "user_data" / "data" / exchange_name
        
        try:
            from freqtrade.data.history.datahandlers import get_datahandler
            from freqtrade.enums import CandleType
            from genetic_algorithm.utils.timerange import parse_timerange, format_date
            
            data_handler = get_datahandler(datadir)
            
            # Find the widest data range across ALL configured timeframes (union).
            # For each timeframe, compute the intersection across pairs (common range
            # that all pairs share), then take the union across timeframes so that
            # any timeframe with more history contributes to the effective range.
            overall_min = None
            overall_max = None
            
            for timeframe in timeframes:
                tf_min = None
                tf_max = None
                
                for pair in pairs:
                    min_date, max_date, length = data_handler.ohlcv_data_min_max(
                        pair, timeframe, CandleType.SPOT
                    )
                    
                    if length == 0:
                        logger.debug(f"No data on disk for {pair} {timeframe}")
                        continue
                    
                    # Make datetimes naive for comparison (ohlcv_data_min_max may return tz-aware)
                    min_dt = min_date.replace(tzinfo=None) if min_date.tzinfo else min_date
                    max_dt = max_date.replace(tzinfo=None) if max_date.tzinfo else max_date
                    
                    # Intersection across pairs for this timeframe
                    if tf_min is None or min_dt > tf_min:
                        tf_min = min_dt
                    if tf_max is None or max_dt < tf_max:
                        tf_max = max_dt
                
                if tf_min is not None and tf_max is not None and tf_min < tf_max:
                    logger.debug(
                        f"Data range for timeframe {timeframe}: "
                        f"{format_date(tf_min)}-{format_date(tf_max)}"
                    )
                    # Union across timeframes: keep the earliest start and latest end
                    if overall_min is None or tf_min < overall_min:
                        overall_min = tf_min
                    if overall_max is None or tf_max > overall_max:
                        overall_max = tf_max
                else:
                    logger.warning(f"No data on disk for timeframe {timeframe} across all pairs")
            
            if overall_min is None or overall_max is None:
                logger.warning("Could not determine data range - no data files found")
                return None
            
            logger.info(f"Actual data range on disk: {format_date(overall_min)}-{format_date(overall_max)}")
            
            # Intersect with configured timerange
            config_timerange = self.backtest_config.get('timerange', '')
            if config_timerange:
                config_start, config_end = parse_timerange(config_timerange)
                
                effective_start = max(overall_min, config_start)
                effective_end = min(overall_max, config_end)
                
                if effective_start >= effective_end:
                    logger.error(
                        f"No overlap between config timerange ({config_timerange}) and "
                        f"available data ({format_date(overall_min)}-{format_date(overall_max)})")
                    return None
                
                effective_timerange = f"{format_date(effective_start)}-{format_date(effective_end)}"
            else:
                effective_timerange = f"{format_date(overall_min)}-{format_date(overall_max)}"
            
            logger.info(f"Effective data range: {effective_timerange}")
            return effective_timerange
            
        except Exception as e:
            logger.warning(f"Failed to detect data range: {e}. Using config timerange as fallback.")
            return self.backtest_config.get('timerange', '')
    
    def _validate_data_exists(self) -> Dict[str, list]:
        """
        Check if required data files exist for backtesting.
        
        Returns:
            Dictionary with 'missing' list of (pair, timeframe) tuples that are missing
        """
        pairs = self.backtest_config.get('pairs', [])
        # Use same config path as StrategyGenerator to ensure we check all timeframes
        # that might be used by generated strategies
        timeframes = list(self.config.get('strategy', {}).get('timeframes', ['5m', '15m', '1h']))
        
        # Include multi-timeframe timeframes when enabled
        multi_tf_config = self.config.get('multi_timeframe', {})
        if multi_tf_config.get('enabled', False):
            multi_tf_available = multi_tf_config.get('available', [])
            for tf in multi_tf_available:
                if tf not in timeframes:
                    timeframes.append(tf)
            logger.info(f"Multi-timeframe enabled, validating timeframes: {timeframes}")
        
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
            from freqtrade.enums import CandleType, TradingMode
            
            # Get unique pairs and timeframes
            pairs = list(set(p for p, _ in missing_data))
            timeframes = list(set(tf for _, tf in missing_data))
            
            exchange_name = self.backtest_config.get('exchange', self.DEFAULT_EXCHANGE)
            datadir = self.freqtrade_root / "user_data" / "data" / exchange_name
            datadir.mkdir(parents=True, exist_ok=True)
            
            # Use the configured timerange so the download covers the full requested
            # history instead of only what the exchange provides by default.
            configured_timerange = self.backtest_config.get('timerange', None)
            
            logger.info(f"Auto-downloading missing data for {len(pairs)} pair(s) and {len(timeframes)} timeframe(s)...")
            logger.info(f"  Pairs: {pairs}")
            logger.info(f"  Timeframes: {timeframes}")
            logger.info(f"  Exchange: {exchange_name}")
            if configured_timerange:
                logger.info(f"  Timerange: {configured_timerange}")
            
            # Determine stake currency from configured pairs
            stake_currency = 'USDT'
            config_pairs = self.backtest_config.get('pairs', [])
            if config_pairs and '/' in config_pairs[0]:
                stake_currency = config_pairs[0].split('/')[1]
            else:
                logger.warning(f"Could not derive stake currency from pairs {config_pairs}, "
                             f"defaulting to '{stake_currency}'")
            
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
                'stake_currency': stake_currency,
                'dry_run': True,
                'runmode': 'other',
                'entry_pricing': {
                'price_side': 'same',
                'use_order_book': False,
                'order_book_top': 1,
                },
                'exit_pricing': {
                    'price_side': 'same',
                    'use_order_book': False,
                    'order_book_top': 1,
                },
            }
            
            # Initialize exchange
            exchange = ExchangeResolver.load_exchange(exchange_config)
            
            # Convert the configured timerange string to a TimeRange object so the
            # download covers the full requested history rather than just the default
            # number of candles the exchange returns without an explicit range.
            from freqtrade.configuration import TimeRange as FTTimeRange
            ft_timerange = (
                FTTimeRange.parse_timerange(configured_timerange)
                if configured_timerange else None
            )
            
            # Download data
            refresh_backtest_ohlcv_data(
                exchange=exchange,
                pairs=pairs,
                timeframes=timeframes,
                datadir=datadir,
                timerange=ft_timerange,
                erase=False,
                trading_mode=TradingMode.SPOT,
                candle_types=[CandleType.SPOT],
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
                         max_retries: int = 2,
                         strategy_max_open_trades: Optional[int] = None) -> BacktestResult:
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
                logger.debug(f"Using cached result for {strategy_name}")
                return cached_result
        
        # Try multiple times in case of transient errors
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry {attempt}/{max_retries} for {strategy_name}")
                    time.sleep(1)
                
                result = self._run_backtest_direct(strategy_code, strategy_name, strategy_max_open_trades)
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
    
    def _run_backtest_direct(self, strategy_code: str, strategy_name: str, strategy_max_open_trades: Optional[int] = None, collect_trades: bool = False) -> BacktestResult:
        """
        Run backtest using FreqTrade Python API with mocked exchange.
        
        Args:
            strategy_code: Python code for strategy
            strategy_name: Name of the strategy
            collect_trades: Whether to collect detailed trade data for visualization
            
        Returns:
            BacktestResult object
        """
        # Write strategy to file atomically to prevent race conditions
        # when multiple workers write to the same strategy file
        strategy_file = self.strategy_dir / f"{strategy_name}.py"
        try:
            # Write to a temp file first, then atomically rename
            fd, tmp_path = tempfile.mkstemp(
                suffix='.py', dir=str(self.strategy_dir), prefix=f".{strategy_name}_"
            )
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(strategy_code)
                os.replace(tmp_path, str(strategy_file))  # atomic on POSIX
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            # Invalidate any cached .pyc for this module
            cache_dir = self.strategy_dir / '__pycache__'
            if cache_dir.exists():
                for pyc in cache_dir.glob(f"{strategy_name}.*.pyc"):
                    try:
                        pyc.unlink()
                    except OSError:
                        pass
            logger.debug(f"Wrote strategy file (atomic): {strategy_file}")
        except Exception as e:
            logger.error(f"Failed to write strategy file: {e}")
            return BacktestResult(
                success=False,
                strategy_name=strategy_name,
                error_message=f"Failed to write strategy file: {e}"
            )
        
        # Validate generated Python syntax before backtesting
        try:
            compile(strategy_code, str(strategy_file), 'exec')
        except SyntaxError as e:
            logger.error(f"Generated strategy has syntax error at line {e.lineno}: {e.msg}")
            return BacktestResult(
                success=False,
                strategy_name=strategy_name,
                error_message=f"Generated strategy syntax error at line {e.lineno}: {e.msg}"
            )
        
        # Deep validation: actually import the module to catch runtime errors
        # (e.g., missing talib functions, NameError, ImportError)
        try:
            import importlib
            import importlib.util
            importlib.invalidate_caches()  # ensure fresh file is picked up
            spec = importlib.util.spec_from_file_location(strategy_name, str(strategy_file))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # Verify the expected class exists in the module
                if not hasattr(module, strategy_name):
                    logger.error(f"Strategy module loaded but class '{strategy_name}' not found")
                    return BacktestResult(
                        success=False,
                        strategy_name=strategy_name,
                        error_message=f"Strategy class '{strategy_name}' not found in module"
                    )
        except (ImportError, ModuleNotFoundError, NameError, AttributeError) as e:
            logger.error(f"Strategy '{strategy_name}' has runtime import error: {e}")
            return BacktestResult(
                success=False,
                strategy_name=strategy_name,
                error_message=f"Strategy runtime import error: {e}"
            )
        except Exception as e:
            # Don't block on unexpected errors during validation — let FreqTrade try
            logger.warning(f"Strategy pre-import validation warning for '{strategy_name}': {e}")
        
        try:
            # Import FreqTrade modules
            from freqtrade.configuration import Configuration
            from freqtrade.optimize.backtesting import Backtesting
            from freqtrade.exchange.exchange import Exchange
            import io
            import sys
            
            # Create configuration
            config_dict = self._create_backtest_config(strategy_name, strategy_max_open_trades)
            
            # Suppress FreqTrade's verbose output by redirecting stdout
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            # Also suppress FreqTrade's verbose logging during backtesting
            freqtrade_loggers = [
                'freqtrade.exchange.exchange',
                'freqtrade.resolvers',
                'freqtrade.resolvers.strategy_resolver',
                'freqtrade.resolvers.exchange_resolver',
                'freqtrade.resolvers.iresolver',
                'freqtrade.configuration',
                'freqtrade.configuration.config_validation',
                'freqtrade.optimize.backtesting',
                'freqtrade.data.dataprovider',
                'freqtrade.data.history',
                'freqtrade.strategy',
                'freqtrade.strategy.hyper',
                'freqtrade.misc',
            ]
            old_log_levels = {}
            for logger_name in freqtrade_loggers:
                ft_logger = logging.getLogger(logger_name)
                old_log_levels[logger_name] = ft_logger.level
                ft_logger.setLevel(logging.WARNING)
            
            try:
                # Mock the exchange to avoid network calls
                with patch.object(Exchange, '_load_async_markets', return_value={}), \
                     patch.object(Exchange, 'markets', PropertyMock(return_value=self._get_mock_markets())), \
                     patch.object(Exchange, 'validate_config', MagicMock()), \
                     patch.object(Exchange, 'validate_timeframes', MagicMock()), \
                     patch.object(Exchange, '_init_ccxt', MagicMock()), \
                     patch.object(Exchange, 'get_fee', return_value=config_dict.get('exchange', {}).get('fee', 0.001)), \
                     patch.object(Exchange, 'precisionMode', PropertyMock(return_value=2)), \
                     patch.object(Exchange, 'precision_mode_price', PropertyMock(return_value=2)), \
                     patch.object(Exchange, 'timeframes', PropertyMock(return_value=["1m", "5m", "15m", "1h", "1d"])), \
                     patch.object(Exchange, 'get_min_pair_stake_amount', return_value=0.0), \
                     patch.object(Exchange, 'get_max_pair_stake_amount', return_value=float('inf')):
                    
                    # Initialize backtesting
                    backtesting = Backtesting(config_dict)
                    
                    # Skip prior backtest loading - GA generates unique strategies
                    # and the parallel workers corrupt each other's .meta files
                    backtesting.load_prior_backtest = lambda: None
                    
                    # Run backtest
                    backtesting.start()
                    
                    # Get results from the backtest results
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
                    
                    # Fallback: if the expected key is missing, use the first available strategy key
                    if not strategy_results and backtesting.results['strategy']:
                        available_keys = list(backtesting.results['strategy'].keys())
                        logger.warning(
                            f"Strategy key '{strategy_name}' not found in results. "
                            f"Available keys: {available_keys}. Using first key."
                        )
                        strategy_results = backtesting.results['strategy'][available_keys[0]]
                    
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
                    
                    # Extract per-pair performance breakdown
                    try:
                        trades_df = strategy_results.get('trades', None)
                        if trades_df is not None and hasattr(trades_df, 'groupby') and len(trades_df) > 0:
                            per_pair = {}
                            for pair, group in trades_df.groupby('pair'):
                                # profit_ratio is per-trade profit as a ratio
                                pair_profit = group['profit_ratio'].sum() * 100 if 'profit_ratio' in group.columns else 0.0
                                per_pair[pair] = pair_profit
                            result.per_pair_profit = per_pair
                            logger.debug(f"Per-pair profits: {per_pair}")
                    except Exception as e:
                        logger.debug(f"Could not extract per-pair profits: {e}")
                    
                    # Extract monthly profit breakdown for stability analysis
                    try:
                        trades_df = strategy_results.get('trades', None)
                        if trades_df is not None and hasattr(trades_df, 'groupby') and len(trades_df) > 0:
                            # Use close_date for monthly grouping
                            date_col = None
                            for col in ['close_date', 'sell_date', 'exit_date']:
                                if col in trades_df.columns:
                                    date_col = col
                                    break
                            
                            if date_col and 'profit_ratio' in trades_df.columns:
                                import pandas as pd
                                trades_copy = trades_df.copy()
                                trades_copy[date_col] = pd.to_datetime(trades_copy[date_col])
                                monthly = trades_copy.set_index(date_col).resample('ME')['profit_ratio'].sum() * 100
                                result.monthly_profits = monthly.tolist()
                                logger.debug(f"Monthly profits: {result.monthly_profits}")
                    except Exception as e:
                        logger.debug(f"Could not extract monthly profits: {e}")
                    
                    # Collect detailed trade data for visualization if requested
                    if collect_trades:
                        trades_list = []
                        ohlcv_dict = {}
                        
                        try:
                            # Extract trades from backtest results
                            # FreqTrade stores trades as a DataFrame in strategy_results
                            trades_df = strategy_results.get('trades', None)
                            if trades_df is not None and hasattr(trades_df, 'to_dict'):
                                # Convert DataFrame to list of dicts
                                trades_list = trades_df.to_dict('records')
                                logger.debug(f"Collected {len(trades_list)} trades for visualization")
                            elif isinstance(strategy_results.get('trades'), list):
                                trades_list = strategy_results['trades']
                            
                            # Try to get OHLCV data from backtesting object
                            # FreqTrade stores data in different attributes depending on version
                            ohlcv_data_source = None
                            if hasattr(backtesting, 'processed') and backtesting.processed:
                                ohlcv_data_source = backtesting.processed
                            elif hasattr(backtesting, 'data') and backtesting.data:
                                ohlcv_data_source = backtesting.data
                            elif hasattr(backtesting, '_data') and backtesting._data:
                                ohlcv_data_source = backtesting._data
                            
                            if ohlcv_data_source:
                                for key, df in ohlcv_data_source.items():
                                    if hasattr(df, 'copy'):
                                        ohlcv_dict[key] = df.copy()
                                logger.debug(f"Collected OHLCV data for {len(ohlcv_dict)} pair(s)")
                            else:
                                # Fallback: load OHLCV data directly from disk
                                logger.debug("No OHLCV data in backtesting object, loading from disk...")
                                ohlcv_dict = self._load_ohlcv_for_pairs(config_dict)
                            
                        except Exception as e:
                            logger.warning(f"Could not extract trade details: {e}")
                        
                        result.trades = trades_list
                        result.ohlcv_data = ohlcv_dict
                    
                    return result
            finally:
                # Restore stdout
                sys.stdout = old_stdout
                # Restore logging levels
                for logger_name, level in old_log_levels.items():
                    logging.getLogger(logger_name).setLevel(level)
                
        except Exception as e:
            logger.error(f"Backtest execution error: {e}")
            import traceback
            traceback.print_exc()
            return BacktestResult(
                success=False,
                strategy_name=strategy_name,
                error_message=f"Execution error: {str(e)}"
            )
    
    def _create_backtest_config(self, strategy_name: str, strategy_max_open_trades: Optional[int] = None) -> Dict[str, Any]:
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
        # Use strategy-specific max_open_trades if provided, otherwise use global config
        if strategy_max_open_trades is not None:
            max_open_trades = strategy_max_open_trades
        else:
            max_open_trades = ga_cfg.get('max_open_trades', 3)
        fee = ga_cfg.get('fee', 0.001)
        
        # Add slippage on top of exchange fee for realistic cost modeling
        # Slippage accounts for spread, market impact, and execution delays
        slippage_pct = ga_cfg.get('slippage_pct', 0.0)
        if slippage_pct > 0:
            fee = fee + slippage_pct
            logger.debug(f"Fee adjusted with slippage: base={ga_cfg.get('fee', 0.001)}, "
                        f"slippage={slippage_pct}, total={fee}")
        
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
        
        # Convert fractional stake_amount to actual currency amount
        # If stake_amount is between 0 and 1, treat it as a fraction of starting balance
        # e.g., 0.15 means 15% of wallet = $1500 for a $10,000 wallet
        # Values >= 1 are treated as literal amounts (e.g., 100 = $100 per trade)
        if isinstance(stake_amount, (int, float)) and 0 < stake_amount < 1:
            actual_stake = stake_amount * starting_balance
            logger.info(f"Stake amount {stake_amount} interpreted as {stake_amount:.0%} of "
                        f"{starting_balance} {stake_currency} = {actual_stake:.2f} {stake_currency} per trade")
            stake_amount = actual_stake
        
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
            
            # Data format - CRITICAL: must match the format of data files on disk
            "dataformat_ohlcv": "feather",  # Use feather format for faster data loading
            "dataformat_trades": "feather",
            
            # Critical config values from GA config
            "stake_currency": stake_currency,  # Calculated from pairs
            "stake_amount": stake_amount,  # From GA config
            "dry_run_wallet": starting_balance,  # Calculated based on stake currency
            "max_open_trades": max_open_trades,  # From GA config
            "fee": fee,  # From GA config
            
            # Allow multiple trades per pair (enables true max_open_trades)
            "position_stacking": True,  # Required to open more than 1 trade per pair
            
            # Don't set timeframe here - let the strategy define it
            # "timeframe": "5m",  # Removed - strategy's timeframe will be used
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
        
        # Log at debug level to avoid spam - full config is shown at initialization
        logger.debug(f"Backtest config for {strategy_name}: pairs={pairs}, timerange={timerange}, max_open_trades={max_open_trades}")
        
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
                
            base, quote = pair.split("/", 1)  # maxsplit=1 to handle pairs like 'BTC/USDT'
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
        
        # Log raw profit values for debugging
        logger.debug(f"Raw profit for {strategy_name}: profit_total={profit_total}, "
                     f"profit_total_abs={profit_total_abs}, "
                     f"max_drawdown_account={stats.get('max_drawdown_account', 'N/A')}, "
                     f"max_drawdown_abs={stats.get('max_drawdown_abs', 'N/A')}")
        
        # Convert profit_total to percentage
        # FreqTrade always returns profit_total as a ratio (e.g., 0.05 = 5%)
        # and profit_total_pct as percentage. We use profit_total * 100 for consistency.
        # See freqtrade/optimize/optimize_reports/optimize_reports.py:
        #   profit_total = result["profit_abs"].sum() / starting_balance  (ratio)
        #   "profit_total_pct": round(profit_total * 100.0, 2)  (percentage)
        profit_percent = stats.get('profit_total_pct', profit_total * 100)
        
        total_trades = stats.get('total_trades', 0)
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        
        # Calculate win rate if not provided
        win_rate = stats.get('winrate', 0.0)
        if win_rate == 0.0 and total_trades > 0:
            win_rate = wins / total_trades
        
        # Log key results at debug level (summary logged elsewhere)
        logger.debug(f"Parsed {strategy_name}: profit={profit_percent:.4f}%, trades={total_trades}, win_rate={win_rate:.2%}")
        
        # Extract metrics from stats
        # Note: FreqTrade uses 'max_drawdown_account' for percentage drawdown (as ratio, e.g., 0.15 = 15%)
        max_drawdown = stats.get('max_drawdown_account', stats.get('max_drawdown', 0.0))
        
        return BacktestResult(
            success=True,
            strategy_name=strategy_name,
            total_profit=profit_total_abs,
            profit_percent=profit_percent,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            max_drawdown_abs=stats.get('max_drawdown_abs', 0.0),
            sharpe_ratio=stats.get('sharpe', 0.0),
            sortino_ratio=stats.get('sortino', 0.0),
            profit_factor=stats.get('profit_factor', 0.0),
            avg_profit=stats.get('profit_mean', 0.0),
            median_profit=stats.get('profit_median', 0.0),
            # FreqTrade uses 'holding_avg' (timedelta) at strategy level, not 'duration_avg'
            avg_duration=str(stats.get('holding_avg', '')) if stats.get('holding_avg') else stats.get('duration_avg', ""),
        )

    def backtest_strategy_with_trades(
        self,
        strategy_code: str,
        strategy_name: str,
        max_retries: int = 2,
        strategy_max_open_trades: Optional[int] = None
    ) -> BacktestResult:
        """
        Run backtest and collect detailed trade data for visualization.
        
        This method is similar to backtest_strategy() but also collects
        the individual trades and OHLCV data needed for trade visualization.
        
        Args:
            strategy_code: Python code for strategy
            strategy_name: Name of the strategy
            max_retries: Maximum number of retries on failure
            strategy_max_open_trades: Optional max open trades override
            
        Returns:
            BacktestResult object with trades and ohlcv_data populated
        """
        start_time = time.time()
        
        # Note: We don't use cache for trade visualization as we need fresh data
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.debug(f"Retry {attempt}/{max_retries} for {strategy_name}")
                    time.sleep(1)
                
                # Run backtest with trade collection enabled
                result = self._run_backtest_direct(
                    strategy_code, 
                    strategy_name, 
                    strategy_max_open_trades,
                    collect_trades=True
                )
                result.execution_time = time.time() - start_time
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Backtest with trades attempt {attempt + 1} failed: {e}")
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

    def _load_ohlcv_for_pairs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load OHLCV data from disk for trade visualization.
        
        Args:
            config: Backtest configuration with pairs, datadir, etc.
            
        Returns:
            Dictionary mapping "pair_timeframe" to DataFrame
        """
        ohlcv_dict = {}
        
        try:
            from freqtrade.data.history import load_pair_history
            from freqtrade.enums import CandleType
            
            pairs = config.get('exchange', {}).get('pair_whitelist', [])
            datadir = config.get('datadir')
            
            # Get the base timeframe from strategy config
            # Use 5m as default since it's the most common
            timeframe = '5m'
            
            for pair in pairs:
                try:
                    df = load_pair_history(
                        pair=pair,
                        timeframe=timeframe,
                        datadir=datadir,
                        candle_type=CandleType.SPOT,
                        timerange=None  # Load all data
                    )
                    
                    if df is not None and len(df) > 0:
                        key = f"{pair.replace('/', '_')}_{timeframe}"
                        ohlcv_dict[key] = df
                        logger.debug(f"Loaded {len(df)} candles for {pair} {timeframe}")
                        
                except Exception as e:
                    logger.warning(f"Could not load OHLCV data for {pair}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to load OHLCV data from disk: {e}")
        
        return ohlcv_dict
