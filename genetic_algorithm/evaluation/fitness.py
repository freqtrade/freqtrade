"""
Fitness Evaluator

Evaluates the fitness of trading strategies through backtesting
and calculating performance metrics. Supports both standard backtesting
and walk-forward optimization for preventing overfitting.
"""

import logging
import hashlib
from collections import OrderedDict
from typing import Tuple, Dict, Any, List, Optional

from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.evaluation.direct_backtester import DirectBacktester, BacktestResult
from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.utils.timerange import (
    create_walk_forward_windows,
    validate_walk_forward_config,
    aggregate_validation_scores,
    parse_timerange,
    format_date,
    TimeWindow
)

logger = logging.getLogger(__name__)


class FitnessEvaluator:
    """
    Evaluates strategy fitness through backtesting.
    
    Responsible for:
    - Running FreqTrade backtests
    - Parsing backtest results
    - Calculating fitness score
    - Computing performance metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fitness evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.fitness_weights = config.get('fitness_weights', {})
        self.fitness_penalties = config.get('fitness_penalties', {})
        self.backtest_config = config.get('backtesting', {})
        self.walk_forward_config = config.get('walk_forward', {})
        
        # Fitness bounds for clamping extreme values
        fitness_bounds = config.get('fitness_bounds', {})
        self.profit_min = fitness_bounds.get('profit_min', -50)
        self.profit_max = fitness_bounds.get('profit_max', 200)
        self.sharpe_min = fitness_bounds.get('sharpe_min', -5)
        self.sharpe_max = fitness_bounds.get('sharpe_max', 10)
        self.sortino_min = fitness_bounds.get('sortino_min', -5)
        self.sortino_max = fitness_bounds.get('sortino_max', 12)
        self.profit_factor_max = fitness_bounds.get('profit_factor_max', 10)
        
        # Trade frequency thresholds
        tf_config = config.get('trade_frequency_thresholds', {})
        self.tf_very_few = tf_config.get('very_few', 5)
        self.tf_few = tf_config.get('few', 10)
        self.tf_ideal_min = tf_config.get('ideal_min', 10)
        self.tf_ideal_max = tf_config.get('ideal_max', 50)
        self.tf_moderate_excess = tf_config.get('moderate_excess', 100)
        
        # Validate walk-forward config if enabled
        if self.walk_forward_config.get('enabled', False):
            validate_walk_forward_config(self.walk_forward_config)
            logger.info("Walk-forward optimization enabled")
        
        # Initialize direct backtester and strategy generator
        self.backtester = DirectBacktester(config)
        self.strategy_generator = StrategyGenerator(config)
        
        # Walk-forward cache: (strategy_hash, window_index) -> BacktestResult
        # Uses OrderedDict for LRU eviction — most-recently-used entries at the end.
        self._wf_cache: OrderedDict[Tuple[str, int], BacktestResult] = OrderedDict()
        self._wf_cache_hits = 0
        self._wf_cache_misses = 0
        self._wf_cache_max_size = self.walk_forward_config.get('cache_max_size', 10000)
        
        # Deflated Sharpe Ratio tracker (anti-overfitting)
        from genetic_algorithm.evaluation.deflated_sharpe import DSRTracker
        self._dsr_tracker = DSRTracker(config)
    
    def evaluate(self, strategy_gene: StrategyGene, strategy_name: str = None) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a strategy's fitness through backtesting.
        
        If walk-forward optimization is enabled in config, uses walk-forward validation.
        Otherwise, uses standard single-period backtesting.
        
        Args:
            strategy_gene: Strategy to evaluate
            strategy_name: Optional name for the strategy (auto-generated if not provided)
            
        Returns:
            Tuple of (fitness_score, metrics_dict)
        """
        # Check if walk-forward is enabled
        if self.walk_forward_config.get('enabled', False):
            return self.evaluate_walk_forward(strategy_gene, strategy_name)
        
        # Standard single-period evaluation
        return self._evaluate_standard(strategy_gene, strategy_name)
    
    def _evaluate_standard(self, strategy_gene: StrategyGene, strategy_name: str = None) -> Tuple[float, Dict[str, float]]:
        """
        Standard single-period evaluation (original evaluate logic).
        
        Args:
            strategy_gene: Strategy to evaluate
            strategy_name: Optional name for the strategy
            
        Returns:
            Tuple of (fitness_score, metrics_dict)
        """
        try:
            # Generate strategy code (strategy name is auto-generated from gene info)
            strategy_code = self.strategy_generator.generate_strategy_code(strategy_gene)
            
            # Use generated name from the gene for consistency
            generated_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
            
            # Run backtest with strategy-specific max_open_trades
            backtest_result = self.backtester.backtest_strategy(
                strategy_code, 
                generated_name,
                strategy_max_open_trades=strategy_gene.max_open_trades
            )
            
            # Check if backtest was successful
            if not backtest_result.success:
                logger.warning(f"Backtest failed for {generated_name}: {backtest_result.error_message}")
                # Return very low fitness for failed strategies
                return 0.0, {
                    'profit': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 1.0,
                    'win_rate': 0.0,
                    'num_trades': 0,
                    'complexity': strategy_gene.calculate_complexity(),
                    'error': backtest_result.error_message
                }
            
            # Convert backtest result to metrics dictionary
            metrics = self._backtest_result_to_metrics(backtest_result)
            
            # Add complexity to metrics
            metrics['complexity'] = strategy_gene.calculate_complexity()
            
            # Calculate fitness (includes complexity penalty)
            fitness = self.calculate_fitness(metrics, strategy_gene)
            
            # Log at debug level - summary is logged by evolution.py
            logger.debug(f"{generated_name}: fitness={fitness:.4f}, profit={metrics['profit']:.2f}%, trades={metrics['num_trades']}")
            
            return fitness, metrics
            
        except Exception as e:
            generated_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
            logger.error(f"Error evaluating strategy {generated_name}: {e}", exc_info=True)
            # Return zero fitness on error
            return 0.0, {
                'profit': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'complexity': strategy_gene.calculate_complexity(),
                'error': str(e)
            }
    
    def _auto_adjust_walk_forward_params(
        self, 
        timerange: str
    ) -> Optional[Dict[str, int]]:
        """
        Auto-adjust walk-forward parameters to fit available data range.
        
        When the available data is shorter than the configured train_days + validation_days,
        this method reduces the parameters proportionally so that at least one window can
        be created.
        
        Args:
            timerange: Effective timerange string (YYYYMMDD-YYYYMMDD)
            
        Returns:
            Adjusted parameters dict with 'train_days', 'validation_days', 'step_days',
            or None if no valid adjustment is possible (data too short).
        """
        from genetic_algorithm.utils.timerange import parse_timerange
        
        start, end = parse_timerange(timerange)
        available_days = (end - start).days
        
        train_days = self.walk_forward_config['train_days']
        validation_days = self.walk_forward_config['validation_days']
        step_days = self.walk_forward_config['step_days']
        required_days = train_days + validation_days
        
        if available_days >= required_days:
            return {
                'train_days': train_days,
                'validation_days': validation_days,
                'step_days': step_days,
            }
        
        # Need to shrink parameters to fit.
        # Keep the train/validation ratio the same, but scale down.
        # Reserve at least 5 days for validation and 7 days for training.
        MIN_TRAIN_DAYS = 7
        MIN_VAL_DAYS = 5
        min_total = MIN_TRAIN_DAYS + MIN_VAL_DAYS
        
        if available_days < min_total:
            logger.warning(
                f"Available data ({available_days} days) is too short for walk-forward "
                f"(minimum {min_total} days needed). Cannot auto-adjust.")
            return None
        
        # Scale proportionally, ensuring both minimums are met
        ratio = train_days / required_days
        adjusted_train = min(
            available_days - MIN_VAL_DAYS,
            max(MIN_TRAIN_DAYS, int(available_days * ratio))
        )
        adjusted_val = max(MIN_VAL_DAYS, available_days - adjusted_train)
        
        # Make sure they actually fit
        if adjusted_train + adjusted_val > available_days:
            adjusted_train = available_days - adjusted_val
        
        if adjusted_train < MIN_TRAIN_DAYS:
            return None
        
        adjusted_step = max(1, adjusted_val)
        
        logger.warning(
            f"⚠️  Walk-forward auto-adjusted: available data is only {available_days} days "
            f"(need {required_days} for configured train={train_days}+val={validation_days}). "
            f"Adjusted to train={adjusted_train}, val={adjusted_val}, step={adjusted_step}.")
        
        return {
            'train_days': adjusted_train,
            'validation_days': adjusted_val,
            'step_days': adjusted_step,
        }
    
    def evaluate_walk_forward(
        self, 
        strategy_gene: StrategyGene, 
        strategy_name: str = None,
        progress_callback: Optional[callable] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate strategy using walk-forward optimization.
        
        Trains on multiple windows and validates on out-of-sample data.
        Final fitness is based on aggregated validation performance, not training performance.
        
        If the available data is too short for the configured walk-forward parameters,
        the parameters are auto-adjusted. If even that is not possible, it falls back
        to standard single-period evaluation with a warning.
        
        Args:
            strategy_gene: Strategy to evaluate
            strategy_name: Optional name for the strategy
            progress_callback: Optional callback(window_idx, total_windows) for progress tracking
            
        Returns:
            Tuple of (aggregated_validation_fitness, aggregated_metrics)
        """
        try:
            # Generate strategy code once (reused for all windows)
            strategy_code = self.strategy_generator.generate_strategy_code(strategy_gene)
            generated_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
            
            # Create a hash for caching (using SHA-256 for robustness)
            strategy_hash = hashlib.sha256(strategy_code.encode()).hexdigest()[:16]
            
            # Create walk-forward windows using actual data range
            original_timerange = self.backtest_config.get('timerange', '')
            
            # Detect actual data range to avoid creating windows outside available data
            effective_timerange = self.backtester.get_available_data_range()
            if effective_timerange and effective_timerange != original_timerange:
                logger.info(f"Adjusted timerange from config ({original_timerange}) "
                           f"to effective data range ({effective_timerange})")
            timerange_for_windows = effective_timerange or original_timerange
            
            # Auto-adjust walk-forward parameters if data is too short
            adjusted = self._auto_adjust_walk_forward_params(timerange_for_windows)
            if adjusted is None:
                logger.warning(
                    f"⚠️  Walk-forward optimization disabled for this run: insufficient data. "
                    f"Falling back to standard single-period evaluation. "
                    f"To use walk-forward, download more historical data.")
                return self._evaluate_standard(strategy_gene, strategy_name)
            
            wf_train_days = adjusted['train_days']
            wf_val_days = adjusted['validation_days']
            wf_step_days = adjusted['step_days']
            
            try:
                windows = create_walk_forward_windows(
                    timerange=timerange_for_windows,
                    train_days=wf_train_days,
                    validation_days=wf_val_days,
                    step_days=wf_step_days,
                    mode=self.walk_forward_config.get('mode', 'rolling'),
                    embargo_days=self.walk_forward_config.get('embargo_days', 0),
                    max_windows=self.walk_forward_config.get('max_windows', None)
                )
            except ValueError as e:
                logger.warning(
                    f"⚠️  Walk-forward window creation failed even after auto-adjust "
                    f"(train={wf_train_days}, val={wf_val_days}, step={wf_step_days}, "
                    f"timerange={timerange_for_windows}): {e}. "
                    f"Falling back to standard single-period evaluation.")
                return self._evaluate_standard(strategy_gene, strategy_name)
            
            logger.info(f"Evaluating {generated_name} with {len(windows)} walk-forward windows")
            
            validation_fitness_scores = []
            train_fitness_scores = []  # For comparison/debugging
            all_window_metrics = []
            failed_windows = 0  # Track failed windows separately
            consecutive_zero_trade = 0  # Early exit after repeated zero-trade windows
            max_consecutive_zero = self.walk_forward_config.get('max_consecutive_zero_windows', 3)
            
            for window in windows:
                if progress_callback:
                    progress_callback(window.window_index, len(windows))
                
                # Check cache first
                cache_key = (strategy_hash, window.window_index)
                if cache_key in self._wf_cache:
                    self._wf_cache_hits += 1
                    # Promote to end for LRU ordering
                    self._wf_cache.move_to_end(cache_key)
                    train_result = self._wf_cache[cache_key]
                    logger.debug(f"Cache hit for window {window.window_index + 1}/{len(windows)}")
                else:
                    self._wf_cache_misses += 1
                    # Run backtest on training window
                    train_result = self._backtest_with_timerange(
                        strategy_code, 
                        generated_name, 
                        window.train_timerange,
                        strategy_max_open_trades=strategy_gene.max_open_trades
                    )
                    # Cache the training result (LRU eviction if over limit)
                    self._wf_cache[cache_key] = train_result
                    while len(self._wf_cache) > self._wf_cache_max_size:
                        self._wf_cache.popitem(last=False)  # evict oldest
                
                # Skip validation if training backtest completely failed
                if not train_result.success:
                    logger.warning(f"Window {window.window_index + 1}/{len(windows)}: Training backtest failed "
                                 f"({train_result.error_message}). Skipping window.")
                    failed_windows += 1
                    continue
                
                # Adaptive min_train_trades: scale based on window size relative to expected
                base_min_train_trades = self.walk_forward_config.get('min_train_trades', 10)
                expected_train_days = self.walk_forward_config.get('train_days', 90)
                # Calculate actual window days from timerange
                try:
                    w_start, w_end = parse_timerange(window.train_timerange)
                    actual_window_days = (w_end - w_start).days
                except Exception:
                    actual_window_days = expected_train_days
                adaptive_min_trades = max(3, int(base_min_train_trades * (actual_window_days / max(expected_train_days, 1))))
                
                # Partial credit system: instead of binary skip, compute a trade-count
                # confidence factor. Windows with few trades get reduced weight.
                train_trade_credit = 1.0
                if train_result.total_trades == 0:
                    consecutive_zero_trade += 1
                    # Rate-limit zero-trade warnings (only log first 3 per evaluation)
                    if consecutive_zero_trade <= 3:
                        logger.warning(f"Window {window.window_index + 1}/{len(windows)}: Zero training trades. "
                                     f"Skipping window. ({consecutive_zero_trade} consecutive)")
                    failed_windows += 1
                    if consecutive_zero_trade >= max_consecutive_zero:
                        logger.warning(f"Early exit: {consecutive_zero_trade} consecutive zero-trade windows. "
                                     f"Skipping remaining {len(windows) - window.window_index - 1} windows.")
                        break
                    continue
                else:
                    consecutive_zero_trade = 0  # Reset on successful window
                if train_result.total_trades < adaptive_min_trades:
                    # Partial credit: scale from 0.3 (1 trade) to 1.0 (at adaptive_min_trades)
                    train_trade_credit = 0.3 + 0.7 * (train_result.total_trades / adaptive_min_trades)
                    logger.info(f"Window {window.window_index + 1}/{len(windows)}: Low training trades "
                               f"({train_result.total_trades} < {adaptive_min_trades}). "
                               f"Applying partial credit: {train_trade_credit:.2f}")
                
                # Run backtest on validation window (never cached - validation is key metric)
                val_result = self._backtest_with_timerange(
                    strategy_code,
                    generated_name,
                    window.val_timerange,
                    strategy_max_open_trades=strategy_gene.max_open_trades
                )
                
                # Calculate fitness for validation data
                if val_result.success and val_result.total_trades > 0:
                    val_metrics = self._backtest_result_to_metrics(val_result)
                    val_metrics['complexity'] = strategy_gene.calculate_complexity()
                    val_fitness = self.calculate_fitness(val_metrics, strategy_gene)
                    # Apply partial credit from training trade count
                    val_fitness *= train_trade_credit
                else:
                    val_fitness = 0.0
                    val_metrics = {
                        'profit': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,  # Zero trades = no drawdown (not 100%)
                        'win_rate': 0.0,
                        'num_trades': 0,
                        'complexity': strategy_gene.calculate_complexity()
                    }
                
                validation_fitness_scores.append(val_fitness)
                
                # Calculate training fitness for logging
                if train_result.success:
                    train_metrics = self._backtest_result_to_metrics(train_result)
                    train_fitness = self.calculate_fitness(train_metrics, strategy_gene)
                else:
                    train_fitness = 0.0
                
                train_fitness_scores.append(train_fitness)
                
                # Store metrics for this window
                all_window_metrics.append({
                    'window_index': window.window_index,
                    'train_fitness': train_fitness,
                    'val_fitness': val_fitness,
                    'train_trades': train_result.total_trades,
                    'val_trades': val_result.total_trades,
                    **val_metrics
                })
                
                logger.debug(f"Window {window.window_index + 1}/{len(windows)}: "
                          f"Train fitness={train_fitness:.4f} ({train_result.total_trades} trades), "
                          f"Val fitness={val_fitness:.4f} ({val_result.total_trades} trades)")
            
            # Aggregate validation scores
            aggregation_method = self.walk_forward_config.get('aggregation', 'mean')
            
            # If all windows failed, fall back to standard eval with heavy penalty
            # instead of returning 0.0 — this keeps genetic material alive but strongly disfavored
            if not validation_fitness_scores:
                logger.warning(f"All {len(windows)} walk-forward windows failed for {generated_name}. "
                              f"Falling back to standard eval with overfitting penalty.")
                fallback_fitness, fallback_metrics = self._evaluate_standard(strategy_gene, strategy_name)
                # Apply heavy penalty: strategy couldn't survive walk-forward at all
                wf_fallback_penalty = 0.3
                fallback_fitness *= wf_fallback_penalty
                fallback_metrics['walk_forward'] = True
                fallback_metrics['walk_forward_fallback'] = True
                fallback_metrics['num_windows'] = len(windows)
                fallback_metrics['failed_windows'] = failed_windows
                fallback_metrics['wf_fallback_penalty'] = wf_fallback_penalty
                logger.info(f"Walk-forward fallback for {generated_name}: "
                           f"standard fitness={fallback_fitness / wf_fallback_penalty:.4f} -> "
                           f"penalized={fallback_fitness:.4f} (x{wf_fallback_penalty})")
                return fallback_fitness, fallback_metrics
            
            # For weighted aggregation, auto-generate recency weights (later windows weighted more)
            if aggregation_method == 'weighted' and validation_fitness_scores:
                n = len(validation_fitness_scores)
                # Linear recency weights: [1, 2, 3, ..., n] normalized to sum to 1
                weights = [i / sum(range(1, n + 1)) for i in range(1, n + 1)]
                final_fitness = aggregate_validation_scores(validation_fitness_scores, method=aggregation_method, weights=weights)
            else:
                final_fitness = aggregate_validation_scores(validation_fitness_scores, method=aggregation_method)
            
            # Calculate average metrics across validation windows
            avg_metrics = self._aggregate_window_metrics(all_window_metrics)
            avg_metrics['walk_forward'] = True
            avg_metrics['num_windows'] = len(windows)
            avg_metrics['failed_windows'] = failed_windows
            avg_metrics['successful_windows'] = len(validation_fitness_scores)
            
            # Apply proportional penalty for failed windows
            # If e.g. 2 out of 5 windows failed, penalty = (5-2)/5 = 0.6 multiplier
            if failed_windows > 0:
                total_windows = len(windows)
                success_ratio = len(validation_fitness_scores) / total_windows
                final_fitness *= success_ratio
                logger.info(f"Walk-forward window failure penalty: {failed_windows}/{total_windows} failed, "
                           f"fitness scaled by {success_ratio:.2f}")
            
            avg_metrics['avg_train_fitness'] = sum(train_fitness_scores) / len(train_fitness_scores) if train_fitness_scores else 0.0
            avg_metrics['avg_val_fitness'] = sum(validation_fitness_scores) / len(validation_fitness_scores) if validation_fitness_scores else 0.0
            # Train-val gap: Positive = training better (potential overfit), Negative = validation better (rare but good)
            avg_metrics['train_val_gap'] = avg_metrics['avg_train_fitness'] - avg_metrics['avg_val_fitness']
            
            # Apply train-validation gap penalty to discourage overfitting
            # A strategy that performs much better on training than validation is likely overfit
            gap_penalty_config = self.walk_forward_config.get('gap_penalty', {})
            gap_penalty_enabled = gap_penalty_config.get('enabled', True)
            gap_penalty_threshold = gap_penalty_config.get('threshold', 0.1)
            gap_penalty_max = gap_penalty_config.get('max_penalty', 0.5)
            
            if gap_penalty_enabled and avg_metrics['train_val_gap'] > gap_penalty_threshold:
                excess_gap = avg_metrics['train_val_gap'] - gap_penalty_threshold
                # Progressive penalty: larger gap = harsher penalty, capped at max_penalty
                gap_penalty_factor = max(1.0 - gap_penalty_max, 1.0 - excess_gap * 2.0)
                final_fitness *= gap_penalty_factor
                avg_metrics['gap_penalty_applied'] = 1.0 - gap_penalty_factor
                logger.info(f"Walk-forward gap penalty: gap={avg_metrics['train_val_gap']:.4f}, "
                           f"penalty={1.0 - gap_penalty_factor:.2%} applied to {generated_name}")
            
            # Log summary only
            logger.debug(f"Walk-forward {generated_name}: fitness={final_fitness:.4f}, gap={avg_metrics['train_val_gap']:.4f}")
            
            return final_fitness, avg_metrics
            
        except Exception as e:
            generated_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
            logger.error(f"Error in walk-forward evaluation for {generated_name}: {e}", exc_info=True)
            return 0.0, {
                'profit': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'complexity': strategy_gene.calculate_complexity(),
                'error': str(e),
                'walk_forward': True
            }
    
    def get_wf_cache_stats(self) -> Dict[str, Any]:
        """Get walk-forward cache statistics."""
        total = self._wf_cache_hits + self._wf_cache_misses
        hit_rate = self._wf_cache_hits / total if total > 0 else 0.0
        return {
            'hits': self._wf_cache_hits,
            'misses': self._wf_cache_misses,
            'total': total,
            'hit_rate': hit_rate,
            'cache_size': len(self._wf_cache),
        }
    
    def log_wf_cache_stats(self):
        """Log walk-forward cache statistics at INFO level."""
        stats = self.get_wf_cache_stats()
        if stats['total'] > 0:
            logger.info(
                f"[WF-CACHE] hits={stats['hits']}, misses={stats['misses']}, "
                f"hit_rate={stats['hit_rate']:.1%}, cache_size={stats['cache_size']}"
            )
    
    def _backtest_with_timerange(
        self, 
        strategy_code: str, 
        strategy_name: str, 
        timerange: str,
        strategy_max_open_trades: Optional[int] = None
    ) -> BacktestResult:
        """
        Run backtest with a specific timerange (helper for walk-forward).
        
        Args:
            strategy_code: Strategy Python code
            strategy_name: Strategy name
            timerange: Timerange string (e.g., '20230101-20230201')
            strategy_max_open_trades: Optional max open trades for this strategy
            
        Returns:
            BacktestResult
        """
        # Temporarily modify backtester config
        original_timerange = self.backtester.backtest_config.get('timerange', '')
        self.backtester.backtest_config['timerange'] = timerange
        
        try:
            result = self.backtester.backtest_strategy(
                strategy_code, 
                strategy_name,
                strategy_max_open_trades=strategy_max_open_trades
            )
            return result
        finally:
            # Restore original timerange
            self.backtester.backtest_config['timerange'] = original_timerange
    
    def _aggregate_window_metrics(self, window_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate metrics across all validation windows.
        
        Args:
            window_metrics: List of metric dictionaries, one per window
            
        Returns:
            Aggregated metrics dictionary
        """
        if not window_metrics:
            return {
                'profit': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'complexity': 0
            }
        
        # Average most metrics
        avg_metrics = {}
        numeric_keys = ['profit', 'sharpe_ratio', 'sortino_ratio', 'win_rate', 'num_trades', 
                       'profit_factor', 'complexity', 'val_trades', 'train_trades',
                       'dsr_penalty', 'dsr']
        
        for key in numeric_keys:
            values = [m.get(key, 0) for m in window_metrics if key in m]
            avg_metrics[key] = sum(values) / len(values) if values else 0.0
        
        # Max drawdown: take the worst (highest) across windows
        drawdowns = [m.get('max_drawdown', 0) for m in window_metrics if 'max_drawdown' in m]
        avg_metrics['max_drawdown'] = max(drawdowns) if drawdowns else 1.0
        
        return avg_metrics
    
    def _backtest_result_to_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """
        Convert BacktestResult to metrics dictionary for fitness calculation.
        
        Args:
            result: BacktestResult object
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'profit': result.profit_percent,
            'sharpe_ratio': max(-10.0, min(50.0, result.sharpe_ratio)),  # Clamp to sane display range
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'num_trades': result.total_trades,
            'profit_factor': result.profit_factor,
            'sortino_ratio': max(-10.0, min(50.0, result.sortino_ratio)),  # Clamp to sane display range
        }
        
        # Include per-pair profits for robustness analysis
        if result.per_pair_profit:
            metrics['per_pair_profit'] = result.per_pair_profit
            # Worst pair profit (most negative)
            metrics['worst_pair_profit'] = min(result.per_pair_profit.values())
            # Pair consistency: std deviation of per-pair profits
            pair_profits = list(result.per_pair_profit.values())
            if len(pair_profits) > 1:
                mean_pp = sum(pair_profits) / len(pair_profits)
                metrics['pair_profit_std'] = (sum((p - mean_pp) ** 2 for p in pair_profits) / len(pair_profits)) ** 0.5
        
        # Include monthly profits for stability analysis
        if result.monthly_profits and len(result.monthly_profits) > 1:
            metrics['monthly_profits'] = result.monthly_profits
            monthly = result.monthly_profits
            mean_monthly = sum(monthly) / len(monthly)
            metrics['monthly_return_std'] = (sum((m - mean_monthly) ** 2 for m in monthly) / len(monthly)) ** 0.5
            # Positive months ratio
            metrics['positive_months_ratio'] = sum(1 for m in monthly if m > 0) / len(monthly)
        
        return metrics
    
    def calculate_fitness(self, metrics: Dict[str, float], strategy_gene: StrategyGene = None) -> float:
        """
        Calculate overall fitness score from metrics.
        
        Uses weighted combination of metrics with penalties and robustness scoring.
        
        Args:
            metrics: Dictionary of performance metrics
            strategy_gene: Optional StrategyGene for complexity penalty calculation
            
        Returns:
            Fitness score (higher is better)
        """
        import math
        
        # Extract and normalize metrics
        profit = metrics.get('profit', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = metrics.get('sortino_ratio', 0)  # New: downside risk focus
        profit_factor = metrics.get('profit_factor', 0)  # New: win/loss ratio
        drawdown = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        trades = metrics.get('num_trades', 0)
        
        # NaN/Inf protection: replace invalid values with 0
        if math.isnan(profit) or math.isinf(profit):
            profit = 0
        if math.isnan(sharpe) or math.isinf(sharpe):
            sharpe = 0
        if math.isnan(sortino) or math.isinf(sortino):
            sortino = 0
        if math.isnan(profit_factor) or math.isinf(profit_factor):
            profit_factor = 0
        if math.isnan(drawdown) or math.isinf(drawdown):
            drawdown = 1.0  # Assume worst case
        if math.isnan(win_rate) or math.isinf(win_rate):
            win_rate = 0
        
        # Clamp values to reasonable ranges to avoid extreme outliers
        # Use configurable bounds from self.profit_min, self.profit_max, etc.
        profit = max(self.profit_min, min(profit, self.profit_max))
        sharpe = max(self.sharpe_min, min(sharpe, self.sharpe_max))
        sortino = max(self.sortino_min, min(sortino, self.sortino_max))
        profit_factor = max(0, min(profit_factor, self.profit_factor_max))
        drawdown = min(drawdown, 1.0)  # 0 to 100%
        win_rate = max(0, min(win_rate, 1.0))  # 0 to 100%
        
        # Normalize to 0-1 range with configurable scaling
        profit_range = self.profit_max - self.profit_min
        norm_profit = (profit - self.profit_min) / profit_range if profit_range > 0 else 0
        sharpe_range = self.sharpe_max - self.sharpe_min
        norm_sharpe = (sharpe - self.sharpe_min) / sharpe_range if sharpe_range > 0 else 0
        sortino_range = self.sortino_max - self.sortino_min
        norm_sortino = (sortino - self.sortino_min) / sortino_range if sortino_range > 0 else 0
        norm_profit_factor = min(1.0, profit_factor / 3.0)  # >3.0 is excellent
        norm_drawdown = 1 - drawdown  # Lower drawdown is better
        norm_win_rate = win_rate  # Already 0-1
        norm_trades = self._normalize_trade_frequency(trades)
        
        # Clamp normalized values
        norm_profit = max(0, min(norm_profit, 1))
        norm_sharpe = max(0, min(norm_sharpe, 1))
        norm_sortino = max(0, min(norm_sortino, 1))
        norm_profit_factor = max(0, min(norm_profit_factor, 1))
        norm_drawdown = max(0, min(norm_drawdown, 1))
        
        # Get weights with defaults (adjusted to include new metrics)
        w = self.fitness_weights
        w_profit = w.get('profit', 0.22)
        w_sharpe = w.get('sharpe_ratio', 0.16)
        w_sortino = w.get('sortino_ratio', 0.13)
        w_profit_factor = w.get('profit_factor', 0.10)
        w_drawdown = w.get('drawdown', 0.13)
        w_win_rate = w.get('win_rate', 0.07)
        w_trades = w.get('trade_frequency', 0.07)
        w_stability = w.get('monthly_stability', 0.06)
        w_cross_pair = w.get('cross_pair', 0.06)
        
        # === Monthly stability score ===
        # Lower monthly return std = higher stability = better
        monthly_return_std = metrics.get('monthly_return_std', 0)
        positive_months = metrics.get('positive_months_ratio', 0.5)
        if monthly_return_std > 0:
            # Normalize: std of 0 gets 1.0, std of 20+ gets ~0
            norm_stability = max(0, 1.0 - monthly_return_std / 20.0)
            # Bonus for high positive months ratio
            norm_stability = norm_stability * 0.7 + positive_months * 0.3
        else:
            norm_stability = 0.5  # Unknown / not enough data
        
        # === Cross-pair consistency score ===
        # Penalize strategies that only work on 1-2 pairs
        pair_profit_std = metrics.get('pair_profit_std', 0)
        per_pair_profit = metrics.get('per_pair_profit', {})
        if per_pair_profit and len(per_pair_profit) > 1:
            # Count pairs with positive profit
            positive_pairs = sum(1 for v in per_pair_profit.values() if v > 0)
            pair_consistency_ratio = positive_pairs / len(per_pair_profit)
            # Low std across pairs = consistent = good
            norm_cross_pair = max(0, 1.0 - pair_profit_std / 30.0) * 0.5 + pair_consistency_ratio * 0.5
        else:
            norm_cross_pair = 0.5  # Single pair or no data
        
        # Normalize weights to sum to 1.0 (handles missing or extra weights in configs)
        weights_dict = {
            'profit': w_profit,
            'sharpe_ratio': w_sharpe,
            'sortino_ratio': w_sortino,
            'profit_factor': w_profit_factor,
            'drawdown': w_drawdown,
            'win_rate': w_win_rate,
            'trade_frequency': w_trades,
            'monthly_stability': w_stability,
            'cross_pair': w_cross_pair
        }
        total_weight = sum(weights_dict.values())
        if total_weight > 0:
            w_profit = weights_dict['profit'] / total_weight
            w_sharpe = weights_dict['sharpe_ratio'] / total_weight
            w_sortino = weights_dict['sortino_ratio'] / total_weight
            w_profit_factor = weights_dict['profit_factor'] / total_weight
            w_drawdown = weights_dict['drawdown'] / total_weight
            w_win_rate = weights_dict['win_rate'] / total_weight
            w_trades = weights_dict['trade_frequency'] / total_weight
            w_stability = weights_dict['monthly_stability'] / total_weight
            w_cross_pair = weights_dict['cross_pair'] / total_weight
        
        # Calculate weighted fitness
        fitness = (
            w_profit * norm_profit + 
            w_sharpe * norm_sharpe + 
            w_sortino * norm_sortino +
            w_profit_factor * norm_profit_factor +
            w_drawdown * norm_drawdown + 
            w_win_rate * norm_win_rate + 
            w_trades * norm_trades +
            w_stability * norm_stability +
            w_cross_pair * norm_cross_pair
        )
        
        # ==================================================================================
        # BONUS STACKING STRATEGY:
        # Multiple bonuses are tracked and capped to avoid excessive amplification.
        # Maximum total bonus: 1.3x (30% boost) to prevent lucky strategies from dominating.
        # ==================================================================================
        
        total_bonus = 1.0
        
        # Robustness bonus: reward consistency (good Sortino and profit factor together)
        if sortino > 1.0 and profit_factor > 1.5:
            robustness_bonus = 0.05 * min(sortino, 3.0)  # Up to 15% bonus
            total_bonus += robustness_bonus
        
        # Bonus for positive profit (encourage profitable strategies)
        if profit > 0:
            total_bonus += 0.05  # 5% bonus for any positive profit
        
        # Extra bonus for significantly profitable strategies
        if profit > 10:
            total_bonus += 0.10  # Additional 10% bonus for >10% profit
        
        # Risk-adjusted excellence bonus: reward exceptional risk-adjusted returns
        if sharpe > 2.0 and drawdown < 0.15:
            total_bonus += 0.10  # 10% bonus for excellent risk management
        
        # Cap total bonus at 1.3x (30% max boost)
        total_bonus = min(total_bonus, 1.3)
        fitness *= total_bonus
        
        # ==================================================================================
        # DEFLATED SHARPE RATIO PENALTY
        # Corrects for selection bias (multiple testing) and non-normal return distributions.
        # A low DSR means the observed Sharpe is likely a statistical artifact.
        # ==================================================================================
        dsr_penalty, dsr_info = self._dsr_tracker.compute_penalty(
            observed_sharpe=sharpe,
            n_returns=int(trades),
            skewness=metrics.get('return_skewness', 0.0),
            kurtosis=metrics.get('return_kurtosis', 3.0),
        )
        fitness *= dsr_penalty
        
        # Store DSR info in metrics for downstream reporting
        metrics['dsr'] = dsr_info.get('dsr', float('nan'))
        metrics['dsr_penalty'] = dsr_info.get('dsr_penalty', 1.0)
        
        # Register this evaluation for future DSR calculations
        self._dsr_tracker.register_evaluation()
        
        # Apply penalties and return
        penalized_fitness = self._apply_penalties(fitness, metrics, strategy_gene)
        
        # Ensure non-negative
        return max(0, penalized_fitness)
    
    def _normalize_trade_frequency(self, num_trades: int) -> float:
        """
        Normalize trade frequency to 0-1 range.
        
        Uses configurable thresholds from self.tf_* attributes.
        Too few trades = unreliable, too many trades = overtrading and high fees.
        """
        if num_trades == 0:
            return 0.0
        elif num_trades < self.tf_very_few:
            # Very few trades - heavily penalized
            return num_trades / (self.tf_very_few * 2)
        elif self.tf_very_few <= num_trades < self.tf_few:
            # Few trades - some penalty
            return 0.5 + (num_trades - self.tf_very_few) / (self.tf_few - self.tf_very_few) * 0.5
        elif self.tf_ideal_min <= num_trades <= self.tf_ideal_max:
            # Ideal range - full score
            return 1.0
        elif self.tf_ideal_max < num_trades <= self.tf_moderate_excess:
            # Moderate overtrading - slight penalty
            return 1.0 - (num_trades - self.tf_ideal_max) / (self.tf_moderate_excess - self.tf_ideal_max) * 0.5
        else:
            # Excessive trading - significant penalty
            return max(0.3, 0.5 - (num_trades - self.tf_moderate_excess) / 200)
    
    def _apply_penalties(self, fitness: float, metrics: Dict[str, float], strategy_gene: StrategyGene = None) -> float:
        """
        Apply penalties for constraint violations.
        
        Penalties are applied multiplicatively to reduce fitness for strategies
        that violate important constraints.
        
        Args:
            fitness: Base fitness score
            metrics: Performance metrics
            strategy_gene: Optional StrategyGene for complexity penalty
            
        Returns:
            Fitness with penalties applied
        """
        penalties = self.fitness_penalties
        
        num_trades = metrics.get('num_trades', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        
        # Soft penalty for low trade count (gradual instead of harsh)
        min_trades = penalties.get('min_trades', 5)
        if num_trades < min_trades:
            if num_trades == 0:
                fitness *= 0.1  # Very low fitness for no trades
            else:
                # Gradual penalty: 50% at 1 trade, increasing to full at min_trades
                # Formula: 0.5 + (num_trades / min_trades) * 0.5
                # E.g., with min_trades=5: 1 trade=60%, 2=70%, 3=80%, 4=90%, 5+=100%
                trade_penalty = 0.5 + (num_trades / min_trades) * 0.5
                fitness *= trade_penalty
        
        # Penalty for excessive drawdown
        max_dd_threshold = penalties.get('max_drawdown', 0.30)
        if max_drawdown > max_dd_threshold:
            # Progressive penalty: worse drawdown = worse penalty
            dd_excess = max_drawdown - max_dd_threshold
            dd_penalty = max(0.3, 1.0 - dd_excess * 2)
            fitness *= dd_penalty
        
        # Penalty for low win rate (but not too harsh)
        min_win_rate = penalties.get('min_win_rate', 0.30)
        if win_rate < min_win_rate and num_trades >= 5:  # Only penalize if enough trades
            # Gradual penalty for low win rate
            wr_penalty = max(0.6, win_rate / min_win_rate)
            fitness *= wr_penalty
        
        # Complexity penalty: penalize overly complex strategies
        # Applied additively (after multiplicative penalties) to allow fine-tuning
        # Additive approach chosen because:
        # - Complexity is a count (discrete), not a rate
        # - Easier to interpret and tune (linear relationship)
        # - Avoids compound effects with other multiplicative penalties
        if strategy_gene is not None:
            complexity_weight = penalties.get('complexity_weight', 0.01)
            if complexity_weight > 0:
                complexity = strategy_gene.calculate_complexity()
                complexity_penalty = complexity_weight * complexity
                # Subtract penalty from fitness
                fitness = max(0, fitness - complexity_penalty)
                logger.debug(f"Applied complexity penalty: {complexity_penalty:.4f} "
                           f"(complexity={complexity}, weight={complexity_weight})")
        
        # Per-pair robustness penalty: penalize strategies with large losses on any single pair
        # This prevents pair-concentration risk where aggregate profit masks individual pair losses
        worst_pair_profit = metrics.get('worst_pair_profit')
        pair_loss_threshold = penalties.get('pair_loss_threshold', -10.0)  # Max acceptable loss on any pair
        if worst_pair_profit is not None and worst_pair_profit < pair_loss_threshold:
            excess_loss = abs(worst_pair_profit - pair_loss_threshold)
            pair_penalty = max(0.5, 1.0 - excess_loss / 100.0)  # Cap at 50% penalty
            fitness *= pair_penalty
            logger.debug(f"Applied per-pair penalty: worst_pair={worst_pair_profit:.2f}%, "
                        f"penalty={1.0 - pair_penalty:.2%}")
        
        # Unused-indicator penalty: penalize indicators that don't contribute to any condition
        # Unused indicators add noise and computational overhead without improving signal quality
        if strategy_gene is not None:
            unused_penalty_weight = penalties.get('unused_indicator_weight', 0.02)
            if unused_penalty_weight > 0:
                total_indicators = len(strategy_gene.indicators)
                if total_indicators > 0:
                    # Collect all indicator references from conditions
                    used_indicators = set()
                    for cond in strategy_gene.entry_conditions:
                        used_indicators.add(cond.indicator)
                    for cond in strategy_gene.exit_conditions:
                        used_indicators.add(cond.indicator)
                    
                    # Count indicators that are actually used
                    used_count = 0
                    for ind in strategy_gene.indicators:
                        ind_ref = ind.instance_id or ind.type
                        if ind_ref in used_indicators:
                            used_count += 1
                    
                    unused_count = total_indicators - used_count
                    if unused_count > 0:
                        unused_ratio = unused_count / total_indicators
                        unused_penalty = unused_ratio * unused_penalty_weight * total_indicators
                        fitness = max(0, fitness - unused_penalty)
                        logger.debug(f"Applied unused-indicator penalty: {unused_penalty:.4f} "
                                   f"({unused_count}/{total_indicators} unused)")
        
        # Dead exit condition penalty: penalize strategies where ALL exit
        # conditions use impossible thresholds (e.g. RSI < 0) — forcing exits
        # to rely entirely on ROI/stoploss, which overfits to training data.
        if strategy_gene is not None and strategy_gene.exit_conditions:
            _BOUNDED = {'RSI': (0, 100), 'STOCH': (0, 100), 'CCI': (-300, 300),
                        'CMF': (-1, 1), 'ADX': (0, 100)}
            dead_count = 0
            bounded_count = 0
            for cond in strategy_gene.exit_conditions:
                base_type = cond.indicator.split('_')[0] if '_' in cond.indicator else cond.indicator
                bounds = _BOUNDED.get(base_type.upper())
                if bounds:
                    bounded_count += 1
                    lo, hi = bounds
                    # Condition is "dead" if threshold is outside the indicator's range
                    if cond.operator in ('<', 'less_than') and cond.threshold <= lo:
                        dead_count += 1
                    elif cond.operator in ('>', 'greater_than') and cond.threshold >= hi:
                        dead_count += 1
            if bounded_count > 0 and dead_count == bounded_count:
                # ALL bounded exit conditions are impossible → heavy penalty
                fitness *= 0.7
                logger.debug(f"Applied dead-exit penalty: all {dead_count} bounded exit conditions "
                           f"use impossible thresholds (fitness x0.7)")
        
        return fitness
    
    def evaluate_holdout(self, strategy_gene: StrategyGene, holdout_timerange: str,
                         strategy_name: str = None) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a strategy on a completely unseen holdout period.
        
        This method is designed to be called ONLY ONCE after evolution is complete,
        on the final top-N strategies. The holdout period should never be seen during
        evolution to provide a true out-of-sample performance estimate.
        
        Args:
            strategy_gene: Strategy to evaluate
            holdout_timerange: Timerange string for holdout period (YYYYMMDD-YYYYMMDD)
            strategy_name: Optional name for the strategy
            
        Returns:
            Tuple of (fitness_score, metrics_dict)
        """
        try:
            strategy_code = self.strategy_generator.generate_strategy_code(strategy_gene)
            generated_name = strategy_name or f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
            
            logger.info(f"[HOLDOUT] Evaluating {generated_name} on holdout period: {holdout_timerange}")
            
            result = self._backtest_with_timerange(
                strategy_code, generated_name, holdout_timerange,
                strategy_max_open_trades=strategy_gene.max_open_trades
            )
            
            if not result.success or result.total_trades == 0:
                logger.warning(f"[HOLDOUT] {generated_name}: backtest failed or zero trades")
                return 0.0, {
                    'profit': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 1.0,
                    'win_rate': 0.0, 'num_trades': 0, 'holdout': True,
                    'error': result.error_message if not result.success else 'zero trades'
                }
            
            metrics = self._backtest_result_to_metrics(result)
            metrics['complexity'] = strategy_gene.calculate_complexity()
            metrics['holdout'] = True
            fitness = self.calculate_fitness(metrics, strategy_gene)
            
            logger.info(f"[HOLDOUT] {generated_name}: fitness={fitness:.4f}, "
                       f"profit={metrics['profit']:.2f}%, trades={metrics['num_trades']}")
            
            return fitness, metrics
            
        except Exception as e:
            logger.error(f"[HOLDOUT] Error evaluating {strategy_name}: {e}", exc_info=True)
            return 0.0, {'profit': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 1.0,
                        'win_rate': 0.0, 'num_trades': 0, 'holdout': True, 'error': str(e)}
    
    @staticmethod
    def split_timerange_for_holdout(timerange: str, holdout_pct: float = 0.15) -> Tuple[str, str]:
        """
        Split a timerange into evolution and holdout periods.
        
        The holdout period is taken from the END of the timerange (most recent data),
        since we want to validate forward generalization.
        
        Args:
            timerange: Full timerange string (YYYYMMDD-YYYYMMDD)
            holdout_pct: Fraction of data to reserve as holdout (default: 15%)
            
        Returns:
            Tuple of (evolution_timerange, holdout_timerange)
        """
        from datetime import timedelta
        start, end = parse_timerange(timerange)
        total_days = (end - start).days
        holdout_days = max(7, int(total_days * holdout_pct))  # minimum 7 days
        
        split_date = end - timedelta(days=holdout_days)
        
        evolution_tr = f"{format_date(start)}-{format_date(split_date)}"
        holdout_tr = f"{format_date(split_date)}-{format_date(end)}"
        
        logger.info(f"[HOLDOUT] Split timerange: evolution={evolution_tr} ({total_days - holdout_days}d), "
                    f"holdout={holdout_tr} ({holdout_days}d)")
        
        return evolution_tr, holdout_tr