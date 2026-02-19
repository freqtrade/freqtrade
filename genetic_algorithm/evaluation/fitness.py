"""
Fitness Evaluator

Evaluates the fitness of trading strategies through backtesting
and calculating performance metrics. Supports both standard backtesting
and walk-forward optimization for preventing overfitting.
"""

import logging
import hashlib
from typing import Tuple, Dict, Any, List, Optional

from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.evaluation.direct_backtester import DirectBacktester, BacktestResult
from genetic_algorithm.strategies.generator import StrategyGenerator
from genetic_algorithm.utils.timerange import (
    create_walk_forward_windows,
    validate_walk_forward_config,
    aggregate_validation_scores,
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
        
        # Validate walk-forward config if enabled
        if self.walk_forward_config.get('enabled', False):
            validate_walk_forward_config(self.walk_forward_config)
            logger.info("Walk-forward optimization enabled")
        
        # Initialize direct backtester and strategy generator
        self.backtester = DirectBacktester(config)
        self.strategy_generator = StrategyGenerator(config)
        
        # Walk-forward cache: (strategy_hash, window_index) -> BacktestResult
        self._wf_cache: Dict[Tuple[str, int], BacktestResult] = {}
        self._wf_cache_hits = 0
        self._wf_cache_misses = 0
    
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
            
            # Run backtest
            backtest_result = self.backtester.backtest_strategy(strategy_code, generated_name)
            
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
            
            logger.info(f"Strategy {generated_name}: fitness={fitness:.4f}, "
                       f"profit={metrics['profit']:.2f}%, trades={metrics['num_trades']}, "
                       f"complexity={metrics['complexity']}")
            
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
                    mode=self.walk_forward_config.get('mode', 'rolling')
                )
            except ValueError as e:
                logger.warning(
                    f"⚠️  Walk-forward window creation failed even after auto-adjust: {e}. "
                    f"Falling back to standard single-period evaluation.")
                return self._evaluate_standard(strategy_gene, strategy_name)
            
            logger.info(f"Evaluating {generated_name} with {len(windows)} walk-forward windows")
            
            validation_fitness_scores = []
            train_fitness_scores = []  # For comparison/debugging
            all_window_metrics = []
            
            for window in windows:
                if progress_callback:
                    progress_callback(window.window_index, len(windows))
                
                # Check cache first
                cache_key = (strategy_hash, window.window_index)
                if cache_key in self._wf_cache:
                    self._wf_cache_hits += 1
                    train_result = self._wf_cache[cache_key]
                    logger.debug(f"Cache hit for window {window.window_index}")
                else:
                    self._wf_cache_misses += 1
                    # Run backtest on training window
                    train_result = self._backtest_with_timerange(
                        strategy_code, 
                        generated_name, 
                        window.train_timerange
                    )
                    # Cache the training result
                    self._wf_cache[cache_key] = train_result
                
                # Skip validation if training failed (e.g., no data for this window)
                # This avoids wasting time on a validation backtest that can't be used
                min_train_trades = self.walk_forward_config.get('min_train_trades', 10)
                if not train_result.success:
                    logger.warning(f"Window {window.window_index}: Training backtest failed "
                                 f"({train_result.error_message}). Skipping window.")
                    validation_fitness_scores.append(0.0)
                    continue
                
                if train_result.total_trades < min_train_trades:
                    logger.warning(f"Window {window.window_index}: Insufficient training trades "
                                 f"({train_result.total_trades} < {min_train_trades}). Using penalty fitness.")
                    validation_fitness_scores.append(0.0)
                    continue
                
                # Run backtest on validation window (never cached - validation is key metric)
                val_result = self._backtest_with_timerange(
                    strategy_code,
                    generated_name,
                    window.val_timerange
                )
                
                # Calculate fitness for validation data
                if val_result.success and val_result.total_trades > 0:
                    val_metrics = self._backtest_result_to_metrics(val_result)
                    val_metrics['complexity'] = strategy_gene.calculate_complexity()
                    val_fitness = self.calculate_fitness(val_metrics, strategy_gene)
                else:
                    val_fitness = 0.0
                    val_metrics = {
                        'profit': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 1.0,
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
                
                logger.info(f"Window {window.window_index}/{len(windows)-1}: "
                          f"Train fitness={train_fitness:.4f} ({train_result.total_trades} trades), "
                          f"Val fitness={val_fitness:.4f} ({val_result.total_trades} trades)")
            
            # Aggregate validation scores
            aggregation_method = self.walk_forward_config.get('aggregation', 'mean')
            final_fitness = aggregate_validation_scores(validation_fitness_scores, method=aggregation_method)
            
            # Calculate average metrics across validation windows
            avg_metrics = self._aggregate_window_metrics(all_window_metrics)
            avg_metrics['walk_forward'] = True
            avg_metrics['num_windows'] = len(windows)
            avg_metrics['avg_train_fitness'] = sum(train_fitness_scores) / len(train_fitness_scores) if train_fitness_scores else 0.0
            avg_metrics['avg_val_fitness'] = sum(validation_fitness_scores) / len(validation_fitness_scores) if validation_fitness_scores else 0.0
            # Train-val gap: Positive = training better (potential overfit), Negative = validation better (rare but good)
            avg_metrics['train_val_gap'] = avg_metrics['avg_train_fitness'] - avg_metrics['avg_val_fitness']
            
            logger.info(f"Walk-forward complete for {generated_name}: "
                       f"Final fitness={final_fitness:.4f} "
                       f"(train avg={avg_metrics['avg_train_fitness']:.4f}, "
                       f"val avg={avg_metrics['avg_val_fitness']:.4f}, "
                       f"gap={avg_metrics['train_val_gap']:.4f})")
            logger.info(f"Walk-forward cache stats: {self._wf_cache_hits} hits, {self._wf_cache_misses} misses")
            
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
    
    def _backtest_with_timerange(
        self, 
        strategy_code: str, 
        strategy_name: str, 
        timerange: str
    ) -> BacktestResult:
        """
        Run backtest with a specific timerange (helper for walk-forward).
        
        Args:
            strategy_code: Strategy Python code
            strategy_name: Strategy name
            timerange: Timerange string (e.g., '20230101-20230201')
            
        Returns:
            BacktestResult
        """
        # Temporarily modify backtester config
        original_timerange = self.backtester.backtest_config.get('timerange', '')
        self.backtester.backtest_config['timerange'] = timerange
        
        try:
            result = self.backtester.backtest_strategy(strategy_code, strategy_name)
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
        numeric_keys = ['profit', 'sharpe_ratio', 'win_rate', 'num_trades', 
                       'complexity', 'val_trades', 'train_trades']
        
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
        return {
            'profit': result.profit_percent,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'num_trades': result.total_trades,
            'profit_factor': result.profit_factor,
            'sortino_ratio': result.sortino_ratio,
        }
    
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
        # Extract and normalize metrics
        profit = metrics.get('profit', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = metrics.get('sortino_ratio', 0)  # New: downside risk focus
        profit_factor = metrics.get('profit_factor', 0)  # New: win/loss ratio
        drawdown = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        trades = metrics.get('num_trades', 0)
        
        # Clamp values to reasonable ranges to avoid extreme outliers
        profit = max(-50, min(profit, 200))  # -50% to +200%
        sharpe = max(-5, min(sharpe, 10))  # -5 to 10
        sortino = max(-5, min(sortino, 12))  # Sortino often higher than Sharpe
        profit_factor = max(0, min(profit_factor, 10))  # 0 to 10
        drawdown = min(drawdown, 1.0)  # 0 to 100%
        win_rate = max(0, min(win_rate, 1.0))  # 0 to 100%
        
        # Normalize to 0-1 range with better scaling
        norm_profit = (profit + 50) / 250  # -50% to +200%
        norm_sharpe = (sharpe + 5) / 15  # -5 to 10
        norm_sortino = (sortino + 5) / 17  # -5 to 12
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
        w_profit = w.get('profit', 0.25)
        w_sharpe = w.get('sharpe_ratio', 0.15)
        w_sortino = w.get('sortino_ratio', 0.15)  # New weight
        w_profit_factor = w.get('profit_factor', 0.10)  # New weight
        w_drawdown = w.get('drawdown', 0.15)
        w_win_rate = w.get('win_rate', 0.10)
        w_trades = w.get('trade_frequency', 0.10)
        
        # Normalize weights to sum to 1.0 (handles missing or extra weights in configs)
        weights_dict = {
            'profit': w_profit,
            'sharpe_ratio': w_sharpe,
            'sortino_ratio': w_sortino,
            'profit_factor': w_profit_factor,
            'drawdown': w_drawdown,
            'win_rate': w_win_rate,
            'trade_frequency': w_trades
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
        
        # Calculate weighted fitness
        fitness = (
            w_profit * norm_profit + 
            w_sharpe * norm_sharpe + 
            w_sortino * norm_sortino +
            w_profit_factor * norm_profit_factor +
            w_drawdown * norm_drawdown + 
            w_win_rate * norm_win_rate + 
            w_trades * norm_trades
        )
        
        # ==================================================================================
        # BONUS STACKING STRATEGY:
        # Multiple bonuses can stack multiplicatively to reward exceptional strategies.
        # Maximum possible bonus: ~2.01x (1.15 × 1.1 × 1.2 × 1.15 = 1.74x to 2.01x)
        # This is intentional - truly exceptional strategies deserve strong amplification.
        # ==================================================================================
        
        # Robustness bonus: reward consistency (good Sortino and profit factor together)
        if sortino > 1.0 and profit_factor > 1.5:
            robustness_bonus = 1.0 + (0.05 * min(sortino, 3.0))  # Up to 15% bonus
            fitness *= robustness_bonus
        
        # Bonus for positive profit (encourage profitable strategies)
        if profit > 0:
            fitness *= 1.1  # 10% bonus for any positive profit
        
        # Extra bonus for significantly profitable strategies
        # Note: This is cumulative with above, so total bonus is 32% (1.1 * 1.2) for >10% profit
        if profit > 10:
            fitness *= 1.2  # Additional 20% bonus (32% total with previous bonus)
        
        # Risk-adjusted excellence bonus: reward exceptional risk-adjusted returns
        if sharpe > 2.0 and drawdown < 0.15:
            fitness *= 1.15  # 15% bonus for excellent risk management
        
        # Apply penalties and return
        penalized_fitness = self._apply_penalties(fitness, metrics, strategy_gene)
        
        # Ensure non-negative
        return max(0, penalized_fitness)
    
    def _normalize_trade_frequency(self, num_trades: int) -> float:
        """
        Normalize trade frequency to 0-1 range.
        
        Prefers 10-50 trades for most strategies. Too few trades = unreliable,
        too many trades = overtrading and high fees.
        """
        if num_trades == 0:
            return 0.0
        elif num_trades < 5:
            # Very few trades - heavily penalized
            return num_trades / 10
        elif 5 <= num_trades < 10:
            # Few trades - some penalty
            return 0.5 + (num_trades - 5) / 10
        elif 10 <= num_trades <= 50:
            # Ideal range - full score
            return 1.0
        elif 50 < num_trades <= 100:
            # Moderate overtrading - slight penalty
            return 1.0 - (num_trades - 50) / 100
        else:
            # Excessive trading - significant penalty
            return max(0.3, 1.0 - (num_trades - 50) / 200)
    
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
        
        return fitness
    

