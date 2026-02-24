"""
Regime-Aware Fitness Evaluator

Extends the FitnessEvaluator to support regime-balanced evaluation.
Strategies are evaluated across multiple market regime segments (bullish, 
bearish, sideways) to prevent overfitting to a single market condition.

This module connects the regime detection pipeline to the GA fitness evaluation.
"""

import logging
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from statistics import harmonic_mean

from genetic_algorithm.core.strategy_gene import StrategyGene
from genetic_algorithm.evaluation.fitness import FitnessEvaluator
from genetic_algorithm.evaluation.direct_backtester import BacktestResult
from genetic_algorithm.utils.regime_detector import (
    RegimeDetector,
    RegimeSegment,
    RegimeType,
    load_ohlcv_data,
)

logger = logging.getLogger(__name__)


@dataclass
class RegimeEvaluationResult:
    """Container for per-regime evaluation results."""
    segment: RegimeSegment
    fitness: float
    metrics: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class RegimeAwareEvaluator:
    """
    Evaluates strategies across multiple regime segments.
    
    This evaluator wraps the standard FitnessEvaluator and extends it to:
    1. Run backtests on specific regime segments (using timeranges)
    2. Aggregate fitness across regimes using configurable methods
    3. Track holdout segments separately for final validation
    4. Cache segment-level results for efficiency
    
    The goal is to produce strategies robust to different market conditions,
    not just strategies that perform well in one type of market (e.g., bull run).
    
    Usage:
        evaluator = RegimeAwareEvaluator(config, segments={'optimization': [...], 'holdout': [...]})
        fitness, metrics = evaluator.evaluate(strategy_gene)
    """
    
    # Supported aggregation methods
    AGGREGATION_METHODS = ['mean', 'min', 'harmonic_mean', 'cvar']
    
    def __init__(
        self,
        config: Dict[str, Any],
        segments: Optional[Dict[str, List[RegimeSegment]]] = None,
    ):
        """
        Initialize regime-aware evaluator.
        
        Args:
            config: Configuration dictionary (includes regime_aware section)
            segments: Optional pre-computed segments dict with keys:
                      'optimization', 'model_selection', 'holdout'
                      If not provided, will auto-detect from data.
        """
        self.config = config
        self.regime_config = config.get('regime_aware', {})
        
        # Initialize base fitness evaluator
        self.base_evaluator = FitnessEvaluator(config)
        
        # Store segments
        self.segments = segments or {}
        self._optimization_segments = self.segments.get('optimization', [])
        self._holdout_segments = self.segments.get('holdout', [])
        self._model_selection_segments = self.segments.get('model_selection', [])
        
        # Aggregation method
        self.aggregation_method = self.regime_config.get('aggregation', 'harmonic_mean')
        if self.aggregation_method not in self.AGGREGATION_METHODS:
            logger.warning(f"Unknown aggregation method '{self.aggregation_method}', using 'harmonic_mean'")
            self.aggregation_method = 'harmonic_mean'
        
        # CVaR parameters (for 'cvar' aggregation)
        self.cvar_alpha = self.regime_config.get('cvar_alpha', 0.2)  # Bottom 20%
        
        # Cache for segment-level results: (strategy_hash, segment_id) -> RegimeEvaluationResult
        self._segment_cache: Dict[Tuple[str, str], RegimeEvaluationResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Holdout protection - prevents access during evolution
        self._holdout_locked = True  # Locked by default
        self._holdout_access_attempts = 0
        
        # Regime weights (optional - for weighted aggregation by regime type)
        self.regime_weights = self.regime_config.get('regime_weights', {
            'bullish': 1.0,
            'bearish': 1.0,
            'sideways': 1.0,
        })
        
        logger.info(
            f"RegimeAwareEvaluator initialized with {len(self._optimization_segments)} "
            f"optimization segments, {len(self._holdout_segments)} holdout segments, "
            f"aggregation={self.aggregation_method}"
        )
        if self._holdout_segments:
            logger.info(f"[HOLDOUT PROTECTION] Holdout segments are LOCKED - call unlock_holdout() for final validation")
    
    def lock_holdout(self) -> None:
        """
        Lock holdout segments to prevent access during evolution.
        
        This is the default state. Use after final validation to re-lock.
        """
        self._holdout_locked = True
        logger.info("[HOLDOUT PROTECTION] Holdout segments LOCKED")
    
    def unlock_holdout(self) -> None:
        """
        Unlock holdout segments for final validation.
        
        Call this ONLY when evolution is complete and you want to
        evaluate the best strategy on holdout data.
        
        WARNING: After calling this, ensure holdout is re-locked if
        you continue evolution.
        """
        self._holdout_locked = False
        logger.info("[HOLDOUT PROTECTION] Holdout segments UNLOCKED for final validation")
    
    def is_holdout_locked(self) -> bool:
        """Check if holdout segments are currently locked."""
        return self._holdout_locked
    
    def evaluate(
        self,
        strategy_gene: StrategyGene,
        strategy_name: Optional[str] = None,
        use_holdout: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate strategy across multiple regime segments.
        
        During evolution, use_holdout=False to evaluate only on optimization segments.
        For final validation, use_holdout=True to get true out-of-sample performance.
        
        Args:
            strategy_gene: Strategy to evaluate
            strategy_name: Optional name (auto-generated if not provided)
            use_holdout: If True, evaluate on holdout segments instead of optimization
            
        Returns:
            Tuple of (aggregated_fitness, aggregated_metrics)
        """
        # HOLDOUT PROTECTION: Prevent access to holdout during evolution
        if use_holdout and self._holdout_locked:
            self._holdout_access_attempts += 1
            logger.error(
                f"[HOLDOUT PROTECTION] Attempted to access locked holdout segments! "
                f"(attempt #{self._holdout_access_attempts}). "
                f"Call unlock_holdout() only after evolution is complete."
            )
            raise RuntimeError(
                "Holdout segments are locked. This protects against data leakage during evolution. "
                "Call evaluator.unlock_holdout() only after GA evolution is complete."
            )
        
        # Select which segments to use
        if use_holdout:
            segments = self._holdout_segments
            segment_type = 'holdout'
        else:
            segments = self._optimization_segments
            segment_type = 'optimization'
        
        # If no segments available, fall back to standard evaluation
        if not segments:
            logger.warning(
                f"No {segment_type} segments available, falling back to standard evaluation"
            )
            return self.base_evaluator.evaluate(strategy_gene, strategy_name)
        
        # Generate strategy code and hash for caching
        strategy_code = self.base_evaluator.strategy_generator.generate_strategy_code(strategy_gene)
        strategy_hash = hashlib.sha256(strategy_code.encode()).hexdigest()[:16]
        generated_name = f"GAStrategy_Gen{strategy_gene.generation}_Ind{strategy_gene.individual_id}"
        
        # Evaluate on each segment
        segment_results: List[RegimeEvaluationResult] = []
        
        for segment in segments:
            result = self._evaluate_segment(
                strategy_gene=strategy_gene,
                strategy_code=strategy_code,
                strategy_hash=strategy_hash,
                generated_name=generated_name,
                segment=segment,
            )
            segment_results.append(result)
        
        # Aggregate results
        aggregated_fitness, aggregated_metrics = self._aggregate_results(
            segment_results, strategy_gene
        )
        
        # Add regime-aware metadata
        aggregated_metrics['regime_aware'] = True
        aggregated_metrics['segment_type'] = segment_type
        aggregated_metrics['num_segments'] = len(segments)
        aggregated_metrics['aggregation'] = self.aggregation_method
        
        # Log per-regime summary
        regime_summary = self._get_regime_summary(segment_results)
        for regime_type, summary in regime_summary.items():
            aggregated_metrics[f'{regime_type}_avg_fitness'] = summary['avg_fitness']
            aggregated_metrics[f'{regime_type}_segment_count'] = summary['count']
        
        logger.debug(
            f"Regime-aware evaluation for {generated_name}: "
            f"fitness={aggregated_fitness:.4f}, segments={len(segment_results)}, "
            f"regime_summary={regime_summary}"
        )
        
        return aggregated_fitness, aggregated_metrics
    
    def _evaluate_segment(
        self,
        strategy_gene: StrategyGene,
        strategy_code: str,
        strategy_hash: str,
        generated_name: str,
        segment: RegimeSegment,
    ) -> RegimeEvaluationResult:
        """
        Evaluate strategy on a single regime segment.
        
        Uses caching to avoid re-evaluating same strategy on same segment.
        
        Args:
            strategy_gene: Strategy being evaluated
            strategy_code: Generated Python code
            strategy_hash: Hash of strategy code for caching
            generated_name: Strategy name for logging
            segment: Regime segment to evaluate on
            
        Returns:
            RegimeEvaluationResult with fitness and metrics
        """
        # Check cache
        cache_key = (strategy_hash, segment.segment_id)
        if cache_key in self._segment_cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for {generated_name} on segment {segment.segment_id}")
            return self._segment_cache[cache_key]
        
        self._cache_misses += 1
        
        try:
            # Run backtest with segment's timerange
            backtest_result = self._backtest_with_segment(
                strategy_code=strategy_code,
                strategy_name=generated_name,
                segment=segment,
                strategy_max_open_trades=strategy_gene.max_open_trades,
            )
            
            if not backtest_result.success:
                logger.warning(
                    f"Backtest failed for {generated_name} on segment {segment.segment_id}: "
                    f"{backtest_result.error_message}"
                )
                result = RegimeEvaluationResult(
                    segment=segment,
                    fitness=0.0,
                    metrics={'error': backtest_result.error_message},
                    success=False,
                    error_message=backtest_result.error_message,
                )
            else:
                # Calculate metrics and fitness
                metrics = self.base_evaluator._backtest_result_to_metrics(backtest_result)
                metrics['complexity'] = strategy_gene.calculate_complexity()
                metrics['regime'] = segment.regime.value
                metrics['segment_id'] = segment.segment_id
                metrics['segment_confidence'] = segment.confidence
                
                fitness = self.base_evaluator.calculate_fitness(metrics, strategy_gene)
                
                result = RegimeEvaluationResult(
                    segment=segment,
                    fitness=fitness,
                    metrics=metrics,
                    success=True,
                )
                
                logger.debug(
                    f"Segment {segment.segment_id} ({segment.regime.value}): "
                    f"fitness={fitness:.4f}, trades={metrics['num_trades']}"
                )
        
        except Exception as e:
            logger.error(f"Error evaluating segment {segment.segment_id}: {e}", exc_info=True)
            result = RegimeEvaluationResult(
                segment=segment,
                fitness=0.0,
                metrics={'error': str(e)},
                success=False,
                error_message=str(e),
            )
        
        # Cache result
        self._segment_cache[cache_key] = result
        
        return result
    
    def _backtest_with_segment(
        self,
        strategy_code: str,
        strategy_name: str,
        segment: RegimeSegment,
        strategy_max_open_trades: Optional[int] = None,
    ) -> BacktestResult:
        """
        Run backtest restricted to a specific regime segment's timerange.
        
        Args:
            strategy_code: Strategy Python code
            strategy_name: Strategy name
            segment: Regime segment with timerange
            strategy_max_open_trades: Optional max trades override
            
        Returns:
            BacktestResult
        """
        # Temporarily modify backtester config with segment's timerange
        original_timerange = self.base_evaluator.backtester.backtest_config.get('timerange', '')
        self.base_evaluator.backtester.backtest_config['timerange'] = segment.timerange
        
        try:
            result = self.base_evaluator.backtester.backtest_strategy(
                strategy_code,
                strategy_name,
                strategy_max_open_trades=strategy_max_open_trades,
            )
            return result
        finally:
            # Restore original timerange
            self.base_evaluator.backtester.backtest_config['timerange'] = original_timerange
    
    def _aggregate_results(
        self,
        results: List[RegimeEvaluationResult],
        strategy_gene: StrategyGene,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Aggregate fitness scores and metrics across segments.
        
        Args:
            results: List of per-segment evaluation results
            strategy_gene: Strategy being evaluated (for complexity)
            
        Returns:
            Tuple of (aggregated_fitness, aggregated_metrics)
        """
        # Extract successful fitness scores with regime weights
        weighted_scores = []
        regime_scores: Dict[str, List[float]] = {}
        
        for result in results:
            if result.success and result.fitness > 0:
                # Apply regime weight
                regime_type = result.segment.regime.value
                weight = self.regime_weights.get(regime_type, 1.0)
                weighted_scores.append((result.fitness, weight))
                
                # Track by regime type
                if regime_type not in regime_scores:
                    regime_scores[regime_type] = []
                regime_scores[regime_type].append(result.fitness)
        
        if not weighted_scores:
            logger.warning("No successful segment evaluations, returning zero fitness")
            return 0.0, {
                'profit': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'complexity': strategy_gene.calculate_complexity(),
            }
        
        # Calculate aggregated fitness
        fitness_values = [score for score, _ in weighted_scores]
        
        if self.aggregation_method == 'mean':
            # Weighted mean
            total_weight = sum(w for _, w in weighted_scores)
            aggregated_fitness = sum(s * w for s, w in weighted_scores) / total_weight
        
        elif self.aggregation_method == 'min':
            # Worst-case performance
            aggregated_fitness = min(fitness_values)
        
        elif self.aggregation_method == 'harmonic_mean':
            # Penalizes inconsistency (geometric interpretation: average rate)
            try:
                aggregated_fitness = harmonic_mean(fitness_values)
            except Exception:
                aggregated_fitness = sum(fitness_values) / len(fitness_values)
        
        elif self.aggregation_method == 'cvar':
            # Conditional Value at Risk: average of worst alpha% outcomes
            sorted_scores = sorted(fitness_values)
            n_worst = max(1, int(len(sorted_scores) * self.cvar_alpha))
            aggregated_fitness = sum(sorted_scores[:n_worst]) / n_worst
        
        else:
            # Fallback to mean
            aggregated_fitness = sum(fitness_values) / len(fitness_values)
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(results)
        aggregated_metrics['complexity'] = strategy_gene.calculate_complexity()
        
        return aggregated_fitness, aggregated_metrics
    
    def _aggregate_metrics(
        self,
        results: List[RegimeEvaluationResult],
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across all segment results.
        
        Args:
            results: List of segment evaluation results
            
        Returns:
            Aggregated metrics dictionary
        """
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {
                'profit': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'win_rate': 0.0,
                'num_trades': 0,
            }
        
        # Average most metrics
        aggregated = {}
        numeric_keys = ['profit', 'sharpe_ratio', 'sortino_ratio', 'profit_factor', 
                        'win_rate', 'num_trades']
        
        for key in numeric_keys:
            values = [r.metrics.get(key, 0) for r in successful_results if key in r.metrics]
            aggregated[key] = sum(values) / len(values) if values else 0.0
        
        # Max drawdown: use worst across segments
        drawdowns = [r.metrics.get('max_drawdown', 0) for r in successful_results]
        aggregated['max_drawdown'] = max(drawdowns) if drawdowns else 1.0
        
        # Add per-segment fitness values
        aggregated['segment_fitness_values'] = [r.fitness for r in results]
        aggregated['segment_success_rate'] = len(successful_results) / len(results)
        
        return aggregated
    
    def _get_regime_summary(
        self,
        results: List[RegimeEvaluationResult],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Summarize results by regime type.
        
        Args:
            results: List of segment evaluation results
            
        Returns:
            Dict mapping regime type to summary stats
        """
        summary: Dict[str, Dict[str, Any]] = {}
        
        for result in results:
            regime_type = result.segment.regime.value
            if regime_type not in summary:
                summary[regime_type] = {'fitness_values': [], 'count': 0}
            
            summary[regime_type]['fitness_values'].append(result.fitness)
            summary[regime_type]['count'] += 1
        
        # Calculate averages
        for regime_type, data in summary.items():
            values = data['fitness_values']
            data['avg_fitness'] = sum(values) / len(values) if values else 0.0
            data['min_fitness'] = min(values) if values else 0.0
            data['max_fitness'] = max(values) if values else 0.0
            del data['fitness_values']  # Remove raw values from summary
        
        return summary
    
    def evaluate_holdout(
        self,
        strategy_gene: StrategyGene,
        strategy_name: Optional[str] = None,
        auto_unlock: bool = True,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Convenience method to evaluate on holdout segments.
        
        Should only be called once at the end of evolution for final validation.
        By default, automatically unlocks holdout, evaluates, and re-locks.
        
        Args:
            strategy_gene: Strategy to evaluate
            strategy_name: Optional name
            auto_unlock: If True (default), temporarily unlock holdout for evaluation
            
        Returns:
            Tuple of (holdout_fitness, holdout_metrics)
        """
        if not self._holdout_segments:
            raise ValueError("No holdout segments configured")
        
        was_locked = self._holdout_locked
        
        try:
            if auto_unlock and was_locked:
                self.unlock_holdout()
            
            return self.evaluate(strategy_gene, strategy_name, use_holdout=True)
        finally:
            # Re-lock if it was locked before
            if auto_unlock and was_locked:
                self.lock_holdout()
    
    def get_holdout_protection_stats(self) -> Dict[str, Any]:
        """Get holdout protection statistics."""
        return {
            'holdout_locked': self._holdout_locked,
            'holdout_access_attempts': self._holdout_access_attempts,
            'holdout_segments_count': len(self._holdout_segments),
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_size': len(self._segment_cache),
            'hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses),
        }
    
    def clear_cache(self):
        """Clear segment cache."""
        self._segment_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Segment cache cleared")


def create_regime_aware_evaluator(
    config: Dict[str, Any],
    auto_detect: bool = True,
    data_path: Optional[str] = None,
    policy: Optional["DatasetPolicy"] = None,
) -> RegimeAwareEvaluator:
    """
    Factory function to create a RegimeAwareEvaluator with auto-detected segments.
    
    Args:
        config: GA configuration dictionary
        auto_detect: If True, auto-detect regimes from data (legacy behavior)
        data_path: Optional path to data directory (uses config if not provided)
        policy: Optional DatasetPolicy to use for segment creation
        
    Returns:
        Configured RegimeAwareEvaluator
    """
    from pathlib import Path
    from genetic_algorithm.utils.dataset_policy import DatasetPolicy, create_policy_from_config
    
    regime_config = config.get('regime_aware', {})
    
    if not regime_config.get('enabled', False):
        logger.info("Regime-aware evaluation disabled in config")
        # Return evaluator with no segments (will fall back to standard evaluation)
        return RegimeAwareEvaluator(config, segments={})
    
    segments = {}
    
    # Use provided policy, or create from config
    if policy is not None:
        logger.info(f"Using provided policy: {policy.describe()}")
        segments = policy.build_segments(
            config,
            data_path=Path(data_path) if data_path else None,
        )
    elif auto_detect:
        # Use DatasetPolicy for consistent behavior
        detected_policy = create_policy_from_config(config)
        logger.info(f"Using auto-detected policy: {detected_policy.describe()}")
        segments = detected_policy.build_segments(
            config,
            data_path=Path(data_path) if data_path else None,
        )
    
    return RegimeAwareEvaluator(config, segments=segments)


def _auto_detect_segments(
    config: Dict[str, Any],
    data_path: Optional[str] = None,
) -> Dict[str, List[RegimeSegment]]:
    """
    Auto-detect regime segments from historical data.
    
    Args:
        config: Configuration dictionary
        data_path: Optional data directory path
        
    Returns:
        Dict with 'optimization', 'model_selection', 'holdout' segment lists
    """
    from pathlib import Path
    
    regime_config = config.get('regime_aware', {})
    backtest_config = config.get('backtesting', {})
    
    # Get data path
    if data_path:
        datadir = Path(data_path)
    else:
        datadir = Path(backtest_config.get('datadir', 'user_data/data/binance'))
    
    # Get primary pair for regime detection (use first pair or benchmark pair)
    pairs = backtest_config.get('pairs', [])
    benchmark_pair = regime_config.get('benchmark_pair')
    if benchmark_pair is None:  # Handle explicit null in YAML
        benchmark_pair = pairs[0] if pairs else 'BTC/USDT'
    
    # Get timeframe (prefer 1h or 4h for regime detection, even if trading on 5m)
    timeframe = regime_config.get('detection_timeframe', '1h')
    
    # Get timerange
    timerange = backtest_config.get('timerange', '')
    
    logger.info(f"Auto-detecting regimes from {benchmark_pair} {timeframe} in {datadir}")
    
    try:
        # Load data
        df = load_ohlcv_data(
            pair=benchmark_pair,
            timeframe=timeframe,
            datadir=datadir,
            timerange=timerange,
        )
        
        if df.empty:
            logger.warning(f"No data loaded for {benchmark_pair} {timeframe}, no segments created")
            return {}
        
        # Create detector
        method = regime_config.get('method', 'sma_adx')
        detector = RegimeDetector(method=method)
        
        # Classify periods
        period_days = regime_config.get('period_days', 90)
        min_period_days = regime_config.get('min_period_days', 60)
        embargo_days = regime_config.get('embargo_days', 5)
        
        segments = detector.classify_periods(
            df=df,
            period_days=period_days,
            min_period_days=min_period_days,
            embargo_days=embargo_days,
        )
        
        if not segments:
            logger.warning("No segments created from regime detection")
            return {}
        
        # Get balanced segments
        segments_per_regime = regime_config.get('segments_per_regime', 3)
        balanced = detector.get_balanced_segments(
            segments,
            segments_per_regime=segments_per_regime,
        )
        
        # Split into train/holdout
        holdout_ratio = regime_config.get('holdout_ratio', 0.20)
        splits = detector.split_segments_by_role(
            balanced,
            optimization_ratio=1.0 - holdout_ratio,
            model_selection_ratio=0.0,
            holdout_ratio=holdout_ratio,
        )
        
        logger.info(
            f"Auto-detected segments: {len(splits['optimization'])} optimization, "
            f"{len(splits['holdout'])} holdout"
        )
        
        return splits
        
    except Exception as e:
        logger.error(f"Failed to auto-detect segments: {e}", exc_info=True)
        return {}
