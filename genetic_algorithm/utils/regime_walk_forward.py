"""
Regime-Aware Walk-Forward Optimization

This module enhances the existing walk-forward optimization to ensure
regime balance across windows, providing better generalization across
different market conditions.

Key features:
- Classify each walk-forward window by dominant regime
- Add/remove windows to ensure regime coverage
- Report per-regime metrics in walk-forward output
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

from genetic_algorithm.utils.timerange import TimeWindow, create_walk_forward_windows, parse_timerange, format_date
from genetic_algorithm.utils.regime_detector import RegimeDetector, RegimeType, RegimeSegment

logger = logging.getLogger(__name__)


@dataclass
class RegimeWindowInfo:
    """
    Extended walk-forward window with regime classification.
    
    Attributes:
        window: Original TimeWindow
        dominant_regime: The primary regime during this window
        regime_confidence: Confidence of the regime classification (0-1)
        regime_distribution: Percentage of each regime type in the window
        metadata: Additional metrics about the window
    """
    window: TimeWindow
    dominant_regime: RegimeType
    regime_confidence: float
    regime_distribution: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def window_index(self) -> int:
        return self.window.window_index
    
    @property
    def train_timerange(self) -> str:
        return self.window.train_timerange
    
    @property
    def val_timerange(self) -> str:
        return self.window.val_timerange
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'window_index': self.window_index,
            'train_timerange': self.train_timerange,
            'val_timerange': self.val_timerange,
            'dominant_regime': self.dominant_regime.value,
            'regime_confidence': self.regime_confidence,
            'regime_distribution': self.regime_distribution,
            **self.metadata
        }


@dataclass
class RegimeWalkForwardMetrics:
    """
    Aggregated metrics from regime-aware walk-forward evaluation.
    
    Attributes:
        overall_fitness: Aggregated fitness across all windows
        overall_metrics: Aggregated performance metrics
        per_window_results: Results for each window
        per_regime_metrics: Aggregated metrics grouped by regime
        regime_coverage: Which regimes were covered and how many windows each
        regime_balance_score: Score indicating how balanced the regime coverage is (0-1)
    """
    overall_fitness: float
    overall_metrics: Dict[str, float]
    per_window_results: List[Dict[str, Any]]
    per_regime_metrics: Dict[str, Dict[str, float]]
    regime_coverage: Dict[str, int]
    regime_balance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overall_fitness': self.overall_fitness,
            'overall_metrics': self.overall_metrics,
            'per_window_results': self.per_window_results,
            'per_regime_metrics': self.per_regime_metrics,
            'regime_coverage': self.regime_coverage,
            'regime_balance_score': self.regime_balance_score,
        }


class RegimeWalkForwardManager:
    """
    Manages regime-aware walk-forward optimization.
    
    Enhances standard walk-forward by:
    1. Classifying each window by dominant market regime
    2. Ensuring balanced regime coverage
    3. Computing per-regime performance metrics
    
    Usage:
        manager = RegimeWalkForwardManager(regime_config, ohlcv_data)
        windows = manager.create_regime_windows(walk_forward_config)
        # ... run backtests on windows ...
        metrics = manager.aggregate_results(window_results)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data: Optional[pd.DataFrame] = None,
        detector: Optional[RegimeDetector] = None
    ):
        """
        Initialize the regime walk-forward manager.
        
        Args:
            config: Configuration dictionary containing 'regime_aware' and 'walk_forward' sections
            data: Optional preloaded OHLCV data for regime detection
            detector: Optional pre-configured RegimeDetector
        """
        self.config = config
        self._data = data
        
        # Initialize regime detector if not provided
        regime_config = config.get('regime_aware', {})
        if detector is None:
            detection_method = regime_config.get('detection_method', 'sma_adx')
            self._detector = RegimeDetector(method=detection_method)
        else:
            self._detector = detector
        
        # Regime balancing configuration
        self._min_windows_per_regime = regime_config.get('min_windows_per_regime', 1)
        self._target_regimes = [
            RegimeType[r.upper()] for r in
            regime_config.get('target_regimes', ['bullish', 'bearish', 'sideways'])
        ]
        self._fill_missing_regimes = regime_config.get('fill_missing_regimes', True)
        
        # Cached regime series
        self._regime_series: Optional[pd.Series] = None
    
    def set_data(self, data: pd.DataFrame) -> None:
        """
        Set OHLCV data for regime detection.
        
        Args:
            data: DataFrame with OHLCV data and datetime index or 'date' column
        """
        self._data = data
        self._regime_series = None  # Clear cache
    
    def _ensure_regime_series(self) -> pd.Series:
        """
        Ensure regime series is computed and cached.
        
        Returns:
            Series with RegimeType for each bar, indexed by datetime
        """
        if self._regime_series is None:
            if self._data is None:
                raise ValueError("No data available. Call set_data() first.")
            
            # Ensure datetime index
            df = self._data.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                else:
                    df.index = pd.to_datetime(df.index)
            
            # Detect regimes
            self._regime_series = self._detector.detect(df)
            logger.info(f"Computed regime series with {len(self._regime_series)} bars")
        
        return self._regime_series
    
    def classify_window(
        self,
        window: TimeWindow,
        use_training: bool = True
    ) -> RegimeWindowInfo:
        """
        Classify a walk-forward window by dominant regime.
        
        Args:
            window: TimeWindow to classify
            use_training: If True, classify based on training period. 
                         If False, classify based on validation period.
        
        Returns:
            RegimeWindowInfo with regime classification
        """
        if self._data is None:
            raise ValueError("No data available. Call set_data() first.")
        
        regime_series = self._ensure_regime_series()
        
        # Determine which timerange to use
        if use_training:
            timerange = window.train_timerange
        else:
            timerange = window.val_timerange
        
        # Parse timerange and handle timezone
        start_str, end_str = timerange.split('-')
        start_date = pd.Timestamp(datetime.strptime(start_str, '%Y%m%d'))
        end_date = pd.Timestamp(datetime.strptime(end_str, '%Y%m%d'))
        
        # Match timezone of the regime_series index if needed
        if hasattr(regime_series.index, 'tz') and regime_series.index.tz is not None:
            start_date = start_date.tz_localize(regime_series.index.tz)
            end_date = end_date.tz_localize(regime_series.index.tz)
        
        # Get regime values for this window
        mask = (regime_series.index >= start_date) & (regime_series.index < end_date)
        window_regimes = regime_series[mask]
        
        if window_regimes.empty:
            logger.warning(f"No regime data for window {window.window_index}")
            return RegimeWindowInfo(
                window=window,
                dominant_regime=RegimeType.UNCERTAIN,
                regime_confidence=0.0,
                regime_distribution={},
                metadata={'error': 'no_data'}
            )
        
        # Calculate regime distribution
        regime_counts = window_regimes.value_counts()
        total_bars = len(window_regimes)
        
        regime_distribution = {
            regime.value: count / total_bars
            for regime, count in regime_counts.items()
        }
        
        # Filter out UNCERTAIN for dominant regime selection
        valid_regimes = regime_counts.drop(RegimeType.UNCERTAIN, errors='ignore')
        
        if valid_regimes.empty:
            dominant_regime = RegimeType.UNCERTAIN
            confidence = 0.0
        else:
            dominant_regime = valid_regimes.idxmax()
            confidence = valid_regimes[dominant_regime] / total_bars
        
        # Calculate additional metadata
        df = self._data
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
        
        # Make start_date/end_date compatible with df index timezone
        df_start = start_date
        df_end = end_date
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            if df_start.tz is None:
                df_start = df_start.tz_localize(df.index.tz)
            if df_end.tz is None:
                df_end = df_end.tz_localize(df.index.tz)
        
        window_mask = (df.index >= df_start) & (df.index < df_end)
        window_df = df[window_mask]
        
        metadata = {}
        if len(window_df) > 1:
            returns = window_df['close'].pct_change().dropna()
            metadata['total_return'] = (window_df['close'].iloc[-1] / window_df['close'].iloc[0] - 1)
            metadata['volatility'] = returns.std() if len(returns) > 0 else 0.0
            metadata['bar_count'] = len(window_df)
        
        return RegimeWindowInfo(
            window=window,
            dominant_regime=dominant_regime,
            regime_confidence=confidence,
            regime_distribution=regime_distribution,
            metadata=metadata
        )
    
    def classify_windows(
        self,
        windows: List[TimeWindow],
        use_training: bool = True
    ) -> List[RegimeWindowInfo]:
        """
        Classify multiple walk-forward windows.
        
        Args:
            windows: List of TimeWindow objects
            use_training: If True, classify based on training periods
        
        Returns:
            List of RegimeWindowInfo objects
        """
        return [self.classify_window(w, use_training) for w in windows]
    
    def create_regime_balanced_windows(
        self,
        walk_forward_config: Dict[str, Any],
        timerange: str
    ) -> Tuple[List[RegimeWindowInfo], Dict[str, int]]:
        """
        Create walk-forward windows with regime balance awareness.
        
        This method:
        1. Creates standard walk-forward windows
        2. Classifies each by dominant regime
        3. Optionally fills gaps to ensure each target regime has minimum coverage
        
        Args:
            walk_forward_config: Walk-forward configuration
            timerange: Full data timerange
        
        Returns:
            Tuple of:
            - List of RegimeWindowInfo (classified windows)
            - Dict mapping regime name to count of windows
        """
        # Create standard windows
        windows = create_walk_forward_windows(
            timerange=timerange,
            train_days=walk_forward_config['train_days'],
            validation_days=walk_forward_config['validation_days'],
            step_days=walk_forward_config['step_days'],
            mode=walk_forward_config.get('mode', 'rolling')
        )
        
        # Classify windows
        classified_windows = self.classify_windows(windows)
        
        # Count regimes
        regime_counts = defaultdict(int)
        for rw in classified_windows:
            regime_counts[rw.dominant_regime.value] += 1
        
        logger.info(f"Created {len(classified_windows)} walk-forward windows:")
        for regime, count in sorted(regime_counts.items()):
            logger.info(f"  {regime}: {count} windows")
        
        # Check if we need to fill missing regimes
        if self._fill_missing_regimes:
            missing_regimes = []
            for regime in self._target_regimes:
                if regime_counts.get(regime.value, 0) < self._min_windows_per_regime:
                    missing_regimes.append(regime)
            
            if missing_regimes:
                logger.warning(
                    f"Missing regime coverage for: {[r.value for r in missing_regimes]}. "
                    f"Consider adjusting walk-forward parameters or adding manual segments."
                )
        
        return classified_windows, dict(regime_counts)
    
    def compute_regime_balance_score(
        self,
        regime_counts: Dict[str, int]
    ) -> float:
        """
        Compute a score indicating how balanced the regime coverage is.
        
        A score of 1.0 means perfectly balanced (equal windows for each regime).
        A score closer to 0 means highly imbalanced.
        
        Args:
            regime_counts: Dict mapping regime name to window count
        
        Returns:
            Balance score between 0 and 1
        """
        if not regime_counts:
            return 0.0
        
        counts = list(regime_counts.values())
        if len(counts) == 0 or max(counts) == 0:
            return 0.0
        
        # Calculate normalized entropy
        total = sum(counts)
        if total == 0:
            return 0.0
        
        probabilities = [c / total for c in counts if c > 0]
        
        # Shannon entropy normalized by max possible entropy
        entropy = -sum(p * np.log(p) for p in probabilities)
        max_entropy = np.log(len(self._target_regimes))  # Max if all equal
        
        if max_entropy == 0:
            return 1.0
        
        return min(1.0, entropy / max_entropy)
    
    def aggregate_per_regime_metrics(
        self,
        window_results: List[Dict[str, Any]],
        classified_windows: List[RegimeWindowInfo]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate performance metrics grouped by regime.
        
        Args:
            window_results: List of result dicts from walk-forward evaluation
            classified_windows: Corresponding classified windows
        
        Returns:
            Dict mapping regime name to aggregated metrics
        """
        # Group results by regime
        regime_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for rw, result in zip(classified_windows, window_results):
            regime_results[rw.dominant_regime.value].append(result)
        
        # Aggregate metrics for each regime
        per_regime_metrics = {}
        
        for regime, results in regime_results.items():
            if not results:
                continue
            
            # Compute aggregated metrics
            metrics = {}
            
            # List of metrics to aggregate
            metric_keys = ['fitness', 'profit', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'num_trades']
            
            for key in metric_keys:
                values = [r.get(key, r.get('val_' + key, 0.0)) for r in results if key in r or 'val_' + key in r]
                if values:
                    metrics[f'{key}_mean'] = np.mean(values)
                    metrics[f'{key}_std'] = np.std(values) if len(values) > 1 else 0.0
                    metrics[f'{key}_min'] = np.min(values)
                    metrics[f'{key}_max'] = np.max(values)
            
            metrics['window_count'] = len(results)
            per_regime_metrics[regime] = metrics
        
        return per_regime_metrics
    
    def create_regime_walk_forward_report(
        self,
        window_results: List[Dict[str, Any]],
        classified_windows: List[RegimeWindowInfo],
        overall_fitness: float,
        overall_metrics: Dict[str, float]
    ) -> RegimeWalkForwardMetrics:
        """
        Create a comprehensive regime walk-forward report.
        
        Args:
            window_results: Results from each walk-forward window
            classified_windows: Classified windows with regime info
            overall_fitness: Aggregated overall fitness
            overall_metrics: Aggregated overall metrics
        
        Returns:
            RegimeWalkForwardMetrics object with full report
        """
        # Count regimes
        regime_counts = defaultdict(int)
        for rw in classified_windows:
            regime_counts[rw.dominant_regime.value] += 1
        
        # Compute per-regime metrics
        per_regime_metrics = self.aggregate_per_regime_metrics(
            window_results, classified_windows
        )
        
        # Compute balance score
        balance_score = self.compute_regime_balance_score(dict(regime_counts))
        
        # Build per-window results with regime info
        per_window_results = []
        for rw, result in zip(classified_windows, window_results):
            window_result = {
                'window_index': rw.window_index,
                'train_timerange': rw.train_timerange,
                'val_timerange': rw.val_timerange,
                'dominant_regime': rw.dominant_regime.value,
                'regime_confidence': rw.regime_confidence,
                'regime_distribution': rw.regime_distribution,
                **result
            }
            per_window_results.append(window_result)
        
        return RegimeWalkForwardMetrics(
            overall_fitness=overall_fitness,
            overall_metrics=overall_metrics,
            per_window_results=per_window_results,
            per_regime_metrics=per_regime_metrics,
            regime_coverage=dict(regime_counts),
            regime_balance_score=balance_score
        )


def add_regime_windows(
    existing_windows: List[RegimeWindowInfo],
    new_segments: List[RegimeSegment],
    timerange_overlaps: bool = False
) -> List[RegimeWindowInfo]:
    """
    Add windows from manual regime segments to fill gaps in coverage.
    
    Args:
        existing_windows: Currently classified walk-forward windows
        new_segments: Additional RegimeSegments to add (e.g., from manual policy)
        timerange_overlaps: If True, allows overlapping timeranges
    
    Returns:
        Extended list of RegimeWindowInfo
    """
    extended = list(existing_windows)
    
    # Track existing timeranges to avoid overlaps
    existing_ranges = set()
    if not timerange_overlaps:
        for rw in existing_windows:
            existing_ranges.add(rw.train_timerange)
    
    # Convert segments to RegimeWindowInfo
    base_idx = max(rw.window_index for rw in existing_windows) + 1 if existing_windows else 0
    
    for i, seg in enumerate(new_segments):
        # Create a pseudo-TimeWindow (training only, no validation)
        timerange = seg.timerange
        
        if not timerange_overlaps and timerange in existing_ranges:
            logger.debug(f"Skipping overlapping segment: {timerange}")
            continue
        
        # Parse segment dates
        start_str = seg.start_date.strftime('%Y%m%d')
        end_str = seg.end_date.strftime('%Y%m%d')
        
        pseudo_window = TimeWindow(
            train_start=start_str,
            train_end=end_str,
            val_start=end_str,  # No validation for manual segments
            val_end=end_str,
            window_index=base_idx + i
        )
        
        regime_window = RegimeWindowInfo(
            window=pseudo_window,
            dominant_regime=seg.regime,
            regime_confidence=seg.confidence,
            regime_distribution={seg.regime.value: 1.0},
            metadata={
                'source': 'manual_segment',
                'segment_id': seg.segment_id
            }
        )
        
        extended.append(regime_window)
    
    return extended


def format_regime_walk_forward_summary(metrics: RegimeWalkForwardMetrics) -> str:
    """
    Format a human-readable summary of regime walk-forward results.
    
    Args:
        metrics: RegimeWalkForwardMetrics from evaluation
    
    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("=" * 70)
    lines.append("REGIME-AWARE WALK-FORWARD SUMMARY")
    lines.append("=" * 70)
    
    # Overall performance
    lines.append(f"\nOverall Fitness: {metrics.overall_fitness:.4f}")
    lines.append(f"Regime Balance Score: {metrics.regime_balance_score:.2%}")
    
    # Regime coverage
    lines.append("\nRegime Coverage:")
    for regime, count in sorted(metrics.regime_coverage.items()):
        lines.append(f"  {regime:10s}: {count} windows")
    
    # Per-regime performance
    lines.append("\nPer-Regime Performance:")
    for regime, regime_metrics in sorted(metrics.per_regime_metrics.items()):
        lines.append(f"\n  {regime.upper()}:")
        windows = regime_metrics.get('window_count', 0)
        fitness = regime_metrics.get('fitness_mean', 0)
        profit = regime_metrics.get('profit_mean', 0)
        lines.append(f"    Windows: {windows}")
        lines.append(f"    Mean Fitness: {fitness:.4f}")
        lines.append(f"    Mean Profit: {profit:.2f}%")
    
    # Individual window breakdown (optional detail)
    lines.append("\nPer-Window Results:")
    for wr in metrics.per_window_results:
        idx = wr.get('window_index', '?')
        regime = wr.get('dominant_regime', 'unknown')
        conf = wr.get('regime_confidence', 0.0)
        fitness = wr.get('val_fitness', wr.get('fitness', 0))
        lines.append(f"  Window {idx}: {regime:10s} ({conf:.0%}) - fitness={fitness:.4f}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)
