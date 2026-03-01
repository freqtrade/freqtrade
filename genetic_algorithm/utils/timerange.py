"""
Timerange Utilities for Walk-Forward Optimization

This module provides utilities for splitting timeranges into training and validation
windows for walk-forward optimization.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TimeWindow:
    """Represents a single training/validation window pair."""
    
    train_start: str  # Format: 'YYYYMMDD'
    train_end: str    # Format: 'YYYYMMDD'
    val_start: str    # Format: 'YYYYMMDD'
    val_end: str      # Format: 'YYYYMMDD'
    window_index: int  # Window number (0-based)
    
    @property
    def train_timerange(self) -> str:
        """Get FreqTrade-compatible training timerange string."""
        return f"{self.train_start}-{self.train_end}"
    
    @property
    def val_timerange(self) -> str:
        """Get FreqTrade-compatible validation timerange string."""
        return f"{self.val_start}-{self.val_end}"
    
    def __str__(self) -> str:
        """String representation for logging."""
        return (f"Window {self.window_index}: "
                f"Train={self.train_timerange}, Val={self.val_timerange}")


def parse_timerange(timerange: str) -> Tuple[datetime, datetime]:
    """
    Parse FreqTrade timerange string to datetime objects.
    
    Args:
        timerange: String in format 'YYYYMMDD-YYYYMMDD'
        
    Returns:
        Tuple of (start_datetime, end_datetime)
        
    Raises:
        ValueError: If timerange format is invalid
    """
    try:
        parts = timerange.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid timerange format: {timerange}. Expected 'YYYYMMDD-YYYYMMDD'")
        
        start_str, end_str = parts
        start_date = datetime.strptime(start_str, '%Y%m%d')
        end_date = datetime.strptime(end_str, '%Y%m%d')
        
        if end_date <= start_date:
            raise ValueError(f"End date must be after start date: {timerange}")
        
        return start_date, end_date
        
    except ValueError as e:
        raise ValueError(f"Error parsing timerange '{timerange}': {str(e)}")


def format_date(dt: datetime) -> str:
    """
    Format datetime to FreqTrade timerange format.
    
    Args:
        dt: Datetime object
        
    Returns:
        Date string in format 'YYYYMMDD'
    """
    return dt.strftime('%Y%m%d')


def create_walk_forward_windows(
    timerange: str,
    train_days: int,
    validation_days: int,
    step_days: int,
    mode: str = 'rolling',
    min_train_days: Optional[int] = None,
    embargo_days: int = 0,
    max_windows: Optional[int] = None
) -> List[TimeWindow]:
    """
    Create walk-forward optimization windows from a timerange.
    
    Args:
        timerange: Full timerange string in format 'YYYYMMDD-YYYYMMDD'
        train_days: Number of days for training window
        validation_days: Number of days for validation window
        step_days: Number of days to slide forward for each window
        mode: Window mode - 'rolling' (fixed window) or 'anchored' (expanding window)
        min_train_days: Minimum training days required (defaults to train_days)
        embargo_days: Number of days gap between training and validation windows
                      to prevent autocorrelated data leakage (default: 0)
        
    Returns:
        List of TimeWindow objects
        
    Raises:
        ValueError: If parameters are invalid
        
    Example:
        >>> windows = create_walk_forward_windows(
        ...     timerange='20230101-20230331',
        ...     train_days=60,
        ...     validation_days=15,
        ...     step_days=15,
        ...     mode='rolling'
        ... )
        >>> len(windows)
        3
        >>> print(windows[0])
        Window 0: Train=20230101-20230302, Val=20230302-20230317
    """
    # Validate parameters
    if train_days <= 0:
        raise ValueError(f"train_days must be positive, got {train_days}")
    if validation_days <= 0:
        raise ValueError(f"validation_days must be positive, got {validation_days}")
    if step_days <= 0:
        raise ValueError(f"step_days must be positive, got {step_days}")
    if mode not in ['rolling', 'anchored']:
        raise ValueError(f"mode must be 'rolling' or 'anchored', got '{mode}'")
    
    if min_train_days is None:
        min_train_days = train_days
    
    if embargo_days < 0:
        raise ValueError(f"embargo_days must be non-negative, got {embargo_days}")
    
    # Parse timerange
    full_start, full_end = parse_timerange(timerange)
    logger.info(f"Creating walk-forward windows from {format_date(full_start)} to {format_date(full_end)}")
    logger.info(f"Parameters: train_days={train_days}, validation_days={validation_days}, "
                f"step_days={step_days}, mode={mode}, embargo_days={embargo_days}")
    
    windows = []
    window_index = 0
    current_start = full_start
    
    while True:
        # Calculate training window
        if mode == 'rolling':
            # Rolling window: fixed size training window
            train_start = current_start
            train_end = train_start + timedelta(days=train_days)
        else:  # anchored
            # Anchored window: expanding training window from start
            train_start = full_start
            train_end = current_start + timedelta(days=train_days)
        
        # Calculate validation window (with embargo gap to prevent data leakage)
        val_start = train_end + timedelta(days=embargo_days)
        val_end = val_start + timedelta(days=validation_days)
        
        # Check if we have enough data for this window
        if val_end > full_end:
            logger.info(f"Stopping: validation window would exceed available data "
                       f"(val_end={format_date(val_end)}, full_end={format_date(full_end)})")
            break
        
        # Check minimum training days
        actual_train_days = (train_end - train_start).days
        if actual_train_days < min_train_days:
            logger.warning(f"Skipping window {window_index}: insufficient training days "
                          f"({actual_train_days} < {min_train_days})")
            current_start += timedelta(days=step_days)
            continue
        
        # Create window
        window = TimeWindow(
            train_start=format_date(train_start),
            train_end=format_date(train_end),
            val_start=format_date(val_start),
            val_end=format_date(val_end),
            window_index=window_index
        )
        
        windows.append(window)
        logger.debug(f"Created {window}")
        
        # Move to next window
        current_start += timedelta(days=step_days)
        window_index += 1
        
        # Configurable max_windows cap
        if max_windows and window_index >= max_windows:
            logger.info(f"Reached max_windows cap ({max_windows}), stopping window creation")
            break
        
        # Safety check to prevent infinite loops
        if window_index > 1000:
            logger.error("Exceeded maximum window count (1000). Check parameters.")
            raise ValueError("Too many windows generated. Check step_days parameter.")
    
    if not windows:
        raise ValueError(
            f"No valid windows could be created with the given parameters. "
            f"Timerange: {timerange}, train_days: {train_days}, "
            f"validation_days: {validation_days}, step_days: {step_days}"
        )
    
    logger.info(f"Created {len(windows)} walk-forward windows")
    return windows


def validate_walk_forward_config(config: Dict[str, Any]) -> None:
    """
    Validate walk-forward configuration.
    
    Args:
        config: Walk-forward configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not config.get('enabled', False):
        return  # Walk-forward disabled, no validation needed
    
    required_keys = ['train_days', 'validation_days', 'step_days']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Walk-forward config missing required key: '{key}'")
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"Walk-forward config '{key}' must be a positive integer, got {config[key]}")
    
    # Validate mode
    mode = config.get('mode', 'rolling')
    if mode not in ['rolling', 'anchored']:
        raise ValueError(f"Walk-forward mode must be 'rolling' or 'anchored', got '{mode}'")
    
    # Validate aggregation method
    aggregation = config.get('aggregation', 'mean')
    valid_aggregations = ['mean', 'min', 'harmonic_mean', 'weighted']
    if aggregation not in valid_aggregations:
        raise ValueError(f"Walk-forward aggregation must be one of {valid_aggregations}, got '{aggregation}'")
    
    # Validate min_train_trades
    min_train_trades = config.get('min_train_trades', 10)
    if not isinstance(min_train_trades, int) or min_train_trades < 0:
        raise ValueError(f"min_train_trades must be a non-negative integer, got {min_train_trades}")
    
    logger.info("Walk-forward configuration validated successfully")


def aggregate_validation_scores(
    scores: List[float],
    method: str = 'mean',
    weights: Optional[List[float]] = None
) -> float:
    """
    Aggregate validation scores from multiple windows.
    
    Args:
        scores: List of validation fitness scores (one per window)
        method: Aggregation method - 'mean', 'min', 'harmonic_mean', or 'weighted'
        weights: Optional weights for 'weighted' method (must sum to 1.0)
        
    Returns:
        Aggregated fitness score
        
    Raises:
        ValueError: If method is invalid or weights don't match scores
        
    Examples:
        >>> aggregate_validation_scores([0.8, 0.7, 0.9], method='mean')
        0.8
        >>> aggregate_validation_scores([0.8, 0.7, 0.9], method='min')
        0.7
        >>> aggregate_validation_scores([0.8, 0.7, 0.9], method='harmonic_mean')
        0.7826...
    """
    if not scores:
        return 0.0
    
    # Filter out invalid scores (NaN, negative, etc.)
    valid_scores = [s for s in scores if isinstance(s, (int, float)) and s >= 0]
    
    if not valid_scores:
        logger.warning("No valid scores to aggregate, returning 0.0")
        return 0.0
    
    if method == 'mean':
        return sum(valid_scores) / len(valid_scores)
    
    elif method == 'min':
        # Conservative: worst-case performance
        return min(valid_scores)
    
    elif method == 'harmonic_mean':
        # Penalizes inconsistency more than arithmetic mean
        if any(s == 0 for s in valid_scores):
            return 0.0
        reciprocal_sum = sum(1.0 / s for s in valid_scores)
        return len(valid_scores) / reciprocal_sum
    
    elif method == 'weighted':
        if weights is None:
            raise ValueError("Weights must be provided for 'weighted' aggregation")
        if len(weights) != len(scores):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of scores ({len(scores)})")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
        
        # Calculate weighted sum, only using valid scores
        weighted_sum = 0.0
        total_weight = 0.0
        for i, score in enumerate(scores):
            # Check validity criteria directly
            if isinstance(score, (int, float)) and score >= 0:
                weighted_sum += weights[i] * score
                total_weight += weights[i]
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}. "
                        f"Must be 'mean', 'min', 'harmonic_mean', or 'weighted'")


if __name__ == '__main__':
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Rolling windows
    print("=== Rolling Window Example ===")
    windows = create_walk_forward_windows(
        timerange='20230101-20230331',
        train_days=60,
        validation_days=15,
        step_days=15,
        mode='rolling'
    )
    
    for window in windows:
        print(f"  {window}")
    
    # Example 2: Anchored windows
    print("\n=== Anchored Window Example ===")
    windows = create_walk_forward_windows(
        timerange='20230101-20230331',
        train_days=30,
        validation_days=10,
        step_days=10,
        mode='anchored'
    )
    
    for window in windows:
        print(f"  {window}")
    
    # Example 3: Aggregation
    print("\n=== Aggregation Example ===")
    scores = [0.8, 0.7, 0.9, 0.75]
    print(f"Scores: {scores}")
    print(f"Mean: {aggregate_validation_scores(scores, 'mean'):.4f}")
    print(f"Min: {aggregate_validation_scores(scores, 'min'):.4f}")
    print(f"Harmonic Mean: {aggregate_validation_scores(scores, 'harmonic_mean'):.4f}")
    print(f"Weighted: {aggregate_validation_scores(scores, 'weighted', [0.1, 0.2, 0.3, 0.4]):.4f}")
