"""Utility functions and helpers."""

from .timerange import (
    TimeWindow,
    create_walk_forward_windows,
    parse_timerange,
    format_date,
)
from .regime_detector import (
    RegimeDetector,
    RegimeType,
    RegimeSegment,
    load_ohlcv_data,
    save_segments_to_yaml,
    load_segments_from_yaml,
)
from .dataset_policy import (
    DatasetPolicy,
    PolicyMode,
    PolicyConfig,
    ManualPolicy,
    AutoRegimePolicy,
    AutoHoldoutPolicy,
    create_policy_from_config,
)
from .regime_walk_forward import (
    RegimeWindowInfo,
    RegimeWalkForwardMetrics,
    RegimeWalkForwardManager,
    add_regime_windows,
    format_regime_walk_forward_summary,
)

__all__ = [
    # Timerange utilities
    'TimeWindow',
    'create_walk_forward_windows',
    'parse_timerange',
    'format_date',
    # Regime detection
    'RegimeDetector',
    'RegimeType',
    'RegimeSegment',
    'load_ohlcv_data',
    'save_segments_to_yaml',
    'load_segments_from_yaml',
    # Dataset policies
    'DatasetPolicy',
    'PolicyMode',
    'PolicyConfig',
    'ManualPolicy',
    'AutoRegimePolicy',
    'AutoHoldoutPolicy',
    'create_policy_from_config',
    # Regime walk-forward
    'RegimeWindowInfo',
    'RegimeWalkForwardMetrics',
    'RegimeWalkForwardManager',
    'add_regime_windows',
    'format_regime_walk_forward_summary',
]
