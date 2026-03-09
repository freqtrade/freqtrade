"""
Dataset Policy Module

Provides a clean abstraction for regime-balanced dataset selection.
This module defines how training/validation/holdout segments are created
based on different policies (manual, auto_regime, auto_holdout).

Usage:
    policy = DatasetPolicy.create(config)
    segments = policy.build_segments(data_path)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum

from genetic_algorithm.utils.regime_detector import (
    RegimeDetector,
    RegimeSegment,
    RegimeType,
    load_ohlcv_data,
)
from genetic_algorithm.utils.mtf_regime_detector import MTFRegimeDetector

logger = logging.getLogger(__name__)


class PolicyMode(Enum):
    """Dataset selection policy modes."""
    MANUAL = "manual"           # User-supplied timeranges
    AUTO_REGIME = "auto_regime" # Automatic regime detection + balanced sampling
    AUTO_HOLDOUT = "auto_holdout"  # Auto-regime + holdout reservation (default)


@dataclass
class PolicyConfig:
    """Configuration for dataset policy."""
    mode: PolicyMode = PolicyMode.AUTO_HOLDOUT
    
    # Manual mode: user-specified segments
    manual_segments: Optional[Dict[str, List[Dict]]] = None
    
    # Auto mode parameters
    benchmark_pair: Optional[str] = None
    detection_timeframe: str = "1h"
    detection_method: str = "adx_di_hysteresis"
    detection_params: Optional[Dict[str, Any]] = None
    
    # Segment creation
    period_days: int = 30
    min_period_days: int = 20
    embargo_days: int = 5
    segments_per_regime: int = 3
    
    # Split ratios
    optimization_ratio: float = 0.70
    model_selection_ratio: float = 0.10
    holdout_ratio: float = 0.20

    # MTF settings
    mtf_enabled: bool = False
    segmentation: str = 'adaptive'  # 'adaptive' or 'fixed'
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PolicyConfig":
        """Create PolicyConfig from GA configuration dict."""
        regime_config = config.get('regime_aware', {})
        backtest_config = config.get('backtesting', {})
        
        mode_str = regime_config.get('policy_mode', 'auto_holdout')
        try:
            mode = PolicyMode(mode_str)
        except ValueError:
            logger.warning(f"Unknown policy mode '{mode_str}', using 'auto_holdout'")
            mode = PolicyMode.AUTO_HOLDOUT
        
        # Get pairs for fallback
        pairs = backtest_config.get('pairs', [])
        default_pair = pairs[0] if pairs else 'BTC/USDT'
        
        return cls(
            mode=mode,
            manual_segments=regime_config.get('manual_segments'),
            benchmark_pair=regime_config.get('benchmark_pair') or default_pair,
            detection_timeframe=regime_config.get('detection_timeframe', '1h'),
            detection_method=regime_config.get('method', 'adx_di_hysteresis'),
            detection_params=regime_config.get('detection_params'),
            period_days=regime_config.get('period_days', 30),
            min_period_days=regime_config.get('min_period_days', 20),
            embargo_days=regime_config.get('embargo_days', 5),
            segments_per_regime=regime_config.get('segments_per_regime', 3),
            optimization_ratio=1.0 - regime_config.get('holdout_ratio', 0.20),
            mtf_enabled=regime_config.get('mtf_enabled', False),
            segmentation=regime_config.get('segmentation', 'adaptive'),
            model_selection_ratio=0.0,
            holdout_ratio=regime_config.get('holdout_ratio', 0.20),
        )


class DatasetPolicy(ABC):
    """
    Abstract base class for dataset selection policies.
    
    A DatasetPolicy determines how market data is split into segments
    for optimization, model selection, and holdout evaluation.
    """
    
    @abstractmethod
    def build_segments(
        self,
        config: Dict[str, Any],
        data_path: Optional[Path] = None,
    ) -> Dict[str, List[RegimeSegment]]:
        """
        Build segments for optimization, model selection, and holdout.
        
        Args:
            config: GA configuration dictionary
            data_path: Optional path to data directory
            
        Returns:
            Dict with keys 'optimization', 'model_selection', 'holdout',
            each containing a list of RegimeSegment objects.
        """
        pass
    
    @abstractmethod
    def describe(self) -> str:
        """Return human-readable description of the policy."""
        pass
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> "DatasetPolicy":
        """
        Factory method to create appropriate policy based on config.
        
        Args:
            config: GA configuration dictionary
            
        Returns:
            Appropriate DatasetPolicy subclass instance
        """
        policy_config = PolicyConfig.from_config(config)
        
        if policy_config.mode == PolicyMode.MANUAL:
            return ManualPolicy(policy_config)
        elif policy_config.mode == PolicyMode.AUTO_REGIME:
            return AutoRegimePolicy(policy_config)
        else:  # AUTO_HOLDOUT (default)
            return AutoHoldoutPolicy(policy_config)


class ManualPolicy(DatasetPolicy):
    """
    Manual dataset policy with user-specified timeranges.
    
    Users provide explicit segment definitions in config:
    
    regime_aware:
      policy_mode: 'manual'
      manual_segments:
        optimization:
          - start: '20230101'
            end: '20230401'
            regime: 'bullish'
          - start: '20230601'
            end: '20230901'
            regime: 'sideways'
        holdout:
          - start: '20231001'
            end: '20231231'
            regime: 'bearish'
    """
    
    def __init__(self, policy_config: PolicyConfig):
        self.config = policy_config
    
    def build_segments(
        self,
        config: Dict[str, Any],
        data_path: Optional[Path] = None,
    ) -> Dict[str, List[RegimeSegment]]:
        """Build segments from manual configuration."""
        from datetime import datetime
        
        result = {
            'optimization': [],
            'model_selection': [],
            'holdout': [],
        }
        
        manual = self.config.manual_segments
        if not manual:
            logger.warning("Manual policy selected but no manual_segments provided")
            return result
        
        def parse_segment(seg_dict: Dict, idx: int, role: str) -> Optional[RegimeSegment]:
            """Parse a segment dictionary into RegimeSegment."""
            try:
                start_str = seg_dict.get('start', '')
                end_str = seg_dict.get('end', '')
                regime_str = seg_dict.get('regime', 'sideways')
                
                # Parse dates (support YYYYMMDD or YYYY-MM-DD)
                start_str = start_str.replace('-', '')
                end_str = end_str.replace('-', '')
                
                start_date = datetime.strptime(start_str, '%Y%m%d')
                end_date = datetime.strptime(end_str, '%Y%m%d')
                
                # Parse regime
                regime_map = {
                    'bullish': RegimeType.BULLISH,
                    'bearish': RegimeType.BEARISH,
                    'sideways': RegimeType.SIDEWAYS,
                    'volatile': RegimeType.VOLATILE,
                }
                regime = regime_map.get(regime_str.lower(), RegimeType.SIDEWAYS)
                
                return RegimeSegment(
                    segment_id=f"manual_{role}_{idx}",
                    start_date=start_date,
                    end_date=end_date,
                    regime=regime,
                    confidence=1.0,
                    metadata={'source': 'manual'},
                    role=role,
                )
            except Exception as e:
                logger.error(f"Failed to parse manual segment {seg_dict}: {e}")
                return None
        
        for role in ['optimization', 'model_selection', 'holdout']:
            segments_list = manual.get(role, [])
            for idx, seg_dict in enumerate(segments_list):
                segment = parse_segment(seg_dict, idx, role)
                if segment:
                    result[role].append(segment)
        
        logger.info(f"Manual policy: {len(result['optimization'])} opt, "
                    f"{len(result['model_selection'])} model_sel, "
                    f"{len(result['holdout'])} holdout segments")
        
        return result
    
    def describe(self) -> str:
        return "Manual: User-specified segment timeranges"


class AutoRegimePolicy(DatasetPolicy):
    """
    Automatic regime detection with balanced sampling.
    
    This policy:
    1. Loads market data from the data directory
    2. Runs regime detection (SMA+ADX, ADX+DI, etc.)
    3. Classifies data into segments by dominant regime
    4. Selects balanced segments (equal from each regime type)
    
    Note: No holdout reservation - all segments used for optimization.
    """
    
    def __init__(self, policy_config: PolicyConfig):
        self.config = policy_config
    
    def build_segments(
        self,
        config: Dict[str, Any],
        data_path: Optional[Path] = None,
    ) -> Dict[str, List[RegimeSegment]]:
        """Build segments using automatic regime detection."""
        backtest_config = config.get('backtesting', {})
        
        # Determine data path
        if data_path:
            datadir = Path(data_path)
        else:
            datadir = Path(backtest_config.get('datadir', 'user_data/data/binance'))
        
        # Load data
        timerange = backtest_config.get('timerange', '')
        
        logger.info(f"AutoRegimePolicy: Loading {self.config.benchmark_pair} "
                    f"{self.config.detection_timeframe} from {datadir}")
        
        try:
            df = load_ohlcv_data(
                pair=self.config.benchmark_pair,
                timeframe=self.config.detection_timeframe,
                datadir=datadir,
                timerange=timerange,
            )
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return {'optimization': [], 'model_selection': [], 'holdout': []}
        
        if df.empty:
            logger.warning("No data loaded, returning empty segments")
            return {'optimization': [], 'model_selection': [], 'holdout': []}
        
        # Create detector with configured params
        params = self.config.detection_params or {}
        detector = RegimeDetector(
            method=self.config.detection_method,
            params=params,
        )
        
        # Classify periods — choose segmentation mode
        segmentation = self.config.segmentation
        if segmentation == 'adaptive':
            all_segments = detector.classify_periods_adaptive(
                df=df,
                min_segment_days=max(14, self.config.min_period_days // 2),
                max_segment_days=self.config.period_days * 2,
                merge_threshold_days=7,
                embargo_days=self.config.embargo_days,
            )
        else:
            all_segments = detector.classify_periods(
                df=df,
                period_days=self.config.period_days,
                min_period_days=self.config.min_period_days,
                embargo_days=self.config.embargo_days,
            )
        
        if not all_segments:
            logger.warning("No segments created from regime detection")
            return {'optimization': [], 'model_selection': [], 'holdout': []}
        
        # Get balanced segments
        balanced = detector.get_balanced_segments(
            all_segments,
            segments_per_regime=self.config.segments_per_regime,
        )
        
        logger.info(f"AutoRegimePolicy ({segmentation}): {len(balanced)} balanced segments (no holdout)")
        
        return {
            'optimization': balanced,
            'model_selection': [],
            'holdout': [],
        }
    
    def describe(self) -> str:
        return (f"Auto-Regime: {self.config.detection_method} detection, "
                f"{self.config.period_days}-day periods, "
                f"{self.config.segments_per_regime} per regime")


class AutoHoldoutPolicy(DatasetPolicy):
    """
    Automatic regime detection with holdout reservation (default).
    
    This policy extends AutoRegimePolicy by:
    1. Reserving a portion of segments for holdout evaluation
    2. Ensuring holdout segments are never seen during optimization
    3. Final validation only uses holdout segments
    
    Supports two segmentation modes:
    - 'adaptive' (default): change-point based, variable-length segments
    - 'fixed': fixed-width windows of period_days length (legacy)
    
    When mtf_enabled=true, uses MTFRegimeDetector for multi-timeframe
    regime classification with continuous scores and enriched metadata.
    
    This is the recommended policy for production use.
    """
    
    def __init__(self, policy_config: PolicyConfig):
        self.config = policy_config
    
    def build_segments(
        self,
        config: Dict[str, Any],
        data_path: Optional[Path] = None,
    ) -> Dict[str, List[RegimeSegment]]:
        """Build segments with train/holdout split."""
        backtest_config = config.get('backtesting', {})
        
        # Determine data path
        if data_path:
            datadir = Path(data_path)
        else:
            datadir = Path(backtest_config.get('datadir', 'user_data/data/binance'))
        
        timerange = backtest_config.get('timerange', '')
        
        # --- MTF path: use MTFRegimeDetector for multi-timeframe detection ---
        if self.config.mtf_enabled:
            return self._build_segments_mtf(config, datadir, timerange)
        
        # --- Single-TF path (existing behavior) ---
        logger.info(f"AutoHoldoutPolicy: Loading {self.config.benchmark_pair} "
                    f"{self.config.detection_timeframe} from {datadir}")
        
        try:
            df = load_ohlcv_data(
                pair=self.config.benchmark_pair,
                timeframe=self.config.detection_timeframe,
                datadir=datadir,
                timerange=timerange,
            )
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return {'optimization': [], 'model_selection': [], 'holdout': []}
        
        if df.empty:
            logger.warning("No data loaded, returning empty segments")
            return {'optimization': [], 'model_selection': [], 'holdout': []}
        
        # Create detector with configured params
        params = self.config.detection_params or {}
        detector = RegimeDetector(
            method=self.config.detection_method,
            params=params,
        )
        
        # Choose segmentation mode
        segmentation = self.config.segmentation
        if segmentation == 'adaptive':
            all_segments = detector.classify_periods_adaptive(
                df=df,
                min_segment_days=max(14, self.config.min_period_days // 2),
                max_segment_days=self.config.period_days * 2,
                merge_threshold_days=7,
                embargo_days=self.config.embargo_days,
            )
        else:
            all_segments = detector.classify_periods(
                df=df,
                period_days=self.config.period_days,
                min_period_days=self.config.min_period_days,
                embargo_days=self.config.embargo_days,
            )
        
        if not all_segments:
            logger.warning("No segments created from regime detection")
            return {'optimization': [], 'model_selection': [], 'holdout': []}
        
        # Get balanced segments
        balanced = detector.get_balanced_segments(
            all_segments,
            segments_per_regime=self.config.segments_per_regime,
        )
        
        # Split into optimization/holdout
        splits = detector.split_segments_by_role(
            balanced,
            optimization_ratio=self.config.optimization_ratio,
            model_selection_ratio=self.config.model_selection_ratio,
            holdout_ratio=self.config.holdout_ratio,
        )
        
        logger.info(f"AutoHoldoutPolicy ({segmentation}): "
                    f"{len(splits.get('optimization', []))} opt, "
                    f"{len(splits.get('holdout', []))} holdout segments")
        
        return splits
    
    def _build_segments_mtf(
        self,
        config: Dict[str, Any],
        datadir: Path,
        timerange: str,
    ) -> Dict[str, List[RegimeSegment]]:
        """
        Build segments using multi-timeframe regime detection.
        
        Runs MTFRegimeDetector with continuous scores, then classifies
        adaptive segments with enriched metadata (trend_score,
        volatility_score, regime_context).
        """
        logger.info(
            f"AutoHoldoutPolicy (MTF): Running multi-timeframe detection for "
            f"{self.config.benchmark_pair}"
        )
        
        try:
            mtf_detector = MTFRegimeDetector(config)
            result = mtf_detector.detect(
                benchmark_pair=self.config.benchmark_pair,
                datadir=datadir,
                timerange=timerange or None,
            )
        except Exception as e:
            logger.error(f"MTF detection failed, falling back to single-TF: {e}")
            # Fall back to single-TF adaptive detection
            self_copy = AutoHoldoutPolicy(PolicyConfig(
                **{k: v for k, v in self.config.__dict__.items() if k != 'mtf_enabled'},
                mtf_enabled=False,
            ))
            return self_copy.build_segments(config, datadir)
        
        # Load base-TF OHLCV for segment metadata computation
        target_tf = result.metadata.get('target_timeframe', self.config.detection_timeframe)
        df = load_ohlcv_data(
            pair=self.config.benchmark_pair,
            timeframe=target_tf,
            datadir=datadir,
            timerange=timerange or None,
        )
        
        # Build adaptive segments from MTF result
        all_segments = mtf_detector.classify_segments(
            result=result,
            df=df,
            min_segment_days=max(14, self.config.min_period_days // 2),
            max_segment_days=self.config.period_days * 2,
            merge_threshold_days=7,
            embargo_days=self.config.embargo_days,
        )
        
        if not all_segments:
            logger.warning("MTF: No segments created from detection")
            return {'optimization': [], 'model_selection': [], 'holdout': []}
        
        # Use the existing RegimeDetector for balancing & splitting
        detector = RegimeDetector(method=self.config.detection_method)
        balanced = detector.get_balanced_segments(
            all_segments,
            segments_per_regime=self.config.segments_per_regime,
        )
        
        splits = detector.split_segments_by_role(
            balanced,
            optimization_ratio=self.config.optimization_ratio,
            model_selection_ratio=self.config.model_selection_ratio,
            holdout_ratio=self.config.holdout_ratio,
        )
        
        logger.info(
            f"AutoHoldoutPolicy (MTF): {len(splits.get('optimization', []))} opt, "
            f"{len(splits.get('holdout', []))} holdout segments"
        )
        
        return splits
    
    def describe(self) -> str:
        mode = 'MTF' if self.config.mtf_enabled else self.config.segmentation
        return (f"Auto-Holdout ({mode}): {self.config.detection_method} detection, "
                f"{self.config.period_days}-day periods, "
                f"{int(self.config.holdout_ratio*100)}% holdout")


def create_policy_from_config(config: Dict[str, Any]) -> DatasetPolicy:
    """
    Convenience function to create a DatasetPolicy from config.
    
    Args:
        config: GA configuration dictionary
        
    Returns:
        Configured DatasetPolicy instance
    """
    return DatasetPolicy.create(config)
