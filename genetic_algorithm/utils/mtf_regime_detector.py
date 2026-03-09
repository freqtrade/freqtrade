"""
Multi-Timeframe (MTF) Regime Detection Engine

Combines regime classifications from multiple timeframes (e.g. 30m, 1h, 4h, 1d)
into a single composite regime signal.  Two fusion strategies are supported:

- **hierarchical**: Higher timeframes anchor the macro regime; lower timeframes
  can only refine within a bounded range of the higher-TF score.  This prevents
  short-term noise from overriding the dominant trend.

- **weighted_voting**: Each timeframe is assigned a weight, and the composite
  score is the weighted average of per-TF continuous scores.  This gives a
  balanced view without strict dominance.

The output is a pair of continuous scores:
  - ``trend_score`` in [-1, +1]
  - ``volatility_score`` in [0, 1]

plus a human-readable ``regime_context`` string (e.g. "strong_bullish",
"bullish_pullback", "volatile_sideways") for richer metadata.

Additionally, the module provides **regime transition detection** that computes
rate-of-change signals on the composite trend_score, enabling strategies to
react to regime *changes* rather than only to the current regime.

Usage:
    detector = MTFRegimeDetector(config)
    result = detector.detect(benchmark_pair, datadir, timerange)
    # result.trend_score, result.volatility_score, result.regime_context,
    # result.transition_speed, result.transition_signal
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from genetic_algorithm.utils.regime_detector import (
    RegimeDetector,
    RegimeSegment,
    RegimeType,
    load_ohlcv_data,
)

logger = logging.getLogger(__name__)

# Default weights per timeframe (higher TF → higher weight)
DEFAULT_MTF_WEIGHTS: Dict[str, float] = {
    '30m': 0.5,
    '1h': 1.0,
    '4h': 2.0,
    '1d': 3.0,
}

# Default timeframes for MTF detection
DEFAULT_MTF_TIMEFRAMES: List[str] = ['30m', '1h', '4h', '1d']

# Combination methods
COMBINATION_METHODS = ['hierarchical', 'weighted_voting']

# Regime context labels derived from composite scores
REGIME_CONTEXT_MAP = {
    # (trend bucket, vol bucket) → context label
    ('strong_bull', 'low'): 'strong_bullish',
    ('strong_bull', 'mid'): 'strong_bullish',
    ('strong_bull', 'high'): 'volatile_bullish',
    ('bull', 'low'): 'steady_bullish',
    ('bull', 'mid'): 'bullish',
    ('bull', 'high'): 'volatile_bullish',
    ('weak_bull', 'low'): 'weak_bullish',
    ('weak_bull', 'mid'): 'bullish_pullback',
    ('weak_bull', 'high'): 'volatile_uncertain',
    ('neutral', 'low'): 'quiet_sideways',
    ('neutral', 'mid'): 'sideways',
    ('neutral', 'high'): 'volatile_sideways',
    ('weak_bear', 'low'): 'weak_bearish',
    ('weak_bear', 'mid'): 'bearish_bounce',
    ('weak_bear', 'high'): 'volatile_uncertain',
    ('bear', 'low'): 'bearish',
    ('bear', 'mid'): 'bearish',
    ('bear', 'high'): 'volatile_bearish',
    ('strong_bear', 'low'): 'strong_bearish',
    ('strong_bear', 'mid'): 'strong_bearish',
    ('strong_bear', 'high'): 'volatile_bearish',
}


@dataclass
class MTFRegimeResult:
    """
    Result container for multi-timeframe regime detection.

    Attributes:
        trend_score: Composite trend score in [-1.0, +1.0].
        volatility_score: Composite volatility score in [0.0, 1.0].
        regime_context: Human-readable regime context label per bar.
        regime_type: Discrete RegimeType per bar (backward compatible).
        transition_speed: Rate of change of trend_score (first derivative).
        transition_signal: Boolean flag when trend_score crosses key thresholds.
        per_tf_trend: Dict mapping timeframe → trend_score Series.
        per_tf_volatility: Dict mapping timeframe → volatility_score Series.
        index: DatetimeIndex of the result (aligned to the lowest available TF).
        metadata: Additional metadata about the detection run.
    """
    trend_score: pd.Series
    volatility_score: pd.Series
    regime_context: pd.Series
    regime_type: pd.Series
    transition_speed: pd.Series
    transition_signal: pd.Series
    per_tf_trend: Dict[str, pd.Series] = field(default_factory=dict)
    per_tf_volatility: Dict[str, pd.Series] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MTFRegimeDetector:
    """
    Multi-Timeframe Regime Detection Engine.

    Runs per-bar continuous regime detection on multiple timeframes and
    combines the results into a single composite signal using configurable
    fusion strategies.

    Usage:
        config = {
            'regime_aware': {
                'mtf_enabled': True,
                'mtf_timeframes': ['30m', '1h', '4h', '1d'],
                'mtf_combination': 'hierarchical',
                'mtf_weights': {'30m': 0.5, '1h': 1.0, '4h': 2.0, '1d': 3.0},
                'method': 'adx_di_hysteresis',
                ...
            },
            'backtesting': {
                'datadir': 'user_data/data/binance',
                'pairs': ['BTC/USDT'],
                'timerange': '20230101-20240101',
            }
        }
        mtf = MTFRegimeDetector(config)
        result = mtf.detect('BTC/USDT', Path('user_data/data/binance'))
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MTF regime detector.

        Args:
            config: Full GA configuration dict.  Reads from
                    ``config['regime_aware']`` for detection settings.
        """
        self.config = config
        regime_cfg = config.get('regime_aware', {})

        # Timeframes to detect on
        self.timeframes: List[str] = regime_cfg.get(
            'mtf_timeframes', DEFAULT_MTF_TIMEFRAMES
        )

        # Combination method
        self.combination: str = regime_cfg.get('mtf_combination', 'hierarchical')
        if self.combination not in COMBINATION_METHODS:
            logger.warning(
                f"Unknown MTF combination '{self.combination}', "
                f"falling back to 'hierarchical'"
            )
            self.combination = 'hierarchical'

        # Per-TF weights (used by weighted_voting, and for hierarchical TF ordering)
        self.weights: Dict[str, float] = regime_cfg.get(
            'mtf_weights', DEFAULT_MTF_WEIGHTS
        )

        # Base detection method used on each timeframe
        self.base_method: str = regime_cfg.get('method', 'adx_di_hysteresis')

        # Hierarchical refinement range — lower TFs can deviate at most this
        # much from the higher-TF anchor score
        self.hierarchical_refine_range: float = regime_cfg.get(
            'mtf_hierarchical_refine', 0.3
        )

        # Transition detection parameters
        self.transition_fast_window: int = regime_cfg.get(
            'mtf_transition_fast_window', 5
        )
        self.transition_slow_window: int = regime_cfg.get(
            'mtf_transition_slow_window', 20
        )
        self.transition_thresholds: List[float] = regime_cfg.get(
            'mtf_transition_thresholds', [-0.5, 0.0, 0.5]
        )

        logger.info(
            f"MTFRegimeDetector initialized: timeframes={self.timeframes}, "
            f"combination={self.combination}, base_method={self.base_method}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        benchmark_pair: str,
        datadir: Path,
        timerange: Optional[str] = None,
        base_timeframe: Optional[str] = None,
    ) -> MTFRegimeResult:
        """
        Run full MTF regime detection pipeline.

        1. Load OHLCV data for each configured timeframe.
        2. Compute continuous (trend, volatility) scores per timeframe.
        3. Align all scores to a common index (lowest available timeframe).
        4. Fuse scores using the configured combination method.
        5. Compute transition signals on the composite trend score.
        6. Derive regime_context labels and backward-compatible RegimeType.

        Args:
            benchmark_pair: Trading pair to analyze (e.g. 'BTC/USDT').
            datadir: Path to FreqTrade data directory.
            timerange: Optional timerange filter ('YYYYMMDD-YYYYMMDD').
            base_timeframe: Optional override for the alignment timeframe.
                            Defaults to the lowest configured timeframe.

        Returns:
            MTFRegimeResult with all computed scores and signals.
        """
        # 1. Load data and compute per-TF scores
        per_tf_trend: Dict[str, pd.Series] = {}
        per_tf_vol: Dict[str, pd.Series] = {}
        loaded_tfs: List[str] = []

        for tf in self.timeframes:
            df = load_ohlcv_data(
                pair=benchmark_pair,
                timeframe=tf,
                datadir=datadir,
                timerange=timerange,
            )
            if df.empty:
                logger.warning(
                    f"No data for {benchmark_pair} {tf}, skipping this timeframe"
                )
                continue

            detector = RegimeDetector(
                method=self.base_method,
                params=self.config.get('regime_aware', {}).get('detection_params'),
                benchmark_pair=benchmark_pair,
            )
            trend, vol = detector.detect_continuous(df)
            per_tf_trend[tf] = trend
            per_tf_vol[tf] = vol
            loaded_tfs.append(tf)

            logger.debug(
                f"Computed continuous scores for {tf}: "
                f"trend range=[{trend.min():.3f}, {trend.max():.3f}], "
                f"vol range=[{vol.min():.3f}, {vol.max():.3f}]"
            )

        if not loaded_tfs:
            raise ValueError(
                f"No data available for any of the configured timeframes "
                f"{self.timeframes} for {benchmark_pair} in {datadir}"
            )

        # 2. Determine target index (lowest available TF for maximum resolution)
        if base_timeframe and base_timeframe in loaded_tfs:
            target_tf = base_timeframe
        else:
            target_tf = self._get_lowest_timeframe(loaded_tfs)
        target_index = per_tf_trend[target_tf].index

        # 3. Align all TF scores to the target index via forward-fill
        aligned_trend = {}
        aligned_vol = {}
        for tf in loaded_tfs:
            aligned_trend[tf] = (
                per_tf_trend[tf]
                .reindex(target_index, method='ffill')
            )
            aligned_vol[tf] = (
                per_tf_vol[tf]
                .reindex(target_index, method='ffill')
            )

        # 4. Fuse scores
        if self.combination == 'hierarchical':
            composite_trend, composite_vol = self._combine_hierarchical(
                aligned_trend, aligned_vol, loaded_tfs
            )
        else:  # weighted_voting
            composite_trend, composite_vol = self._combine_weighted(
                aligned_trend, aligned_vol, loaded_tfs
            )

        # 5. Compute transitions
        transition_speed, transition_signal = self._detect_transitions(
            composite_trend
        )

        # 6. Derive context labels and discrete regime
        regime_context = self._compute_regime_context(
            composite_trend, composite_vol
        )
        regime_type = RegimeDetector.continuous_to_regime(
            composite_trend, composite_vol
        )

        metadata = {
            'timeframes': loaded_tfs,
            'combination': self.combination,
            'base_method': self.base_method,
            'target_timeframe': target_tf,
            'bar_count': len(target_index),
        }

        logger.info(
            f"MTF detection complete: {len(loaded_tfs)} timeframes, "
            f"{len(target_index)} bars, combination={self.combination}"
        )

        return MTFRegimeResult(
            trend_score=composite_trend,
            volatility_score=composite_vol,
            regime_context=regime_context,
            regime_type=regime_type,
            transition_speed=transition_speed,
            transition_signal=transition_signal,
            per_tf_trend=per_tf_trend,
            per_tf_volatility=per_tf_vol,
            metadata=metadata,
        )

    def detect_from_dataframes(
        self,
        dataframes: Dict[str, pd.DataFrame],
    ) -> MTFRegimeResult:
        """
        Run MTF detection from pre-loaded DataFrames (avoids re-loading data).

        Useful when the caller already has OHLCV data for multiple timeframes
        (e.g., from FreqTrade's dataprovider in a running strategy).

        Args:
            dataframes: Dict mapping timeframe → OHLCV DataFrame.

        Returns:
            MTFRegimeResult
        """
        per_tf_trend: Dict[str, pd.Series] = {}
        per_tf_vol: Dict[str, pd.Series] = {}
        loaded_tfs: List[str] = []

        for tf, df in dataframes.items():
            if df.empty:
                continue
            detector = RegimeDetector(method=self.base_method)
            trend, vol = detector.detect_continuous(df)
            per_tf_trend[tf] = trend
            per_tf_vol[tf] = vol
            loaded_tfs.append(tf)

        if not loaded_tfs:
            raise ValueError("No non-empty DataFrames provided")

        target_tf = self._get_lowest_timeframe(loaded_tfs)
        target_index = per_tf_trend[target_tf].index

        aligned_trend = {}
        aligned_vol = {}
        for tf in loaded_tfs:
            aligned_trend[tf] = per_tf_trend[tf].reindex(target_index, method='ffill')
            aligned_vol[tf] = per_tf_vol[tf].reindex(target_index, method='ffill')

        if self.combination == 'hierarchical':
            composite_trend, composite_vol = self._combine_hierarchical(
                aligned_trend, aligned_vol, loaded_tfs
            )
        else:
            composite_trend, composite_vol = self._combine_weighted(
                aligned_trend, aligned_vol, loaded_tfs
            )

        transition_speed, transition_signal = self._detect_transitions(composite_trend)
        regime_context = self._compute_regime_context(composite_trend, composite_vol)
        regime_type = RegimeDetector.continuous_to_regime(composite_trend, composite_vol)

        return MTFRegimeResult(
            trend_score=composite_trend,
            volatility_score=composite_vol,
            regime_context=regime_context,
            regime_type=regime_type,
            transition_speed=transition_speed,
            transition_signal=transition_signal,
            per_tf_trend=per_tf_trend,
            per_tf_volatility=per_tf_vol,
            metadata={
                'timeframes': loaded_tfs,
                'combination': self.combination,
                'base_method': self.base_method,
                'target_timeframe': target_tf,
                'bar_count': len(target_index),
            },
        )

    # ------------------------------------------------------------------
    # Segment classification using MTF scores
    # ------------------------------------------------------------------

    def classify_segments(
        self,
        result: MTFRegimeResult,
        df: pd.DataFrame,
        min_segment_days: int = 14,
        max_segment_days: int = 180,
        merge_threshold_days: int = 7,
        embargo_days: int = 5,
        bullish_min: float = 0.35,
        bearish_max: float = -0.35,
    ) -> List[RegimeSegment]:
        """
        Build adaptive regime segments from MTF detection results.

        Unlike fixed-window classification, this method finds regime change
        points from the composite trend_score and creates variable-length
        segments with enriched metadata (continuous scores + context).

        Args:
            result: MTFRegimeResult from detect() or detect_from_dataframes().
            df: OHLCV DataFrame aligned to the same index as result.
            min_segment_days: Minimum segment duration.
            max_segment_days: Maximum segment duration.
            merge_threshold_days: Merge segments shorter than this.
            embargo_days: Gap between segments.

        Returns:
            List of RegimeSegment objects with enriched metadata.
        """
        # Use the underlying RegimeDetector's adaptive segmentation
        # but with our composite regime_type series
        detector = RegimeDetector(method=self.base_method)

        # We build segments from the composite regime_type
        # using the adaptive change-point approach
        regime_series = result.regime_type
        trend_series = result.trend_score
        vol_series = result.volatility_score
        context_series = result.regime_context

        if regime_series.empty:
            return []

        # Ensure datetime index
        if not isinstance(regime_series.index, pd.DatetimeIndex):
            logger.error("MTF result must have DatetimeIndex")
            return []

        # Find change points based on regime_context changes
        # (more granular than regime_type changes)
        prev_context = context_series.shift(1)
        change_mask = (context_series != prev_context) & prev_context.notna()
        change_indices = context_series.index[change_mask].tolist()

        all_boundaries = [regime_series.index[0]] + change_indices + [regime_series.index[-1]]

        # Build raw segments
        raw_segments: List[Dict[str, Any]] = []
        for i in range(len(all_boundaries) - 1):
            seg_start = all_boundaries[i]
            seg_end = all_boundaries[i + 1]

            mask = (regime_series.index >= seg_start) & (regime_series.index < seg_end)
            seg_regimes = regime_series[mask]
            seg_trend = trend_series[mask]
            seg_vol = vol_series[mask]
            seg_context = context_series[mask]

            if seg_regimes.empty:
                continue

            avg_trend = float(seg_trend.mean()) if not seg_trend.empty else 0.0
            avg_vol = float(seg_vol.mean()) if not seg_vol.empty else 0.0
            trend_std = float(seg_trend.std()) if len(seg_trend) > 1 else 0.0

            # Score-band regime assignment: use avg_trend_score with
            # configurable band boundaries instead of discrete vote counting.
            # This guarantees every regime gets representation when data
            # covers diverse market conditions.
            if avg_trend >= bullish_min:
                dominant = RegimeType.BULLISH
            elif avg_trend <= bearish_max:
                dominant = RegimeType.BEARISH
            else:
                dominant = RegimeType.SIDEWAYS

            # Confidence from score consistency within segment
            confidence = max(0.0, min(1.0, 1.0 - trend_std))

            duration_days = (seg_end - seg_start).days

            raw_segments.append({
                'start': seg_start.to_pydatetime() if hasattr(seg_start, 'to_pydatetime') else seg_start,
                'end': seg_end.to_pydatetime() if hasattr(seg_end, 'to_pydatetime') else seg_end,
                'regime': dominant,
                'confidence': confidence,
                'duration_days': duration_days,
                'bar_count': len(seg_regimes),
                'avg_trend_score': avg_trend,
                'avg_volatility_score': avg_vol,
                'trend_score_std': trend_std,
                'dominant_context': seg_context.mode().iloc[0] if not seg_context.empty else 'uncertain',
            })

        # Merge short segments
        merged = RegimeDetector._merge_short_segments(raw_segments, merge_threshold_days)

        # Split excessively long segments
        final_raw = []
        for seg in merged:
            if seg['duration_days'] > max_segment_days:
                n_splits = (seg['duration_days'] // max_segment_days) + 1
                from datetime import timedelta
                split_duration = timedelta(days=seg['duration_days'] / n_splits)
                for j in range(n_splits):
                    sub_start = seg['start'] + split_duration * j
                    sub_end = seg['start'] + split_duration * (j + 1)
                    if j == n_splits - 1:
                        sub_end = seg['end']
                    final_raw.append({
                        **seg,
                        'start': sub_start,
                        'end': sub_end,
                        'duration_days': (sub_end - sub_start).days,
                    })
            else:
                final_raw.append(seg)

        # Remove segments shorter than minimum
        final_raw = [s for s in final_raw if s['duration_days'] >= min_segment_days]

        # Build RegimeSegment objects with enriched metadata
        segments = []
        for idx, seg in enumerate(final_raw):
            # Re-derive regime from avg_trend_score using score-band
            # boundaries (may have shifted after merge/split)
            avg_t = seg.get('avg_trend_score', 0.0)
            if avg_t >= bullish_min:
                regime = RegimeType.BULLISH
            elif avg_t <= bearish_max:
                regime = RegimeType.BEARISH
            else:
                regime = RegimeType.SIDEWAYS

            metadata = {
                'avg_trend_score': avg_t,
                'avg_volatility_score': seg.get('avg_volatility_score', 0.0),
                'trend_score_std': seg.get('trend_score_std', 0.0),
                'regime_context': seg.get('dominant_context', 'unknown'),
                'bar_count': seg.get('bar_count', 0),
                'source': 'mtf_score_band',
                'bullish_min': bullish_min,
                'bearish_max': bearish_max,
            }

            # Add OHLCV metadata if df is available
            if df is not None and not df.empty:
                df_mask = (df.index >= seg['start']) & (df.index < seg['end'])
                seg_df = df[df_mask]
                if not seg_df.empty and 'close' in seg_df.columns:
                    rets = seg_df['close'].pct_change().dropna()
                    metadata['mean_return'] = float(rets.mean()) if len(rets) > 0 else 0.0
                    metadata['volatility'] = float(rets.std()) if len(rets) > 0 else 0.0
                    metadata['total_return'] = float(
                        seg_df['close'].iloc[-1] / seg_df['close'].iloc[0] - 1
                    ) if len(seg_df) > 1 else 0.0

            segment_id = f"mtf_{idx:03d}_{seg['start'].strftime('%Y%m%d')}_{seg['end'].strftime('%Y%m%d')}"
            segments.append(RegimeSegment(
                segment_id=segment_id,
                start_date=seg['start'],
                end_date=seg['end'],
                regime=regime,
                confidence=seg['confidence'],
                metadata=metadata,
            ))

        logger.info(
            f"MTF adaptive segmentation: {len(segments)} segments "
            f"from {len(raw_segments)} raw change points"
        )
        regime_dist: Dict[str, int] = {}
        for s in segments:
            regime_dist[s.regime.value] = regime_dist.get(s.regime.value, 0) + 1
        logger.info(f"Regime distribution (MTF): {regime_dist}")

        return segments

    # ------------------------------------------------------------------
    # Combination methods
    # ------------------------------------------------------------------

    def _combine_hierarchical(
        self,
        aligned_trend: Dict[str, pd.Series],
        aligned_vol: Dict[str, pd.Series],
        loaded_tfs: List[str],
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Hierarchical combination: higher TFs dominate, lower TFs refine.

        Algorithm:
        1. Sort timeframes from highest to lowest.
        2. Start with the highest-TF trend score as the anchor.
        3. For each subsequent (lower) TF, allow it to refine the running
           composite by up to ±hierarchical_refine_range from the anchor.
        4. Volatility: weighted average (hierarchical doesn't change semantics).
        """
        sorted_tfs = self._sort_timeframes_descending(loaded_tfs)

        if len(sorted_tfs) == 1:
            tf = sorted_tfs[0]
            return aligned_trend[tf], aligned_vol[tf]

        # Start with highest-TF as anchor
        anchor_tf = sorted_tfs[0]
        composite_trend = aligned_trend[anchor_tf].copy()
        total_weight = self.weights.get(anchor_tf, 1.0)
        composite_vol = aligned_vol[anchor_tf] * total_weight

        # Iteratively refine with lower timeframes
        for tf in sorted_tfs[1:]:
            lower_trend = aligned_trend[tf]
            lower_vol = aligned_vol[tf]
            w = self.weights.get(tf, 1.0)

            # Constrained refinement: lower TF can pull the composite by at
            # most refine_range away from the current anchor
            delta = lower_trend - composite_trend
            refine = self.hierarchical_refine_range
            clamped_delta = delta.clip(-refine, refine)

            # Weight the contribution by relative weight
            # Higher weight TF (already in composite) resists change more
            relative_influence = w / (total_weight + w)
            composite_trend = composite_trend + clamped_delta * relative_influence

            # Volatility: simple running weighted average
            composite_vol = composite_vol + lower_vol * w
            total_weight += w

        # Normalize volatility
        composite_vol = composite_vol / total_weight

        # Clip final trend to [-1, 1]
        composite_trend = composite_trend.clip(-1.0, 1.0)
        composite_vol = composite_vol.clip(0.0, 1.0)

        return composite_trend, composite_vol

    def _combine_weighted(
        self,
        aligned_trend: Dict[str, pd.Series],
        aligned_vol: Dict[str, pd.Series],
        loaded_tfs: List[str],
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Weighted voting combination: weighted average of all TF scores.

        Each timeframe contributes proportionally to its configured weight.
        """
        total_weight = 0.0
        composite_trend = None
        composite_vol = None

        for tf in loaded_tfs:
            w = self.weights.get(tf, 1.0)
            if composite_trend is None:
                composite_trend = aligned_trend[tf] * w
                composite_vol = aligned_vol[tf] * w
            else:
                composite_trend = composite_trend + aligned_trend[tf] * w
                composite_vol = composite_vol + aligned_vol[tf] * w
            total_weight += w

        if composite_trend is None:
            raise ValueError("No timeframes to combine")

        composite_trend = (composite_trend / total_weight).clip(-1.0, 1.0)
        composite_vol = (composite_vol / total_weight).clip(0.0, 1.0)

        return composite_trend, composite_vol

    # ------------------------------------------------------------------
    # Transition detection
    # ------------------------------------------------------------------

    def _detect_transitions(
        self,
        composite_trend: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect regime transitions from the composite trend score.

        Computes:
        - **transition_speed**: Smoothed first derivative of the composite
          trend score.  Positive = market turning bullish, negative = turning
          bearish.  Uses a fast EMA minus slow EMA (MACD-like) approach for
          robustness.
        - **transition_signal**: Boolean flag raised when the composite trend
          score crosses key thresholds (configurable, default: -0.5, 0.0, 0.5).
          Signals regime *changes* in progress.

        Args:
            composite_trend: The fused trend score Series.

        Returns:
            Tuple of (transition_speed, transition_signal) Series.
        """
        fast_w = self.transition_fast_window
        slow_w = self.transition_slow_window

        # Smoothed first derivative (MACD-like: fast EMA - slow EMA)
        fast_ema = composite_trend.ewm(span=fast_w, adjust=False).mean()
        slow_ema = composite_trend.ewm(span=slow_w, adjust=False).mean()
        transition_speed = fast_ema - slow_ema

        # Threshold crossing signals
        transition_signal = pd.Series(False, index=composite_trend.index, dtype=bool)
        prev_trend = composite_trend.shift(1)

        for thresh in self.transition_thresholds:
            # Upward crossing
            crosses_up = (prev_trend <= thresh) & (composite_trend > thresh)
            # Downward crossing
            crosses_down = (prev_trend >= thresh) & (composite_trend < thresh)
            transition_signal = transition_signal | crosses_up | crosses_down

        return transition_speed, transition_signal

    # ------------------------------------------------------------------
    # Context labeling
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_regime_context(
        trend: pd.Series,
        vol: pd.Series,
    ) -> pd.Series:
        """
        Compute human-readable regime context label for each bar.

        Maps the continuous (trend, volatility) tuple to a descriptive string
        like 'strong_bullish', 'bullish_pullback', 'volatile_sideways', etc.
        """
        context = pd.Series('uncertain', index=trend.index, dtype=object)

        def _trend_bucket(t: float) -> str:
            if np.isnan(t):
                return 'neutral'
            if t > 0.6:
                return 'strong_bull'
            if t > 0.25:
                return 'bull'
            if t > 0.1:
                return 'weak_bull'
            if t >= -0.1:
                return 'neutral'
            if t >= -0.25:
                return 'weak_bear'
            if t >= -0.6:
                return 'bear'
            return 'strong_bear'

        def _vol_bucket(v: float) -> str:
            if np.isnan(v):
                return 'mid'
            if v > 0.75:
                return 'high'
            if v < 0.25:
                return 'low'
            return 'mid'

        # Vectorized approach using numpy
        trend_vals = trend.values
        vol_vals = vol.values
        contexts = np.empty(len(trend_vals), dtype=object)

        for i in range(len(trend_vals)):
            tb = _trend_bucket(trend_vals[i])
            vb = _vol_bucket(vol_vals[i])
            contexts[i] = REGIME_CONTEXT_MAP.get((tb, vb), 'uncertain')

        context = pd.Series(contexts, index=trend.index, dtype=object)
        return context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_lowest_timeframe(timeframes: List[str]) -> str:
        """Return the lowest (shortest duration) timeframe from a list."""
        from genetic_algorithm.core.strategy_gene import timeframe_to_minutes
        return min(timeframes, key=lambda tf: timeframe_to_minutes(tf))

    def _sort_timeframes_descending(self, timeframes: List[str]) -> List[str]:
        """Sort timeframes from highest to lowest."""
        from genetic_algorithm.core.strategy_gene import timeframe_to_minutes
        return sorted(timeframes, key=lambda tf: timeframe_to_minutes(tf), reverse=True)
