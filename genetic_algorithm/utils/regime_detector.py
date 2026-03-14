"""
Regime Detection Module

Provides market regime classification for regime-aware dataset selection.
Supports multiple detection methods to classify market periods as:
- bullish (uptrend)
- bearish (downtrend)
- sideways (ranging/low trend)

This is the foundation for regime-balanced evaluation in the GA fitness evaluator.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNCERTAIN = "uncertain"


@dataclass
class RegimeSegment:
    """
    Represents a classified market regime segment.
    
    Attributes:
        segment_id: Unique identifier for caching
        start_date: Segment start date
        end_date: Segment end date
        regime: Classified regime type
        confidence: Detection confidence (0-1)
        metadata: Additional metrics about the segment
        role: Segment role ('optimization', 'model_selection', 'holdout')
    """
    segment_id: str
    start_date: datetime
    end_date: datetime
    regime: RegimeType
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    role: str = "optimization"
    
    @property
    def timerange(self) -> str:
        """Get FreqTrade-compatible timerange string."""
        return f"{self.start_date.strftime('%Y%m%d')}-{self.end_date.strftime('%Y%m%d')}"
    
    @property
    def duration_days(self) -> int:
        """Get segment duration in days."""
        return (self.end_date - self.start_date).days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'segment_id': self.segment_id,
            'timerange': self.timerange,
            'regime': self.regime.value,
            'confidence': self.confidence,
            'duration_days': self.duration_days,
            'metadata': self.metadata,
            'role': self.role
        }


class RegimeDetector:
    """
    Market regime detection engine.
    
    Classifies historical market data into distinct regimes (bullish, bearish, sideways)
    using configurable detection methods. This is the foundation for regime-balanced
    dataset selection in GA evolution.
    
    Supported detection methods:
    - 'sma_adx': SMA crossover + ADX trend strength (recommended)
    - 'adx_di': ADX with Directional Movement indicators
    - 'returns': Rolling return distribution analysis
    - 'bollinger': Bollinger Band position analysis
    - 'ensemble': Voting combination of multiple methods
    
    Usage:
        detector = RegimeDetector(method='sma_adx')
        regimes = detector.detect(close_prices)  # Per-bar regime
        segments = detector.classify_periods(df, period_days=90)  # Period classification
    """
    
    # Default parameters for each detection method
    DEFAULT_PARAMS = {
        'sma_adx': {
            'sma_fast': 50,
            'sma_slow': 200,
            'adx_period': 14,
            'adx_threshold': 25,
        },
        'adx_di': {
            'adx_period': 14,
            'adx_threshold': 25,
            'adx_sideways_threshold': 20,
        },
        'adx_di_hysteresis': {
            'adx_period': 14,
            'adx_enter': 25,      # Enter trend mode when ADX > this
            'adx_exit': 20,       # Exit trend mode when ADX < this
        },
        'returns': {
            'lookback_period': 20,
            'trend_threshold': 0.001,  # Daily return threshold
            'volatility_cap': 0.03,    # Max volatility for trending classification
        },
        'rolling_returns': {
            'window': 50,              # Rolling window size
            'threshold': 0.0005,       # Return threshold per bar (0.05%)
            'hysteresis': 0.3,         # Hysteresis factor for threshold
        },
        'bollinger': {
            'period': 20,
            'std_dev': 2.0,
            'lookback_bars': 10,  # Bars to assess "consistently above/below"
            'consistency_threshold': 0.7,  # 70% of bars must be in position
        },
        'hmm': {
            'n_states': 3,
            'min_dwell': 1,       # Minimum bars to stay in regime (1=no lag, higher=smoother but lagged)
            'vol_window': 20,     # Rolling volatility window
        },
        'volatility_cluster': {
            'vol_window': 20,         # Rolling volatility window
            'vol_lookback': 60,       # Lookback for vol percentile thresholds
            'high_vol_pct': 75,       # Percentile threshold for "high volatility"
            'low_vol_pct': 25,        # Percentile threshold for "low volatility"
            'trend_window': 50,       # Window for trend direction detection
            'trend_threshold': 0.001, # Min rolling return for trend classification
        },
        'sma_slope': {
            'sma_period': 20,         # SMA lookback period (use 20 on 1d candles)
            'slope_window': 5,        # Bars over which to compute slope
            'tanh_k': 0.3,            # tanh sensitivity multiplier (k * median_slope)
        },
    }
    
    def __init__(
        self,
        method: str = 'adx_di_hysteresis',  # Changed from 'sma_adx' - best performer
        params: Optional[Dict[str, Any]] = None,
        benchmark_pair: Optional[str] = None  # For consistent regime labeling across pairs
    ):
        """
        Initialize regime detector.
        
        Args:
            method: Detection method ('sma_adx', 'adx_di', 'adx_di_hysteresis', 
                    'returns', 'rolling_returns', 'bollinger', 'hmm', 'ensemble')
            params: Custom parameters for the detection method (overrides defaults)
            benchmark_pair: Optional benchmark pair for market-wide regime labeling
        """
        valid_methods = ['sma_adx', 'adx_di', 'adx_di_hysteresis', 'returns', 
                         'rolling_returns', 'bollinger', 'hmm', 'volatility_cluster',
                         'sma_slope', 'ensemble', 'advanced_ensemble', 'ml_lgbm']
        if method not in valid_methods:
            raise ValueError(f"Unknown detection method: {method}. Valid: {valid_methods}")
        
        self.method = method
        self.benchmark_pair = benchmark_pair
        
        # Merge default params with custom params
        self.params = self.DEFAULT_PARAMS.get(method, {}).copy()
        if params:
            self.params.update(params)
        
        logger.info(f"RegimeDetector initialized with method='{method}', params={self.params}")
    
    def detect(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market regime for each bar in the dataframe.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
                Must be sorted by date (ascending).
        
        Returns:
            pd.Series of RegimeType for each bar (same index as df)
        """
        if df.empty:
            return pd.Series(dtype=object)
        
        # Ensure column names are lowercase
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Validate required columns
        required = ['close']
        if self.method in ['adx_di', 'adx_di_hysteresis', 'sma_adx', 'ml_lgbm', 'advanced_ensemble']:
            required.extend(['high', 'low'])
        
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Dispatch to detection method
        if self.method == 'sma_adx':
            return self._detect_sma_adx(df)
        elif self.method == 'adx_di':
            return self._detect_adx_di(df)
        elif self.method == 'adx_di_hysteresis':
            return self._detect_adx_di_hysteresis(df)
        elif self.method == 'returns':
            return self._detect_returns(df)
        elif self.method == 'rolling_returns':
            return self._detect_rolling_returns(df)
        elif self.method == 'bollinger':
            return self._detect_bollinger(df)
        elif self.method == 'hmm':
            return self._detect_hmm(df)
        elif self.method == 'volatility_cluster':
            return self._detect_volatility_cluster(df)
        elif self.method == 'sma_slope':
            return self._detect_sma_slope(df)
        elif self.method == 'ensemble':
            return self._detect_ensemble(df)
        elif self.method == 'advanced_ensemble':
            return self._detect_advanced_ensemble(df)
        elif self.method == 'ml_lgbm':
            return self._detect_ml_lgbm(df)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def detect_continuous(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Detect market regime as continuous scores instead of discrete categories.

        Returns two continuous series that capture regime nuances beyond simple
        bullish/bearish/sideways categories:

        - **trend_score** in [-1.0, +1.0]:
          -1.0 = strong bearish, 0.0 = sideways/neutral, +1.0 = strong bullish.
          Derived from the underlying indicator values (e.g. ADX * DI direction,
          normalized rolling return) and clipped to [-1, 1].

        - **volatility_score** in [0.0, 1.0]:
          0.0 = calm/low volatility, 1.0 = extreme volatility.
          Derived from rolling realized volatility ranked against its own
          trailing distribution (percentile rank).

        These scores allow downstream consumers (MTF fusion, strategy code gen,
        fitness evaluation) to operate on a gradient rather than hard buckets.

        The existing ``detect()`` method is equivalent to thresholding these
        continuous scores.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume).
                Must be sorted by date ascending.

        Returns:
            Tuple of (trend_score: pd.Series, volatility_score: pd.Series)
            with the same index as *df*.  NaN for warmup bars.
        """
        if df.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        df = df.copy()
        df.columns = df.columns.str.lower()

        trend_score = self._compute_trend_score(df)
        volatility_score = self._compute_volatility_score(df)

        return trend_score, volatility_score

    def detect_transition_speed(self, df: pd.DataFrame, lookback: int = 5) -> pd.Series:
        """
        Compute regime transition speed — how rapidly the market regime is changing.
        
        Returns a Series in [0.0, 1.0] where:
        - 0.0 = regime is completely stable (no change)
        - 1.0 = regime is changing at maximum observed rate
        
        Calculated as the rolling absolute change in the continuous trend score
        over `lookback` bars, normalized to [0, 1] via percentile rank.
        
        Useful for:
        - Filtering entries near regime transitions (high speed = risky)
        - Adjusting position sizing based on regime stability
        - As an indicator in generated strategies
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to measure change over (default: 5)
            
        Returns:
            pd.Series of transition speed values [0.0, 1.0], same index as df
        """
        if df.empty:
            return pd.Series(dtype=float)
        
        trend_score, _ = self.detect_continuous(df)
        
        # Rate of change in trend score over lookback bars
        delta = trend_score.diff(lookback).abs()
        
        # Normalize to [0, 1] using rolling percentile rank
        # This adapts to the data's own volatility characteristics
        window = max(50, lookback * 10)
        rank = delta.rolling(window=window, min_periods=lookback).rank(pct=True)
        
        # Fill NaN warmup bars with 0.5 (neutral)
        rank = rank.fillna(0.5)
        
        return rank

    def _compute_trend_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a continuous trend score in [-1, +1].

        Strategy depends on the active detection method:
        - advanced_ensemble / ensemble:
            Confidence-weighted average of 5 sub-detector continuous scores
            (ADX-DI, rolling-returns, Bollinger, volatility-cluster, HMM).
            Directly mirrors the discrete voting logic but stays continuous.
        - adx_di_hysteresis / adx_di / sma_adx:
            score = (plus_di - minus_di) / (plus_di + minus_di) * (adx / 50)
            This captures both *direction* (DI differential) and *strength* (ADX).
        - returns / rolling_returns:
            score = tanh(rolling_mean_return / threshold)
            Smooth sigmoid mapping of returns to [-1, 1].
        - bollinger:
            score = (close - sma) / (2 * std)  clipped to [-1, 1]
        - hmm / volatility_cluster / ml_lgbm / other:
            Adaptive returns-based continuous score with data-derived threshold.
        """
        close = df['close']

        # ── Advanced / standard ensemble: confidence-weighted multi-score ──
        if self.method in ('advanced_ensemble', 'ensemble'):
            return self._compute_ensemble_trend_score(df)

        if self.method in (
            'adx_di_hysteresis', 'adx_di', 'sma_adx',
        ):
            adx_period = self.params.get('adx_period', 14)
            adx, plus_di, minus_di = self._calculate_adx(df, adx_period)

            # Direction: normalized DI differential in [-1, 1]
            di_sum = plus_di + minus_di
            di_direction = (plus_di - minus_di) / di_sum.replace(0, np.nan)

            # Strength: ADX normalized by 50 (ADX > 50 is very strong), clipped to [0, 1]
            adx_strength = (adx / 50.0).clip(0, 1)

            # Combined: direction * strength → [-1, 1]
            score = di_direction * adx_strength

            # For adx_di_hysteresis: apply hysteresis dampening
            # When ADX is below the exit threshold, dampen the score towards zero
            if self.method == 'adx_di_hysteresis':
                adx_enter = self.params.get('adx_enter', 25)
                adx_exit = self.params.get('adx_exit', 20)
                # Smooth dampening: below adx_exit → score * 0, above adx_enter → score * 1
                dampen = ((adx - adx_exit) / (adx_enter - adx_exit)).clip(0, 1)
                score = score * dampen

            return score.clip(-1.0, 1.0)

        elif self.method in ('returns', 'rolling_returns'):
            returns = close.pct_change()
            window = self.params.get('window', self.params.get('lookback_period', 50))
            threshold = self.params.get('threshold', self.params.get('trend_threshold', 0.0005))
            rolling_mean = returns.rolling(window=window, min_periods=window).mean()
            # tanh mapping: smooth transition, threshold controls sensitivity
            score = np.tanh(rolling_mean / max(threshold, 1e-8))
            return pd.Series(score, index=df.index).clip(-1.0, 1.0)

        elif self.method == 'bollinger':
            period = self.params.get('period', 20)
            std_dev = self.params.get('std_dev', 2.0)
            sma = self._calculate_sma(close, period)
            std = close.rolling(window=period, min_periods=period).std()
            score = (close - sma) / (std_dev * std.replace(0, np.nan))
            return score.clip(-1.0, 1.0)

        elif self.method == 'sma_slope':
            return self._compute_sma_slope_score(df)

        else:
            # Generic fallback: adaptive returns-based continuous score
            # Uses the asset's own rolling volatility to set sensitivity,
            # ensuring the score spreads across [-1, +1] with adequate
            # coverage in all three bands regardless of asset/timeframe.
            #
            # Calibration: threshold = rolling_std.median() * k
            #   k=0.5 → ~78% sideways (too conservative)
            #   k=0.2 → ~40-50% sideways (balanced for GA island splits)
            # The lower multiplier ensures rolling mean returns that are
            # even modestly directional get mapped to the bull/bear zones.
            returns = close.pct_change()
            rolling_mean = returns.rolling(window=50, min_periods=50).mean()
            rolling_std = returns.rolling(window=50, min_periods=50).std()
            threshold = rolling_std.median() * 0.2
            threshold = max(threshold, 1e-6)
            score = np.tanh(rolling_mean / threshold)
            return pd.Series(score, index=df.index).clip(-1.0, 1.0)

    def _compute_ensemble_trend_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a continuous trend score for advanced_ensemble / ensemble.

        Instead of falling to a single generic formula, this method mirrors
        the discrete ensemble approach but stays continuous:

        1. Compute each sub-detector's continuous score in [-1, +1].
        2. Compute a rolling agreement-based confidence weight per method.
        3. Return the confidence-weighted average across all sub-detectors.

        This produces a well-calibrated score with natural spread across
        the [-1, +1] range because different methods contribute different
        sensitivities (e.g. ADX-DI is conservative, returns-based is
        responsive, Bollinger is mean-reversion aware).

        Sub-detectors and base weights:
          1. adx_di_hysteresis  (w=2.0)  — stable trend with hysteresis
          2. rolling_returns    (w=2.0)  — momentum/directional bias
          3. bollinger          (w=1.0)  — mean-reversion / range
          4. volatility_cluster (w=1.0)  — volatility-adjusted (via returns fallback)
          5. hmm                (w=1.5)  — statistical state model (via returns fallback)
        """
        n = len(df)

        # Sub-detector specs: (method, params, base_weight)
        # Propagate user-configured params to each sub-detector by namespace
        # so that custom settings (e.g. adx_period=20) reach the right detector.
        def _params_for(prefix: str) -> dict:
            return {k: v for k, v in self.params.items() if k.startswith(prefix)}

        sub_specs = [
            ('adx_di_hysteresis', _params_for('adx_'), 2.0),
            ('rolling_returns', {**{'window': 50, 'threshold': 0.0005}, **_params_for('rolling_')}, 2.0),
            ('bollinger', _params_for('bb_'), 1.0),
            ('volatility_cluster', _params_for('vol_'), 1.0),
            ('hmm', _params_for('hmm_'), 1.5),
        ]

        sub_scores: Dict[str, Tuple[pd.Series, float]] = {}
        for method_name, params, base_weight in sub_specs:
            try:
                det = RegimeDetector(method=method_name, params=params)
                score = det._compute_trend_score(df)
                sub_scores[method_name] = (score, base_weight)
            except Exception as e:
                logger.debug(
                    "Ensemble trend score: %s failed, skipping: %s",
                    method_name, e,
                )

        if not sub_scores:
            # Ultimate fallback — adaptive returns
            logger.warning("Ensemble trend score: all sub-detectors failed, using fallback")
            close = df['close']
            returns = close.pct_change()
            rolling_mean = returns.rolling(window=50, min_periods=50).mean()
            rolling_std = returns.rolling(window=50, min_periods=50).std()
            threshold = max(rolling_std.median() * 0.2, 1e-6)
            return pd.Series(np.tanh(rolling_mean / threshold), index=df.index).clip(-1, 1)

        # Stack all sub-scores into a DataFrame
        score_df = pd.DataFrame(
            {name: s for name, (s, _) in sub_scores.items()},
            index=df.index,
        )

        # Compute rolling confidence per sub-detector:
        # Agreement = correlation with the mean of other detectors
        # over a rolling window.  High correlation → high confidence.
        conf_window = min(100, max(20, n // 10))
        weights_df = pd.DataFrame(index=df.index, columns=score_df.columns, dtype=float)

        for method_name in score_df.columns:
            others = score_df.drop(columns=[method_name])
            others_mean = others.mean(axis=1)

            # Rolling correlation with consensus (others' average)
            rolling_corr = score_df[method_name].rolling(
                window=conf_window, min_periods=max(10, conf_window // 4),
            ).corr(others_mean)

            # Transform: corr in [-1, 1] → confidence in [0.2, 1.0]
            # Low/negative correlation → low weight but not zero
            confidence = (rolling_corr.clip(-1, 1) + 1) / 2  # → [0, 1]
            confidence = confidence * 0.8 + 0.2  # → [0.2, 1.0]
            confidence = confidence.fillna(0.5)

            base_weight = sub_scores[method_name][1]
            weights_df[method_name] = base_weight * confidence

        # Compute weighted average score
        weighted_sum = (score_df * weights_df).sum(axis=1)
        weight_total = weights_df.sum(axis=1).replace(0, np.nan)
        final_score = weighted_sum / weight_total

        return final_score.clip(-1.0, 1.0)

    def _compute_volatility_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a continuous volatility score in [0, 1].

        Uses rolling realized volatility ranked against its own trailing
        distribution.  A rolling percentile rank ensures the score adapts
        to changing volatility regimes (e.g., a 2% daily std is extreme
        in calm markets but normal in a crash).

        Score = rolling_percentile_rank(realized_vol, lookback=252)
        """
        close = df['close']
        returns = close.pct_change()

        # Exponentially weighted realized volatility
        vol_window = self.params.get('vol_window', 20)
        realized_vol = returns.ewm(span=vol_window, adjust=False).std()

        # Percentile rank over a trailing window (default 252 bars ≈ 1 year on daily)
        vol_lookback = self.params.get('vol_lookback', 252)

        def _percentile_rank(series: pd.Series, lookback: int) -> pd.Series:
            """Rolling percentile rank — fraction of trailing values <= current."""
            result = pd.Series(np.nan, index=series.index)
            vals = series.values
            n = len(vals)
            for i in range(lookback, n):
                window = vals[max(0, i - lookback):i + 1]
                valid = window[~np.isnan(window)]
                if len(valid) < 2:
                    continue
                result.iloc[i] = np.sum(valid <= vals[i]) / len(valid)
            return result

        # Vectorized percentile rank (faster approximation)
        vol_min = realized_vol.rolling(window=vol_lookback, min_periods=min(vol_lookback, 60)).min()
        vol_max = realized_vol.rolling(window=vol_lookback, min_periods=min(vol_lookback, 60)).max()
        vol_range = vol_max - vol_min
        score = (realized_vol - vol_min) / vol_range.replace(0, np.nan)

        return score.clip(0.0, 1.0)

    @staticmethod
    def continuous_to_regime(
        trend_score: pd.Series,
        volatility_score: pd.Series,
        trend_threshold: float = 0.25,
        volatility_threshold: float = 0.75,
        bullish_min: Optional[float] = None,
        bearish_max: Optional[float] = None,
    ) -> pd.Series:
        """
        Convert continuous scores to discrete RegimeType labels.

        This is the inverse of detect_continuous() — useful for backward
        compatibility with code that expects RegimeType enum values.

        When ``bullish_min`` / ``bearish_max`` are provided (score-band mode)
        they take precedence over the legacy symmetric ``trend_threshold``.
        Default band boundaries: bullish >= 0.35, bearish <= -0.35,
        sideways = everything in between.

        Args:
            trend_score: Continuous trend score in [-1, 1]
            volatility_score: Continuous volatility score in [0, 1]
            trend_threshold: Legacy symmetric threshold (used when bands
                             are not specified). |trend_score| above this
                             → trending (bull/bear).
            volatility_threshold: volatility_score above this → VOLATILE
            bullish_min: Score-band lower bound for bullish regime.
                         When set, ``trend_score >= bullish_min`` → BULLISH.
            bearish_max: Score-band upper bound for bearish regime.
                         When set, ``trend_score <= bearish_max`` → BEARISH.

        Returns:
            pd.Series of RegimeType
        """
        # Resolve band boundaries
        if bullish_min is not None or bearish_max is not None:
            bull_threshold = bullish_min if bullish_min is not None else 0.35
            bear_threshold = bearish_max if bearish_max is not None else -0.35
        else:
            # Legacy symmetric mode
            bull_threshold = trend_threshold
            bear_threshold = -trend_threshold

        # Support scalar inputs
        if not isinstance(trend_score, pd.Series):
            trend_score = pd.Series([trend_score])
            volatility_score = pd.Series([volatility_score])
            scalar = True
        else:
            scalar = False

        regime = pd.Series(RegimeType.UNCERTAIN, index=trend_score.index, dtype=object)
        valid = trend_score.notna() & volatility_score.notna()

        is_volatile = valid & (volatility_score > volatility_threshold)
        is_trending_up = valid & (trend_score >= bull_threshold)
        is_trending_down = valid & (trend_score <= bear_threshold)
        is_sideways = valid & ~is_trending_up & ~is_trending_down & ~is_volatile

        # High volatility + trend → keep direction; high vol + flat → VOLATILE
        regime[is_volatile & is_trending_up] = RegimeType.BULLISH
        regime[is_volatile & is_trending_down] = RegimeType.BEARISH
        regime[is_volatile & ~is_trending_up & ~is_trending_down] = RegimeType.VOLATILE

        # Normal volatility
        regime[~is_volatile & is_trending_up] = RegimeType.BULLISH
        regime[~is_volatile & is_trending_down] = RegimeType.BEARISH
        regime[is_sideways] = RegimeType.SIDEWAYS

        if scalar:
            return regime.iloc[0]
        return regime

    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return series.rolling(window=period, min_periods=period).mean()
    
    def _detect_sma_slope(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime using SMA slope thresholding.

        Converts the continuous sma_slope score in [-1, 1] to discrete
        RegimeType labels:
          - Bullish:  score > +0.25
          - Bearish:  score < -0.25
          - Sideways: |score| <= 0.25
        NaN warmup bars are labeled UNCERTAIN.
        """
        score = self._compute_sma_slope_score(df)

        regime = pd.Series(index=df.index, dtype=object)
        regime[:] = RegimeType.UNCERTAIN

        valid = score.notna()
        regime[valid & (score > 0.25)] = RegimeType.BULLISH
        regime[valid & (score < -0.25)] = RegimeType.BEARISH
        regime[valid & (score >= -0.25) & (score <= 0.25)] = RegimeType.SIDEWAYS

        return regime

    @staticmethod
    def _infer_bar_interval_hours(df: pd.DataFrame) -> float:
        """Infer the bar interval in hours from the DataFrame's datetime index.

        Uses the median difference between consecutive timestamps.
        Returns 24.0 (daily) if the index has fewer than 2 rows or
        is not a DatetimeIndex.
        """
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
            return 24.0
        diffs = df.index.to_series().diff().dropna()
        if diffs.empty:
            return 24.0
        median_td = diffs.median()
        hours = median_td.total_seconds() / 3600.0
        return max(hours, 0.01)  # safety floor

    def _compute_sma_slope_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a continuous trend score using the slope of an SMA.

        The default ``sma_period`` and ``slope_window`` are calibrated for
        **daily (1d)** candles.  When the data has a sub-daily bar interval
        the parameters are automatically scaled up so that the SMA and slope
        window cover equivalent calendar durations (e.g. SMA(20) on 1d
        becomes SMA(480) on 1h).  This prevents the noisy micro-slopes that
        caused all-bearish regime detection on hourly data.

        Steps:
        1. Auto-scale ``sma_period`` and ``slope_window`` to match the
           data's bar frequency (if sub-daily).
        2. Compute SMA(sma_period) on close prices.
        3. Compute percentage slope over ``slope_window`` bars.
        4. Shift forward by ``sma_period // 2`` to correct for SMA lag.
        5. Normalize to [-1, 1] via tanh with an adaptive threshold
           derived from the median absolute slope.

        Parameters (via self.params):
            sma_period:    SMA lookback in *daily* bars (default 20).
                           Auto-scaled for sub-daily timeframes.
            slope_window:  Bars over which to measure slope in *daily* bars
                           (default 5).  Auto-scaled likewise.
            tanh_k:        Sensitivity multiplier applied to median absolute
                           slope for the tanh denominator (default 0.3).
                           Lower = more sensitive (more bull/bear labels).
        """
        close = df['close']
        sma_period_cfg = self.params.get('sma_period', 20)
        slope_window_cfg = self.params.get('slope_window', 5)
        tanh_k = self.params.get('tanh_k', 0.3)

        # ── Auto-scale for sub-daily timeframes ──
        bar_hours = self._infer_bar_interval_hours(df)
        if bar_hours < 23.0:  # sub-daily data detected
            scale = 24.0 / bar_hours
            sma_period = max(2, int(round(sma_period_cfg * scale)))
            slope_window = max(1, int(round(slope_window_cfg * scale)))
            logger.debug(
                "SMA-slope auto-scaled for %.1fh bars: sma_period %d→%d, "
                "slope_window %d→%d",
                bar_hours, sma_period_cfg, sma_period,
                slope_window_cfg, slope_window,
            )
        else:
            sma_period = sma_period_cfg
            slope_window = slope_window_cfg

        # 1. SMA
        sma = self._calculate_sma(close, sma_period)

        # 2. Percentage slope: (sma[t] - sma[t - slope_window]) / sma[t - slope_window]
        sma_shifted = sma.shift(slope_window)
        slope = (sma - sma_shifted) / sma_shifted.replace(0, np.nan)

        # 3. Shift forward by half the SMA period to correct for lag
        # SMA(n) at candle t describes the average of candles [t-n+1 .. t],
        # whose midpoint is t - n//2. Shifting forward by n//2 aligns the
        # score with the candle it best describes.
        shift_amount = sma_period // 2
        slope = slope.shift(-shift_amount)

        # 4. Adaptive tanh normalization
        # Use median of absolute slope as the reference scale so that the
        # score naturally spreads across [-1, +1] regardless of the
        # asset's average drift. The multiplier k controls sensitivity:
        # k=0.3 → the tanh denominator is 0.3 * median, meaning a slope
        # equal to ~0.3 * median maps to ~tanh(1)≈0.76.
        abs_slope = slope.abs()
        median_slope = abs_slope.median()
        threshold = max(median_slope * tanh_k, 1e-8)
        score = np.tanh(slope / threshold)

        return pd.Series(score, index=df.index).clip(-1.0, 1.0)
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        
        return atr
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate ADX and Directional Movement indicators.
        
        Returns:
            Tuple of (ADX, +DI, -DI) Series
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # Calculate DX and ADX
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        dx = 100 * di_diff / di_sum.replace(0, np.nan)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx, plus_di, minus_di
    
    def _detect_sma_adx(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime using SMA crossover + ADX trend strength.
        
        Rules:
        - Bullish: SMA_fast > SMA_slow AND ADX > threshold
        - Bearish: SMA_fast < SMA_slow AND ADX > threshold
        - Sideways: ADX < threshold
        """
        params = self.params
        close = df['close']
        
        # Calculate indicators
        sma_fast = self._calculate_sma(close, params['sma_fast'])
        sma_slow = self._calculate_sma(close, params['sma_slow'])
        adx, _, _ = self._calculate_adx(df, params['adx_period'])
        
        # Initialize regime series
        regime = pd.Series(index=df.index, dtype=object)
        regime[:] = RegimeType.UNCERTAIN
        
        # Apply classification rules
        is_trending = adx >= params['adx_threshold']
        is_bullish_trend = sma_fast > sma_slow
        
        # Sideways when ADX is low
        regime[adx < params['adx_threshold']] = RegimeType.SIDEWAYS
        
        # Bullish when trending up
        regime[is_trending & is_bullish_trend] = RegimeType.BULLISH
        
        # Bearish when trending down
        regime[is_trending & ~is_bullish_trend] = RegimeType.BEARISH
        
        return regime
    
    def _detect_adx_di(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime using ADX + Directional Movement.
        
        Rules:
        - Bullish: ADX > threshold AND +DI > -DI
        - Bearish: ADX > threshold AND -DI > +DI
        - Sideways: ADX < sideways_threshold
        - Uncertain: sideways_threshold <= ADX < threshold
        """
        params = self.params
        
        adx, plus_di, minus_di = self._calculate_adx(df, params['adx_period'])
        
        regime = pd.Series(index=df.index, dtype=object)
        regime[:] = RegimeType.UNCERTAIN
        
        # Sideways when ADX is low
        regime[adx < params['adx_sideways_threshold']] = RegimeType.SIDEWAYS
        
        # Trending markets
        is_strong_trend = adx >= params['adx_threshold']
        regime[is_strong_trend & (plus_di > minus_di)] = RegimeType.BULLISH
        regime[is_strong_trend & (minus_di > plus_di)] = RegimeType.BEARISH
        
        return regime
    
    def _detect_returns(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime using rolling return distribution.
        
        Rules:
        - Bullish: mean(returns) > threshold AND std(returns) < vol_cap
        - Bearish: mean(returns) < -threshold AND std(returns) < vol_cap
        - Volatile: std(returns) > vol_cap
        - Sideways: |mean(returns)| < threshold
        """
        params = self.params
        close = df['close']
        
        # Calculate rolling returns
        returns = close.pct_change()
        
        # Rolling statistics
        lookback = params['lookback_period']
        rolling_mean = returns.rolling(window=lookback, min_periods=lookback).mean()
        rolling_std = returns.rolling(window=lookback, min_periods=lookback).std()
        
        regime = pd.Series(index=df.index, dtype=object)
        regime[:] = RegimeType.UNCERTAIN
        
        threshold = params['trend_threshold']
        vol_cap = params['volatility_cap']
        
        # Volatile markets
        is_volatile = rolling_std > vol_cap
        regime[is_volatile] = RegimeType.VOLATILE
        
        # Non-volatile trending markets
        is_calm = ~is_volatile & rolling_std.notna()
        regime[is_calm & (rolling_mean > threshold)] = RegimeType.BULLISH
        regime[is_calm & (rolling_mean < -threshold)] = RegimeType.BEARISH
        regime[is_calm & (abs(rolling_mean) <= threshold)] = RegimeType.SIDEWAYS
        
        return regime
    
    def _detect_rolling_returns(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime using rolling returns with improved thresholds.
        
        This method provides better balance than the basic 'returns' method
        by using optimized window sizes and thresholds for crypto markets.
        
        Rules:
        - Bullish: mean rolling return > threshold
        - Bearish: mean rolling return < -threshold
        - Sideways: |mean return| <= threshold
        """
        params = self.params
        close = df['close']
        
        # Calculate returns
        returns = close.pct_change()
        
        # Rolling mean
        window = params.get('window', 50)
        threshold = params.get('threshold', 0.0005)  # 0.05% per bar
        
        rolling_mean = returns.rolling(window=window, min_periods=window).mean()
        
        regime = pd.Series(index=df.index, dtype=object)
        regime[:] = RegimeType.UNCERTAIN
        
        # Classify based on rolling mean returns
        valid = rolling_mean.notna()
        regime[valid & (rolling_mean > threshold)] = RegimeType.BULLISH
        regime[valid & (rolling_mean < -threshold)] = RegimeType.BEARISH
        regime[valid & (rolling_mean >= -threshold) & (rolling_mean <= threshold)] = RegimeType.SIDEWAYS
        
        return regime
    
    def _detect_adx_di_hysteresis(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime using ADX + DI with hysteresis.
        
        This is an improved version of adx_di that uses hysteresis to
        prevent rapid flipping between trend and sideways modes.
        
        Rules:
        - Enter trend mode when ADX > adx_enter
        - Stay in trend mode until ADX < adx_exit
        - In trend mode: +DI > -DI = bullish, else bearish
        - In range mode: sideways
        
        Uses numpy vectorization with a single pass for hysteresis state tracking.
        """
        params = self.params
        adx_enter = params.get('adx_enter', 25)
        adx_exit = params.get('adx_exit', 20)
        adx_period = params.get('adx_period', 14)
        
        # Calculate ADX and DI
        adx, plus_di, minus_di = self._calculate_adx(df, adx_period)
        
        n = len(df)
        adx_vals = adx.values
        plus_di_vals = plus_di.values
        minus_di_vals = minus_di.values
        
        # Vectorized hysteresis: compute trigger/release signals first
        enters_trend = adx_vals > adx_enter   # Signal to enter trend mode
        exits_trend = adx_vals < adx_exit      # Signal to exit trend mode
        
        # Single pass for hysteresis state (unavoidable due to state dependency)
        # But using numpy arrays for speed instead of .iloc[] lookup
        in_trend = np.zeros(n, dtype=np.bool_)
        trend_state = False
        for i in range(n):
            if np.isnan(adx_vals[i]):
                continue
            if not trend_state and enters_trend[i]:
                trend_state = True
            elif trend_state and exits_trend[i]:
                trend_state = False
            in_trend[i] = trend_state
        
        # Vectorized regime classification
        bullish_mask = in_trend & (plus_di_vals > minus_di_vals)
        bearish_mask = in_trend & ~(plus_di_vals > minus_di_vals)
        sideways_mask = ~in_trend & ~np.isnan(adx_vals)
        
        regime = pd.Series(RegimeType.UNCERTAIN, index=df.index, dtype=object)
        regime[bullish_mask] = RegimeType.BULLISH
        regime[bearish_mask] = RegimeType.BEARISH
        regime[sideways_mask] = RegimeType.SIDEWAYS
        
        return regime
    
    def _detect_hmm(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime using Hidden Markov Model with multiple features.
        
        IMPROVED VERSION based on QuantStart research:
        - Uses both returns AND volatility as features (2D observation)
        - Maps states by mean return AND variance
        - Applies smoothing to prevent rapid flipping
        
        The HMM identifies volatility regimes first, then we overlay direction.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed, falling back to returns method")
            return self._detect_returns(df)
        
        params = self.params
        n_states = params.get('n_states', 3)
        min_dwell = params.get('min_dwell', 10)  # Increased from 5
        vol_window = params.get('vol_window', 20)  # For rolling volatility
        
        # Prepare features: returns + rolling volatility
        close = df['close']
        returns = close.pct_change()
        volatility = returns.rolling(window=vol_window).std()
        
        # Create feature matrix
        features_df = pd.DataFrame({
            'returns': returns,
            'volatility': volatility
        }).dropna()
        
        if len(features_df) < 250:
            logger.warning("Insufficient data for HMM, falling back to rolling_returns method")
            return self._detect_rolling_returns(df)
        
        # Standardize features to avoid numerical issues
        features = features_df.values.copy()
        feature_means = features.mean(axis=0)
        feature_stds = features.std(axis=0)
        feature_stds[feature_stds == 0] = 1  # Prevent division by zero
        features_normalized = (features - feature_means) / feature_stds
        
        # Handle NaN/Inf
        features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=3.0, neginf=-3.0)
        
        try:
            # Fit HMM with diagonal covariance (more stable than full)
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",  # Changed from "full" - more stable
                n_iter=200,
                random_state=42,
                init_params="stmc",
            )
            model.fit(features_normalized)
            
            # Get predictions using normalized features
            hidden_states = model.predict(features_normalized)
            
            # Analyze each state's characteristics
            state_stats = []
            for state in range(n_states):
                mask = hidden_states == state
                if mask.sum() > 0:
                    state_returns = features_df['returns'].values[mask]
                    state_vol = features_df['volatility'].values[mask]
                    state_stats.append({
                        'state': state,
                        'mean_return': np.mean(state_returns),
                        'mean_vol': np.mean(state_vol),
                        'count': mask.sum()
                    })
                else:
                    state_stats.append({
                        'state': state,
                        'mean_return': 0,
                        'mean_vol': 0,
                        'count': 0
                    })
            
            # Map states to regimes using RELATIVE ranking (adaptive to data)
            # Since HMM primarily captures volatility regimes, we use:
            # - Rank states by mean_return relative to overall mean
            # - Consider volatility levels for volatile/sideways distinction
            
            state_map = {}
            vol_threshold = np.median([s['mean_vol'] for s in state_stats])
            
            # Sort states by mean_return for relative ranking
            sorted_states = sorted(state_stats, key=lambda x: x['mean_return'], reverse=True)
            
            # For 3 states: Best=BULLISH, Worst=BEARISH, Middle=SIDEWAYS/VOLATILE
            # This ensures we always have bullish and bearish in the output
            
            for rank, s in enumerate(sorted_states):
                if rank == 0:  # Highest return
                    state_map[s['state']] = RegimeType.BULLISH
                elif rank == len(sorted_states) - 1:  # Lowest return
                    # Lowest returning state = BEARISH (always)
                    state_map[s['state']] = RegimeType.BEARISH
                else:  # Middle state(s)
                    # Mid return + high volatility = VOLATILE
                    # Mid return + low volatility = SIDEWAYS
                    if s['mean_vol'] > vol_threshold * 1.3:
                        state_map[s['state']] = RegimeType.VOLATILE
                    else:
                        state_map[s['state']] = RegimeType.SIDEWAYS
            
            # Apply smoothing: require minimum consecutive bars to switch
            regime_values = []
            current_regime = RegimeType.UNCERTAIN
            pending_regime = None
            pending_count = 0
            
            for i in range(len(hidden_states)):
                predicted_state = hidden_states[i]
                predicted_regime = state_map.get(predicted_state, RegimeType.SIDEWAYS)
                
                if predicted_regime != current_regime:
                    if pending_regime == predicted_regime:
                        pending_count += 1
                    else:
                        pending_regime = predicted_regime
                        pending_count = 1
                    
                    # Check if we have enough consecutive bars to switch
                    if pending_count >= min_dwell:
                        current_regime = predicted_regime
                        pending_regime = None
                        pending_count = 0
                else:
                    pending_regime = None
                    pending_count = 0
                
                regime_values.append(current_regime)
            
            # Create regime series aligned with features index
            regime = pd.Series(regime_values, index=features_df.index)
            
            # Reindex to full dataframe
            regime = regime.reindex(df.index, fill_value=RegimeType.UNCERTAIN)
            
            # Log state statistics for debugging
            logger.debug(f"HMM State Statistics: {state_stats}")
            logger.debug(f"HMM State Mapping: {state_map}")
            
            return regime
            
        except Exception as e:
            logger.warning(f"HMM fitting failed: {e}, falling back to rolling_returns method")
            return self._detect_rolling_returns(df)

    def _detect_volatility_cluster(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime using volatility clustering and trend direction.

        Uses the well-known stylized fact that volatility clusters in financial
        time series (ARCH/GARCH effects). Instead of relying solely on ADX
        (which over-classifies sideways), this method:

        1. Computes rolling realized volatility (exponentially weighted)
        2. Classifies each bar into volatility regime using adaptive quantile
           thresholds computed over a trailing lookback window
        3. Overlays trend direction from smoothed returns

        Regime mapping:
        - High vol + positive trend → BULLISH (strong rally with expansion)
        - High vol + negative trend → BEARISH (crash / sell-off)
        - High vol + no trend → VOLATILE (choppy high-vol market)
        - Low/normal vol + positive trend → BULLISH (steady uptrend)
        - Low/normal vol + negative trend → BEARISH (steady downtrend)
        - Low/normal vol + no trend → SIDEWAYS

        This avoids the ADX binary threshold problem and naturally captures
        volatility regime changes that are invisible to trend-only methods.
        """
        params = self.params
        vol_window = params.get('vol_window', 20)
        vol_lookback = params.get('vol_lookback', 60)
        high_vol_pct = params.get('high_vol_pct', 75)
        low_vol_pct = params.get('low_vol_pct', 25)
        trend_window = params.get('trend_window', 50)
        trend_threshold = params.get('trend_threshold', 0.001)

        close = df['close']
        returns = close.pct_change()

        # Exponentially weighted rolling volatility (captures clustering)
        rolling_vol = returns.ewm(span=vol_window, adjust=False).std()

        # Adaptive quantile thresholds (trailing lookback)
        high_vol_threshold = rolling_vol.rolling(
            window=vol_lookback, min_periods=vol_lookback
        ).quantile(high_vol_pct / 100.0)
        low_vol_threshold = rolling_vol.rolling(
            window=vol_lookback, min_periods=vol_lookback
        ).quantile(low_vol_pct / 100.0)

        # Trend direction from smoothed rolling returns
        rolling_return = returns.rolling(window=trend_window, min_periods=trend_window).mean()

        # Classification
        regime = pd.Series(RegimeType.UNCERTAIN, index=df.index, dtype=object)

        valid = rolling_vol.notna() & high_vol_threshold.notna() & rolling_return.notna()
        is_high_vol = valid & (rolling_vol > high_vol_threshold)
        is_low_vol = valid & (rolling_vol < low_vol_threshold)
        is_normal_vol = valid & ~is_high_vol & ~is_low_vol

        is_uptrend = rolling_return > trend_threshold
        is_downtrend = rolling_return < -trend_threshold
        is_flat = ~is_uptrend & ~is_downtrend

        # High volatility regimes
        regime[is_high_vol & is_uptrend] = RegimeType.BULLISH
        regime[is_high_vol & is_downtrend] = RegimeType.BEARISH
        regime[is_high_vol & is_flat] = RegimeType.VOLATILE

        # Normal/low volatility regimes
        regime[(is_normal_vol | is_low_vol) & is_uptrend] = RegimeType.BULLISH
        regime[(is_normal_vol | is_low_vol) & is_downtrend] = RegimeType.BEARISH
        regime[(is_normal_vol | is_low_vol) & is_flat] = RegimeType.SIDEWAYS

        return regime

    def _detect_bollinger(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime using Bollinger Band position.
        
        Rules:
        - Bullish: Price consistently above middle band
        - Bearish: Price consistently below middle band
        - Sideways: Price oscillates around middle band
        """
        params = self.params
        close = df['close']
        
        # Calculate Bollinger Bands
        sma = self._calculate_sma(close, params['period'])
        std = close.rolling(window=params['period'], min_periods=params['period']).std()
        upper_band = sma + params['std_dev'] * std
        lower_band = sma - params['std_dev'] * std
        
        # Calculate position relative to middle band
        above_middle = (close > sma).astype(int)
        below_middle = (close < sma).astype(int)
        
        # Rolling consistency check
        lookback = params['lookback_bars']
        threshold = params['consistency_threshold']
        
        above_ratio = above_middle.rolling(window=lookback, min_periods=lookback).mean()
        below_ratio = below_middle.rolling(window=lookback, min_periods=lookback).mean()
        
        regime = pd.Series(index=df.index, dtype=object)
        regime[:] = RegimeType.UNCERTAIN
        
        # Classify based on consistency
        regime[above_ratio >= threshold] = RegimeType.BULLISH
        regime[below_ratio >= threshold] = RegimeType.BEARISH
        regime[(above_ratio < threshold) & (below_ratio < threshold)] = RegimeType.SIDEWAYS
        
        return regime
    
    def _detect_ensemble(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime using ensemble voting of multiple methods.
        
        IMPROVED: Uses weighted voting with the best-performing methods:
        - ADX+DI Hysteresis (weight=2, most stable)
        - Rolling Returns (weight=2, best balanced)
        - HMM (weight=1, captures volatility regimes)
        
        The ensemble provides robustness by combining:
        1. Technical indicators (ADX) for trend strength
        2. Statistical approach (rolling returns) for directional bias
        3. Machine learning (HMM) for volatility-based regimes
        """
        # Create temporary detectors
        adx_detector = RegimeDetector(method='adx_di_hysteresis')
        returns_detector = RegimeDetector(method='rolling_returns', 
                                          params={'window': 50, 'threshold': 0.0005})
        
        # Run detection methods
        methods_weights = {
            'adx_di_hysteresis': (adx_detector.detect(df), 2),
            'rolling_returns': (returns_detector.detect(df), 2),
        }
        
        # Try HMM if available (lower weight due to potential noise)
        try:
            hmm_detector = RegimeDetector(method='hmm', params={'n_states': 3, 'min_dwell': 10})
            methods_weights['hmm'] = (hmm_detector.detect(df), 1)
        except Exception as e:
            logger.warning(f"HMM regime detection failed, skipping: {type(e).__name__}: {e}")
        
        regime = pd.Series(RegimeType.UNCERTAIN, index=df.index, dtype=object)
        
        # Vectorized weighted voting using numeric encoding
        regime_map = {
            RegimeType.BULLISH: 0,
            RegimeType.BEARISH: 1,
            RegimeType.SIDEWAYS: 2,
        }
        n_regimes = 3
        n = len(df.index)
        
        # Accumulate weighted votes per regime type
        vote_matrix = np.zeros((n, n_regimes), dtype=np.float32)
        
        for method_name, (method_regimes, weight) in methods_weights.items():
            # Align method_regimes to df.index
            aligned = method_regimes.reindex(df.index)
            for regime_type, regime_idx in regime_map.items():
                mask = (aligned == regime_type)
                # Also treat VOLATILE as SIDEWAYS
                if regime_type == RegimeType.SIDEWAYS:
                    mask = mask | (aligned == RegimeType.VOLATILE)
                vote_matrix[mask.values, regime_idx] += weight
        
        # Optionally add ML model as an ensemble voter (Phase 1B)
        try:
            ml_model_path = Path(__file__).parent.parent / 'ml' / 'models' / 'regime_lgbm.pkl'
            if ml_model_path.exists():
                from genetic_algorithm.ml.regime_detector import MLRegimeDetector
                ml_detector = MLRegimeDetector(model_path=str(ml_model_path))
                ml_regimes = ml_detector.detect(df)
                ml_weight = 2  # Same weight as the top rule-based methods
                methods_weights['ml_lgbm'] = (ml_regimes, ml_weight)
                # Re-accumulate votes for ML method
                aligned_ml = ml_regimes.reindex(df.index)
                for regime_type, regime_idx in regime_map.items():
                    mask = (aligned_ml == regime_type)
                    if regime_type == RegimeType.SIDEWAYS:
                        mask = mask | (aligned_ml == RegimeType.VOLATILE)
                    vote_matrix[mask.values, regime_idx] += ml_weight
                logger.debug("ML regime model included in ensemble voting")
        except Exception as e:
            logger.debug(f"ML regime model not available for ensemble: {e}")

        # Find winner: highest weighted vote per row
        has_votes = vote_matrix.sum(axis=1) > 0
        winners = np.argmax(vote_matrix, axis=1)
        
        # Map back to RegimeType
        idx_to_regime = {0: RegimeType.BULLISH, 1: RegimeType.BEARISH, 2: RegimeType.SIDEWAYS}
        for regime_idx, regime_type in idx_to_regime.items():
            mask = has_votes & (winners == regime_idx)
            regime[mask] = regime_type
        
        return regime

    def _detect_advanced_ensemble(self, df: pd.DataFrame) -> pd.Series:
        """
        Advanced ensemble — combines 6 detection methods with confidence-weighted voting.

        This is the recommended 'standard' detection method that maximizes
        robustness by combining complementary approaches:

        1. ADX+DI Hysteresis (weight=2) — stable trend detection
        2. Rolling Returns (weight=2) — momentum / directional bias
        3. HMM (weight=1.5) — statistical state model
        4. Volatility Cluster (weight=1) — volatility regime overlay
        5. Bollinger (weight=1) — mean-reversion / range detection
        6. ML LightGBM (weight=3 when available) — learned consensus

        Each method's contribution is scaled by a rolling confidence score
        (agreement rate over a lookback window), so noisy methods are
        automatically down-weighted.
        """
        # Define methods and base weights
        method_specs = [
            ('adx_di_hysteresis', {}, 2.0),
            ('rolling_returns', {'window': 50, 'threshold': 0.0005}, 2.0),
            ('hmm', {'n_states': 3, 'min_dwell': 10}, 1.5),
            ('volatility_cluster', {}, 1.0),
            ('bollinger', {}, 1.0),
        ]

        methods_results = {}
        for method_name, params, base_weight in method_specs:
            try:
                det = RegimeDetector(method=method_name, params=params)
                result = det.detect(df)
                methods_results[method_name] = (result, base_weight)
            except Exception as e:
                logger.debug(f"Advanced ensemble: {method_name} failed, skipping: {e}")

        # Try ML model (highest weight when available)
        try:
            ml_model_path = Path(__file__).parent.parent / 'ml' / 'models' / 'regime_lgbm.pkl'
            if ml_model_path.exists():
                from genetic_algorithm.ml.regime_detector import MLRegimeDetector
                ml_detector = MLRegimeDetector(model_path=str(ml_model_path))
                ml_regimes = ml_detector.detect(df)
                methods_results['ml_lgbm'] = (ml_regimes, 3.0)
                logger.info("Advanced ensemble: ML model included (weight=3.0)")
        except Exception as e:
            logger.debug(f"Advanced ensemble: ML model not available: {e}")

        if not methods_results:
            logger.warning("Advanced ensemble: all methods failed, falling back to adx_di_hysteresis")
            det = RegimeDetector(method='adx_di_hysteresis')
            return det.detect(df)

        logger.info(
            f"Advanced ensemble: {len(methods_results)} methods active: "
            f"{list(methods_results.keys())}"
        )

        # Regime encoding
        regime_map = {
            RegimeType.BULLISH: 0,
            RegimeType.BEARISH: 1,
            RegimeType.SIDEWAYS: 2,
        }
        n_regimes = 3
        n = len(df.index)

        # Compute confidence-adjusted weights per method
        # Confidence = rolling agreement rate with the majority over a lookback window
        confidence_window = min(100, max(20, n // 10))  # adaptive window

        # First pass: collect all results aligned to df.index
        aligned_results = {}
        for method_name, (result, _) in methods_results.items():
            aligned = result.reindex(df.index)
            # Encode to numeric
            encoded = pd.Series(np.nan, index=df.index)
            for regime_type, regime_idx in regime_map.items():
                mask = (aligned == regime_type)
                if regime_type == RegimeType.SIDEWAYS:
                    mask = mask | (aligned == RegimeType.VOLATILE)
                encoded[mask] = regime_idx
            aligned_results[method_name] = encoded

        # Compute per-bar majority vote (unweighted) for confidence calc
        all_encoded = pd.DataFrame(aligned_results)
        majority = all_encoded.mode(axis=1).iloc[:, 0]  # most common regime per bar

        # Compute per-method rolling confidence (agreement with majority)
        method_confidence = {}
        for method_name in aligned_results:
            agreement = (aligned_results[method_name] == majority).astype(float)
            rolling_conf = agreement.rolling(
                window=confidence_window, min_periods=max(1, confidence_window // 4)
            ).mean()
            # Fill leading NaN with global agreement rate
            global_conf = agreement.mean()
            rolling_conf = rolling_conf.fillna(global_conf)
            method_confidence[method_name] = rolling_conf

        # Build confidence-weighted vote matrix
        vote_matrix = np.zeros((n, n_regimes), dtype=np.float64)

        for method_name, (result, base_weight) in methods_results.items():
            conf = method_confidence[method_name].values  # shape (n,)
            effective_weight = base_weight * conf  # per-bar adaptive weight

            aligned = result.reindex(df.index)
            for regime_type, regime_idx in regime_map.items():
                mask = (aligned == regime_type)
                if regime_type == RegimeType.SIDEWAYS:
                    mask = mask | (aligned == RegimeType.VOLATILE)
                vote_matrix[mask.values, regime_idx] += effective_weight[mask.values]

        # Determine winners
        regime = pd.Series(RegimeType.UNCERTAIN, index=df.index, dtype=object)
        has_votes = vote_matrix.sum(axis=1) > 0
        winners = np.argmax(vote_matrix, axis=1)

        idx_to_regime = {0: RegimeType.BULLISH, 1: RegimeType.BEARISH, 2: RegimeType.SIDEWAYS}
        for regime_idx, regime_type in idx_to_regime.items():
            mask = has_votes & (winners == regime_idx)
            regime[mask] = regime_type

        # Log confidence stats
        for method_name, conf_series in method_confidence.items():
            avg_conf = conf_series.mean()
            base_w = methods_results[method_name][1]
            logger.debug(
                f"  {method_name}: base_weight={base_w:.1f}, "
                f"avg_confidence={avg_conf:.3f}, effective_avg={base_w * avg_conf:.3f}"
            )

        return regime

    def _detect_ml_lgbm(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime using a pre-trained LightGBM classifier.

        Requires a trained model at genetic_algorithm/ml/models/regime_lgbm.pkl.
        Train it first with: python -m genetic_algorithm.ml.train_regime

        The ML model uses raw TA features and/or rule-based method outputs
        as features (configurable via feature_mode during training).
        """
        from genetic_algorithm.ml.regime_detector import MLRegimeDetector

        model_path = self.params.get(
            'model_path',
            str(Path(__file__).parent.parent / 'ml' / 'models' / 'regime_lgbm.pkl'),
        )
        feature_mode = self.params.get('feature_mode', 'combined')

        ml_detector = MLRegimeDetector(
            model_path=model_path,
            feature_mode=feature_mode,
        )
        return ml_detector.detect(df)
    
    def classify_periods(
        self,
        df: pd.DataFrame,
        period_days: int = 90,
        min_period_days: int = 60,
        embargo_days: int = 5,
        warmup_bars: int = 200,
    ) -> List[RegimeSegment]:
        """
        Classify historical data into regime-labeled segments.
        
        Splits the data into segments of approximately period_days length,
        classifies each segment by its dominant regime, and returns a list
        of RegimeSegment objects.
        
        Args:
            df: DataFrame with OHLCV data and datetime index
            period_days: Target segment length in days
            min_period_days: Minimum acceptable segment length
            embargo_days: Gap between segments to prevent leakage
            warmup_bars: Indicator warmup period (excluded from regime calculation)
        
        Returns:
            List of RegimeSegment objects, sorted by start date
        """
        if df.empty:
            return []
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            # Check if 'date' column exists (FreqTrade format)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    logger.error(f"Cannot convert index to datetime: {e}")
                    return []
        
        # Get date range
        start_date = df.index.min().to_pydatetime()
        end_date = df.index.max().to_pydatetime()
        total_days = (end_date - start_date).days
        
        logger.info(f"Classifying periods from {start_date.date()} to {end_date.date()} ({total_days} days)")
        
        # Detect regime for all bars
        regime_series = self.detect(df)
        
        # Create segments
        segments = []
        current_start = start_date + timedelta(days=warmup_bars // (24 * 4))  # Approx warmup in days
        segment_idx = 0
        
        while current_start < end_date:
            segment_end = min(current_start + timedelta(days=period_days), end_date)
            
            # Check minimum period length
            if (segment_end - current_start).days < min_period_days:
                logger.debug(f"Skipping short segment: {current_start.date()} to {segment_end.date()}")
                break
            
            # Get regime values for this segment
            mask = (df.index >= current_start) & (df.index < segment_end)
            segment_regimes = regime_series[mask]
            
            if segment_regimes.empty:
                current_start = segment_end + timedelta(days=embargo_days)
                continue
            
            # Calculate dominant regime and confidence
            regime_counts = segment_regimes.value_counts()
            
            # Filter out UNCERTAIN
            valid_regimes = regime_counts.drop(RegimeType.UNCERTAIN, errors='ignore')
            
            if valid_regimes.empty:
                dominant_regime = RegimeType.UNCERTAIN
                confidence = 0.0
            else:
                dominant_regime = valid_regimes.idxmax()
                confidence = valid_regimes[dominant_regime] / len(segment_regimes)
            
            # Calculate segment metadata
            segment_df = df[mask]
            returns = segment_df['close'].pct_change().dropna()
            
            metadata = {
                'mean_return': returns.mean() if len(returns) > 0 else 0.0,
                'volatility': returns.std() if len(returns) > 0 else 0.0,
                'total_return': (segment_df['close'].iloc[-1] / segment_df['close'].iloc[0] - 1) if len(segment_df) > 1 else 0.0,
                'bar_count': len(segment_df),
                'regime_distribution': {k.value: v for k, v in regime_counts.items()},
            }
            
            # Create segment ID (hash of timerange for stable caching)
            segment_id = f"seg_{segment_idx:03d}_{current_start.strftime('%Y%m%d')}_{segment_end.strftime('%Y%m%d')}"
            
            segment = RegimeSegment(
                segment_id=segment_id,
                start_date=current_start,
                end_date=segment_end,
                regime=dominant_regime,
                confidence=confidence,
                metadata=metadata,
            )
            
            segments.append(segment)
            logger.debug(f"Created segment: {segment.segment_id} - {segment.regime.value} ({confidence:.1%} confidence)")
            
            # Move to next segment with embargo gap
            current_start = segment_end + timedelta(days=embargo_days)
            segment_idx += 1
        
        logger.info(f"Created {len(segments)} regime segments")
        
        # Log regime distribution
        regime_dist = {}
        for seg in segments:
            regime_dist[seg.regime.value] = regime_dist.get(seg.regime.value, 0) + 1
        logger.info(f"Regime distribution: {regime_dist}")
        
        return segments

    def classify_periods_by_score(
        self,
        df: pd.DataFrame,
        bullish_min: float = 0.35,
        bearish_max: float = -0.35,
        min_segment_days: int = 14,
        max_segment_days: int = 180,
        merge_threshold_days: int = 7,
        embargo_days: int = 5,
        warmup_bars: int = 200,
    ) -> List[RegimeSegment]:
        """
        Score-band adaptive segmentation.

        Uses continuous ``trend_score`` from ``detect_continuous()`` to
        build variable-length segments.  Instead of classifying each bar
        discretely and finding change-points in the discrete labels, this
        method tracks when the continuous score crosses configurable band
        boundaries:

            trend_score >= bullish_min   →  BULLISH
            bearish_max < score < bullish_min  →  SIDEWAYS
            trend_score <= bearish_max   →  BEARISH

        Because the bands partition the full [-1, +1] range exhaustively,
        **every regime is guaranteed to receive data** whenever the history
        covers diverse market conditions (3+ years recommended).

        Segments are created at band crossings, merged when too short,
        split when too long, and enriched with continuous score metadata
        (``avg_trend_score``, ``avg_volatility_score``).

        Args:
            df: OHLCV DataFrame with datetime index
            bullish_min: Lower trend_score boundary for bullish regime
            bearish_max: Upper trend_score boundary for bearish regime
            min_segment_days: Merge segments shorter than this
            max_segment_days: Split segments longer than this
            merge_threshold_days: Short-segment merge threshold
            embargo_days: Gap between segments to prevent data leakage
            warmup_bars: Indicator warmup bars to skip

        Returns:
            List of RegimeSegment objects with score-based regime labels
        """
        if df.empty:
            return []

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    logger.error(f"Cannot convert index to datetime: {e}")
                    return []

        start_date = df.index.min().to_pydatetime()
        end_date = df.index.max().to_pydatetime()
        total_days = (end_date - start_date).days

        logger.info(
            "Score-band segmentation: %s to %s (%d days), "
            "bands=[bear<=%.2f | sideways | bull>=%.2f]",
            start_date.date(), end_date.date(), total_days,
            bearish_max, bullish_min,
        )

        # ── Compute continuous scores ──
        trend_score, volatility_score = self.detect_continuous(df)

        # Skip warmup period
        warmup_end = df.index[min(warmup_bars, len(df) - 1)]
        trend_score = trend_score.loc[trend_score.index >= warmup_end]
        volatility_score = volatility_score.loc[volatility_score.index >= warmup_end]
        df_valid = df.loc[df.index >= warmup_end]

        if trend_score.empty:
            logger.warning("No data after warmup — cannot create segments")
            return []

        # ── Assign each bar to a band ──
        def _score_to_band(score: float) -> str:
            if np.isnan(score):
                return 'sideways'  # safe default
            if score >= bullish_min:
                return 'bullish'
            if score <= bearish_max:
                return 'bearish'
            return 'sideways'

        band_series = trend_score.map(_score_to_band)

        # ── Find band crossings (change points) ──
        prev_band = band_series.shift(1)
        change_mask = (band_series != prev_band) & prev_band.notna()
        change_indices = band_series.index[change_mask].tolist()

        all_boundaries = [band_series.index[0]] + change_indices + [band_series.index[-1]]

        # ── Build raw segments between crossings ──
        raw_segments: List[Dict[str, Any]] = []
        for i in range(len(all_boundaries) - 1):
            seg_start = all_boundaries[i]
            seg_end = all_boundaries[i + 1]

            mask = (trend_score.index >= seg_start) & (trend_score.index < seg_end)
            seg_trend = trend_score[mask]
            seg_vol = volatility_score[mask]

            if seg_trend.empty:
                continue

            avg_trend = float(seg_trend.mean())
            avg_vol = float(seg_vol.mean())
            trend_std = float(seg_trend.std()) if len(seg_trend) > 1 else 0.0

            # Regime from average score in the band
            if avg_trend >= bullish_min:
                regime = RegimeType.BULLISH
            elif avg_trend <= bearish_max:
                regime = RegimeType.BEARISH
            else:
                regime = RegimeType.SIDEWAYS

            # Confidence: high when trend_score is consistent within segment
            # 1 - std  → score=1 when perfectly flat, lower when noisy
            confidence = max(0.0, min(1.0, 1.0 - trend_std))

            s = seg_start.to_pydatetime() if hasattr(seg_start, 'to_pydatetime') else seg_start
            e = seg_end.to_pydatetime() if hasattr(seg_end, 'to_pydatetime') else seg_end
            duration_days = (e - s).days

            raw_segments.append({
                'start': s,
                'end': e,
                'regime': regime,
                'confidence': confidence,
                'duration_days': duration_days,
                'bar_count': len(seg_trend),
                'avg_trend_score': avg_trend,
                'avg_volatility_score': avg_vol,
                'trend_score_std': trend_std,
            })

        # ── Merge short segments into neighbours ──
        merged = self._merge_short_segments(raw_segments, merge_threshold_days)

        # ── Split excessively long segments ──
        final_raw: List[Dict[str, Any]] = []
        for seg in merged:
            if seg['duration_days'] > max_segment_days:
                n_splits = (seg['duration_days'] // max_segment_days) + 1
                split_dur = timedelta(days=seg['duration_days'] / n_splits)
                for j in range(n_splits):
                    sub_start = seg['start'] + split_dur * j
                    sub_end = seg['start'] + split_dur * (j + 1)
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

        # ── Remove segments below minimum length ──
        final_raw = [s for s in final_raw if s['duration_days'] >= min_segment_days]

        # ── Apply embargo gaps ──
        if embargo_days > 0 and len(final_raw) > 1:
            gap = timedelta(days=embargo_days)
            for i in range(len(final_raw)):
                if i > 0:
                    prev_end = final_raw[i - 1]['end']
                    if final_raw[i]['start'] < prev_end + gap:
                        final_raw[i]['start'] = prev_end + gap
                        d = (final_raw[i]['end'] - final_raw[i]['start']).days
                        final_raw[i]['duration_days'] = d
            # Remove segments that became too short after embargo
            final_raw = [s for s in final_raw if s['duration_days'] >= min_segment_days]

        # ── Build RegimeSegment objects ──
        segments: List[RegimeSegment] = []
        for idx, seg in enumerate(final_raw):
            # Re-derive regime from avg score (may have shifted after merge)
            avg_t = seg.get('avg_trend_score', 0.0)
            if avg_t >= bullish_min:
                regime = RegimeType.BULLISH
            elif avg_t <= bearish_max:
                regime = RegimeType.BEARISH
            else:
                regime = RegimeType.SIDEWAYS

            # Enrich metadata with continuous scores
            seg_df = df_valid.loc[
                (df_valid.index >= seg['start']) & (df_valid.index < seg['end'])
            ]
            rets = seg_df['close'].pct_change().dropna() if not seg_df.empty else pd.Series(dtype=float)

            metadata = {
                'avg_trend_score': seg.get('avg_trend_score', 0.0),
                'avg_volatility_score': seg.get('avg_volatility_score', 0.0),
                'trend_score_std': seg.get('trend_score_std', 0.0),
                'bar_count': seg.get('bar_count', 0),
                'mean_return': float(rets.mean()) if len(rets) > 0 else 0.0,
                'volatility': float(rets.std()) if len(rets) > 0 else 0.0,
                'total_return': float(
                    seg_df['close'].iloc[-1] / seg_df['close'].iloc[0] - 1
                ) if len(seg_df) > 1 else 0.0,
                'source': 'score_band',
                'bullish_min': bullish_min,
                'bearish_max': bearish_max,
            }

            segment_id = (
                f"sb_{idx:03d}_{seg['start'].strftime('%Y%m%d')}"
                f"_{seg['end'].strftime('%Y%m%d')}"
            )
            segments.append(RegimeSegment(
                segment_id=segment_id,
                start_date=seg['start'],
                end_date=seg['end'],
                regime=regime,
                confidence=seg.get('confidence', 0.5),
                metadata=metadata,
            ))

        # ── Log results ──
        regime_dist: Dict[str, int] = {}
        total_bars = 0
        for s in segments:
            regime_dist[s.regime.value] = regime_dist.get(s.regime.value, 0) + 1
            total_bars += s.metadata.get('bar_count', 0)

        logger.info(
            "Score-band segmentation produced %d segments (%d bars): %s",
            len(segments), total_bars, regime_dist,
        )

        # Warn about empty regimes
        for regime_name in ['bullish', 'bearish', 'sideways']:
            if regime_name not in regime_dist or regime_dist[regime_name] == 0:
                logger.warning(
                    "⚠ Score-band: NO segments for regime '%s'. "
                    "Consider widening the timerange (3-6 years recommended) "
                    "or adjusting band boundaries (current: bull>=%.2f, bear<=%.2f).",
                    regime_name, bullish_min, bearish_max,
                )

        return segments

    def classify_periods_adaptive(
        self,
        df: pd.DataFrame,
        min_segment_days: int = 14,
        max_segment_days: int = 180,
        merge_threshold_days: int = 7,
        warmup_bars: int = 200,
        embargo_days: int = 5,
    ) -> List[RegimeSegment]:
        """
        Adaptive segmentation: split data at regime change points.

        Instead of fixed-width windows (which mix regimes), this method:
        1. Detects per-bar regime labels across the full dataset
        2. Finds regime change points (transitions)
        3. Creates variable-length segments between change points
        4. Merges very short segments into their neighbors
        5. Computes per-segment confidence from label homogeneity

        This produces segments that are internally consistent (each segment
        contains predominantly one regime) and avoids the fixed-window problem
        where a single 90-day window straddles a bull→bear transition.

        Args:
            df: DataFrame with OHLCV data and datetime index
            min_segment_days: Minimum segment duration (merge shorter ones)
            max_segment_days: Maximum segment duration (split longer ones)
            merge_threshold_days: Segments shorter than this get merged
            warmup_bars: Indicator warmup period to skip
            embargo_days: Gap between segments to prevent leakage

        Returns:
            List of RegimeSegment objects with high internal consistency
        """
        if df.empty:
            return []

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    logger.error(f"Cannot convert index to datetime: {e}")
                    return []

        # Detect per-bar regime labels
        regime_series = self.detect(df)

        # Skip warmup bars
        if warmup_bars > 0 and len(df) > warmup_bars:
            regime_series = regime_series.iloc[warmup_bars:]
            df = df.iloc[warmup_bars:]

        if len(regime_series) < 2:
            return []

        # Find change points: indices where regime differs from previous bar
        prev_regime = regime_series.shift(1)
        change_mask = (regime_series != prev_regime) & prev_regime.notna()
        change_indices = regime_series.index[change_mask].tolist()

        # Include start and end as boundaries
        all_boundaries = [regime_series.index[0]] + change_indices + [regime_series.index[-1]]

        # Build raw segments between change points
        raw_segments: List[dict] = []
        for i in range(len(all_boundaries) - 1):
            seg_start = all_boundaries[i]
            seg_end = all_boundaries[i + 1]

            # Get regime labels in this span
            mask = (regime_series.index >= seg_start) & (regime_series.index < seg_end)
            seg_labels = regime_series[mask]

            if seg_labels.empty:
                continue

            # Dominant regime
            counts = seg_labels.value_counts()
            valid = counts.drop(RegimeType.UNCERTAIN, errors='ignore')
            if valid.empty:
                dominant = RegimeType.UNCERTAIN
                confidence = 0.0
            else:
                dominant = valid.idxmax()
                confidence = float(valid[dominant]) / len(seg_labels)

            duration_days = (seg_end - seg_start).days

            raw_segments.append({
                'start': seg_start.to_pydatetime() if hasattr(seg_start, 'to_pydatetime') else seg_start,
                'end': seg_end.to_pydatetime() if hasattr(seg_end, 'to_pydatetime') else seg_end,
                'regime': dominant,
                'confidence': confidence,
                'duration_days': duration_days,
                'bar_count': len(seg_labels),
            })

        # Merge very short segments into neighbors
        merged = self._merge_short_segments(raw_segments, merge_threshold_days)

        # Split excessively long segments
        final_raw = []
        for seg in merged:
            if seg['duration_days'] > max_segment_days:
                # Split at midpoint(s)
                n_splits = (seg['duration_days'] // max_segment_days) + 1
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

        # Build RegimeSegment objects
        segments = []
        for idx, seg in enumerate(final_raw):
            # Compute metadata from df slice
            mask = (df.index >= seg['start']) & (df.index < seg['end'])
            seg_df = df[mask]

            metadata = {}
            if not seg_df.empty and 'close' in seg_df.columns:
                rets = seg_df['close'].pct_change().dropna()
                metadata = {
                    'mean_return': float(rets.mean()) if len(rets) > 0 else 0.0,
                    'volatility': float(rets.std()) if len(rets) > 0 else 0.0,
                    'total_return': float(seg_df['close'].iloc[-1] / seg_df['close'].iloc[0] - 1) if len(seg_df) > 1 else 0.0,
                    'bar_count': len(seg_df),
                }

            segment_id = f"aseg_{idx:03d}_{seg['start'].strftime('%Y%m%d')}_{seg['end'].strftime('%Y%m%d')}"

            segments.append(RegimeSegment(
                segment_id=segment_id,
                start_date=seg['start'],
                end_date=seg['end'],
                regime=seg['regime'],
                confidence=seg['confidence'],
                metadata=metadata,
            ))

        logger.info(f"Adaptive segmentation: {len(segments)} segments from {len(raw_segments)} raw change points")
        regime_dist = {}
        for s in segments:
            regime_dist[s.regime.value] = regime_dist.get(s.regime.value, 0) + 1
        logger.info(f"Regime distribution (adaptive): {regime_dist}")
        avg_conf = sum(s.confidence for s in segments) / len(segments) if segments else 0
        logger.info(f"Average segment confidence: {avg_conf:.2%}")

        return segments

    @staticmethod
    def _merge_short_segments(
        segments: List[dict],
        min_duration_days: int,
    ) -> List[dict]:
        """
        Merge segments shorter than min_duration_days into adjacent same-regime
        neighbors, or into the longer neighbor if regimes differ.

        IMPORTANT: avg_trend_score, avg_volatility_score, and trend_score_std
        are recomputed as bar-count-weighted averages on every merge so that
        the final regime re-derivation (which uses avg_trend_score) reflects
        the actual data composition of the merged segment.
        """
        if not segments:
            return []

        def _weighted_merge_scores(target: dict, source: dict) -> None:
            """Recompute score fields as bar-count-weighted averages."""
            tb = target.get('bar_count', 1) or 1
            sb = source.get('bar_count', 1) or 1
            total = tb + sb
            for key in ('avg_trend_score', 'avg_volatility_score', 'trend_score_std'):
                tv = target.get(key, 0.0)
                sv = source.get(key, 0.0)
                target[key] = (tv * tb + sv * sb) / total
            target['bar_count'] = total

        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]

            # Merge condition: current segment is too short
            if seg['duration_days'] < min_duration_days:
                # Same regime → merge into previous
                if seg['regime'] == prev['regime']:
                    prev['end'] = seg['end']
                    prev['duration_days'] = (prev['end'] - prev['start']).days
                    _weighted_merge_scores(prev, seg)
                    prev['confidence'] = max(prev['confidence'], seg['confidence'])
                else:
                    # Different regime, short segment → absorb into previous
                    # (alternative: could absorb into next, but previous is simpler)
                    prev['end'] = seg['end']
                    prev['duration_days'] = (prev['end'] - prev['start']).days
                    _weighted_merge_scores(prev, seg)
                    # Reduce confidence since we're mixing regimes
                    prev['confidence'] *= 0.9
            else:
                merged.append(seg)

        # Second pass: merge previous segment into current if previous is too short
        if len(merged) > 1:
            final = [merged[0]]
            for seg in merged[1:]:
                prev = final[-1]
                if prev['duration_days'] < min_duration_days:
                    seg['start'] = prev['start']
                    seg['duration_days'] = (seg['end'] - seg['start']).days
                    _weighted_merge_scores(seg, prev)
                    seg['confidence'] *= 0.9
                    final[-1] = seg
                else:
                    final.append(seg)
            merged = final

        return merged

    def get_balanced_segments(
        self,
        segments: List[RegimeSegment],
        segments_per_regime: int = 3,
        target_regimes: Optional[List[RegimeType]] = None,
    ) -> List[RegimeSegment]:
        """
        Select a balanced set of segments with equal representation per regime.
        
        Args:
            segments: List of all available segments
            segments_per_regime: Number of segments to select per regime type
            target_regimes: Specific regimes to include (default: bullish, bearish, sideways)
        
        Returns:
            Balanced list of segments
        """
        if target_regimes is None:
            target_regimes = [RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS]
        
        # Group segments by regime
        regime_groups: Dict[RegimeType, List[RegimeSegment]] = {}
        for seg in segments:
            if seg.regime in target_regimes:
                if seg.regime not in regime_groups:
                    regime_groups[seg.regime] = []
                regime_groups[seg.regime].append(seg)
        
        # Sort each group by confidence (highest first)
        for regime in regime_groups:
            regime_groups[regime].sort(key=lambda x: x.confidence, reverse=True)
        
        # Select top N from each regime
        balanced = []
        missing_regimes = []
        for regime in target_regimes:
            regime_segs = regime_groups.get(regime, [])
            selected = regime_segs[:segments_per_regime]
            balanced.extend(selected)
            
            if len(selected) == 0:
                missing_regimes.append(regime.value)
            elif len(selected) < segments_per_regime:
                logger.warning(
                    f"Only {len(selected)} segments available for {regime.value} "
                    f"(requested {segments_per_regime})"
                )
        
        # If entire regimes are missing, warn prominently — this likely means
        # the detection method/params need adjustment
        if missing_regimes:
            logger.warning(
                f"⚠ REGIME DETECTION GAP: No segments found for {missing_regimes}. "
                f"Detection method '{self.method}' with current params may be too restrictive. "
                f"Consider switching to 'adx_di_hysteresis' or lowering adx thresholds."
            )
        
        logger.info(f"Selected {len(balanced)} balanced segments across {len(target_regimes)} regimes")
        
        return balanced
    
    def split_segments_by_role(
        self,
        segments: List[RegimeSegment],
        optimization_ratio: float = 0.60,
        model_selection_ratio: float = 0.20,
        holdout_ratio: float = 0.20,
        min_holdout_segments: int = 3,
    ) -> Dict[str, List[RegimeSegment]]:
        """
        Split segments into optimization, model-selection, and holdout sets.
        
        Maintains regime balance within each split. Newer segments are preferentially
        assigned to holdout to simulate forward-testing.

        Smart holdout: When a regime has fewer than `min_holdout_segments` segments,
        ALL segments go to optimization (holdout is waived for that regime).
        This prevents the scenario where scarce regimes get zero optimization data.
        
        Args:
            segments: List of segments to split
            optimization_ratio: Fraction for GA evolution (default 60%)
            model_selection_ratio: Fraction for elite re-ranking (default 20%)
            holdout_ratio: Fraction for final holdout (default 20%)
            min_holdout_segments: Minimum segments per regime to reserve holdout
                                 (below this, all segments go to optimization)
        
        Returns:
            Dict with keys 'optimization', 'model_selection', 'holdout'
        """
        if abs(optimization_ratio + model_selection_ratio + holdout_ratio - 1.0) > 0.01:
            raise ValueError("Ratios must sum to 1.0")
        
        # Sort segments by start date (oldest first)
        sorted_segments = sorted(segments, key=lambda x: x.start_date)
        
        # Group by regime
        regime_groups: Dict[RegimeType, List[RegimeSegment]] = {}
        for seg in sorted_segments:
            if seg.regime not in regime_groups:
                regime_groups[seg.regime] = []
            regime_groups[seg.regime].append(seg)
        
        result = {
            'optimization': [],
            'model_selection': [],
            'holdout': []
        }

        holdout_waived_regimes = []
        
        # For each regime, split maintaining temporal order (holdout = newest)
        for regime, segs in regime_groups.items():
            n = len(segs)
            if n == 0:
                continue

            # Smart holdout: skip holdout when segments are scarce
            if n < min_holdout_segments:
                holdout_waived_regimes.append(f"{regime.value}({n})")
                for seg in segs:
                    seg.role = 'optimization'
                    result['optimization'].append(seg)
                continue
            
            n_holdout = max(1, int(n * holdout_ratio))
            n_model_sel = max(0, int(n * model_selection_ratio))
            n_opt = n - n_holdout - n_model_sel

            # Safety: ensure at least 1 optimization segment
            if n_opt <= 0:
                n_opt = 1
                n_holdout = n - n_opt - n_model_sel
                if n_holdout < 0:
                    n_model_sel = 0
                    n_holdout = n - n_opt
            
            # Assign: oldest to optimization, middle to model_selection, newest to holdout
            for i, seg in enumerate(segs):
                if i < n_opt:
                    seg.role = 'optimization'
                    result['optimization'].append(seg)
                elif i < n_opt + n_model_sel:
                    seg.role = 'model_selection'
                    result['model_selection'].append(seg)
                else:
                    seg.role = 'holdout'
                    result['holdout'].append(seg)

        if holdout_waived_regimes:
            logger.warning(
                f"Smart holdout: waived holdout for regimes with <{min_holdout_segments} "
                f"segments: {', '.join(holdout_waived_regimes)}. "
                f"All segments assigned to optimization to maximize training data."
            )
        
        logger.info(
            f"Split segments: {len(result['optimization'])} optimization, "
            f"{len(result['model_selection'])} model_selection, "
            f"{len(result['holdout'])} holdout"
        )
        
        return result


def load_ohlcv_data(
    pair: str,
    timeframe: str,
    datadir: Path,
    timerange: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLCV data from FreqTrade data directory.
    
    Args:
        pair: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
        datadir: Path to FreqTrade data directory
        timerange: Optional timerange filter ('YYYYMMDD-YYYYMMDD')
    
    Returns:
        DataFrame with OHLCV data and datetime index
    """
    try:
        from freqtrade.data.history import load_pair_history
        from freqtrade.enums import CandleType
        from freqtrade.configuration import TimeRange
        
        tr = None
        if timerange:
            tr = TimeRange.parse_timerange(timerange)
        
        df = load_pair_history(
            pair=pair,
            timeframe=timeframe,
            datadir=datadir,
            candle_type=CandleType.SPOT,
            timerange=tr,
        )
        
        if df is not None and len(df) > 0:
            # Ensure DatetimeIndex — load_pair_history may return RangeIndex
            # with 'date' as a column; the rest of the pipeline (MTF fusion,
            # classify_segments) requires a DatetimeIndex.
            if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index('date')
                df.index = pd.to_datetime(df.index)
            logger.info(f"Loaded {len(df)} candles for {pair} {timeframe}")
            return df
        else:
            logger.warning(f"No data found for {pair} {timeframe}")
            return pd.DataFrame()
            
    except ImportError as e:
        logger.error(f"FreqTrade import error: {e}")
        logger.info("Falling back to manual file loading...")
        
        # Fallback: try to load JSON files directly
        pair_filename = pair.replace('/', '_')
        file_path = datadir / f"{pair_filename}-{timeframe}.json"
        
        if file_path.exists():
            df = pd.read_json(file_path)
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df['date'] = pd.to_datetime(df['date'], unit='ms')
            df.set_index('date', inplace=True)
            return df
        
        return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error loading data for {pair}: {e}")
        return pd.DataFrame()


def save_segments_to_yaml(
    segments: Dict[str, List[RegimeSegment]],
    filepath: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save segment configuration to YAML file for reproducibility.
    
    Args:
        segments: Dict with 'optimization', 'model_selection', 'holdout' lists
        filepath: Path to save YAML file
        metadata: Optional metadata to include (run_id, detector config, etc.)
    """
    import yaml
    from datetime import datetime
    
    output = {
        'created_at': datetime.now().isoformat(),
        'metadata': metadata or {},
        'segments': {}
    }
    
    for role, seg_list in segments.items():
        output['segments'][role] = [seg.to_dict() for seg in seg_list]
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved segment configuration to {filepath}")


def load_segments_from_yaml(filepath: Path) -> Dict[str, List[RegimeSegment]]:
    """
    Load segment configuration from YAML file.
    
    Args:
        filepath: Path to YAML file
    
    Returns:
        Dict with 'optimization', 'model_selection', 'holdout' segment lists
    """
    import yaml
    from datetime import datetime
    
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    result = {}
    
    for role, seg_list in data.get('segments', {}).items():
        segments = []
        for seg_dict in seg_list:
            # Parse timerange to dates
            timerange = seg_dict['timerange']
            start_str, end_str = timerange.split('-')
            start_date = datetime.strptime(start_str, '%Y%m%d')
            end_date = datetime.strptime(end_str, '%Y%m%d')
            
            segment = RegimeSegment(
                segment_id=seg_dict['segment_id'],
                start_date=start_date,
                end_date=end_date,
                regime=RegimeType(seg_dict['regime']),
                confidence=seg_dict.get('confidence', 0.0),
                metadata=seg_dict.get('metadata', {}),
                role=seg_dict.get('role', role),
            )
            segments.append(segment)
        
        result[role] = segments
    
    logger.info(f"Loaded segment configuration from {filepath}")
    
    return result

def save_labels_to_parquet(
    df: pd.DataFrame,
    labels: pd.Series,
    filepath: Path,
    method: str = 'unknown',
    confidence: Optional[pd.Series] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save regime labels to Parquet file for offline analysis.
    
    Creates a Parquet file with timestamps, regime labels, and metadata
    for reproducibility and efficient data sharing.
    
    Args:
        df: Original DataFrame with 'date' column
        labels: Series of RegimeType or string regime labels
        filepath: Path to save Parquet file
        method: Detection method name (e.g., 'adx_di_hysteresis')
        confidence: Optional per-bar confidence scores
        metadata: Optional dict of additional metadata
    
    Example:
        >>> detector = RegimeDetector(method='adx_di_hysteresis')
        >>> labels = detector.detect(df)
        >>> save_labels_to_parquet(df, labels, Path('labels.parquet'), 'adx_di_hysteresis')
    """
    # Prepare output DataFrame
    date_col = 'date' if 'date' in df.columns else df.index.name or 'index'
    dates = df[date_col] if date_col in df.columns else df.index
    
    output_df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'regime': labels.apply(lambda x: x.value if hasattr(x, 'value') else str(x)),
    })
    
    if confidence is not None:
        output_df['confidence'] = confidence.values
    
    # Add metadata as Parquet file metadata
    parquet_metadata = {
        'method': method,
        'created_at': datetime.now().isoformat(),
        'version': '1.0',
        'num_bars': str(len(output_df)),
    }
    if metadata:
        parquet_metadata.update({k: str(v) for k, v in metadata.items()})
    
    # Save with metadata
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    table = pa.Table.from_pandas(output_df)
    existing_meta = table.schema.metadata or {}
    merged_meta = {**existing_meta, **{k.encode(): v.encode() for k, v in parquet_metadata.items()}}
    table = table.replace_schema_metadata(merged_meta)
    
    pq.write_table(table, filepath)
    logger.info(f"Saved {len(output_df)} regime labels to {filepath}")


def load_labels_from_parquet(filepath: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load regime labels from Parquet file.
    
    Args:
        filepath: Path to Parquet file
    
    Returns:
        Tuple of (DataFrame with date and regime columns, metadata dict)
    
    Example:
        >>> df, meta = load_labels_from_parquet(Path('labels.parquet'))
        >>> print(meta['method'])
        'adx_di_hysteresis'
    """
    import pyarrow.parquet as pq
    
    table = pq.read_table(filepath)
    df = table.to_pandas()
    
    # Extract metadata
    metadata = {}
    if table.schema.metadata:
        metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items() if k != b'pandas'}
    
    logger.info(f"Loaded {len(df)} regime labels from {filepath}")
    return df, metadata