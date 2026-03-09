"""
ML Regime Detector — LightGBM-based market regime classifier.

Sits as a configurable meta-ensemble on top of the existing rule-based
RegimeDetector methods. Three feature modes:
  - 'combined': raw TA features + rule-based method outputs (default)
  - 'raw_only': only raw TA features (no rule-based dependency)
  - 'rules_only': only rule-based method outputs as features

Produces the same pd.Series[RegimeType] output as RegimeDetector.detect()
so it can be used as a drop-in replacement or ensemble voter.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from genetic_algorithm.utils.regime_detector import RegimeDetector, RegimeType

logger = logging.getLogger(__name__)

# Integer encoding for regime types (used in LightGBM labels)
REGIME_TO_INT = {
    RegimeType.BULLISH: 0,
    RegimeType.BEARISH: 1,
    RegimeType.SIDEWAYS: 2,
    RegimeType.VOLATILE: 3,
}
INT_TO_REGIME = {v: k for k, v in REGIME_TO_INT.items()}

# Feature names for raw TA features
RAW_FEATURE_NAMES = [
    'return_5', 'return_20', 'return_60',
    'volatility_5', 'volatility_20', 'volatility_60',
    'atr_ratio',
    'bb_width', 'bb_position',
    'adx', 'plus_di', 'minus_di', 'di_diff',
    'rsi', 'rsi_slope',
    'volume_ratio', 'volume_sma_ratio',
]

# Cross-timeframe feature names (appended when MTF data is available)
MTF_FEATURE_NAMES = [
    # Per-TF trend/vol scores from continuous regime detector
    'trend_score_{tf}', 'vol_score_{tf}',
    # Cross-TF divergences
    'trend_div_low_high',        # low-TF trend minus high-TF trend
    'trend_alignment',           # product of all TF trend signs (1=aligned, <0=diverging)
    'vol_spread',                # max(vol) - min(vol) across TFs
    # Transition features
    'transition_speed',
    'transition_accel',          # derivative of transition speed
]

# Rule-based methods used as feature providers
RULE_METHODS = ['adx_di_hysteresis', 'rolling_returns', 'bollinger', 'volatility_cluster']


class MLRegimeDetector:
    """
    LightGBM-based market regime classifier.

    Loads a pre-trained model and produces regime labels per bar, matching
    the RegimeDetector.detect() interface.

    Usage:
        detector = MLRegimeDetector(model_path='genetic_algorithm/ml/models/regime_lgbm.pkl')
        regimes = detector.detect(df)  # pd.Series of RegimeType
        regimes, confidence = detector.detect_with_confidence(df)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_mode: str = 'combined',
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model_path: Path to saved LightGBM model (.pkl). If None, uses
                        default path from config or 'genetic_algorithm/ml/models/regime_lgbm.pkl'.
            feature_mode: 'combined', 'raw_only', or 'rules_only'
            config: Optional config dict (regime_aware.ml_regime section)
        """
        self.config = config or {}
        self.feature_mode = feature_mode

        if model_path is None:
            model_path = self.config.get(
                'model_path', 'genetic_algorithm/ml/models/regime_lgbm.pkl'
            )
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names: List[str] = []

        self._load_model()

    def _load_model(self) -> None:
        """Load the pre-trained LightGBM model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ML regime model not found at {self.model_path}. "
                f"Train it first: python -m genetic_algorithm.ml.train_regime "
                f"--config genetic_algorithm/config/ga_config.yaml"
            )

        import joblib

        artifact = joblib.load(self.model_path)
        self.model = artifact['model']
        self.feature_names = artifact.get('feature_names', [])
        self.feature_mode = artifact.get('feature_mode', self.feature_mode)
        logger.info(
            f"Loaded ML regime model from {self.model_path} "
            f"(feature_mode={self.feature_mode}, {len(self.feature_names)} features)"
        )

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market regime for each bar using the trained LightGBM model.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume).
                Must be sorted by date ascending.

        Returns:
            pd.Series of RegimeType for each bar (same index as df).
            Bars without enough data for feature computation are labeled UNCERTAIN.
        """
        regimes, _ = self.detect_with_confidence(df)
        return regimes

    def detect_with_confidence(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect regime with per-bar confidence scores.

        Returns:
            Tuple of (regime_series, confidence_series).
            confidence_series contains the max class probability (0-1).
        """
        if df.empty:
            return pd.Series(dtype=object), pd.Series(dtype=float)

        features = self.compute_features(df)

        # Identify rows with NaN features (warmup period)
        valid_mask = features.notna().all(axis=1)

        regime_series = pd.Series(RegimeType.UNCERTAIN, index=df.index, dtype=object)
        confidence_series = pd.Series(0.0, index=df.index, dtype=float)

        if valid_mask.sum() == 0:
            return regime_series, confidence_series

        X_valid = features.loc[valid_mask]

        # Predict
        probas = self.model.predict_proba(X_valid)
        preds = np.argmax(probas, axis=1)
        confidences = np.max(probas, axis=1)

        # Map back to RegimeType
        for i, idx in enumerate(X_valid.index):
            regime_series[idx] = INT_TO_REGIME.get(preds[i], RegimeType.UNCERTAIN)
            confidence_series[idx] = confidences[i]

        return regime_series, confidence_series

    def compute_features(self, df: pd.DataFrame,
                         mtf_scores: Optional[Dict[str, pd.DataFrame]] = None,
                         ) -> pd.DataFrame:
        """
        Compute the feature matrix for the given OHLCV data.

        Args:
            df: OHLCV DataFrame (base timeframe).
            mtf_scores: Optional dict mapping timeframe strings to DataFrames
                        with 'trend_score' and 'volatility_score' columns,
                        already aligned/reindexed to df.index (ffilled).

        Returns a DataFrame with the same index as df and feature columns.
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        features = pd.DataFrame(index=df.index)

        if self.feature_mode in ('combined', 'raw_only'):
            raw = self._compute_raw_features(df)
            features = pd.concat([features, raw], axis=1)

        if self.feature_mode in ('combined', 'rules_only'):
            rules = self._compute_rule_features(df)
            features = pd.concat([features, rules], axis=1)

        # Add MTF cross-timeframe features when available
        if mtf_scores:
            mtf_feats = self._compute_mtf_features(df, mtf_scores)
            features = pd.concat([features, mtf_feats], axis=1)

        # Reorder to match training feature names if available
        if self.feature_names:
            # Add any missing columns as NaN (model may have been trained with
            # features not computable here, e.g. different rule methods)
            for col in self.feature_names:
                if col not in features.columns:
                    features[col] = np.nan
            features = features[self.feature_names]

        return features

    # ------------------------------------------------------------------
    # Raw TA features
    # ------------------------------------------------------------------

    def _compute_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute raw technical-analysis features from OHLCV data."""
        return self._compute_raw_features_static(df)

    @staticmethod
    def _compute_raw_features_static(df: pd.DataFrame) -> pd.DataFrame:
        """Compute raw TA features — static version usable without model."""
        close = df['close']
        high = df.get('high', close)
        low = df.get('low', close)
        volume = df.get('volume', pd.Series(0, index=df.index))

        feats: Dict[str, pd.Series] = {}

        # Rolling returns
        for w in [5, 20, 60]:
            feats[f'return_{w}'] = close.pct_change(w)

        # Rolling volatility (std of returns)
        returns = close.pct_change()
        for w in [5, 20, 60]:
            feats[f'volatility_{w}'] = returns.rolling(w, min_periods=w).std()

        # ATR ratio (ATR / close)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(14, min_periods=14).mean()
        feats['atr_ratio'] = atr_14 / close

        # Bollinger Band width and position
        sma_20 = close.rolling(20, min_periods=20).mean()
        std_20 = close.rolling(20, min_periods=20).std()
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        bb_width = bb_upper - bb_lower
        feats['bb_width'] = bb_width / sma_20  # normalized
        feats['bb_position'] = (close - bb_lower) / bb_width.replace(0, np.nan)

        # ADX + DI
        adx, plus_di, minus_di = MLRegimeDetector._calculate_adx(df, 14)
        feats['adx'] = adx
        feats['plus_di'] = plus_di
        feats['minus_di'] = minus_di
        feats['di_diff'] = plus_di - minus_di

        # RSI + RSI slope
        rsi = MLRegimeDetector._calculate_rsi(close, 14)
        feats['rsi'] = rsi
        feats['rsi_slope'] = rsi.diff(5)

        # Volume features
        vol_sma = volume.rolling(20, min_periods=20).mean()
        feats['volume_ratio'] = volume / vol_sma.replace(0, np.nan)
        short_vol = volume.rolling(5, min_periods=5).mean()
        feats['volume_sma_ratio'] = short_vol / vol_sma.replace(0, np.nan)

        return pd.DataFrame(feats, index=df.index)

    # ------------------------------------------------------------------
    # Multi-timeframe cross-TF features
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_mtf_features(
        df: pd.DataFrame,
        mtf_scores: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Compute cross-timeframe features from pre-computed per-TF trend and
        volatility scores.

        Args:
            df: Base-timeframe OHLCV data (used only for index).
            mtf_scores: Dict[tf_string -> DataFrame] where each DataFrame has
                        'trend_score' and 'volatility_score' columns aligned
                        to df.index via ffill.

        Returns:
            DataFrame of MTF features with df.index.
        """
        feats: Dict[str, pd.Series] = {}
        ordered_tfs = sorted(
            mtf_scores.keys(),
            key=lambda x: {'30m': 0, '1h': 1, '4h': 2, '1d': 3}.get(x, 4),
        )

        # Per-TF scores
        trend_series = {}
        vol_series = {}
        for tf in ordered_tfs:
            scores_df = mtf_scores[tf]
            t = scores_df['trend_score'].reindex(df.index).ffill()
            v = scores_df['volatility_score'].reindex(df.index).ffill()
            feats[f'trend_score_{tf}'] = t
            feats[f'vol_score_{tf}'] = v
            trend_series[tf] = t
            vol_series[tf] = v

        if len(ordered_tfs) >= 2:
            lowest_tf = ordered_tfs[0]
            highest_tf = ordered_tfs[-1]

            # Divergence: low-TF minus high-TF (positive = low TF more bullish)
            feats['trend_div_low_high'] = (
                trend_series[lowest_tf] - trend_series[highest_tf]
            )

            # Alignment: product of signs (+1 when all agree, < 0 when diverging)
            sign_product = pd.Series(1.0, index=df.index)
            for tf in ordered_tfs:
                sign_product *= np.sign(trend_series[tf]).replace(0, 1)
            feats['trend_alignment'] = sign_product

            # Volatility spread: max - min across TFs
            vol_stack = pd.concat(
                [vol_series[tf] for tf in ordered_tfs], axis=1
            )
            feats['vol_spread'] = vol_stack.max(axis=1) - vol_stack.min(axis=1)
        else:
            feats['trend_div_low_high'] = pd.Series(0.0, index=df.index)
            feats['trend_alignment'] = pd.Series(1.0, index=df.index)
            feats['vol_spread'] = pd.Series(0.0, index=df.index)

        # Transition features (computed from composite trend if available,
        # else from highest-TF trend)
        ref_trend = trend_series.get(
            ordered_tfs[-1], pd.Series(0.0, index=df.index)
        )
        fast_ema = ref_trend.ewm(span=5, min_periods=3).mean()
        slow_ema = ref_trend.ewm(span=20, min_periods=10).mean()
        transition_speed = fast_ema - slow_ema
        feats['transition_speed'] = transition_speed
        feats['transition_accel'] = transition_speed.diff()

        return pd.DataFrame(feats, index=df.index)

    # ------------------------------------------------------------------
    # Rule-based method outputs as features
    # ------------------------------------------------------------------

    def _compute_rule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run each rule-based method and encode outputs as integer features."""
        feats: Dict[str, pd.Series] = {}

        for method_name in RULE_METHODS:
            try:
                detector = RegimeDetector(method=method_name)
                result = detector.detect(df)
                # Encode as integer
                feats[f'rule_{method_name}'] = result.map(
                    lambda r: REGIME_TO_INT.get(r, -1)
                )
            except Exception as e:
                logger.warning(f"Rule method '{method_name}' failed, skipping: {e}")
                feats[f'rule_{method_name}'] = pd.Series(-1, index=df.index)

        return pd.DataFrame(feats, index=df.index)

    # ------------------------------------------------------------------
    # TA helper calculations (self-contained, no external dependencies)
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_adx(
        df: pd.DataFrame, period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX, +DI, -DI."""
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        pdm_smooth = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
        mdm_smooth = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

        plus_di = 100 * pdm_smooth / atr
        minus_di = 100 * mdm_smooth / atr

        di_diff = (plus_di - minus_di).abs()
        di_sum = plus_di + minus_di
        dx = 100 * di_diff / di_sum.replace(0, np.nan)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        return adx, plus_di, minus_di

    @staticmethod
    def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
