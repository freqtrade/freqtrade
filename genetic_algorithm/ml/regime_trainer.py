"""
Regime Trainer — training pipeline for the ML regime classifier.

Supports two labeling modes:
  - 'rules': distill the existing rule-based ensemble detector (supervised)
  - 'price': self-label from raw price data using return/volatility thresholds

Walk-forward cross-validation ensures no future leakage.  Saves the trained
LightGBM model + feature importance + validation metrics.

Usage (CLI):
    python -m genetic_algorithm.ml.train_regime \\
        --config genetic_algorithm/config/ga_config.yaml \\
        --pairs BTC/USDT ETH/USDT \\
        --timeframe 4h \\
        --timerange 20230101-20260101
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from genetic_algorithm.ml.regime_detector import (
    INT_TO_REGIME,
    REGIME_TO_INT,
    MLRegimeDetector,
    RegimeType,
)
from genetic_algorithm.utils.regime_detector import RegimeDetector, load_ohlcv_data

logger = logging.getLogger(__name__)


class RegimeTrainer:
    """
    Training pipeline for the LightGBM regime classifier.

    Example:
        trainer = RegimeTrainer(config)
        report = trainer.train(
            pairs=['BTC/USDT'],
            timeframe='4h',
            timerange='20230101-20260101',
        )
        print(report['overall_accuracy'])
    """

    # Default self-labeling thresholds for 'price' mode
    DEFAULT_LABEL_PARAMS = {
        'return_window': 20,
        'bullish_threshold': 0.02,
        'bearish_threshold': -0.02,
        'vol_window': 20,
        'vol_lookback': 60,
        'volatile_pct': 75,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Full GA config dict.  The ml_regime sub-section is read
                    from config['regime_aware']['ml_regime'].
        """
        self.config = config or {}
        regime_config = self.config.get('regime_aware', {})
        self.ml_config = regime_config.get('ml_regime', {})

        self.label_mode: str = self.ml_config.get('label_mode', 'rules')
        self.feature_mode: str = self.ml_config.get('feature_mode', 'combined')
        self.model_path: Path = Path(
            self.ml_config.get('model_path', 'genetic_algorithm/ml/models/regime_lgbm.pkl')
        )
        self.label_params = {
            **self.DEFAULT_LABEL_PARAMS,
            **self.ml_config.get('label_params', {}),
        }
        self.n_cv_folds: int = self.ml_config.get('cv_folds', 5)

        # LightGBM hyperparameters
        self.lgbm_params: Dict[str, Any] = {
            'objective': 'multiclass',
            'num_class': 4,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'class_weight': 'balanced',
            'verbose': -1,
            'random_state': 42,
            **self.ml_config.get('lgbm_params', {}),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        pairs: Optional[List[str]] = None,
        timeframe: str = '4h',
        timerange: Optional[str] = None,
        data_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the regime classifier and save the model.

        Args:
            pairs: Trading pairs to use for training data.
            timeframe: Candle timeframe for regime detection.
            timerange: Optional timerange filter (YYYYMMDD-YYYYMMDD).
            data_path: Optional path to FreqTrade data directory.

        Returns:
            Training report dict with accuracy, F1, confusion matrix, etc.
        """
        import lightgbm as lgb
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
        )

        # 1. Load data
        if pairs is None:
            pairs = self.config.get('backtesting', {}).get('pairs', ['BTC/USDT'])

        datadir = Path(
            data_path
            or self.config.get('backtesting', {}).get('datadir', 'user_data/data/binance')
        )

        all_features = []
        all_labels = []

        for pair in pairs:
            logger.info(f"Loading data for {pair} {timeframe}...")
            df = load_ohlcv_data(
                pair=pair,
                timeframe=timeframe,
                datadir=datadir,
                timerange=timerange,
            )
            if df.empty:
                logger.warning(f"No data for {pair}, skipping")
                continue

            # Compute features
            feature_computer = _FeatureComputer(self.feature_mode)
            features = feature_computer.compute(df)

            # Generate labels
            labels = self._generate_labels(df)

            # Align and drop NaN
            combined = pd.concat([features, labels.rename('label')], axis=1).dropna()
            if combined.empty:
                logger.warning(f"No valid samples for {pair} after NaN drop")
                continue

            all_features.append(combined.drop(columns=['label']))
            all_labels.append(combined['label'].astype(int))
            logger.info(f"  {pair}: {len(combined)} valid samples")

        if not all_features:
            raise ValueError("No training data available — check data paths and pairs")

        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)

        feature_names = list(X.columns)
        logger.info(
            f"Training data: {len(X)} samples, {len(feature_names)} features, "
            f"label distribution: {dict(y.value_counts().sort_index())}"
        )

        # 2. Walk-forward cross-validation
        fold_reports = []
        fold_size = len(X) // self.n_cv_folds

        for fold in range(self.n_cv_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.n_cv_folds - 1 else len(X)

            # Chronological split: use everything before val_start as train,
            # val_start:val_end as validation.  For fold 0 we use folds 1-end
            # as training (future data) — but walk-forward means fold 0 has
            # no training history, so we skip it when fold == 0 and use folds
            # 1+ as validation with expanding training window.
            if fold == 0:
                # Not enough history for first fold — use it as initial training data
                continue

            train_end = val_start
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_val = X.iloc[val_start:val_end]
            y_val = y.iloc[val_start:val_end]

            if len(X_train) < 50 or len(X_val) < 10:
                logger.warning(f"Fold {fold}: too few samples (train={len(X_train)}, val={len(X_val)}), skipping")
                continue

            model = lgb.LGBMClassifier(**self.lgbm_params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
            cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3])

            fold_reports.append({
                'fold': fold,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'accuracy': float(acc),
                'f1_macro': float(f1_macro),
                'confusion_matrix': cm.tolist(),
            })
            logger.info(f"  Fold {fold}: acc={acc:.3f}, F1={f1_macro:.3f}")

        # 3. Train final model on all data
        final_model = lgb.LGBMClassifier(**self.lgbm_params)
        final_model.fit(X, y)

        # Feature importance
        importance = dict(zip(feature_names, final_model.feature_importances_.tolist()))
        importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        # 4. Save model artifact
        import joblib

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            'model': final_model,
            'feature_names': feature_names,
            'feature_mode': self.feature_mode,
            'label_mode': self.label_mode,
            'label_params': self.label_params,
            'lgbm_params': self.lgbm_params,
            'trained_at': datetime.now().isoformat(),
            'training_samples': len(X),
            'pairs': pairs,
            'timeframe': timeframe,
            'timerange': timerange,
        }
        joblib.dump(artifact, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

        # 5. Save training report
        report = {
            'model_path': str(self.model_path),
            'feature_mode': self.feature_mode,
            'label_mode': self.label_mode,
            'training_samples': len(X),
            'feature_count': len(feature_names),
            'label_distribution': {
                INT_TO_REGIME[k].value: int(v) for k, v in y.value_counts().sort_index().items()
            },
            'cv_folds': fold_reports,
            'overall_accuracy': float(np.mean([f['accuracy'] for f in fold_reports])) if fold_reports else 0.0,
            'overall_f1_macro': float(np.mean([f['f1_macro'] for f in fold_reports])) if fold_reports else 0.0,
            'feature_importance': importance_sorted,
            'trained_at': datetime.now().isoformat(),
        }

        report_path = self.model_path.parent / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Training report saved to {report_path}")

        # Log top features
        top_n = min(10, len(importance_sorted))
        top_features = list(importance_sorted.items())[:top_n]
        logger.info(f"Top {top_n} features: {top_features}")

        return report

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    def _generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate regime labels for the given OHLCV data.

        Returns:
            pd.Series of integer labels (REGIME_TO_INT encoding).
        """
        if self.label_mode == 'rules':
            return self._label_from_rules(df)
        elif self.label_mode == 'price':
            return self._label_from_price(df)
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}. Use 'rules' or 'price'.")

    def _label_from_rules(self, df: pd.DataFrame) -> pd.Series:
        """Generate labels by running the existing ensemble rule-based detector."""
        detector = RegimeDetector(method='ensemble')
        regime_series = detector.detect(df)
        # Convert RegimeType → integer labels
        labels = regime_series.map(lambda r: REGIME_TO_INT.get(r, -1))
        # Drop UNCERTAIN (-1) — these will become NaN and be dropped during alignment
        labels = labels.replace(-1, np.nan)
        return labels

    def _label_from_price(self, df: pd.DataFrame) -> pd.Series:
        """
        Self-label from raw price data using return/volatility thresholds.

        Rules:
        - Rolling return > bullish_threshold → BULLISH
        - Rolling return < bearish_threshold → BEARISH
        - ATR spike above volatile_pct percentile AND no clear trend → VOLATILE
        - Otherwise → SIDEWAYS
        """
        df = df.copy()
        df.columns = df.columns.str.lower()
        close = df['close']

        p = self.label_params
        window = p['return_window']
        bull_th = p['bullish_threshold']
        bear_th = p['bearish_threshold']

        # Rolling return
        rolling_ret = close.pct_change(window)

        # Volatility (ATR ratio)
        high = df.get('high', close)
        low = df.get('low', close)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(p['vol_window'], min_periods=p['vol_window']).mean()
        atr_ratio = atr / close

        # Volatility percentile threshold
        vol_threshold = atr_ratio.rolling(
            p['vol_lookback'], min_periods=p['vol_lookback']
        ).quantile(p['volatile_pct'] / 100.0)

        labels = pd.Series(np.nan, index=df.index)

        # VOLATILE: high volatility AND no clear directional trend
        is_volatile = (atr_ratio > vol_threshold) & (rolling_ret.abs() < bull_th)
        labels[is_volatile] = REGIME_TO_INT[RegimeType.VOLATILE]

        # BULLISH: strong positive return
        is_bull = (rolling_ret > bull_th) & ~is_volatile
        labels[is_bull] = REGIME_TO_INT[RegimeType.BULLISH]

        # BEARISH: strong negative return
        is_bear = (rolling_ret < bear_th) & ~is_volatile
        labels[is_bear] = REGIME_TO_INT[RegimeType.BEARISH]

        # SIDEWAYS: everything else (that has valid data)
        valid = rolling_ret.notna() & atr_ratio.notna()
        is_sideways = valid & labels.isna()
        labels[is_sideways] = REGIME_TO_INT[RegimeType.SIDEWAYS]

        return labels


class _FeatureComputer:
    """
    Stateless feature computation helper.

    Wraps MLRegimeDetector's feature computation without requiring a
    trained model (used during training when no model exists yet).
    """

    def __init__(self, feature_mode: str = 'combined'):
        self.feature_mode = feature_mode

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features (same logic as MLRegimeDetector but without model load)."""
        df = df.copy()
        df.columns = df.columns.str.lower()

        features = pd.DataFrame(index=df.index)

        if self.feature_mode in ('combined', 'raw_only'):
            raw = MLRegimeDetector._compute_raw_features_static(df)
            features = pd.concat([features, raw], axis=1)

        if self.feature_mode in ('combined', 'rules_only'):
            from genetic_algorithm.ml.regime_detector import RULE_METHODS

            feats: Dict[str, pd.Series] = {}
            for method_name in RULE_METHODS:
                try:
                    detector = RegimeDetector(method=method_name)
                    result = detector.detect(df)
                    feats[f'rule_{method_name}'] = result.map(
                        lambda r: REGIME_TO_INT.get(r, -1)
                    )
                except Exception as e:
                    logger.warning(f"Rule method '{method_name}' failed: {e}")
                    feats[f'rule_{method_name}'] = pd.Series(-1, index=df.index)
            features = pd.concat([features, pd.DataFrame(feats, index=df.index)], axis=1)

        return features
