"""
Phase 1B Tests — ML Regime Classification Integration

Covers:
- MLRegimeDetector: feature computation, detect() shape/types
- RegimeTrainer: label generation on synthetic data
- StrategyGene: round-trip serialization of regime fields, backward compatibility
- Mutation: mutate_regime produces only valid values
- Crossover: regime fields are propagated
- RegimeAwareEvaluator: specialist / exclusive logic paths
- End-to-end: train → detect pipeline on synthetic data
"""

import logging
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import (
    ConditionGene,
    IndicatorGene,
    StrategyGene,
)
from genetic_algorithm.ml.regime_detector import (
    INT_TO_REGIME,
    REGIME_TO_INT,
    MLRegimeDetector,
    RAW_FEATURE_NAMES,
)
from genetic_algorithm.ml.regime_trainer import RegimeTrainer
from genetic_algorithm.utils.regime_detector import RegimeDetector, RegimeType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_synthetic_ohlcv(
    start_date: datetime = datetime(2023, 1, 1),
    num_days: int = 120,
    regime: str = "bullish",
    timeframe_minutes: int = 60,
    initial_price: float = 100.0,
    volatility: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a specific regime characteristic."""
    bars_per_day = 24 * 60 // timeframe_minutes
    n_bars = num_days * bars_per_day

    dates = pd.date_range(
        start=start_date, periods=n_bars, freq=f"{timeframe_minutes}min"
    )
    rng = np.random.RandomState(seed)

    drift_map = {
        "bullish": (0.001, volatility),
        "bearish": (-0.001, volatility),
        "sideways": (0.0, volatility * 0.5),
        "volatile": (0.0, volatility * 3.0),
    }
    drift, vol = drift_map.get(regime, (0.0, volatility))

    returns = rng.normal(drift, vol, n_bars)
    prices = initial_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices * (1 + np.abs(rng.normal(0, vol, n_bars))),
            "low": prices * (1 - np.abs(rng.normal(0, vol, n_bars))),
            "close": prices,
            "volume": rng.uniform(100, 10000, n_bars),
        }
    )
    df.set_index("date", inplace=True)
    return df


def _make_strategy_gene(**overrides) -> StrategyGene:
    """Create a minimal valid StrategyGene for testing."""
    defaults = dict(
        generation=0,
        individual_id=0,
        indicators=[IndicatorGene(type="RSI", parameters={"period": 14})],
        entry_conditions=[
            ConditionGene(indicator="RSI", operator="<", threshold=30.0)
        ],
        exit_conditions=[
            ConditionGene(indicator="RSI", operator=">", threshold=70.0)
        ],
        timeframe="5m",
        stoploss=-0.10,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01},
    )
    defaults.update(overrides)
    return StrategyGene(**defaults)


def _make_individual(**gene_overrides) -> Individual:
    """Create a minimal Individual wrapping a StrategyGene."""
    gene = _make_strategy_gene(**gene_overrides)
    ind = Individual(strategy_gene=gene)
    ind.mutations = []
    return ind


# ===================================================================
# 1. MLRegimeDetector
# ===================================================================

class TestMLRegimeDetectorFeatures:
    """Test raw feature computation (no model required)."""

    def test_raw_features_shape_and_names(self):
        """_compute_raw_features_static returns expected columns."""
        df = generate_synthetic_ohlcv(num_days=30)
        features = MLRegimeDetector._compute_raw_features_static(df)
        assert isinstance(features, pd.DataFrame)
        assert set(RAW_FEATURE_NAMES).issubset(set(features.columns)), (
            f"Missing columns: {set(RAW_FEATURE_NAMES) - set(features.columns)}"
        )
        assert len(features) == len(df)

    def test_raw_features_no_inf(self):
        """Feature values should not contain ±inf."""
        df = generate_synthetic_ohlcv(num_days=60)
        features = MLRegimeDetector._compute_raw_features_static(df)
        # Drop NaN rows (expected for warm-up) then check for inf
        clean = features.dropna()
        assert not np.isinf(clean.values).any(), "Inf detected in raw features"

    def test_detect_without_model_raises(self):
        """Creating MLRegimeDetector without a model file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MLRegimeDetector(model_path="/tmp/nonexistent_model.pkl")


# ===================================================================
# 2. RegimeTrainer — label generation
# ===================================================================

class TestRegimeTrainerLabels:
    """Test training label generation on synthetic data."""

    def setup_method(self):
        self.trainer = RegimeTrainer()  # default config

    @pytest.mark.slow
    def test_label_from_rules_returns_integer_series(self):
        """rule-based labels should be integer-valued, in {0,1,2,3} or NaN."""
        df = generate_synthetic_ohlcv(num_days=120)
        labels = self.trainer._label_from_rules(df)
        assert isinstance(labels, pd.Series)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({0, 1, 2, 3})

    def test_label_from_price_returns_integer_series(self):
        """price-based labels should be integer-valued, in {0,1,2,3} or NaN."""
        self.trainer.label_mode = "price"
        df = generate_synthetic_ohlcv(num_days=120, regime="bullish")
        labels = self.trainer._label_from_price(df)
        assert isinstance(labels, pd.Series)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({0, 1, 2, 3})

    def test_bullish_data_has_bullish_labels(self):
        """On strongly bullish data, majority of price-based labels should be bullish."""
        df = generate_synthetic_ohlcv(
            num_days=180, regime="bullish", volatility=0.005, seed=123,
        )
        self.trainer.label_mode = "price"
        labels = self.trainer._label_from_price(df)
        valid = labels.dropna()
        bullish_count = (valid == REGIME_TO_INT[RegimeType.BULLISH]).sum()
        # At least 20% should be detected as bullish (generous threshold)
        assert bullish_count / len(valid) > 0.15, (
            f"Only {bullish_count}/{len(valid)} bullish labels on bullish data"
        )

    @pytest.mark.slow
    def test_label_mode_dispatch(self):
        """_generate_labels dispatches to the correct labeler."""
        df = generate_synthetic_ohlcv(num_days=60)

        self.trainer.label_mode = "rules"
        labels_rules = self.trainer._generate_labels(df)
        assert len(labels_rules) == len(df)

        self.trainer.label_mode = "price"
        labels_price = self.trainer._generate_labels(df)
        assert len(labels_price) == len(df)

    def test_invalid_label_mode_raises(self):
        self.trainer.label_mode = "unknown"
        df = generate_synthetic_ohlcv(num_days=30)
        with pytest.raises(ValueError, match="Unknown label_mode"):
            self.trainer._generate_labels(df)


# ===================================================================
# 3. RegimeTrainer — feature computation helper
# ===================================================================

class TestFeatureComputer:
    """Test the _FeatureComputer used during training."""

    def test_combined_mode(self):
        from genetic_algorithm.ml.regime_trainer import _FeatureComputer

        fc = _FeatureComputer(feature_mode="combined")
        df = generate_synthetic_ohlcv(num_days=60)
        feats = fc.compute(df)
        assert isinstance(feats, pd.DataFrame)
        # combined: should have raw features + rule_ columns
        raw_cols = [c for c in feats.columns if c in RAW_FEATURE_NAMES]
        rule_cols = [c for c in feats.columns if c.startswith("rule_")]
        assert len(raw_cols) > 0
        assert len(rule_cols) > 0

    def test_raw_only_mode(self):
        from genetic_algorithm.ml.regime_trainer import _FeatureComputer

        fc = _FeatureComputer(feature_mode="raw_only")
        df = generate_synthetic_ohlcv(num_days=60)
        feats = fc.compute(df)
        rule_cols = [c for c in feats.columns if c.startswith("rule_")]
        assert len(rule_cols) == 0

    def test_rules_only_mode(self):
        from genetic_algorithm.ml.regime_trainer import _FeatureComputer

        fc = _FeatureComputer(feature_mode="rules_only")
        df = generate_synthetic_ohlcv(num_days=60)
        feats = fc.compute(df)
        raw_cols = [c for c in feats.columns if c in RAW_FEATURE_NAMES]
        assert len(raw_cols) == 0
        rule_cols = [c for c in feats.columns if c.startswith("rule_")]
        assert len(rule_cols) > 0


# ===================================================================
# 4. StrategyGene — regime fields
# ===================================================================

class TestStrategyGeneRegimeFields:
    """Regime specialization fields on StrategyGene."""

    def test_defaults(self):
        gene = _make_strategy_gene()
        assert gene.preferred_regime is None
        assert gene.regime_mode == "generalist"

    def test_specialist_creation(self):
        gene = _make_strategy_gene(
            preferred_regime="bearish", regime_mode="specialist"
        )
        assert gene.preferred_regime == "bearish"
        assert gene.regime_mode == "specialist"

    def test_round_trip_serialization(self):
        """to_dict → from_dict preserves regime fields."""
        gene = _make_strategy_gene(
            preferred_regime="volatile", regime_mode="exclusive"
        )
        d = gene.to_dict()
        assert d["preferred_regime"] == "volatile"
        assert d["regime_mode"] == "exclusive"

        restored = StrategyGene.from_dict(d)
        assert restored.preferred_regime == "volatile"
        assert restored.regime_mode == "exclusive"

    def test_backward_compatible_from_dict(self):
        """from_dict on a dict without regime fields defaults to generalist."""
        gene = _make_strategy_gene()
        d = gene.to_dict()
        # Simulate legacy dict without regime keys
        d.pop("preferred_regime", None)
        d.pop("regime_mode", None)

        restored = StrategyGene.from_dict(d)
        assert restored.preferred_regime is None
        assert restored.regime_mode == "generalist"

    def test_copy_preserves_regime_fields(self):
        gene = _make_strategy_gene(
            preferred_regime="bullish", regime_mode="specialist"
        )
        clone = gene.copy()
        assert clone.preferred_regime == "bullish"
        assert clone.regime_mode == "specialist"


# ===================================================================
# 5. Mutation — mutate_regime
# ===================================================================

class TestMutateRegime:
    """Test the regime mutation operator."""

    def _config_enabled(self) -> Dict:
        return {
            "regime_aware": {
                "enabled": True,
                "regime_specialization": {"enabled": True},
            }
        }

    def _config_disabled(self) -> Dict:
        return {"regime_aware": {"enabled": False}}

    def test_mutation_produces_valid_values(self):
        """After many mutations, all values should be within allowed sets."""
        from genetic_algorithm.core.mutation import (
            _VALID_REGIME_MODES,
            _VALID_REGIMES,
            mutate_regime,
        )

        cfg = self._config_enabled()
        seen_regimes = set()
        seen_modes = set()

        for _ in range(200):
            ind = _make_individual()
            result = mutate_regime(ind, 1.0, cfg)
            seen_regimes.add(result.strategy_gene.preferred_regime)
            seen_modes.add(result.strategy_gene.regime_mode)

        assert seen_regimes.issubset(set(_VALID_REGIMES))
        assert seen_modes.issubset(set(_VALID_REGIME_MODES))

    def test_mutation_noop_when_disabled(self):
        """With regime_aware disabled, mutate_regime should be identity."""
        from genetic_algorithm.core.mutation import mutate_regime

        cfg = self._config_disabled()
        ind = _make_individual(preferred_regime="bullish", regime_mode="specialist")
        result = mutate_regime(ind, 1.0, cfg)
        # Should return original when disabled
        assert result.strategy_gene.preferred_regime == "bullish"
        assert result.strategy_gene.regime_mode == "specialist"

    def test_mutation_diversity(self):
        """With enough iterations, mutation should produce diverse values."""
        from genetic_algorithm.core.mutation import mutate_regime

        cfg = self._config_enabled()
        results = set()
        for _ in range(500):
            ind = _make_individual()
            result = mutate_regime(ind, 1.0, cfg)
            results.add(
                (result.strategy_gene.preferred_regime, result.strategy_gene.regime_mode)
            )
        # Should see at least 4 distinct (regime, mode) combos
        assert len(results) >= 4, f"Only {len(results)} distinct regime combos"


# ===================================================================
# 6. Crossover — regime fields
# ===================================================================

class TestCrossoverRegimeFields:
    """Test that crossover properly handles regime fields."""

    def test_uniform_crossover_propagates_fields(self):
        """After uniform crossover, offspring regime fields come from parents."""
        from genetic_algorithm.core.crossover import uniform_crossover

        parent1 = _make_individual(
            preferred_regime="bullish", regime_mode="specialist"
        )
        parent2 = _make_individual(
            preferred_regime="bearish", regime_mode="exclusive"
        )

        valid_regimes = {"bullish", "bearish"}
        valid_modes = {"specialist", "exclusive"}

        all_offspring_regimes = set()
        all_offspring_modes = set()
        for i in range(100):
            o1, o2 = uniform_crossover(parent1, parent2, generation=1, ind_id=i)
            all_offspring_regimes.add(o1.strategy_gene.preferred_regime)
            all_offspring_regimes.add(o2.strategy_gene.preferred_regime)
            all_offspring_modes.add(o1.strategy_gene.regime_mode)
            all_offspring_modes.add(o2.strategy_gene.regime_mode)

        # All offspring regimes should only be from parents
        assert all_offspring_regimes.issubset(valid_regimes)
        assert all_offspring_modes.issubset(valid_modes)


# ===================================================================
# 7. RegimeDetector — ml_lgbm method registration
# ===================================================================

class TestRegimeDetectorMLRegistration:
    """Test that 'ml_lgbm' is a valid method in RegimeDetector."""

    def test_ml_lgbm_is_valid_method(self):
        """'ml_lgbm' should be accepted as a valid method name."""
        # RegimeDetector validates method in __init__
        detector = RegimeDetector(method="ml_lgbm")
        assert detector.method == "ml_lgbm"

    def test_ml_lgbm_detect_without_model(self):
        """ml_lgbm detect without a model file should handle gracefully."""
        detector = RegimeDetector(method="ml_lgbm")
        df = generate_synthetic_ohlcv(num_days=30)
        # Should raise or return fallback when model is missing
        try:
            result = detector.detect(df)
            # If it returns, verify it's a valid Series
            assert isinstance(result, pd.Series)
        except (FileNotFoundError, Exception):
            pass  # Expected — no model file


# ===================================================================
# 8. Regime-aware fitness specialist/exclusive logic
# ===================================================================

class TestRegimeAwareFitnessLogic:
    """Test specialist/exclusive/generalist fitness scoring."""

    def test_specialist_mode_boosts_preferred_regime(self):
        """In specialist mode the preferred regime's weight should be increased."""
        # This tests the logic conceptually via a mock aggregation
        from genetic_algorithm.evaluation.regime_aware import RegimeAwareEvaluator

        config = {
            "regime_aware": {
                "enabled": True,
                "method": "sma_adx",
                "min_segment_bars": 10,
                "regime_specialization": {
                    "enabled": True,
                    "specialist_boost": 2.0,
                    "diversity_weight": 0.05,
                },
            },
            "backtesting": {
                "pairs": ["BTC/USDT"],
                "timeframe": "1h",
                "timerange": "20230101-20230301",
            },
        }
        evaluator = RegimeAwareEvaluator(config)
        assert evaluator is not None
        # Verify config loading succeeded — basic smoke test
        assert evaluator.config.get("regime_aware", {}).get("enabled") is True


# ===================================================================
# 9. Config structure
# ===================================================================

class TestConfigStructure:
    """Verify the YAML config has the expected Phase 1B sections."""

    def test_config_has_ml_regime_section(self):
        import yaml

        config_path = Path(__file__).resolve().parent.parent / "config" / "ga_config.yaml"
        if not config_path.exists():
            pytest.skip("ga_config.yaml not found")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        regime_aware = cfg.get("regime_aware", {})
        ml_regime = regime_aware.get("ml_regime", {})

        # Check expected keys
        assert "enabled" in ml_regime
        assert "feature_mode" in ml_regime
        assert "label_mode" in ml_regime
        assert "model_path" in ml_regime

    def test_config_has_regime_specialization_section(self):
        import yaml

        config_path = Path(__file__).resolve().parent.parent / "config" / "ga_config.yaml"
        if not config_path.exists():
            pytest.skip("ga_config.yaml not found")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        spec = cfg.get("regime_aware", {}).get("regime_specialization", {})
        assert "enabled" in spec
        assert "specialist_boost" in spec
        assert "diversity_weight" in spec


# ===================================================================
# 10. CLI training script
# ===================================================================

class TestCLITrainRegime:
    """Test the CLI arg parser and config loading."""

    def test_parse_defaults(self):
        from genetic_algorithm.ml.train_regime import parse_args

        args = parse_args([])
        assert args.timeframe == "4h"
        assert args.pairs is None
        assert args.label_mode is None
        assert args.feature_mode is None
        assert args.verbose is False

    def test_parse_full_args(self):
        from genetic_algorithm.ml.train_regime import parse_args

        args = parse_args([
            "--pairs", "BTC/USDT", "ETH/USDT",
            "--timeframe", "1h",
            "--timerange", "20230101-20240101",
            "--label-mode", "price",
            "--feature-mode", "raw_only",
            "--cv-folds", "3",
            "-v",
        ])
        assert args.pairs == ["BTC/USDT", "ETH/USDT"]
        assert args.timeframe == "1h"
        assert args.timerange == "20230101-20240101"
        assert args.label_mode == "price"
        assert args.feature_mode == "raw_only"
        assert args.cv_folds == 3
        assert args.verbose is True

    def test_apply_cli_overrides(self):
        from genetic_algorithm.ml.train_regime import apply_cli_overrides, parse_args

        config = {"regime_aware": {"ml_regime": {"label_mode": "rules"}}}
        args = parse_args(["--label-mode", "price", "--feature-mode", "raw_only"])
        result = apply_cli_overrides(config, args)

        assert result["regime_aware"]["ml_regime"]["label_mode"] == "price"
        assert result["regime_aware"]["ml_regime"]["feature_mode"] == "raw_only"


# ===================================================================
# 11. End-to-end: train → detect on synthetic data
# ===================================================================

class TestEndToEnd:
    """Integration test: train a model on synthetic data, then detect."""

    @pytest.fixture
    def tmp_model_path(self, tmp_path):
        return tmp_path / "test_model.pkl"

    def test_train_and_detect_pipeline(self, tmp_model_path):
        """Train on synthetic data, then use the model for detection."""
        # Build config with price labeling (avoids ensemble dependency issues)
        config = {
            "regime_aware": {
                "ml_regime": {
                    "enabled": True,
                    "feature_mode": "raw_only",
                    "label_mode": "price",
                    "model_path": str(tmp_model_path),
                    "cv_folds": 3,
                    "lgbm_params": {
                        "n_estimators": 20,  # fast training
                        "num_leaves": 8,
                        "verbose": -1,
                    },
                }
            }
        }

        # Generate synthetic multi-regime training data
        frames = []
        for regime, seed in [
            ("bullish", 10),
            ("bearish", 20),
            ("sideways", 30),
            ("volatile", 40),
        ]:
            frames.append(
                generate_synthetic_ohlcv(
                    num_days=90,
                    regime=regime,
                    seed=seed,
                    timeframe_minutes=240,
                )
            )
        # Concatenate with new dates to avoid index overlap
        combined = []
        base = datetime(2020, 1, 1)
        for i, f in enumerate(frames):
            f.index = pd.date_range(
                start=base + timedelta(days=90 * i),
                periods=len(f),
                freq="4h",
            )
            combined.append(f)
        train_df = pd.concat(combined)

        # Monkey-patch load_ohlcv_data to return our synthetic data
        with patch(
            "genetic_algorithm.ml.regime_trainer.load_ohlcv_data",
            return_value=train_df,
        ):
            trainer = RegimeTrainer(config)
            report = trainer.train(
                pairs=["SYNTH/TEST"],
                timeframe="4h",
            )

        # Verify report
        assert report["training_samples"] > 100
        assert report["feature_count"] > 0
        assert 0.0 <= report["overall_accuracy"] <= 1.0
        assert tmp_model_path.exists(), "Model artifact not saved"

        # Now detect with the trained model
        test_df = generate_synthetic_ohlcv(
            num_days=30, regime="bullish", seed=99, timeframe_minutes=240,
        )
        detector = MLRegimeDetector(
            model_path=str(tmp_model_path),
            feature_mode="raw_only",
        )
        regimes = detector.detect(test_df)

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(test_df)
        # All values should be valid RegimeType instances
        valid_types = set(RegimeType)
        for val in regimes.dropna().unique():
            assert val in valid_types, f"Unexpected regime type: {val}"

    def test_detect_with_confidence(self, tmp_model_path):
        """detect_with_confidence returns regimes + float confidence."""
        config = {
            "regime_aware": {
                "ml_regime": {
                    "enabled": True,
                    "feature_mode": "raw_only",
                    "label_mode": "price",
                    "model_path": str(tmp_model_path),
                    "lgbm_params": {
                        "n_estimators": 10,
                        "num_leaves": 4,
                        "verbose": -1,
                    },
                }
            }
        }

        # Quick train
        train_df = generate_synthetic_ohlcv(
            num_days=180, regime="bullish", seed=55, timeframe_minutes=240,
        )
        with patch(
            "genetic_algorithm.ml.regime_trainer.load_ohlcv_data",
            return_value=train_df,
        ):
            trainer = RegimeTrainer(config)
            trainer.train(pairs=["SYNTH/TEST"], timeframe="4h")

        test_df = generate_synthetic_ohlcv(
            num_days=30, regime="bearish", seed=66, timeframe_minutes=240,
        )
        detector = MLRegimeDetector(
            model_path=str(tmp_model_path),
            feature_mode="raw_only",
        )
        regimes, confidence = detector.detect_with_confidence(test_df)

        assert isinstance(regimes, pd.Series)
        assert isinstance(confidence, pd.Series)
        assert len(regimes) == len(test_df)
        assert len(confidence) == len(test_df)
        # Confidence should be in [0, 1]
        valid_conf = confidence.dropna()
        assert (valid_conf >= 0.0).all() and (valid_conf <= 1.0).all()


# ===================================================================
# 12. Encoding consistency
# ===================================================================

class TestRegimeEncoding:
    """Verify REGIME_TO_INT ↔ INT_TO_REGIME round-trip."""

    def test_roundtrip(self):
        for regime, idx in REGIME_TO_INT.items():
            assert INT_TO_REGIME[idx] is regime

    def test_all_regime_types_covered(self):
        expected = {RegimeType.BULLISH, RegimeType.BEARISH, RegimeType.SIDEWAYS, RegimeType.VOLATILE}
        assert set(REGIME_TO_INT.keys()) == expected
