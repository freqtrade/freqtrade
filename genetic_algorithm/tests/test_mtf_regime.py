"""
Tests for Multi-Timeframe Regime Detection, Continuous Scoring,
RegimeGene, and In-Strategy Regime Code Generation.
"""

import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genetic_algorithm.utils.regime_detector import RegimeDetector, RegimeType
from genetic_algorithm.utils.mtf_regime_detector import (
    MTFRegimeDetector,
    MTFRegimeResult,
    DEFAULT_MTF_WEIGHTS,
)
from genetic_algorithm.core.strategy_gene import (
    StrategyGene, IndicatorGene, ConditionGene, RegimeGene,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 500, drift: float = 0.0003, vol: float = 0.01,
                seed: int = 42, freq: str = '1h',
                start: str = '2023-01-01') -> pd.DataFrame:
    """Synthetic OHLCV."""
    rng = np.random.RandomState(seed)
    returns = rng.normal(drift, vol, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    noise = rng.uniform(0.002, 0.008, n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(100, 10000, n)
    dates = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({
        'date': dates,
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
    }).set_index('date')


def _make_multi_tf_dfs(seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Create aligned multi-TF DataFrames for testing."""
    return {
        '30m': _make_ohlcv(2000, freq='30min', seed=seed),
        '1h':  _make_ohlcv(1000, freq='1h', seed=seed + 1),
        '4h':  _make_ohlcv(250, freq='4h', seed=seed + 2),
        '1d':  _make_ohlcv(60, freq='1D', seed=seed + 3),
    }


def _make_minimal_strategy_gene(**kwargs) -> StrategyGene:
    """Create a minimal valid StrategyGene for tests."""
    defaults = dict(
        generation=0,
        individual_id=0,
        indicators=[IndicatorGene(type='RSI', parameters={'timeperiod': 14})],
        entry_conditions=[ConditionGene(indicator='RSI', operator='<', threshold=30)],
        exit_conditions=[ConditionGene(indicator='RSI', operator='>', threshold=70)],
    )
    defaults.update(kwargs)
    return StrategyGene(**defaults)


# ===========================================================================
# 1. Continuous Scoring
# ===========================================================================

class TestContinuousScoring:
    """Tests for RegimeDetector.detect_continuous()."""

    def test_returns_two_series(self):
        df = _make_ohlcv()
        det = RegimeDetector(method='adx_di_hysteresis')
        trend, vol = det.detect_continuous(df)
        assert isinstance(trend, pd.Series)
        assert isinstance(vol, pd.Series)
        assert len(trend) == len(df)
        assert len(vol) == len(df)

    def test_trend_score_range(self):
        df = _make_ohlcv()
        det = RegimeDetector(method='adx_di_hysteresis')
        trend, _ = det.detect_continuous(df)
        valid = trend.dropna()
        assert valid.min() >= -1.0 - 1e-9
        assert valid.max() <= 1.0 + 1e-9

    def test_volatility_score_range(self):
        df = _make_ohlcv()
        det = RegimeDetector(method='adx_di_hysteresis')
        _, vol = det.detect_continuous(df)
        valid = vol.dropna()
        assert valid.min() >= 0.0 - 1e-9
        assert valid.max() <= 1.0 + 1e-9

    def test_bullish_has_positive_trend(self):
        """Strong uptrend should yield mostly positive trend scores."""
        df = _make_ohlcv(drift=0.002, vol=0.005)
        det = RegimeDetector(method='adx_di_hysteresis')
        trend, _ = det.detect_continuous(df)
        valid = trend.dropna()
        assert valid.mean() > 0, "Bullish drift should produce positive avg trend score"

    def test_bearish_has_negative_trend(self):
        df = _make_ohlcv(drift=-0.002, vol=0.005)
        det = RegimeDetector(method='adx_di_hysteresis')
        trend, _ = det.detect_continuous(df)
        valid = trend.dropna()
        assert valid.mean() < 0, "Bearish drift should produce negative avg trend score"

    def test_continuous_to_regime_roundtrip(self):
        """continuous_to_regime should map scores back to known RegimeType values."""
        from genetic_algorithm.utils.regime_detector import RegimeDetector as RD
        regime = RD.continuous_to_regime(0.5, 0.3)
        assert regime == RegimeType.BULLISH
        regime = RD.continuous_to_regime(-0.5, 0.3)
        assert regime == RegimeType.BEARISH
        regime = RD.continuous_to_regime(0.0, 0.3)
        assert regime == RegimeType.SIDEWAYS
        regime = RD.continuous_to_regime(0.0, 0.9)
        assert regime == RegimeType.VOLATILE

    def test_multiple_methods(self):
        """detect_continuous should work for several rule-based methods."""
        df = _make_ohlcv()
        for method in ['adx_di_hysteresis', 'rolling_returns', 'bollinger']:
            det = RegimeDetector(method=method)
            trend, vol = det.detect_continuous(df)
            assert len(trend) == len(df), f"Failed for {method}"


# ===========================================================================
# 2. MTFRegimeDetector
# ===========================================================================

class TestMTFRegimeDetector:
    """Tests for the multi-timeframe regime detector."""

    @pytest.fixture()
    def default_config(self) -> dict:
        return {
            'regime_aware': {
                'method': 'adx_di_hysteresis',
                'mtf_enabled': True,
                'mtf_timeframes': ['1h', '4h', '1d'],
                'mtf_combination': 'hierarchical',
                'mtf_weights': {'1h': 1.0, '4h': 2.0, '1d': 3.0},
                'mtf_hierarchical_refine': 0.3,
                'mtf_transition_fast_window': 5,
                'mtf_transition_slow_window': 20,
                'mtf_transition_thresholds': [-0.5, 0.0, 0.5],
            }
        }

    def test_detect_from_dataframes_hierarchical(self, default_config):
        dfs = _make_multi_tf_dfs()
        # Only use TFs in config
        subset = {tf: dfs[tf] for tf in ['1h', '4h', '1d']}
        det = MTFRegimeDetector(default_config)
        result = det.detect_from_dataframes(subset)
        assert isinstance(result, MTFRegimeResult)
        assert len(result.trend_score) > 0
        assert len(result.volatility_score) == len(result.trend_score)

    def test_detect_from_dataframes_weighted(self, default_config):
        default_config['regime_aware']['mtf_combination'] = 'weighted_voting'
        dfs = _make_multi_tf_dfs()
        subset = {tf: dfs[tf] for tf in ['1h', '4h', '1d']}
        det = MTFRegimeDetector(default_config)
        result = det.detect_from_dataframes(subset)
        assert isinstance(result, MTFRegimeResult)
        assert len(result.trend_score) > 0

    def test_result_has_transition_signal(self, default_config):
        dfs = _make_multi_tf_dfs()
        subset = {tf: dfs[tf] for tf in ['1h', '4h', '1d']}
        det = MTFRegimeDetector(default_config)
        result = det.detect_from_dataframes(subset)
        assert result.transition_speed is not None
        assert result.transition_signal is not None
        # transition_signal should have integer labels
        unique = result.transition_signal.dropna().unique()
        assert len(unique) > 0

    def test_result_regime_context(self, default_config):
        dfs = _make_multi_tf_dfs()
        subset = {tf: dfs[tf] for tf in ['1h', '4h', '1d']}
        det = MTFRegimeDetector(default_config)
        result = det.detect_from_dataframes(subset)
        assert result.regime_context is not None
        # Should contain known context labels
        valid_contexts = result.regime_context.dropna().unique()
        assert len(valid_contexts) > 0

    def test_classify_segments(self, default_config):
        dfs = _make_multi_tf_dfs()
        subset = {tf: dfs[tf] for tf in ['1h', '4h', '1d']}
        det = MTFRegimeDetector(default_config)
        result = det.detect_from_dataframes(subset)
        # Need a base DF for segmentation
        base_df = subset['1h'].copy()
        segments = det.classify_segments(result, base_df)
        assert len(segments) > 0
        # Each segment should have start/end/regime
        for seg in segments:
            assert hasattr(seg, 'start_date')
            assert hasattr(seg, 'end_date')
            assert hasattr(seg, 'regime')

    def test_per_tf_scores_populated(self, default_config):
        dfs = _make_multi_tf_dfs()
        subset = {tf: dfs[tf] for tf in ['1h', '4h', '1d']}
        det = MTFRegimeDetector(default_config)
        result = det.detect_from_dataframes(subset)
        for tf in ['1h', '4h', '1d']:
            assert tf in result.per_tf_trend
            assert tf in result.per_tf_volatility

    def test_missing_tf_graceful(self, default_config):
        """Should handle missing TF data gracefully with partial fallback."""
        dfs = _make_multi_tf_dfs()
        # Only provide 1h — missing 4h and 1d
        partial = {'1h': dfs['1h']}
        det = MTFRegimeDetector(default_config)
        result = det.detect_from_dataframes(partial)
        # Should still produce a result with just the 1h data
        assert len(result.trend_score) > 0


# ===========================================================================
# 3. RegimeGene Serialization
# ===========================================================================

class TestRegimeGene:
    """Tests for RegimeGene dataclass serialization."""

    def test_defaults(self):
        rg = RegimeGene()
        assert rg.enabled is False
        assert rg.combination == 'weighted_voting'
        assert rg.entry_trend_min == -1.0
        assert rg.entry_trend_max == 1.0

    def test_to_dict_from_dict_roundtrip(self):
        rg = RegimeGene(
            enabled=True,
            regime_timeframes=['4h', '1d'],
            entry_trend_min=-0.3,
            entry_trend_max=0.7,
            exit_on_regime_change=True,
            combination='weighted_voting',
        )
        d = rg.to_dict()
        rg2 = RegimeGene.from_dict(d)
        assert rg2.enabled == rg.enabled
        assert rg2.regime_timeframes == rg.regime_timeframes
        assert abs(rg2.entry_trend_min - rg.entry_trend_min) < 1e-9
        assert abs(rg2.entry_trend_max - rg.entry_trend_max) < 1e-9
        assert rg2.exit_on_regime_change == rg.exit_on_regime_change
        assert rg2.combination == rg.combination

    def test_from_dict_none_returns_none(self):
        assert RegimeGene.from_dict(None) is None

    def test_strategy_gene_with_regime(self):
        """RegimeGene integrates into StrategyGene serialization."""
        rg = RegimeGene(enabled=True, regime_timeframes=['1h'])
        sg = _make_minimal_strategy_gene(regime_gene=rg)
        d = sg.to_dict()
        assert 'regime_gene' in d
        sg2 = StrategyGene.from_dict(d)
        assert sg2.regime_gene is not None
        assert sg2.regime_gene.enabled is True


# ===========================================================================
# 4. Strategy Code Generator — Regime injection
# ===========================================================================

class TestRegimeCodeGeneration:
    """Tests that regime-aware code injected into strategies compiles."""

    def test_regime_indicator_code_compiles(self):
        """The generated regime indicator code should be valid Python."""
        from genetic_algorithm.strategies.generator import StrategyGenerator

        gen = StrategyGenerator.__new__(StrategyGenerator)
        rg = RegimeGene(
            enabled=True,
            regime_timeframes=['4h', '1d'],
            combination='weighted_voting',
        )
        code = gen._generate_regime_indicator_code(rg, '5m')
        assert code != ''
        # Code is indented for embedding; dedent before compiling
        compile(textwrap.dedent(code), '<regime_indicator>', 'exec')

    def test_regime_entry_filter_compiles(self):
        from genetic_algorithm.strategies.generator import StrategyGenerator

        gen = StrategyGenerator.__new__(StrategyGenerator)
        rg = RegimeGene(
            enabled=True,
            entry_trend_min=-0.3,
            entry_trend_max=0.8,
        )
        code = gen._generate_regime_entry_filter(rg)
        assert code != ''
        compile(textwrap.dedent(code), '<regime_entry>', 'exec')

    def test_regime_exit_filter_compiles(self):
        from genetic_algorithm.strategies.generator import StrategyGenerator

        gen = StrategyGenerator.__new__(StrategyGenerator)
        rg = RegimeGene(enabled=True, exit_on_regime_change=True)
        code = gen._generate_regime_exit_filter(rg)
        assert code != ''
        compile(textwrap.dedent(code), '<regime_exit>', 'exec')

    def test_disabled_regime_gene_no_code(self):
        from genetic_algorithm.strategies.generator import StrategyGenerator

        gen = StrategyGenerator.__new__(StrategyGenerator)
        rg = RegimeGene(enabled=False)
        code = gen._generate_regime_indicator_code(rg, '5m')
        assert code == ''
        code = gen._generate_regime_entry_filter(rg)
        assert code == ''


# ===========================================================================
# 5. ML Detector — MTF Feature Computation
# ===========================================================================

class TestMLDetectorMTFFeatures:
    """Tests for MTF cross-TF feature computation in MLRegimeDetector."""

    def test_mtf_features_shape(self):
        from genetic_algorithm.ml.regime_detector import MLRegimeDetector

        df = _make_ohlcv(300)
        # Simulate pre-computed MTF scores aligned to df.index
        mtf_scores = {}
        for tf in ['1h', '4h']:
            mtf_scores[tf] = pd.DataFrame({
                'trend_score': np.random.uniform(-1, 1, len(df)),
                'volatility_score': np.random.uniform(0, 1, len(df)),
            }, index=df.index)

        feats = MLRegimeDetector._compute_mtf_features(df, mtf_scores)
        assert 'trend_score_1h' in feats.columns
        assert 'trend_score_4h' in feats.columns
        assert 'trend_div_low_high' in feats.columns
        assert 'trend_alignment' in feats.columns
        assert 'vol_spread' in feats.columns
        assert 'transition_speed' in feats.columns
        assert 'transition_accel' in feats.columns
        assert len(feats) == len(df)

    def test_alignment_values(self):
        """When all TFs are bullish, alignment should be +1."""
        from genetic_algorithm.ml.regime_detector import MLRegimeDetector

        df = _make_ohlcv(100)
        mtf_scores = {}
        for tf in ['1h', '4h']:
            mtf_scores[tf] = pd.DataFrame({
                'trend_score': np.full(len(df), 0.5),  # all positive
                'volatility_score': np.full(len(df), 0.3),
            }, index=df.index)

        feats = MLRegimeDetector._compute_mtf_features(df, mtf_scores)
        assert (feats['trend_alignment'] == 1.0).all()


# ===========================================================================
# 6. Mutation — RegimeGene mutation
# ===========================================================================

class TestRegimeGeneMutation:
    """Tests for _mutate_regime_gene in mutation.py."""

    def test_mutation_returns_gene_and_list(self):
        from genetic_algorithm.core.mutation import _mutate_regime_gene

        gene = _make_minimal_strategy_gene(
            regime_gene=RegimeGene(enabled=True, regime_timeframes=['4h', '1d']),
        )
        in_cfg = {'enabled': True, 'regime_timeframes': ['4h', '1d']}
        mutated_gene, mutations = _mutate_regime_gene(gene, in_cfg)
        # Should return the gene and a list of mutation descriptions
        assert isinstance(mutations, list)
        assert hasattr(mutated_gene, 'regime_gene')

    def test_mutation_creates_regime_gene_if_none(self):
        from genetic_algorithm.core.mutation import _mutate_regime_gene

        gene = _make_minimal_strategy_gene(regime_gene=None)
        in_cfg = {'enabled': True, 'regime_timeframes': ['4h', '1d']}
        mutated_gene, _ = _mutate_regime_gene(gene, in_cfg)
        assert mutated_gene.regime_gene is not None

    def test_trend_bounds_stay_valid(self):
        """After many mutations, entry_trend_min < entry_trend_max."""
        from genetic_algorithm.core.mutation import _mutate_regime_gene

        gene = _make_minimal_strategy_gene(
            regime_gene=RegimeGene(enabled=True),
        )
        in_cfg = {'enabled': True, 'regime_timeframes': ['4h', '1d']}

        for _ in range(200):
            gene, _ = _mutate_regime_gene(gene, in_cfg)
            rg = gene.regime_gene
            assert rg.entry_trend_min < rg.entry_trend_max
            assert rg.entry_trend_min >= -1.0
            assert rg.entry_trend_max <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
