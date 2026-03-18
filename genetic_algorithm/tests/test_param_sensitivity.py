"""
Tests for Parameter Sensitivity Analysis.

Tests cover:
- Parameter extraction from strategy genes
- Perturbation value generation (correct ranges)
- Parameter setting via dotted paths
- Sensitivity score calculation
- Robustness classification (fragile/stable)
- Analysis with mock evaluator
"""

import pytest
from unittest.mock import MagicMock, patch

from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene
from genetic_algorithm.evaluation.param_sensitivity import (
    ParameterSensitivityAnalyzer,
    SensitivityReport,
    ParameterResult,
    DEFAULT_PERTURBATION_PCTS,
)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _make_gene(**overrides):
    """Create a test strategy gene with known numeric parameters."""
    defaults = dict(
        generation=0,
        individual_id=0,
        indicators=[
            IndicatorGene(type='RSI', parameters={'period': 14}),
            IndicatorGene(type='BBANDS', parameters={'period': 20, 'std_dev': 2.0}),
        ],
        entry_conditions=[
            ConditionGene(indicator='RSI', operator='<', threshold=30.0, logic='AND'),
        ],
        exit_conditions=[
            ConditionGene(indicator='RSI', operator='>', threshold=70.0, logic='AND'),
        ],
        stoploss=-0.10,
        minimal_roi={"0": 0.04, "30": 0.02, "60": 0.01},
        max_open_trades=3,
        trailing_stop=True,
        trailing_stop_positive=0.01,
        trailing_stop_positive_offset=0.02,
    )
    defaults.update(overrides)
    return StrategyGene(**defaults)


# ══════════════════════════════════════════════════════════════════════
# Tests: Parameter Extraction
# ══════════════════════════════════════════════════════════════════════

class TestParameterExtraction:
    """Tests for extracting numeric parameters from strategy genes."""

    def test_extracts_indicator_params(self):
        gene = _make_gene()
        params = ParameterSensitivityAnalyzer.extract_parameters(gene)
        param_names = [p[0] for p in params]

        assert 'indicators[0].parameters.period' in param_names
        assert 'indicators[1].parameters.period' in param_names
        assert 'indicators[1].parameters.std_dev' in param_names

    def test_extracts_condition_thresholds(self):
        gene = _make_gene()
        params = ParameterSensitivityAnalyzer.extract_parameters(gene)
        param_names = [p[0] for p in params]

        assert 'entry_conditions[0].threshold' in param_names
        assert 'exit_conditions[0].threshold' in param_names

    def test_extracts_stoploss(self):
        gene = _make_gene()
        params = ParameterSensitivityAnalyzer.extract_parameters(gene)
        param_dict = dict(params)

        assert 'stoploss' in param_dict
        assert param_dict['stoploss'] == -0.10

    def test_extracts_roi_values(self):
        gene = _make_gene()
        params = ParameterSensitivityAnalyzer.extract_parameters(gene)
        param_names = [p[0] for p in params]

        assert 'minimal_roi.0' in param_names
        assert 'minimal_roi.30' in param_names
        assert 'minimal_roi.60' in param_names

    def test_extracts_trailing_stop_params(self):
        gene = _make_gene()
        params = ParameterSensitivityAnalyzer.extract_parameters(gene)
        param_dict = dict(params)

        assert 'trailing_stop_positive' in param_dict
        assert param_dict['trailing_stop_positive'] == 0.01
        assert 'trailing_stop_positive_offset' in param_dict

    def test_skips_trailing_when_none(self):
        gene = _make_gene(
            trailing_stop=False,
            trailing_stop_positive=None,
            trailing_stop_positive_offset=None,
        )
        params = ParameterSensitivityAnalyzer.extract_parameters(gene)
        param_names = [p[0] for p in params]

        assert 'trailing_stop_positive' not in param_names
        assert 'trailing_stop_positive_offset' not in param_names

    def test_total_param_count(self):
        gene = _make_gene()
        params = ParameterSensitivityAnalyzer.extract_parameters(gene)

        # 3 indicator params + 1 entry threshold + 1 exit threshold +
        # 1 stoploss + 3 ROI + 2 trailing = 11
        assert len(params) == 11

    def test_minimal_gene_few_params(self):
        gene = StrategyGene(
            generation=0,
            individual_id=0,
            indicators=[
                IndicatorGene(type='RSI', parameters={'period': 14}),
            ],
            entry_conditions=[
                ConditionGene(indicator='RSI', operator='<', threshold=30.0),
            ],
            exit_conditions=[],
            stoploss=-0.05,
            minimal_roi={"0": 0.01},
        )
        params = ParameterSensitivityAnalyzer.extract_parameters(gene)
        # 1 indicator param + 1 entry threshold + 1 stoploss + 1 ROI = 4
        assert len(params) == 4


# ══════════════════════════════════════════════════════════════════════
# Tests: Parameter Setting
# ══════════════════════════════════════════════════════════════════════

class TestParameterSetting:
    """Tests for setting parameters via dotted path names."""

    def test_set_indicator_param(self):
        gene = _make_gene()
        ParameterSensitivityAnalyzer._set_parameter(
            gene, 'indicators[0].parameters.period', 20.0,
        )
        # Period should be rounded to int since original was int
        assert gene.indicators[0].parameters['period'] == 20

    def test_set_indicator_float_param(self):
        gene = _make_gene()
        ParameterSensitivityAnalyzer._set_parameter(
            gene, 'indicators[1].parameters.std_dev', 2.5,
        )
        assert gene.indicators[1].parameters['std_dev'] == 2.5

    def test_set_condition_threshold(self):
        gene = _make_gene()
        ParameterSensitivityAnalyzer._set_parameter(
            gene, 'entry_conditions[0].threshold', 25.0,
        )
        assert gene.entry_conditions[0].threshold == 25.0

    def test_set_stoploss(self):
        gene = _make_gene()
        ParameterSensitivityAnalyzer._set_parameter(gene, 'stoploss', -0.15)
        assert gene.stoploss == -0.15

    def test_set_roi(self):
        gene = _make_gene()
        ParameterSensitivityAnalyzer._set_parameter(
            gene, 'minimal_roi.0', 0.05,
        )
        assert gene.minimal_roi['0'] == 0.05

    def test_set_trailing_stop_positive(self):
        gene = _make_gene()
        ParameterSensitivityAnalyzer._set_parameter(
            gene, 'trailing_stop_positive', 0.015,
        )
        assert gene.trailing_stop_positive == 0.015


# ══════════════════════════════════════════════════════════════════════
# Tests: Analysis with Mock Evaluator
# ══════════════════════════════════════════════════════════════════════

class TestAnalysis:
    """Tests for the full sensitivity analysis pipeline."""

    def _mock_evaluator(self, base_fitness=0.5, sensitivity=0.0):
        """Create a mock evaluator that returns predictable fitness values.

        If sensitivity=0, all perturbations return base_fitness (perfectly robust).
        If sensitivity>0, fitness drops by sensitivity * perturbation_fraction.
        """
        evaluator = MagicMock()
        call_count = [0]

        def fake_evaluate(gene, strategy_name=None):
            call_count[0] += 1
            # Vary fitness slightly based on call count to simulate perturbation
            noise = sensitivity * (call_count[0] % 10) * 0.01
            return (base_fitness - noise, {'profit': 1.0})

        evaluator.evaluate = fake_evaluate
        return evaluator

    def test_analysis_returns_report(self):
        gene = _make_gene()
        evaluator = self._mock_evaluator(base_fitness=0.5)
        config = {}  # Not used since evaluator is provided
        analyzer = ParameterSensitivityAnalyzer(
            config, evaluator=evaluator,
        )
        report = analyzer.analyze(gene, base_fitness=0.5)

        assert isinstance(report, SensitivityReport)
        assert report.base_fitness == 0.5
        assert report.parameters_tested > 0
        assert report.total_backtests > 0

    def test_robust_strategy_high_score(self):
        gene = _make_gene()
        evaluator = self._mock_evaluator(base_fitness=0.5, sensitivity=0.0)
        analyzer = ParameterSensitivityAnalyzer(
            {}, evaluator=evaluator,
        )
        report = analyzer.analyze(gene, base_fitness=0.5)

        # With zero sensitivity, robustness should be very high
        assert report.overall_robustness >= 0.9

    def test_fragile_strategy_detected(self):
        gene = _make_gene()

        # Mock evaluator that drops fitness significantly
        evaluator = MagicMock()
        call_count = [0]

        def fragile_evaluate(gene, strategy_name=None):
            call_count[0] += 1
            # Every perturbation causes a 30% drop
            return (0.35, {'profit': 0.5})

        evaluator.evaluate = fragile_evaluate
        analyzer = ParameterSensitivityAnalyzer({}, evaluator=evaluator)
        report = analyzer.analyze(gene, base_fitness=0.5)

        assert len(report.fragile_params) > 0
        assert report.overall_robustness < 0.8

    def test_analysis_with_zero_value_params_skipped(self):
        """Params with value=0 are skipped; only non-zero ones are tested."""
        gene = StrategyGene(
            generation=0, individual_id=0,
            indicators=[
                IndicatorGene(type='RSI', parameters={'period': 14}),
            ],
            entry_conditions=[
                ConditionGene(indicator='RSI', operator='<', threshold=30.0),
            ],
            exit_conditions=[],
            stoploss=0.0,  # Will be skipped (value=0)
            minimal_roi={},
        )
        evaluator = self._mock_evaluator()
        analyzer = ParameterSensitivityAnalyzer({}, evaluator=evaluator)
        report = analyzer.analyze(gene, base_fitness=0.5)

        # stoploss=0 is skipped, minimal_roi is empty
        # indicator param + entry threshold remain = 2
        assert report.parameters_tested == 2
        assert report.total_backtests > 0

    def test_perturbation_count(self):
        gene = _make_gene()
        evaluator = self._mock_evaluator(base_fitness=0.5)
        pcts = [0.10]  # Only ±10%
        analyzer = ParameterSensitivityAnalyzer(
            {}, perturbation_pcts=pcts, evaluator=evaluator,
        )
        report = analyzer.analyze(gene, base_fitness=0.5)

        # 2 perturbations per param (±10%)
        assert report.perturbations_per_param == 2

    def test_evaluator_exception_handled(self):
        gene = _make_gene()

        evaluator = MagicMock()
        evaluator.evaluate.side_effect = Exception("Backtest crashed")
        analyzer = ParameterSensitivityAnalyzer({}, evaluator=evaluator)

        # Should not raise
        report = analyzer.analyze(gene, base_fitness=0.5)
        assert report.total_backtests > 0

    def test_report_fields_populated(self):
        gene = _make_gene()
        evaluator = self._mock_evaluator(base_fitness=0.5, sensitivity=1.0)
        analyzer = ParameterSensitivityAnalyzer({}, evaluator=evaluator)
        report = analyzer.analyze(gene, base_fitness=0.5)

        assert report.strategy_name is not None
        assert report.robustness_band >= 0
        assert isinstance(report.fragile_params, list)
        assert isinstance(report.stable_params, list)


# ══════════════════════════════════════════════════════════════════════
# Tests: Edge cases
# ══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_negative_stoploss_perturbation(self):
        """Stoploss is negative — perturbations should still work correctly."""
        gene = _make_gene(stoploss=-0.10)
        params = ParameterSensitivityAnalyzer.extract_parameters(gene)
        stoploss_val = dict(params)['stoploss']
        assert stoploss_val == -0.10

        # Check that setting perturbed value works
        perturbed = gene.copy()
        ParameterSensitivityAnalyzer._set_parameter(perturbed, 'stoploss', -0.11)
        assert perturbed.stoploss == -0.11

    def test_integer_param_preserved(self):
        """Integer parameters should stay integers after perturbation."""
        gene = _make_gene()
        perturbed = gene.copy()
        ParameterSensitivityAnalyzer._set_parameter(
            perturbed, 'indicators[0].parameters.period', 15.7,
        )
        assert isinstance(perturbed.indicators[0].parameters['period'], int)
        assert perturbed.indicators[0].parameters['period'] == 16
