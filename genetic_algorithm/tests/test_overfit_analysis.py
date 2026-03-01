"""
Tests for Overfitting Analysis Utilities

Tests classify_overfitting, OverfitThresholds, GenerationHoldoutStats,
generate_detailed_results, and summary computation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import tempfile
import pytest
from genetic_algorithm.utils.overfit_analysis import (
    classify_overfitting,
    OverfitThresholds,
    OverfitAssessment,
    GenerationHoldoutStats,
    generate_detailed_results,
    save_detailed_results,
    print_overfit_summary,
    _compute_summary,
    _extract_fitness_history,
    LABEL_SAFE,
    LABEL_WARNING,
    LABEL_OVERFIT,
    LABEL_UNKNOWN,
)
from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene, IndicatorGene, ConditionGene


# ============================================================================
# Helpers
# ============================================================================

def _make_gene():
    return StrategyGene(
        generation=0, individual_id=0,
        indicators=[IndicatorGene(type='RSI', parameters={'period': 14}),
                     IndicatorGene(type='EMA', parameters={'period': 20})],
        entry_conditions=[ConditionGene(indicator='RSI', operator='<', threshold=30),
                          ConditionGene(indicator='EMA', operator='cross_above', threshold=0)],
        exit_conditions=[ConditionGene(indicator='RSI', operator='>', threshold=70)],
    )


def _make_individual(fitness=1.0, metrics=None):
    ind = Individual(strategy_gene=_make_gene())
    ind.fitness = fitness
    ind.raw_fitness = fitness
    ind.evaluated = True
    ind.metrics = metrics or {}
    return ind


# ============================================================================
# OverfitThresholds
# ============================================================================

class TestOverfitThresholds:
    def test_defaults(self):
        t = OverfitThresholds()
        assert t.holdout_degradation_warning == 0.25
        assert t.holdout_degradation_overfit == 0.50
        assert t.mc_robustness_warning == 0.70
        assert t.composite_warning == 0.25
        assert t.composite_overfit == 0.50
    
    def test_from_config_empty(self):
        t = OverfitThresholds.from_config({})
        assert t.holdout_degradation_warning == 0.25  # defaults
    
    def test_from_config_with_values(self):
        config = {
            'overfit_analysis': {
                'thresholds': {
                    'holdout_degradation_warning': 0.30,
                    'mc_robustness_overfit': 0.40,
                }
            }
        }
        t = OverfitThresholds.from_config(config)
        assert t.holdout_degradation_warning == 0.30
        assert t.mc_robustness_overfit == 0.40
        # Other defaults unchanged
        assert t.holdout_degradation_overfit == 0.50


# ============================================================================
# classify_overfitting
# ============================================================================

class TestClassifyOverfitting:
    def test_safe_individual(self):
        """Low degradation, high MC robustness → SAFE."""
        metrics = {
            'holdout_degradation': 0.10,
            'holdout_fitness': 0.90,
            'mc_robustness': 0.90,
            'mc_mean_profit': 50.0,
        }
        result = classify_overfitting(metrics, fitness=1.0)
        assert result.overall_label == LABEL_SAFE
        assert result.composite_score < 0.35
        assert result.holdout_label == LABEL_SAFE
        assert result.mc_label == LABEL_SAFE
    
    def test_warning_individual(self):
        """Moderate degradation → WARNING."""
        metrics = {
            'holdout_degradation': 0.35,
            'holdout_fitness': 0.65,
            'mc_robustness': 0.65,
        }
        result = classify_overfitting(metrics, fitness=1.0)
        assert result.overall_label == LABEL_WARNING
        assert result.holdout_label == LABEL_WARNING
        assert result.mc_label == LABEL_WARNING
    
    def test_overfit_individual(self):
        """High degradation, low MC robustness → OVERFIT."""
        metrics = {
            'holdout_degradation': 0.70,
            'holdout_fitness': 0.30,
            'mc_robustness': 0.30,
            'train_val_gap': 0.40,
        }
        result = classify_overfitting(metrics, fitness=1.0)
        assert result.overall_label == LABEL_OVERFIT
        assert result.composite_score >= 0.60
        assert result.holdout_label == LABEL_OVERFIT
        assert result.wf_label == LABEL_OVERFIT
    
    def test_no_signals_unknown(self):
        """No holdout/MC/WF data → UNKNOWN."""
        result = classify_overfitting({}, fitness=1.0)
        assert result.overall_label == LABEL_UNKNOWN
        assert result.composite_score is None
    
    def test_only_holdout(self):
        """Only holdout data available."""
        metrics = {'holdout_degradation': 0.10, 'holdout_fitness': 0.90}
        result = classify_overfitting(metrics, fitness=1.0)
        assert result.holdout_label == LABEL_SAFE
        assert result.mc_label == LABEL_UNKNOWN
        assert result.wf_label == LABEL_UNKNOWN
        assert result.overall_label == LABEL_SAFE
    
    def test_only_mc(self):
        """Only Monte Carlo data available."""
        metrics = {'mc_robustness': 0.85}
        result = classify_overfitting(metrics, fitness=1.0)
        assert result.mc_label == LABEL_SAFE
        assert result.holdout_label == LABEL_UNKNOWN
    
    def test_only_wf(self):
        """Only walk-forward data available."""
        metrics = {'train_val_gap': 0.10}
        result = classify_overfitting(metrics, fitness=1.0)
        assert result.wf_label == LABEL_SAFE
        assert result.holdout_label == LABEL_UNKNOWN
    
    def test_custom_thresholds(self):
        """Custom thresholds should change classification."""
        metrics = {'holdout_degradation': 0.20}
        # Default: 0.20 < 0.25 warning → SAFE
        result_default = classify_overfitting(metrics, fitness=1.0)
        assert result_default.holdout_label == LABEL_SAFE
        
        # Custom: warning at 0.15 → WARNING
        strict = OverfitThresholds(holdout_degradation_warning=0.15)
        result_strict = classify_overfitting(metrics, fitness=1.0, thresholds=strict)
        assert result_strict.holdout_label == LABEL_WARNING
    
    def test_strategy_gene_info(self):
        """Should extract indicator/condition counts from gene."""
        gene = _make_gene()
        metrics = {'holdout_degradation': 0.10}
        result = classify_overfitting(metrics, fitness=1.0, strategy_gene=gene)
        assert result.indicator_count == 2
        assert result.condition_count == 2
    
    def test_metrics_fields_populated(self):
        metrics = {
            'profit': 150.0,
            'sharpe_ratio': 2.5,
            'max_drawdown': -0.15,
            'num_trades': 50,
            'win_rate': 0.65,
        }
        result = classify_overfitting(metrics, fitness=1.0)
        assert result.profit == 150.0
        assert result.sharpe_ratio == 2.5
        assert result.win_rate == 0.65
    
    def test_to_dict(self):
        metrics = {'holdout_degradation': 0.10, 'mc_robustness': 0.80}
        result = classify_overfitting(metrics, fitness=1.0)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert 'overall_label' in d
        assert 'composite_score' in d


# ============================================================================
# GenerationHoldoutStats
# ============================================================================

class TestGenerationHoldoutStats:
    def test_creation(self):
        stats = GenerationHoldoutStats(
            generation=5, avg_degradation=0.15, best_degradation=0.05,
            worst_degradation=0.40, num_evaluated=10, num_profitable=8,
        )
        assert stats.generation == 5
        assert stats.avg_degradation == 0.15
    
    def test_to_dict(self):
        stats = GenerationHoldoutStats(
            generation=1, avg_degradation=0.20, best_degradation=0.10,
            worst_degradation=0.50, num_evaluated=5, num_profitable=3,
        )
        d = stats.to_dict()
        assert d['generation'] == 1
        assert d['num_profitable'] == 3


# ============================================================================
# generate_detailed_results
# ============================================================================

class TestGenerateDetailedResults:
    def test_basic_report(self):
        strategies = [_make_individual(fitness=1.0, metrics={'holdout_degradation': 0.10})]
        config = {'genetic_algorithm': {'population_size': 20, 'generations': 10}}
        report = generate_detailed_results(strategies, config)
        
        assert 'metadata' in report
        assert 'strategies' in report
        assert 'summary' in report
        assert len(report['strategies']) == 1
    
    def test_empty_strategies(self):
        config = {}
        report = generate_detailed_results([], config)
        assert report['summary']['total'] == 0
    
    def test_multiple_strategies(self):
        strategies = [
            _make_individual(fitness=1.0, metrics={'holdout_degradation': 0.10, 'mc_robustness': 0.90}),
            _make_individual(fitness=0.5, metrics={'holdout_degradation': 0.60, 'mc_robustness': 0.30}),
        ]
        config = {}
        report = generate_detailed_results(strategies, config)
        assert len(report['strategies']) == 2
        assert report['summary']['total'] == 2
    
    def test_with_holdout_history(self):
        strategies = [_make_individual()]
        config = {}
        history = [
            GenerationHoldoutStats(1, 0.10, 0.05, 0.20, 5, 4),
            GenerationHoldoutStats(2, 0.12, 0.06, 0.25, 5, 4),
        ]
        report = generate_detailed_results(strategies, config, generation_holdout_history=history)
        assert 'holdout_history' in report
        assert len(report['holdout_history']) == 2
    
    def test_with_generation_stats_dicts(self):
        strategies = [_make_individual()]
        config = {}
        gen_stats = [
            {'generation': 0, 'best_fitness': 1.0, 'avg_fitness': 0.5},
            {'generation': 1, 'best_fitness': 1.2, 'avg_fitness': 0.6},
        ]
        report = generate_detailed_results(strategies, config, generation_stats=gen_stats)
        assert 'generation_fitness_history' in report
        assert len(report['generation_fitness_history']) == 2
    
    def test_metadata_structure(self):
        config = {
            'genetic_algorithm': {'population_size': 30, 'generations': 15},
            'backtesting': {'pairs': ['BTC/USDT'], 'timerange': '20200101-20210101'},
            'walk_forward': {'enabled': True},
            'holdout_validation': {'enabled': True, 'holdout_pct': 0.2},
        }
        report = generate_detailed_results([_make_individual()], config)
        meta = report['metadata']
        assert 'timestamp' in meta
        assert meta['config_summary']['population_size'] == 30
        assert meta['config_summary']['walk_forward_enabled'] is True


# ============================================================================
# _compute_summary
# ============================================================================

class TestComputeSummary:
    def test_empty(self):
        result = _compute_summary([])
        assert result['total'] == 0
    
    def test_counts(self):
        assessments = [
            {'assessment': {'overall_label': LABEL_SAFE, 'composite_score': 0.1}},
            {'assessment': {'overall_label': LABEL_SAFE, 'composite_score': 0.2}},
            {'assessment': {'overall_label': LABEL_WARNING, 'composite_score': 0.4}},
            {'assessment': {'overall_label': LABEL_OVERFIT, 'composite_score': 0.8}},
        ]
        result = _compute_summary(assessments)
        assert result['total'] == 4
        assert result['safe'] == 2
        assert result['warning'] == 1
        assert result['overfit'] == 1
        assert result['avg_composite_score'] == pytest.approx(0.375)


# ============================================================================
# save_detailed_results
# ============================================================================

class TestSaveDetailedResults:
    def test_save_to_file(self):
        report = {'test': True, 'strategies': []}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_detailed_results(report, Path(tmpdir), 'test_report.json')
            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded['test'] is True
    
    def test_auto_filename(self):
        report = {'strategies': []}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_detailed_results(report, Path(tmpdir))
            assert path.exists()
            assert path.name.startswith('results_detailed_')
    
    def test_creates_directory(self):
        report = {'strategies': []}
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / 'sub' / 'dir'
            path = save_detailed_results(report, nested, 'test.json')
            assert path.exists()


# ============================================================================
# print_overfit_summary
# ============================================================================

class TestPrintOverfitSummary:
    def test_no_crash_empty(self, capsys):
        print_overfit_summary([])
        captured = capsys.readouterr()
        assert 'OVERFITTING ANALYSIS SUMMARY' in captured.out
    
    def test_formats_assessments(self, capsys):
        a1 = OverfitAssessment(
            individual_id='test_1', fitness=1.0,
            holdout_fitness=0.9, holdout_degradation=0.10,
            mc_robustness=0.85, train_val_gap=0.05,
            composite_score=0.10, overall_label=LABEL_SAFE,
        )
        a2 = OverfitAssessment(
            individual_id='test_2', fitness=0.8,
            composite_score=0.70, overall_label=LABEL_OVERFIT,
        )
        print_overfit_summary([a1, a2])
        captured = capsys.readouterr()
        assert 'test_1' in captured.out
        assert 'test_2' in captured.out
        assert 'SAFE' in captured.out
        assert 'OVERFIT' in captured.out


# ============================================================================
# _extract_fitness_history
# ============================================================================

class TestExtractFitnessHistory:
    def test_from_dicts(self):
        stats = [
            {'generation': 0, 'best_fitness': 1.0, 'avg_fitness': 0.5, 'genetic_diversity': 0.8},
            {'generation': 1, 'best_fitness': 1.2, 'avg_fitness': 0.6, 'genetic_diversity': 0.7},
        ]
        result = _extract_fitness_history(stats)
        assert len(result) == 2
        assert result[0]['generation'] == 0
        assert result[1]['best_fitness'] == 1.2
    
    def test_from_dataclass_like(self):
        """Test with objects that have attributes (duck-typed)."""
        class MockStats:
            def __init__(self, gen, best, avg):
                self.generation = gen
                self.best_fitness = best
                self.avg_fitness = avg
                self.genetic_diversity = None
                self.holdout_avg_degradation = None
                self.holdout_best_degradation = None
        
        stats = [MockStats(0, 1.0, 0.5), MockStats(1, 1.5, 0.7)]
        result = _extract_fitness_history(stats)
        assert len(result) == 2
        assert result[1]['best_fitness'] == 1.5
