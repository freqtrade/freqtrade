"""
Tests for run_diagnostics and analyze_run modules.
"""

import csv
import json
import tempfile
import time
from pathlib import Path

import pytest

from genetic_algorithm.utils.run_diagnostics import (
    TimingTracker,
    GenerationTiming,
    GenerationCSVWriter,
    RunDiagnostics,
    build_run_metadata,
    save_run_metadata,
    _config_hash,
)
from genetic_algorithm.utils.analyze_run import (
    analyze_convergence,
    analyze_timing,
    analyze_diversity,
    analyze_holdout_trend,
    compare_runs,
    _safe_float,
    _format_duration,
)


# ========================================================================
# Helpers
# ========================================================================

def _make_config():
    """Minimal config dict for testing."""
    return {
        'genetic_algorithm': {
            'population_size': 20,
            'generations': 10,
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'elite_size': 2,
            'selection_method': 'tournament',
            'crossover_method': 'uniform',
        },
        'backtesting': {
            'pairs': ['BTC/USDT'],
            'timerange': '20230101-20260228',
            'stake_currency': 'USDT',
        },
        'walk_forward': {'enabled': False},
        'holdout_validation': {'enabled': False},
        'monte_carlo': {'enabled': False},
        'parallel_evaluation': {'enabled': False},
    }


class FakeStats:
    """Mimics PopulationStats for CSV writing."""
    def __init__(self, gen, best=0.5, avg=0.3, worst=0.1, **kw):
        self.generation = gen
        self.size = 20
        self.best_fitness = best
        self.avg_fitness = avg
        self.worst_fitness = worst
        self.median_fitness = (best + avg) / 2
        self.best_raw_fitness = best
        self.avg_raw_fitness = avg
        self.diversity_score = 0.05
        self.genetic_diversity = 0.4
        self.holdout_avg_degradation = kw.get('holdout_avg_degradation')
        self.holdout_best_degradation = kw.get('holdout_best_degradation')
        self.holdout_num_evaluated = kw.get('holdout_num_evaluated')
        self.holdout_num_profitable = kw.get('holdout_num_profitable')


# ========================================================================
# TimingTracker tests
# ========================================================================

class TestTimingTracker:
    def test_single_generation(self):
        tt = TimingTracker()
        tt.start_generation(0)
        time.sleep(0.01)
        tt.start_phase('eval')
        time.sleep(0.01)
        tt.end_phase('eval')
        result = tt.end_generation(0)
        assert isinstance(result, GenerationTiming)
        assert result.generation == 0
        assert result.wall_seconds >= 0.01
        assert result.eval_seconds >= 0.005

    def test_multiple_generations(self):
        tt = TimingTracker()
        for g in range(3):
            tt.start_generation(g)
            tt.start_phase('eval')
            tt.end_phase('eval')
            tt.end_generation(g)
        assert len(tt.history) == 3

    def test_summary(self):
        tt = TimingTracker()
        for g in range(5):
            tt.start_generation(g)
            tt.start_phase('eval')
            time.sleep(0.02)
            tt.end_phase('eval')
            tt.end_generation(g)
        s = tt.get_summary()
        assert s['generations_timed'] == 5
        assert s['total_wall_seconds'] >= 0.1
        assert s['avg_wall_per_gen'] > 0
        assert 'fastest_gen' in s
        assert 'slowest_gen' in s

    def test_empty_summary(self):
        tt = TimingTracker()
        assert tt.get_summary() == {}

    def test_overhead_calculation(self):
        tt = TimingTracker()
        tt.start_generation(0)
        time.sleep(0.02)
        tt.start_phase('eval')
        time.sleep(0.01)
        tt.end_phase('eval')
        result = tt.end_generation(0)
        assert result.overhead_seconds >= 0

    def test_multiple_phases(self):
        tt = TimingTracker()
        tt.start_generation(0)
        for phase in ['eval', 'selection', 'crossover', 'mutation', 'holdout']:
            tt.start_phase(phase)
            tt.end_phase(phase)
        result = tt.end_generation(0)
        assert result.eval_seconds >= 0
        assert result.selection_seconds >= 0
        assert result.crossover_seconds >= 0
        assert result.mutation_seconds >= 0
        assert result.holdout_seconds >= 0


# ========================================================================
# GenerationCSVWriter tests
# ========================================================================

class TestGenerationCSVWriter:
    def test_write_and_read(self, tmp_path):
        w = GenerationCSVWriter(tmp_path)
        w.write_row({'generation': 0, 'best_fitness': 0.5, 'avg_fitness': 0.3})
        w.write_row({'generation': 1, 'best_fitness': 0.6, 'avg_fitness': 0.4})
        w.close()

        with open(w.filepath, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]['generation'] == '0'
        assert rows[1]['best_fitness'] == '0.6'

    def test_none_values_become_empty(self, tmp_path):
        w = GenerationCSVWriter(tmp_path)
        w.write_row({'generation': 0, 'best_fitness': None, 'holdout_avg_degradation': None})
        w.close()

        with open(w.filepath, newline='') as f:
            rows = list(csv.DictReader(f))
        assert rows[0]['best_fitness'] == ''
        assert rows[0]['holdout_avg_degradation'] == ''

    def test_extra_keys_ignored(self, tmp_path):
        w = GenerationCSVWriter(tmp_path)
        w.write_row({'generation': 0, 'unknown_key': 999})
        w.close()
        # Should not raise — extrasaction='ignore'
        with open(w.filepath, newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_filepath_property(self, tmp_path):
        w = GenerationCSVWriter(tmp_path, filename='custom.csv')
        assert w.filepath == tmp_path / 'custom.csv'
        w.close()

    def test_double_close(self, tmp_path):
        w = GenerationCSVWriter(tmp_path)
        w.write_row({'generation': 0})
        w.close()
        w.close()  # Should not raise


# ========================================================================
# Run metadata tests
# ========================================================================

class TestRunMetadata:
    def test_build_metadata_structure(self):
        config = _make_config()
        meta = build_run_metadata(config)
        assert 'run_id' in meta
        assert meta['run_id'].startswith('run_')
        assert 'start_time' in meta
        assert 'python_version' in meta
        assert 'config_hash' in meta
        assert meta['config']['population_size'] == 20
        assert meta['config']['generations'] == 10
        assert meta['config']['pairs'] == ['BTC/USDT']

    def test_config_hash_deterministic(self):
        config = _make_config()
        h1 = _config_hash(config)
        h2 = _config_hash(config)
        assert h1 == h2
        assert len(h1) == 12

    def test_config_hash_changes(self):
        c1 = _make_config()
        c2 = _make_config()
        c2['genetic_algorithm']['population_size'] = 999
        assert _config_hash(c1) != _config_hash(c2)

    def test_save_metadata(self, tmp_path):
        meta = build_run_metadata(_make_config())
        path = save_run_metadata(meta, tmp_path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded['run_id'] == meta['run_id']


# ========================================================================
# RunDiagnostics integration tests
# ========================================================================

class TestRunDiagnostics:
    def test_full_lifecycle(self, tmp_path):
        diag = RunDiagnostics(tmp_path)
        config = _make_config()
        diag.start_run(config)

        for gen in range(3):
            diag.start_generation(gen)
            diag.start_phase('eval')
            time.sleep(0.02)
            diag.end_phase('eval')
            stats = FakeStats(gen, best=0.5 + gen * 0.1, avg=0.3 + gen * 0.05)
            diag.end_generation(gen, stats, extras={'mutation_rate': 0.15})

        summary = diag.end_run()

        # CSV written
        csv_path = tmp_path / 'generation_stats.csv'
        assert csv_path.exists()
        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert float(rows[2]['best_fitness']) == pytest.approx(0.7)

        # Metadata updated with end time
        meta_path = tmp_path / 'run_metadata.json'
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta['end_time'] is not None
        assert meta['duration_seconds'] >= 0
        assert meta['timing_summary']['generations_timed'] == 3

        # Timing summary returned
        assert summary['generations_timed'] == 3

    def test_start_run_creates_metadata(self, tmp_path):
        diag = RunDiagnostics(tmp_path)
        diag.start_run(_make_config())
        assert (tmp_path / 'run_metadata.json').exists()
        diag.csv_writer.close()

    def test_end_run_with_top_strategies(self, tmp_path):
        """end_run should include result_summary if top_strategies given."""
        diag = RunDiagnostics(tmp_path)
        diag.start_run(_make_config())
        diag.start_generation(0)
        diag.end_generation(0, FakeStats(0))
        
        # Mock top strategies
        class FakeInd:
            fitness = 0.75
        
        diag.end_run(top_strategies=[FakeInd(), FakeInd()])
        meta = json.loads((tmp_path / 'run_metadata.json').read_text())
        assert meta['result_summary']['top_strategies_returned'] == 2
        assert meta['result_summary']['best_fitness'] == 0.75


# ========================================================================
# analyze_run tests
# ========================================================================

class TestAnalyzeConvergence:
    def _make_rows(self, best_vals, avg_vals=None):
        rows = []
        for i, b in enumerate(best_vals):
            row = {'generation': str(i), 'best_fitness': str(b)}
            if avg_vals:
                row['avg_fitness'] = str(avg_vals[i])
            else:
                row['avg_fitness'] = str(b * 0.8)
            rows.append(row)
        return rows

    def test_basic_improvement(self):
        rows = self._make_rows([0.1, 0.2, 0.3, 0.4, 0.5])
        result = analyze_convergence(rows)
        assert result['generations_total'] == 5
        assert result['first_best'] == 0.1
        assert result['final_best'] == 0.5
        assert result['total_improvement'] == pytest.approx(0.4)

    def test_no_improvement_stagnation(self):
        rows = self._make_rows([0.5, 0.5, 0.5, 0.5, 0.5])
        result = analyze_convergence(rows)
        assert result['total_improvement'] == 0
        assert result['num_stagnation_windows'] >= 1

    def test_rapid_convergence(self):
        rows = self._make_rows([0.0, 0.9, 0.95, 0.96, 0.96])
        result = analyze_convergence(rows)
        assert result['convergence_gen_95pct'] <= 2

    def test_empty_rows(self):
        assert 'error' in analyze_convergence([])

    def test_single_generation(self):
        rows = self._make_rows([0.5])
        result = analyze_convergence(rows)
        assert result['generations_total'] == 1
        assert result['total_improvement'] == 0


class TestAnalyzeTiming:
    def _make_rows(self, wall_times, eval_times=None):
        rows = []
        for i, w in enumerate(wall_times):
            row = {'wall_seconds': str(w)}
            if eval_times:
                row['eval_seconds'] = str(eval_times[i])
            rows.append(row)
        return rows

    def test_basic_timing(self):
        rows = self._make_rows([10, 12, 8, 15, 11])
        result = analyze_timing(rows)
        assert result['total_wall_seconds'] == 56
        assert result['avg_per_gen'] == pytest.approx(11.2)
        assert result['min'] == 8
        assert result['max'] == 15

    def test_eval_percentage(self):
        rows = self._make_rows([10, 10], [7, 8])
        result = analyze_timing(rows)
        assert result['total_eval_seconds'] == 15
        assert result['eval_pct'] == 75

    def test_no_timing_data(self):
        rows = [{'generation': '0'}]
        result = analyze_timing(rows)
        assert 'error' in result

    def test_percentiles(self):
        rows = self._make_rows(list(range(1, 101)))
        result = analyze_timing(rows)
        assert result['p50'] == 51  # index 50 of sorted 1..100
        assert result['total_wall_human'] == '1h 24m 10s'


class TestAnalyzeDiversity:
    def test_declining_diversity(self):
        rows = [
            {'diversity_score': '0.10', 'genetic_diversity': '0.5'},
            {'diversity_score': '0.08', 'genetic_diversity': '0.4'},
            {'diversity_score': '0.05', 'genetic_diversity': '0.3'},
        ]
        result = analyze_diversity(rows)
        assert result['initial_diversity'] == 0.10
        assert result['final_diversity'] == 0.05
        assert result['diversity_change'] == pytest.approx(-0.05)

    def test_no_diversity_data(self):
        rows = [{'diversity_score': '', 'genetic_diversity': ''}]
        result = analyze_diversity(rows)
        assert 'error' in result


class TestAnalyzeHoldout:
    def test_no_holdout(self):
        rows = [{'generation': '0'}]
        result = analyze_holdout_trend(rows)
        assert result['holdout_monitored'] is False

    def test_worsening_trend(self):
        rows = [
            {'generation': '0', 'holdout_avg_degradation': '10'},
            {'generation': '5', 'holdout_avg_degradation': '15'},
            {'generation': '10', 'holdout_avg_degradation': '20'},
            {'generation': '15', 'holdout_avg_degradation': '40'},
            {'generation': '20', 'holdout_avg_degradation': '50'},
            {'generation': '25', 'holdout_avg_degradation': '60'},
        ]
        result = analyze_holdout_trend(rows)
        assert result['holdout_monitored'] is True
        assert result['checks_performed'] == 6
        assert result['trend'] == 'worsening'

    def test_stable_trend(self):
        rows = [
            {'generation': '0', 'holdout_avg_degradation': '10'},
            {'generation': '5', 'holdout_avg_degradation': '10'},
            {'generation': '10', 'holdout_avg_degradation': '10'},
            {'generation': '15', 'holdout_avg_degradation': '10'},
        ]
        result = analyze_holdout_trend(rows)
        assert result['trend'] == 'stable_or_improving'


class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float('3.14') == pytest.approx(3.14)

    def test_empty_string(self):
        assert _safe_float('') == 0.0

    def test_none(self):
        assert _safe_float(None) == 0.0

    def test_invalid(self):
        assert _safe_float('abc') == 0.0

    def test_custom_default(self):
        assert _safe_float('', -1.0) == -1.0


class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(45.5) == '45.5s'

    def test_minutes(self):
        assert _format_duration(125) == '2m 5s'

    def test_hours(self):
        assert _format_duration(3665) == '1h 1m 5s'


class TestCompareRuns:
    def _create_run(self, tmp_path, name, best_vals, wall_times):
        d = tmp_path / name
        d.mkdir()
        # Write CSV
        with open(d / 'generation_stats.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['generation', 'best_fitness', 'avg_fitness', 
                                                     'diversity_score', 'genetic_diversity',
                                                     'wall_seconds'])
            writer.writeheader()
            for i, (b, w) in enumerate(zip(best_vals, wall_times)):
                writer.writerow({
                    'generation': i, 'best_fitness': b, 'avg_fitness': b * 0.8,
                    'diversity_score': 0.05, 'genetic_diversity': 0.4,
                    'wall_seconds': w,
                })
        # Write metadata
        meta = {'run_id': f'run_{name}', 'config_hash': 'abc123'}
        with open(d / 'run_metadata.json', 'w') as f:
            json.dump(meta, f)
        return d

    def test_basic_comparison(self, tmp_path):
        dir_a = self._create_run(tmp_path, 'a', [0.1, 0.3, 0.5], [10, 12, 11])
        dir_b = self._create_run(tmp_path, 'b', [0.2, 0.5, 0.7], [8, 9, 7])
        result = compare_runs(dir_a, dir_b)
        assert 'a_convergence' in result
        assert 'b_convergence' in result
        assert 'delta' in result
        assert result['delta']['best_fitness_change'] == pytest.approx(0.2)
        assert result['delta']['speedup_pct'] > 0  # B is faster

    def test_missing_csv(self, tmp_path):
        dir_a = tmp_path / 'empty_a'
        dir_a.mkdir()
        dir_b = tmp_path / 'empty_b'
        dir_b.mkdir()
        result = compare_runs(dir_a, dir_b)
        assert 'a_error' in result
