"""
Tests for validation-run preparation changes:
- CPCV integration into run_ga.py post-evolution
- DSR column tracking in generation CSV
- Direct backtester timerange_override
- Holdout phase timing fix
- Config validation (DSR/CPCV sections, LTC→BNB fix)
"""

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

# ── Test 1: CPCV config is loaded by CPCVValidator ──


class TestCPCVIntegration:
    """Verify CPCV plumbing: config loading, block splitting, PBO computation."""

    def test_cpcv_validator_reads_config(self):
        """CPCVValidator picks up n_groups, pbo_threshold from config dict."""
        from genetic_algorithm.evaluation.cpcv import CPCVValidator

        config = {
            'cpcv': {
                'enabled': True,
                'n_groups': 8,
                'n_test_groups': 3,
                'max_paths': 50,
                'pbo_threshold': 0.4,
                'penalty_weight': 0.25,
            }
        }
        v = CPCVValidator(config)
        assert v.enabled is True
        assert v.n_groups == 8
        assert v.n_test_groups == 3
        assert v.max_paths == 50
        assert v.pbo_threshold == 0.4
        assert v.penalty_weight == 0.25

    def test_cpcv_disabled_returns_skipped(self):
        """When CPCV is disabled, validate_strategies returns skipped."""
        from genetic_algorithm.evaluation.cpcv import CPCVValidator

        v = CPCVValidator({'cpcv': {'enabled': False}})
        result = v.validate_strategies({'a': np.array([1, 2, 3])})
        assert result['skipped'] is True
        assert result['pbo'] == 0.0

    def test_cpcv_validate_strategies_computes_pbo(self):
        """Full CPCV path: provide per-block results → get PBO score."""
        from genetic_algorithm.evaluation.cpcv import CPCVValidator

        config = {
            'cpcv': {
                'enabled': True,
                'n_groups': 4,
                'n_test_groups': 2,
                'max_paths': 100,
                'pbo_threshold': 0.5,
                'penalty_weight': 0.20,
            }
        }
        v = CPCVValidator(config)

        # Strategy A is consistently good; Strategy B is overfitted (great on block 0, bad elsewhere)
        results = {
            'strat_a': np.array([5.0, 4.0, 3.0, 6.0]),
            'strat_b': np.array([15.0, -2.0, -1.0, -3.0]),
        }
        out = v.validate_strategies(results)
        assert out['skipped'] is False
        assert 'pbo' in out
        assert 0.0 <= out['pbo'] <= 1.0
        assert 'penalty' in out
        assert 'per_strategy_oos' in out
        assert 'strat_a' in out['per_strategy_oos']
        assert 'strat_b' in out['per_strategy_oos']

    def test_cpcv_needs_at_least_two_strategies(self):
        """CPCV with only 1 strategy returns skipped."""
        from genetic_algorithm.evaluation.cpcv import CPCVValidator

        v = CPCVValidator({'cpcv': {'enabled': True, 'n_groups': 4, 'n_test_groups': 2}})
        result = v.validate_strategies({'only_one': np.array([1, 2, 3, 4])})
        assert result['skipped'] is True

    def test_cpcv_penalty_no_penalty_low_pbo(self):
        """cpcv_penalty returns 1.0 (no penalty) for PBO < 0.2."""
        from genetic_algorithm.evaluation.cpcv import cpcv_penalty

        # PBO < 0.2 → no penalty
        assert cpcv_penalty(0.0) == 1.0
        assert cpcv_penalty(0.1) == 1.0
        assert cpcv_penalty(0.19) == 1.0

    def test_cpcv_penalty_significant_at_high_pbo(self):
        """cpcv_penalty returns significant penalty for high PBO."""
        from genetic_algorithm.evaluation.cpcv import cpcv_penalty

        p_high = cpcv_penalty(0.9, pbo_threshold=0.5, penalty_weight=0.20)
        assert p_high < 1.0
        assert p_high >= 0.80  # At most 20% penalty

    def test_cpcv_penalty_monotonic(self):
        """Higher PBO → lower (more severe) penalty multiplier."""
        from genetic_algorithm.evaluation.cpcv import cpcv_penalty

        p_low = cpcv_penalty(0.3, pbo_threshold=0.5, penalty_weight=0.20)
        p_mid = cpcv_penalty(0.5, pbo_threshold=0.5, penalty_weight=0.20)
        p_high = cpcv_penalty(0.8, pbo_threshold=0.5, penalty_weight=0.20)
        assert p_low >= p_mid >= p_high


# ── Test 2: Backtester timerange_override ──


class TestTimerangeOverride:
    """Verify DirectBacktester respects timerange_override."""

    @patch('genetic_algorithm.evaluation.direct_backtester.DirectBacktester._validate_and_download_data')
    def test_create_backtest_config_uses_override(self, mock_validate):
        """_create_backtest_config uses timerange_override when provided."""
        from genetic_algorithm.evaluation.direct_backtester import DirectBacktester

        config = {
            'backtesting': {
                'pairs': ['BTC/USDT'],
                'timerange': '20240101-20260101',
                'stake_amount': 100,
                'max_open_trades': 3,
                'fee': 0.001,
                'exchange': 'binance',
                'enable_cache': False,
            }
        }
        bt = DirectBacktester(config)

        # Without override — uses config timerange
        cfg_default = bt._create_backtest_config('TestStrat')
        assert cfg_default['timerange'] == '20240101-20260101'

        # With override — uses override
        cfg_override = bt._create_backtest_config('TestStrat', timerange_override='20250101-20250301')
        assert cfg_override['timerange'] == '20250101-20250301'

    @patch('genetic_algorithm.evaluation.direct_backtester.DirectBacktester._validate_and_download_data')
    def test_cache_skipped_with_override(self, mock_validate):
        """Cache is bypassed when timerange_override is in use."""
        from genetic_algorithm.evaluation.direct_backtester import DirectBacktester

        config = {
            'backtesting': {
                'pairs': ['BTC/USDT'],
                'timerange': '20240101-20260101',
                'stake_amount': 100,
                'max_open_trades': 3,
                'fee': 0.001,
                'exchange': 'binance',
                'enable_cache': True,
            }
        }
        bt = DirectBacktester(config)

        # Mock cache
        bt.cache = MagicMock()
        bt.cache.get.return_value = None  # No cache hit

        # Mock _run_backtest_direct to avoid real backtesting
        mock_result = MagicMock()
        mock_result.success = True
        bt._run_backtest_direct = MagicMock(return_value=mock_result)

        # With override, cache.get should NOT be called
        bt.backtest_strategy('code', 'TestStrat', timerange_override='20250101-20250301')
        bt.cache.get.assert_not_called()
        bt.cache.put.assert_not_called()

    @patch('genetic_algorithm.evaluation.direct_backtester.DirectBacktester._validate_and_download_data')
    def test_cache_used_without_override(self, mock_validate):
        """Without timerange_override, cache IS consulted."""
        from genetic_algorithm.evaluation.direct_backtester import DirectBacktester

        config = {
            'backtesting': {
                'pairs': ['BTC/USDT'],
                'timerange': '20240101-20260101',
                'stake_amount': 100,
                'max_open_trades': 3,
                'fee': 0.001,
                'exchange': 'binance',
                'enable_cache': True,
            }
        }
        bt = DirectBacktester(config)

        # Mock cache with a cache hit
        from genetic_algorithm.evaluation.direct_backtester import BacktestResult
        cached = BacktestResult(success=True, strategy_name='TestStrat')
        bt.cache = MagicMock()
        bt.cache.get.return_value = cached

        result = bt.backtest_strategy('code', 'TestStrat')
        bt.cache.get.assert_called_once()
        assert result is cached


# ── Test 3: DSR column tracking ──


class TestDSRColumnTracking:
    """Verify generation_stats.csv includes DSR penalty columns."""

    def test_csv_columns_include_dsr(self):
        """_CSV_COLUMNS list contains avg_dsr_penalty and best_dsr_penalty."""
        from genetic_algorithm.utils.run_diagnostics import _CSV_COLUMNS

        assert 'avg_dsr_penalty' in _CSV_COLUMNS
        assert 'best_dsr_penalty' in _CSV_COLUMNS

    def test_csv_writer_writes_dsr_columns(self):
        """GenerationCSVWriter writes DSR columns when provided."""
        from genetic_algorithm.utils.run_diagnostics import GenerationCSVWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = GenerationCSVWriter(Path(tmpdir))
            writer.write_row({
                'generation': 0,
                'best_fitness': 0.75,
                'avg_dsr_penalty': 0.92,
                'best_dsr_penalty': 0.88,
            })
            writer.close()

            # Read back and verify
            csv_file = Path(tmpdir) / 'generation_stats.csv'
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                row = next(reader)
                assert row['avg_dsr_penalty'] == '0.92'
                assert row['best_dsr_penalty'] == '0.88'

    def test_csv_columns_order_preserved(self):
        """DSR columns appear after the existing columns."""
        from genetic_algorithm.utils.run_diagnostics import _CSV_COLUMNS

        # DSR columns should be at the end of the list
        dsr_idx = _CSV_COLUMNS.index('avg_dsr_penalty')
        best_dsr_idx = _CSV_COLUMNS.index('best_dsr_penalty')
        assert best_dsr_idx == dsr_idx + 1
        assert best_dsr_idx == len(_CSV_COLUMNS) - 1  # last column


# ── Test 4: Holdout phase timing ──


class TestHoldoutPhaseTiming:
    """Verify TimingTracker records holdout phase correctly."""

    def test_timing_tracker_records_holdout(self):
        """TimingTracker records holdout phase when start/end_phase are called."""
        from genetic_algorithm.utils.run_diagnostics import TimingTracker
        import time

        tt = TimingTracker()
        tt.start_generation(0)

        tt.start_phase('eval')
        time.sleep(0.01)
        tt.end_phase('eval')

        tt.start_phase('holdout')
        time.sleep(0.01)
        tt.end_phase('holdout')

        timing = tt.end_generation(0)
        assert timing.holdout_seconds > 0
        assert timing.eval_seconds > 0

    def test_timing_tracker_holdout_zero_without_end_phase(self):
        """Without end_phase('holdout'), holdout_seconds is 0."""
        from genetic_algorithm.utils.run_diagnostics import TimingTracker

        tt = TimingTracker()
        tt.start_generation(0)

        tt.start_phase('eval')
        tt.end_phase('eval')

        # Only start_phase('holdout') — no end_phase (the bug we fixed)
        tt.start_phase('holdout')
        # No tt.end_phase('holdout') call

        timing = tt.end_generation(0)
        assert timing.holdout_seconds == 0


# ── Test 5: Config validation ──


class TestConfigValidation:
    """Verify config files are valid YAML and have correct content."""

    def test_default_config_has_dsr_section(self):
        """ga_config.yaml contains deflated_sharpe section."""
        config_path = Path('genetic_algorithm/config/ga_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert 'deflated_sharpe' in config
        assert config['deflated_sharpe']['enabled'] is True
        assert config['deflated_sharpe']['penalty_weight'] == 0.15

    def test_default_config_has_cpcv_section(self):
        """ga_config.yaml contains cpcv section (disabled by default)."""
        config_path = Path('genetic_algorithm/config/ga_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert 'cpcv' in config
        assert config['cpcv']['enabled'] is False

    def test_benchmark_config_no_ltc(self):
        """ga_config_benchmark.yaml uses BNB/USDT instead of LTC/USDT."""
        config_path = Path('genetic_algorithm/config/ga_config_benchmark.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        pairs = config['backtesting']['pairs']
        assert 'LTC/USDT' not in pairs
        assert 'BNB/USDT' in pairs

    def test_feature_test_config_no_ltc(self):
        """ga_config_feature_test.yaml uses SOL/USDT instead of LTC/USDT."""
        config_path = Path('genetic_algorithm/config/ga_config_feature_test.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        pairs = config['backtesting']['pairs']
        assert 'LTC/USDT' not in pairs
        assert 'SOL/USDT' in pairs

    @pytest.mark.parametrize("config_name", [
        'ga_config_val_wf_parallel.yaml',
        'ga_config_val_regime.yaml',
        'ga_config_val_antioverfit.yaml',
        'ga_config_val_comprehensive.yaml',
    ])
    def test_validation_configs_are_valid_yaml(self, config_name):
        """Each validation config is parseable YAML with required sections."""
        config_path = Path(f'genetic_algorithm/config/{config_name}')
        assert config_path.exists(), f"{config_name} does not exist"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        # Required top-level sections
        assert 'genetic_algorithm' in config
        assert 'backtesting' in config
        assert 'fitness_weights' in config
        assert 'deflated_sharpe' in config
        assert 'cpcv' in config
        # Fitness weights sum to 1.0
        weights = config['fitness_weights']
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"weights sum to {total}"

    def test_val_antioverfit_has_cpcv_enabled(self):
        """Anti-overfit config has CPCV enabled."""
        config_path = Path('genetic_algorithm/config/ga_config_val_antioverfit.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config['cpcv']['enabled'] is True
        assert config['deflated_sharpe']['penalty_weight'] == 0.25

    def test_val_comprehensive_has_everything_enabled(self):
        """Comprehensive config has WF + regime + CPCV + MC all enabled."""
        config_path = Path('genetic_algorithm/config/ga_config_val_comprehensive.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config['walk_forward']['enabled'] is True
        assert config['regime_aware']['enabled'] is True
        assert config['cpcv']['enabled'] is True
        assert config['monte_carlo']['enabled'] is True
        assert config['deflated_sharpe']['enabled'] is True
        assert config['backtesting']['pairs'] == [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT'
        ]

    def test_val_wf_parallel_regime_disabled(self):
        """WF+parallel config has regime disabled (isolation)."""
        config_path = Path('genetic_algorithm/config/ga_config_val_wf_parallel.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config['regime_aware']['enabled'] is False
        assert config['walk_forward']['enabled'] is True
        assert config['parallel_evaluation']['num_workers'] == 8

    def test_val_regime_uses_ensemble(self):
        """Regime config uses ensemble detection method."""
        config_path = Path('genetic_algorithm/config/ga_config_val_regime.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config['regime_aware']['enabled'] is True
        assert config['regime_aware']['method'] == 'ensemble'
        assert config['regime_aware']['aggregation'] == 'mean'

    def test_all_validation_configs_use_available_pairs(self):
        """Validation configs only reference pairs with existing data files."""
        available_pairs = {'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT'}
        config_dir = Path('genetic_algorithm/config')
        for cfg_file in config_dir.glob('ga_config_val_*.yaml'):
            with open(cfg_file) as f:
                config = yaml.safe_load(f)
            pairs = set(config['backtesting']['pairs'])
            missing = pairs - available_pairs
            assert not missing, f"{cfg_file.name} uses unavailable pairs: {missing}"


# ── Test 6: Summary report includes CPCV ──


class TestSummaryReportCPCV:
    """Verify save_summary_report includes CPCV metrics when present."""

    def test_summary_report_writes_cpcv_fields(self):
        """When individual.metrics has cpcv_pbo, the summary report includes it."""
        from genetic_algorithm.run_ga import save_summary_report
        from unittest.mock import MagicMock

        # Create a mock individual with CPCV metrics
        mock_ind = MagicMock()
        mock_ind.id = 'test_strat_1'
        mock_ind.fitness = 0.75
        mock_ind.strategy_gene.generation = 5
        mock_ind.strategy_gene.individual_id = 3
        mock_ind.strategy_gene.indicators = []
        mock_ind.strategy_gene.entry_conditions = []
        mock_ind.strategy_gene.exit_conditions = []
        mock_ind.metrics = {
            'profit': 12.5,
            'sharpe_ratio': 2.1,
            'max_drawdown': 0.08,
            'win_rate': 0.55,
            'num_trades': 40,
            'cpcv_pbo': 0.35,
            'cpcv_penalty': 0.92,
            'cpcv_mean_oos': 4.2,
        }

        config = {
            'genetic_algorithm': {
                'population_size': 30,
                'generations': 10,
                'mutation_rate': 0.20,
                'crossover_rate': 0.75,
                'elite_size': 4,
            },
            'overfit_analysis': {'thresholds': {}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_summary_report([mock_ind], Path(tmpdir), config)

            # Find and read the summary file
            summary_files = list(Path(tmpdir).glob('ga_summary_*.txt'))
            assert len(summary_files) == 1
            content = summary_files[0].read_text()

            assert 'CPCV PBO: 0.350' in content
            assert 'CPCV Penalty: 0.920' in content
            assert 'CPCV Mean OOS: 4.200' in content

    def test_summary_report_skips_cpcv_when_absent(self):
        """When metrics lack cpcv_pbo, those lines don't appear."""
        from genetic_algorithm.run_ga import save_summary_report
        from unittest.mock import MagicMock

        mock_ind = MagicMock()
        mock_ind.id = 'test_strat_2'
        mock_ind.fitness = 0.60
        mock_ind.strategy_gene.generation = 2
        mock_ind.strategy_gene.individual_id = 1
        mock_ind.strategy_gene.indicators = []
        mock_ind.strategy_gene.entry_conditions = []
        mock_ind.strategy_gene.exit_conditions = []
        mock_ind.metrics = {
            'profit': 5.0,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.12,
            'win_rate': 0.48,
            'num_trades': 25,
        }

        config = {
            'genetic_algorithm': {
                'population_size': 20,
                'generations': 5,
                'mutation_rate': 0.15,
                'crossover_rate': 0.70,
                'elite_size': 3,
            },
            'overfit_analysis': {'thresholds': {}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_summary_report([mock_ind], Path(tmpdir), config)

            summary_files = list(Path(tmpdir).glob('ga_summary_*.txt'))
            assert len(summary_files) == 1
            content = summary_files[0].read_text()

            assert 'CPCV PBO' not in content
            assert 'CPCV Penalty' not in content
