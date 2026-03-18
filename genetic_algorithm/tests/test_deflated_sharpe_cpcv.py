"""
Tests for Deflated Sharpe Ratio (DSR) and CPCV/PBO modules.

Validates:
- DSR computation correctness with known inputs
- DSR penalty curve behavior
- DSRTracker accumulation
- CPCV path generation
- PBO computation with synthetic data
- CPCV penalty curve
"""

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genetic_algorithm.evaluation.deflated_sharpe import (
    expected_max_sharpe,
    calculate_dsr,
    compute_return_statistics,
    deflated_sharpe_penalty,
    DSRTracker,
)
from genetic_algorithm.evaluation.cpcv import (
    generate_cpcv_paths,
    create_time_blocks,
    get_train_test_indices,
    compute_pbo,
    cpcv_penalty,
    CPCVValidator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# DSR Tests
# ═══════════════════════════════════════════════════════════════════

class TestExpectedMaxSharpe:
    """Tests for expected_max_sharpe()."""

    def test_single_trial_returns_zero(self):
        """With 1 trial, expected max should be zero (no selection bias)."""
        result = expected_max_sharpe(n_trials=1)
        assert result == 0.0

    def test_increases_with_trials(self):
        """More trials → higher expected max SR (more selection bias)."""
        sr_10 = expected_max_sharpe(n_trials=10)
        sr_100 = expected_max_sharpe(n_trials=100)
        sr_1000 = expected_max_sharpe(n_trials=1000)
        assert sr_10 < sr_100 < sr_1000

    def test_positive_for_multiple_trials(self):
        """Expected max SR should be positive for N > 1."""
        assert expected_max_sharpe(n_trials=5) > 0
        assert expected_max_sharpe(n_trials=50) > 0

    def test_known_value_approx(self):
        """For N=100, unit variance, normal returns, expected max ≈ 2.3-2.7."""
        result = expected_max_sharpe(n_trials=100, variance_sr=1.0)
        assert 1.5 < result < 3.5  # Broad range for approximation


class TestCalculateDSR:
    """Tests for calculate_dsr()."""

    def test_high_sharpe_high_dsr(self):
        """A genuinely high Sharpe with few trials should have high DSR."""
        dsr = calculate_dsr(
            observed_sharpe=3.0,
            n_trials=10,
            n_returns=200,
        )
        assert dsr > 0.8

    def test_mediocre_sharpe_many_trials_low_dsr(self):
        """Mediocre Sharpe after many trials → low DSR (selection bias)."""
        dsr = calculate_dsr(
            observed_sharpe=1.5,
            n_trials=500,
            n_returns=50,
        )
        assert dsr < 0.5

    def test_insufficient_data(self):
        """With very few returns, DSR should be 0."""
        dsr = calculate_dsr(
            observed_sharpe=2.0,
            n_trials=10,
            n_returns=5,  # Too few
        )
        assert dsr == 0.0

    def test_zero_sharpe_low_dsr(self):
        """Zero observed Sharpe → low DSR regardless of trials."""
        dsr = calculate_dsr(
            observed_sharpe=0.0,
            n_trials=10,
            n_returns=100,
        )
        assert dsr < 0.5

    def test_fat_tails_reduce_dsr(self):
        """High kurtosis (fat tails) should reduce DSR."""
        dsr_normal = calculate_dsr(
            observed_sharpe=2.0,
            n_trials=20,
            n_returns=100,
            kurtosis=3.0,  # Normal
        )
        dsr_fat = calculate_dsr(
            observed_sharpe=2.0,
            n_trials=20,
            n_returns=100,
            kurtosis=8.0,  # Fat tails
        )
        assert dsr_fat < dsr_normal

    def test_dsr_in_range(self):
        """DSR should always be in [0, 1]."""
        for sr in [-2.0, 0.0, 1.0, 3.0, 10.0]:
            for n in [1, 10, 100, 1000]:
                dsr = calculate_dsr(
                    observed_sharpe=sr,
                    n_trials=n,
                    n_returns=100,
                )
                assert 0.0 <= dsr <= 1.0, f"DSR={dsr} out of range for SR={sr}, N={n}"


class TestComputeReturnStatistics:
    """Tests for compute_return_statistics()."""

    def test_empty_trades(self):
        stats = compute_return_statistics([])
        assert stats['n_returns'] == 0
        assert stats['sharpe_ratio'] == 0.0

    def test_single_trade(self):
        stats = compute_return_statistics([{'profit_ratio': 0.05}])
        assert stats['n_returns'] == 1
        assert stats['mean'] == 0.05

    def test_normal_distribution(self):
        """With many normally-distributed returns, kurtosis ≈ 3."""
        np.random.seed(42)
        trades = [{'profit_ratio': r} for r in np.random.normal(0.001, 0.02, 500)]
        stats = compute_return_statistics(trades)
        assert stats['n_returns'] == 500
        assert 2.0 < stats['kurtosis'] < 5.0  # Near-normal

    def test_handles_numeric_list(self):
        """Should handle a list of raw numbers."""
        stats = compute_return_statistics([0.01, 0.02, -0.01, 0.03])
        assert stats['n_returns'] == 4


class TestDeflatedSharpePenalty:
    """Tests for deflated_sharpe_penalty()."""

    def test_high_dsr_no_penalty(self):
        """Genuinely strong Sharpe → penalty ≈ 1.0."""
        penalty, info = deflated_sharpe_penalty(
            observed_sharpe=4.0,
            n_trials=10,
            n_returns=200,
        )
        assert penalty > 0.95
        assert not info['dsr_skipped']

    def test_low_dsr_applies_penalty(self):
        """Weak Sharpe after many trials → penalty < 1.0."""
        penalty, info = deflated_sharpe_penalty(
            observed_sharpe=0.5,
            n_trials=500,
            n_returns=30,
            penalty_weight=0.15,
        )
        assert penalty < 1.0
        assert penalty >= 0.85  # Floor = 1 - penalty_weight

    def test_insufficient_data_skip(self):
        """With too few returns or trials, penalty should be skipped."""
        penalty, info = deflated_sharpe_penalty(
            observed_sharpe=2.0,
            n_trials=1,
            n_returns=100,
        )
        assert penalty == 1.0
        assert info['dsr_skipped']

    def test_penalty_in_range(self):
        """Penalty should always be in [1 - weight, 1.0]."""
        weight = 0.15
        for sr in [0.0, 1.0, 2.0, 5.0]:
            penalty, _ = deflated_sharpe_penalty(
                observed_sharpe=sr,
                n_trials=50,
                n_returns=100,
                penalty_weight=weight,
            )
            assert (1.0 - weight - 0.01) <= penalty <= 1.0


class TestDSRTracker:
    """Tests for DSRTracker class."""

    def test_initialization(self):
        tracker = DSRTracker({})
        assert tracker.enabled  # Default enabled
        assert tracker.n_trials == 1  # min floor of 1 (prevents div-by-zero)

    def test_register_evaluation(self):
        tracker = DSRTracker({})
        tracker.register_evaluation("hash1")
        tracker.register_evaluation("hash2")
        tracker.register_evaluation("hash1")  # Duplicate — not counted again
        assert tracker.n_trials == 2  # Only unique strategy hashes
        assert len(tracker._strategy_hashes) == 2

    def test_disabled(self):
        tracker = DSRTracker({'deflated_sharpe': {'enabled': False}})
        penalty, info = tracker.compute_penalty(2.0, 100)
        assert penalty == 1.0
        assert info['dsr_skipped']

    def test_reset(self):
        tracker = DSRTracker({})
        tracker.register_evaluation("hash1")
        tracker.reset()
        assert tracker.n_trials == 1  # min floor of 1 after reset


# ═══════════════════════════════════════════════════════════════════
# CPCV Tests
# ═══════════════════════════════════════════════════════════════════

class TestGenerateCPCVPaths:
    """Tests for generate_cpcv_paths()."""

    def test_basic_combination_count(self):
        """C(6,2) = 15 paths."""
        paths = generate_cpcv_paths(n_groups=6, n_test_groups=2)
        assert len(paths) == 15

    def test_path_structure(self):
        """Each path should have train and test groups that partition all groups."""
        paths = generate_cpcv_paths(n_groups=6, n_test_groups=2)
        for train, test in paths:
            assert len(test) == 2
            assert len(train) == 4
            assert set(train + test) == set(range(6))

    def test_max_paths_subsampling(self):
        """With max_paths, should return limited number."""
        paths = generate_cpcv_paths(n_groups=10, n_test_groups=5, max_paths=20)
        assert len(paths) == 20

    def test_single_test_group(self):
        """C(5,1) = 5 paths."""
        paths = generate_cpcv_paths(n_groups=5, n_test_groups=1)
        assert len(paths) == 5


class TestCreateTimeBlocks:
    """Tests for create_time_blocks()."""

    def test_basic_blocks(self):
        info = create_time_blocks(n_samples=1000, n_groups=5)
        assert len(info['blocks']) == 5
        assert info['blocks'][0] == (0, 200)
        assert info['blocks'][-1][1] == 1000

    def test_purge_embargo_sizes(self):
        info = create_time_blocks(n_samples=10000, n_groups=5, purge_pct=0.02, embargo_pct=0.01)
        assert info['purge_size'] == 200
        assert info['embargo_size'] == 100


class TestGetTrainTestIndices:
    """Tests for get_train_test_indices()."""

    def test_no_overlap(self):
        """Train and test indices should not overlap."""
        info = create_time_blocks(n_samples=500, n_groups=5)
        train_idx, test_idx = get_train_test_indices(info, [0, 1, 2], [3, 4])
        assert len(set(train_idx) & set(test_idx)) == 0

    def test_purge_removes_boundary(self):
        """Purge should remove train samples near test boundaries."""
        info = create_time_blocks(n_samples=500, n_groups=5, purge_pct=0.02)
        train_idx, test_idx = get_train_test_indices(info, [0, 1, 2], [3, 4])
        
        # Test block 3 starts at 300. Purge of 10 samples means
        # train indices 290-299 should be removed
        test_start = 300
        purge_range = set(range(test_start - info['purge_size'], test_start))
        assert len(purge_range & set(train_idx)) == 0


class TestComputePBO:
    """Tests for compute_pbo()."""

    def test_no_overfitting(self):
        """When IS winner is also OOS winner, PBO should be low."""
        np.random.seed(42)
        n_paths = 20
        n_strats = 5
        
        # Strategy 0 is genuinely best in both IS and OOS
        is_perfs = np.random.uniform(0, 1, (n_paths, n_strats))
        oos_perfs = is_perfs + np.random.normal(0, 0.1, (n_paths, n_strats))
        
        # Make strategy 0 consistently the best
        is_perfs[:, 0] += 2.0
        oos_perfs[:, 0] += 2.0
        
        pbo, details = compute_pbo(is_perfs, oos_perfs)
        assert pbo < 0.3  # Low overfitting

    def test_clear_overfitting(self):
        """When IS and OOS are uncorrelated, PBO should be high."""
        np.random.seed(42)
        n_paths = 50
        n_strats = 10
        
        # Completely random — no relationship between IS and OOS
        is_perfs = np.random.uniform(0, 1, (n_paths, n_strats))
        oos_perfs = np.random.uniform(0, 1, (n_paths, n_strats))
        
        pbo, details = compute_pbo(is_perfs, oos_perfs)
        assert pbo > 0.3  # Should show overfitting

    def test_insufficient_strategies(self):
        """With < 2 strategies, PBO should be 0."""
        pbo, details = compute_pbo(np.array([[1.0]]), np.array([[1.0]]))
        assert pbo == 0.0

    def test_pbo_in_range(self):
        """PBO should always be in [0, 1]."""
        np.random.seed(42)
        is_perfs = np.random.uniform(0, 1, (30, 8))
        oos_perfs = np.random.uniform(0, 1, (30, 8))
        pbo, _ = compute_pbo(is_perfs, oos_perfs)
        assert 0.0 <= pbo <= 1.0


class TestCPCVPenalty:
    """Tests for cpcv_penalty()."""

    def test_low_pbo_no_penalty(self):
        assert cpcv_penalty(pbo=0.1) == 1.0

    def test_high_pbo_penalty(self):
        penalty = cpcv_penalty(pbo=0.9, penalty_weight=0.20)
        assert penalty < 0.85

    def test_penalty_in_range(self):
        for pbo in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
            p = cpcv_penalty(pbo, penalty_weight=0.20)
            assert 0.8 <= p <= 1.0


class TestCPCVValidator:
    """Tests for CPCVValidator class."""

    def test_disabled_skips(self):
        validator = CPCVValidator({'cpcv': {'enabled': False}})
        result = validator.validate_strategies({'s1': np.zeros(6), 's2': np.zeros(6)})
        assert result['skipped']

    def test_enabled_runs(self):
        validator = CPCVValidator({
            'cpcv': {
                'enabled': True,
                'n_groups': 6,
                'n_test_groups': 2,
                'max_paths': 15,
            }
        })
        np.random.seed(42)
        results = {
            f's{i}': np.random.uniform(0, 1, 6)
            for i in range(5)
        }
        output = validator.validate_strategies(results)
        assert not output['skipped']
        assert 0.0 <= output['pbo'] <= 1.0
        assert output['n_strategies'] == 5

    def test_quick_pbo_estimate(self):
        validator = CPCVValidator({})
        pbo = validator.quick_pbo_estimate(
            in_sample_fitness=[0.8, 0.6, 0.4, 0.2],
            out_of_sample_fitness=[0.3, 0.7, 0.5, 0.1],
        )
        assert 0.0 <= pbo <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
