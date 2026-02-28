"""
Tests for Monte Carlo Robustness Scoring

Tests bootstrap_trades, shuffle_trade_order, jitter_slippage,
run_monte_carlo, and edge cases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from genetic_algorithm.evaluation.monte_carlo import (
    bootstrap_trades,
    shuffle_trade_order,
    jitter_slippage,
    run_monte_carlo,
    _build_result,
    MonteCarloResult,
)


# ============================================================================
# Fixtures / Helpers
# ============================================================================

def _make_trades(profits):
    """Create a list of trade dicts from a list of profit_ratio values."""
    return [{'profit_ratio': p} for p in profits]


PROFITABLE_TRADES = _make_trades([0.05, 0.03, 0.02, 0.04, 0.01, 0.03, 0.02, 0.06, 0.01, 0.04])
LOSING_TRADES = _make_trades([-0.05, -0.03, -0.02, -0.04, -0.01, -0.03, -0.02, -0.06, -0.01, -0.04])
MIXED_TRADES = _make_trades([0.05, -0.03, 0.02, -0.04, 0.01, 0.06, -0.02, 0.04, -0.01, 0.03])
SINGLE_TRADE = _make_trades([0.05])


# ============================================================================
# _build_result
# ============================================================================

class TestBuildResult:
    def test_empty_profits(self):
        result = _build_result([], 0)
        assert result.robustness_score == 0.0
        assert result.num_permutations == 0
    
    def test_all_positive(self):
        result = _build_result([10.0, 20.0, 30.0, 40.0, 50.0], 5)
        assert result.robustness_score == 1.0  # all profitable
        assert result.mean_profit == 30.0
        assert result.num_permutations == 5
        assert result.profit_p5 <= result.profit_p95
    
    def test_all_negative(self):
        result = _build_result([-10.0, -20.0, -30.0], 3)
        assert result.robustness_score == 0.0
        assert result.mean_profit < 0
    
    def test_mixed(self):
        result = _build_result([10.0, -5.0, 20.0, -3.0], 4)
        assert 0.0 < result.robustness_score < 1.0
        assert result.robustness_score == 0.5  # 2/4 profitable
    
    def test_std_with_identical_values(self):
        result = _build_result([5.0, 5.0, 5.0], 3)
        assert result.profit_std == 0.0
        assert result.mean_sharpe == 0.0  # std=0 -> sharpe=0 (not inf)
    
    def test_percentiles_ordered(self):
        profits = list(range(-50, 51))
        result = _build_result(profits, len(profits))
        assert result.profit_p5 <= result.mean_profit <= result.profit_p95


# ============================================================================
# bootstrap_trades
# ============================================================================

class TestBootstrapTrades:
    def test_empty_trades(self):
        result = bootstrap_trades([])
        assert result.robustness_score == 0.0
        assert result.num_permutations == 0
    
    def test_profitable_trades_high_robustness(self):
        result = bootstrap_trades(PROFITABLE_TRADES, num_permutations=50, random_seed=42)
        assert result.robustness_score >= 0.9  # All trades are positive
        assert result.mean_profit > 0
        assert len(result.permutation_profits) == 50
    
    def test_losing_trades_zero_robustness(self):
        result = bootstrap_trades(LOSING_TRADES, num_permutations=50, random_seed=42)
        assert result.robustness_score <= 0.1
        assert result.mean_profit < 0
    
    def test_reproducible_with_seed(self):
        r1 = bootstrap_trades(MIXED_TRADES, num_permutations=20, random_seed=123)
        r2 = bootstrap_trades(MIXED_TRADES, num_permutations=20, random_seed=123)
        assert r1.permutation_profits == r2.permutation_profits
        assert r1.robustness_score == r2.robustness_score
    
    def test_different_seeds_different_results(self):
        r1 = bootstrap_trades(MIXED_TRADES, num_permutations=20, random_seed=1)
        r2 = bootstrap_trades(MIXED_TRADES, num_permutations=20, random_seed=2)
        assert r1.permutation_profits != r2.permutation_profits
    
    def test_single_trade(self):
        result = bootstrap_trades(SINGLE_TRADE, num_permutations=10, random_seed=42)
        # Bootstrapping single trade should always produce the same profit
        assert result.robustness_score == 1.0  # positive trade
        assert all(p > 0 for p in result.permutation_profits)
    
    def test_sample_fraction(self):
        result = bootstrap_trades(MIXED_TRADES, num_permutations=20, 
                                   sample_fraction=0.5, random_seed=42)
        assert result.num_permutations == 20
        assert len(result.permutation_profits) == 20
    
    def test_zero_profit_trades(self):
        """Trades with exactly 0 profit should not count as profitable."""
        trades = _make_trades([0.0, 0.0, 0.0])
        result = bootstrap_trades(trades, num_permutations=10, random_seed=42)
        assert result.robustness_score == 0.0  # 0 is not > 0


# ============================================================================
# shuffle_trade_order
# ============================================================================

class TestShuffleTradeOrder:
    def test_empty_trades(self):
        result = shuffle_trade_order([])
        assert result.robustness_score == 0.0
    
    def test_profitable_trades(self):
        result = shuffle_trade_order(PROFITABLE_TRADES, num_permutations=50, random_seed=42)
        # With compounding, all-positive trades should still be profitable in any order
        assert result.robustness_score >= 0.9
        assert result.mean_profit > 0
    
    def test_losing_trades(self):
        result = shuffle_trade_order(LOSING_TRADES, num_permutations=50, random_seed=42)
        assert result.robustness_score <= 0.1
        assert result.mean_profit < 0
    
    def test_compounding_creates_variance(self):
        """Different orderings should produce different profits due to compounding."""
        trades = _make_trades([0.10, -0.05, 0.15, -0.08, 0.20])
        result = shuffle_trade_order(trades, num_permutations=100, random_seed=42)
        assert result.profit_std > 0  # variance from order effects
    
    def test_single_trade(self):
        result = shuffle_trade_order(SINGLE_TRADE, num_permutations=10, random_seed=42)
        # Single trade: shuffling doesn't change anything
        assert all(abs(p - result.permutation_profits[0]) < 0.001 
                    for p in result.permutation_profits)
    
    def test_reproducible_with_seed(self):
        r1 = shuffle_trade_order(MIXED_TRADES, num_permutations=20, random_seed=99)
        r2 = shuffle_trade_order(MIXED_TRADES, num_permutations=20, random_seed=99)
        assert r1.permutation_profits == r2.permutation_profits


# ============================================================================
# jitter_slippage
# ============================================================================

class TestJitterSlippage:
    def test_empty_trades(self):
        result = jitter_slippage([])
        assert result.robustness_score == 0.0
    
    def test_profitable_trades_survive_small_jitter(self):
        result = jitter_slippage(PROFITABLE_TRADES, slippage_std=0.0001,
                                  num_permutations=50, random_seed=42)
        # Strong profits should survive tiny slippage
        assert result.robustness_score >= 0.8
    
    def test_losing_trades_stay_losing(self):
        result = jitter_slippage(LOSING_TRADES, slippage_std=0.0005,
                                  num_permutations=50, random_seed=42)
        assert result.robustness_score <= 0.2
    
    def test_large_slippage_destroys_weak_edge(self):
        """Very small positive edge should be destroyed by large slippage."""
        tiny_edge = _make_trades([0.001] * 20)  # tiny positive edge
        result = jitter_slippage(tiny_edge, slippage_std=0.005,
                                  num_permutations=100, random_seed=42)
        # Large slippage relative to edge should destroy most permutations
        assert result.robustness_score < 0.9
    
    def test_reproducible_with_seed(self):
        r1 = jitter_slippage(MIXED_TRADES, slippage_std=0.0005,
                              num_permutations=20, random_seed=77)
        r2 = jitter_slippage(MIXED_TRADES, slippage_std=0.0005,
                              num_permutations=20, random_seed=77)
        assert r1.permutation_profits == r2.permutation_profits
    
    def test_zero_slippage_std(self):
        """Zero slippage std should produce identical results each time."""
        result = jitter_slippage(PROFITABLE_TRADES, slippage_std=0.0,
                                  num_permutations=10, random_seed=42)
        # All permutations should give the same total (Gaussian with std=0 is just 0)
        assert result.profit_std < 0.01


# ============================================================================
# run_monte_carlo  
# ============================================================================

class TestRunMonteCarlo:
    def test_empty_trades(self):
        config = {'num_permutations': 50}
        result = run_monte_carlo([], config)
        assert result.robustness_score == 0.0
    
    def test_all_methods(self):
        config = {
            'num_permutations': 60,
            'methods': ['bootstrap', 'shuffle', 'slippage_jitter'],
            'slippage_std': 0.0005,
            'sample_fraction': 1.0,
            'random_seed': 42,
        }
        result = run_monte_carlo(PROFITABLE_TRADES, config)
        assert result.robustness_score > 0
        assert result.num_permutations > 0
        # 60 permutations / 3 methods = 20 each, so total = 60
        assert len(result.permutation_profits) == 60
    
    def test_single_method(self):
        config = {
            'num_permutations': 30,
            'methods': ['bootstrap'],
            'random_seed': 42,
        }
        result = run_monte_carlo(MIXED_TRADES, config)
        assert result.num_permutations == 30
        assert len(result.permutation_profits) == 30
    
    def test_default_methods(self):
        """Default should use all 3 methods."""
        config = {'num_permutations': 30, 'random_seed': 42}
        result = run_monte_carlo(PROFITABLE_TRADES, config)
        assert result.num_permutations > 0
    
    def test_no_methods_empty_result(self):
        config = {'methods': []}
        result = run_monte_carlo(PROFITABLE_TRADES, config)
        assert result.robustness_score == 0.0
    
    def test_result_fields_populated(self):
        config = {
            'num_permutations': 30,
            'methods': ['bootstrap'],
            'random_seed': 42,
        }
        result = run_monte_carlo(PROFITABLE_TRADES, config)
        assert isinstance(result, MonteCarloResult)
        assert result.profit_p5 <= result.profit_p95
        assert result.profit_std >= 0
        assert 0.0 <= result.robustness_score <= 1.0


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_trade_missing_profit_ratio_key(self):
        """Trades without 'profit_ratio' should use 0.0 default."""
        trades = [{'some_key': 1}, {'profit_ratio': 0.05}]
        result = bootstrap_trades(trades, num_permutations=10, random_seed=42)
        assert result.num_permutations == 10
    
    def test_very_large_num_permutations(self):
        """Should handle many permutations without crashing."""
        result = bootstrap_trades(MIXED_TRADES, num_permutations=1000, random_seed=42)
        assert result.num_permutations == 1000
        assert len(result.permutation_profits) == 1000
    
    def test_negative_profit_ratio_edge(self):
        """Trades with profit_ratio = -1.0 (100% loss) shouldn't crash compounding."""
        trades = _make_trades([0.10, -0.99, 0.05])  # near-total loss
        result = shuffle_trade_order(trades, num_permutations=20, random_seed=42)
        assert result.num_permutations == 20
