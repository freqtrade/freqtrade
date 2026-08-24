import numpy as np

from research.statistics import benjamini_hochberg, deflated_sharpe_ratio, permutation_test


class TestDeflatedSharpeRatio:
    def test_returns_zero_on_degenerate_n_obs(self):
        assert deflated_sharpe_ratio(sharpe_ratio=2.0, n_obs=1, n_trials=10) == 0.0

    def test_bounded_in_unit_interval(self):
        for sharpe in (-3.0, -0.5, 0.0, 0.5, 1.5, 3.0):
            for n_obs in (10, 100, 1000):
                for n_trials in (1, 10, 1000):
                    result = deflated_sharpe_ratio(sharpe, n_obs, n_trials)
                    assert 0.0 <= result <= 1.0, (sharpe, n_obs, n_trials, result)

    def test_more_trials_never_increases_the_score(self):
        """Regression test for the exact bug this project's spec (§16, lesson 1) flags:
        a DSR with no real trial-count term would score n_trials=1 and n_trials=1000
        identically. Holding sharpe and n_obs fixed, searching harder must not look
        MORE convincing."""
        few_trials = deflated_sharpe_ratio(sharpe_ratio=2.0, n_obs=500, n_trials=1)
        many_trials = deflated_sharpe_ratio(sharpe_ratio=2.0, n_obs=500, n_trials=1000)
        assert many_trials <= few_trials

    def test_more_observations_never_decreases_the_score(self):
        """Regression test for the same lesson from the other axis: a 6-trade and a
        10,000-trade backtest must not deflate identically."""
        few_obs = deflated_sharpe_ratio(sharpe_ratio=1.0, n_obs=20, n_trials=50)
        many_obs = deflated_sharpe_ratio(sharpe_ratio=1.0, n_obs=5000, n_trials=50)
        assert many_obs >= few_obs


class TestBenjaminiHochberg:
    def test_matches_hand_worked_example(self):
        # sorted ascending already; thresholds are (rank/m)*q = 0.01,0.02,0.03,0.04,0.05
        p_values = [0.01, 0.02, 0.03, 0.04, 0.5]
        result = benjamini_hochberg(p_values, q=0.05)
        assert result == [True, True, True, True, False]

    def test_empty_input_returns_empty_output(self):
        assert benjamini_hochberg([], q=0.05) == []

    def test_all_p_values_above_q_rejects_nothing(self):
        assert benjamini_hochberg([0.9, 0.8, 0.99], q=0.05) == [False, False, False]


class TestPermutationTest:
    def test_p_value_in_unit_interval(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0, 0.01, 40)
        observed = float(returns.mean() / returns.std())
        p = permutation_test(observed, returns, n_permutations=200, seed=1)
        assert 0.0 <= p <= 1.0

    def test_low_p_value_for_a_strong_consistent_positive_edge(self):
        """All-positive, low-noise returns: virtually every sign-flip permutation
        produces a worse Sharpe than the unflipped (real) series, since flipping any
        positive return can only hurt the mean. p must be small."""
        rng = np.random.default_rng(42)
        returns = 0.02 + rng.normal(0, 0.001, 30)
        observed = float(returns.mean() / returns.std())
        p = permutation_test(observed, returns, n_permutations=2000, seed=42)
        assert p < 0.05

    def test_seeded_calls_are_reproducible(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(0.001, 0.02, 50)
        observed = float(returns.mean() / returns.std())
        p1 = permutation_test(observed, returns, n_permutations=500, seed=123)
        p2 = permutation_test(observed, returns, n_permutations=500, seed=123)
        assert p1 == p2
