import numpy as np

from research.pbo import choose_n_splits, probability_of_backtest_overfitting


class TestChooseNSplits:
    def test_returns_an_even_divisor_of_n_periods(self):
        n_periods = 16
        s = choose_n_splits(n_periods)
        assert s % 2 == 0
        assert n_periods % s == 0

    def test_never_exceeds_max_splits(self):
        assert choose_n_splits(100, max_splits=8) <= 8


class TestProbabilityOfBacktestOverfitting:
    def test_bounded_in_unit_interval(self):
        rng = np.random.default_rng(3)
        matrix = rng.normal(0, 0.01, size=(6, 16))
        result = probability_of_backtest_overfitting(matrix)
        assert 0.0 <= result["pbo"] <= 1.0

    def test_low_when_one_variant_dominates_every_period(self):
        """Variant 0 has a real, consistent edge (positive mean, low noise) in every
        period; the others are pure noise. Picking the best-in-sample variant should
        reliably also do well out-of-sample -> low overfitting probability."""
        rng = np.random.default_rng(11)
        n_variants, n_periods = 5, 16
        matrix = rng.normal(0.0, 0.01, size=(n_variants, n_periods))
        matrix[0] = rng.normal(0.05, 0.005, size=n_periods)
        result = probability_of_backtest_overfitting(matrix)
        assert result["pbo"] < 0.3

    def test_near_coin_flip_when_all_variants_are_pure_noise(self):
        """No variant has a real edge -> which one looks best in-sample is arbitrary
        and should NOT reliably predict out-of-sample rank. PBO should be high."""
        rng = np.random.default_rng(6)
        matrix = rng.normal(0.0, 0.01, size=(6, 16))
        result = probability_of_backtest_overfitting(matrix)
        assert result["pbo"] > 0.35

    def test_degenerate_input_fails_closed(self):
        result = probability_of_backtest_overfitting(np.zeros((1, 2)))
        assert result["pbo"] == 1.0
