import numpy as np

from research.pbo import choose_n_splits, probability_of_backtest_overfitting


class TestChooseNSplits:
    def test_returns_an_even_number_when_n_periods_has_an_even_divisor(self):
        n_periods = 16
        s = choose_n_splits(n_periods)
        assert s % 2 == 0
        assert s >= 2

    def test_never_exceeds_max_splits(self):
        assert choose_n_splits(100, max_splits=8) <= 8

    def test_returns_an_even_number_when_n_periods_is_prime(self):
        """n_periods=17 (prime) has no even divisor at all -- the old implementation
        fell back to s=2, which STILL doesn't divide 17 evenly, so the caller's
        n_periods % s != 0 guard silently failed every PBO computation closed to 1.0
        regardless of the real data (confirmed against three real walk-forward gate
        runs, all using a 17-window setup, all reporting a fake PBO=1.0). An exact
        divisor was never actually required: np.array_split handles a remainder by
        giving the first few blocks one extra column, so only evenness (for a
        symmetric IS/OOS half-split) and s >= 2 matter."""
        s = choose_n_splits(17)
        assert s % 2 == 0
        assert s >= 2


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

    def test_low_when_one_variant_dominates_every_period_and_n_periods_is_prime(self):
        """Direct reproduction of the real bug: same dominant-variant setup as
        test_low_when_one_variant_dominates_every_period, but with 17 (prime) periods
        instead of 16. Before the fix, this silently returned the fail-closed PBO=1.0
        regardless of variant 0's real, consistent edge -- exactly what happened on
        three real gate runs this session, all using a 17-window walk-forward setup."""
        rng = np.random.default_rng(11)
        n_variants, n_periods = 5, 17
        matrix = rng.normal(0.0, 0.01, size=(n_variants, n_periods))
        matrix[0] = rng.normal(0.05, 0.005, size=n_periods)
        result = probability_of_backtest_overfitting(matrix)
        assert result["pbo"] < 0.3
        assert result["n_splits"] >= 2, "must not have fallen back to the degenerate path"

    def test_explicit_odd_n_splits_fails_closed(self):
        """An odd split count can't form a symmetric half IS/half OOS split (s // 2
        would silently produce an uneven 2-vs-3-style split for e.g. s=5) -- an
        explicitly-passed odd n_splits is a caller error and must still fail closed,
        even though choose_n_splits itself never returns one."""
        rng = np.random.default_rng(3)
        matrix = rng.normal(0, 0.01, size=(6, 16))
        result = probability_of_backtest_overfitting(matrix, n_splits=5)
        assert result["pbo"] == 1.0
        assert result["n_splits"] == 0
