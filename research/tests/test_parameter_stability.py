# research/tests/test_parameter_stability.py
import numpy as np
import pytest

from research.parameter_stability import parameter_stability


def test_all_variants_profitable_returns_one():
    matrix = np.array([[0.01, 0.02, 0.01], [0.03, 0.01, 0.02]])
    assert parameter_stability(matrix) == pytest.approx(1.0)


def test_no_variants_profitable_returns_zero():
    matrix = np.array([[-0.01, -0.02, -0.01], [-0.03, -0.01, -0.02]])
    assert parameter_stability(matrix) == pytest.approx(0.0)


def test_mixed_variants_returns_exact_fraction():
    # row means: 0.01 (>0), -0.01 (<0), 0.04/3 (>0, cells straddle zero), -0.05/3 (<0)
    # -> 2 of 4 variants profitable
    matrix = np.array(
        [
            [0.01, 0.02, 0.00],
            [-0.01, -0.02, 0.00],
            [0.05, -0.03, 0.02],
            [-0.05, 0.01, -0.01],
        ]
    )
    assert parameter_stability(matrix) == pytest.approx(0.5)


def test_row_mean_exactly_zero_is_not_profitable():
    # strict ">", not ">=" -- a row sitting exactly on zero doesn't count as profitable.
    matrix = np.array([[0.0, 0.0], [0.01, 0.01]])
    assert parameter_stability(matrix) == pytest.approx(0.5)


def test_single_variant_grid_fails_open_to_one():
    matrix = np.array([[-0.5, -0.5, -0.5]])  # unprofitable, but no region to test
    assert parameter_stability(matrix) == pytest.approx(1.0)


def test_raises_on_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        parameter_stability(np.array([0.01, 0.02]))


def test_raises_on_zero_rows():
    with pytest.raises(ValueError, match="2-D"):
        parameter_stability(np.zeros((0, 3)))
