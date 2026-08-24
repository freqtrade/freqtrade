# research/tests/test_scoring.py
import pytest

from research.gate import GateResult
from research.scoring import WEIGHTS, robustness_score, strategy_report


def _core_result(deflated_sharpe=0.90, permutation_p=0.02, pbo=0.15, passed=True, **kwargs):
    """A GateResult with only the three always-present statistics populated,
    unless overridden."""
    return GateResult(
        strategy_id="TestStrategy",
        passed=passed,
        deflated_sharpe=deflated_sharpe,
        permutation_p=permutation_p,
        pbo=pbo,
        mean_test_sharpe=1.1,
        n_trials=10,
        reasons=[],
        **kwargs,
    )


def test_robustness_score_with_only_core_stats():
    result = _core_result()

    score = robustness_score(result)

    # Independently computed, not calling robustness_score recursively.
    weighted_sum = (
        WEIGHTS["deflated_sharpe"] * 0.90
        + WEIGHTS["significance"] * (1.0 - 0.02)
        + WEIGHTS["pbo_inverse"] * (1.0 - 0.15)
    )
    weight_total = WEIGHTS["deflated_sharpe"] + WEIGHTS["significance"] + WEIGHTS["pbo_inverse"]
    assert score == pytest.approx(weighted_sum / weight_total)
    assert score == pytest.approx(0.9088235294117647)


def test_robustness_score_includes_cost_sensitivity_when_fee_sensitivity_present():
    result = _core_result(
        fee_sensitivity={
            1.0: {"mean_test_sharpe": 0.8, "deflated_sharpe": 0.90, "n_windows": 5},
            1.5: {"mean_test_sharpe": 0.4, "deflated_sharpe": 0.60, "n_windows": 5},
        }
    )

    score = robustness_score(result)

    assert score == pytest.approx(0.8833333333333333)


def test_robustness_score_includes_regime_consistency_when_regime_breakdown_present():
    result = _core_result(
        regime_breakdown={
            "Bull/High": {
                "n_windows": 2,
                "n_trades": 10,
                "mean_test_sharpe": 0.5,
                "total_return": 0.01,
            },
            "Bull/Low": {
                "n_windows": 1,
                "n_trades": 5,
                "mean_test_sharpe": 0.3,
                "total_return": 0.005,
            },
            "Bear/Low": {
                "n_windows": 1,
                "n_trades": 4,
                "mean_test_sharpe": 0.1,
                "total_return": 0.002,
            },
            "Bear/High": {
                "n_windows": 1,
                "n_trades": 3,
                "mean_test_sharpe": -0.2,
                "total_return": -0.004,
            },
        }
    )

    score = robustness_score(result)

    assert score == pytest.approx(0.9)


def test_robustness_score_with_both_fee_sensitivity_and_regime_breakdown():
    result = _core_result(
        fee_sensitivity={
            1.0: {"mean_test_sharpe": 0.8, "deflated_sharpe": 0.90, "n_windows": 5},
            1.5: {"mean_test_sharpe": 0.4, "deflated_sharpe": 0.60, "n_windows": 5},
        },
        regime_breakdown={
            "Bull/High": {
                "n_windows": 2,
                "n_trades": 10,
                "mean_test_sharpe": 0.5,
                "total_return": 0.01,
            },
            "Bull/Low": {
                "n_windows": 1,
                "n_trades": 5,
                "mean_test_sharpe": 0.3,
                "total_return": 0.005,
            },
            "Bear/Low": {
                "n_windows": 1,
                "n_trades": 4,
                "mean_test_sharpe": 0.1,
                "total_return": 0.002,
            },
            "Bear/High": {
                "n_windows": 1,
                "n_trades": 3,
                "mean_test_sharpe": -0.2,
                "total_return": -0.004,
            },
        },
    )

    score = robustness_score(result)

    assert score == pytest.approx(0.8766666666666667)


def test_robustness_score_cost_sensitivity_single_multiplier_is_one():
    result = _core_result(
        fee_sensitivity={1.0: {"mean_test_sharpe": 0.5, "deflated_sharpe": 0.5, "n_windows": 5}}
    )

    score = robustness_score(result)

    assert score == pytest.approx(0.9184210526315789)


def test_robustness_score_cost_sensitivity_zero_baseline_is_zero():
    result = _core_result(
        fee_sensitivity={
            1.0: {"mean_test_sharpe": 0.0, "deflated_sharpe": 0.0, "n_windows": 5},
            2.0: {"mean_test_sharpe": 0.1, "deflated_sharpe": 0.3, "n_windows": 5},
        }
    )

    score = robustness_score(result)

    assert score == pytest.approx(0.8131578947368421)


def test_robustness_score_stays_in_unit_interval():
    worst = _core_result(deflated_sharpe=0.0, permutation_p=1.0, pbo=1.0)
    best = _core_result(deflated_sharpe=1.0, permutation_p=0.0, pbo=0.0)

    assert robustness_score(worst) == pytest.approx(0.0)
    assert robustness_score(best) == pytest.approx(1.0)


def test_robustness_score_renormalization_makes_scores_incomparable_across_optional_flags():
    """Documents a deliberate, spec-approved limitation (not a bug to fix): identical
    core statistics score differently once an optional component is added, because the
    weighted-average denominator shifts. See the spec's "Known limitation" paragraph."""
    without_regime = _core_result(deflated_sharpe=0.7, permutation_p=0.1, pbo=0.2)
    with_regime = _core_result(
        deflated_sharpe=0.7,
        permutation_p=0.1,
        pbo=0.2,
        regime_breakdown={
            "Bull/High": {
                "n_windows": 3,
                "n_trades": 20,
                "mean_test_sharpe": 0.6,
                "total_return": 0.02,
            },
        },
    )

    assert robustness_score(without_regime) == pytest.approx(0.788235294117647)
    assert robustness_score(with_regime) == pytest.approx(0.8)
    assert robustness_score(without_regime) != pytest.approx(robustness_score(with_regime))


def test_strategy_report_pass_case_shows_verdict_core_stats_and_score():
    result = _core_result(passed=True, deflated_sharpe=0.97, permutation_p=0.01, pbo=0.1)

    report = strategy_report(result)

    assert "TestStrategy: PASS" in report
    assert "robustness score" in report
    assert "deflated_sharpe   0.970" in report
    assert "permutation p     0.010" in report
    assert "PBO               0.100" in report
    assert "mean OOS sharpe   1.100" in report
    assert "trials (ledger)   10" in report


def test_strategy_report_fail_case_shows_reasons():
    result = GateResult(
        strategy_id="TestStrategy",
        passed=False,
        deflated_sharpe=0.4,
        permutation_p=0.3,
        pbo=0.7,
        mean_test_sharpe=0.1,
        n_trials=12,
        reasons=["deflated_sharpe 0.400 below threshold 0.95"],
    )

    report = strategy_report(result)

    assert "TestStrategy: FAIL" in report
    assert "deflated_sharpe 0.400 below threshold 0.95" in report


def test_strategy_report_includes_fee_sensitivity_table_when_present():
    result = _core_result(
        fee_sensitivity={
            1.0: {"mean_test_sharpe": 0.87, "deflated_sharpe": 0.91, "n_windows": 5},
            1.5: {"mean_test_sharpe": 0.33, "deflated_sharpe": 0.52, "n_windows": 5},
        }
    )

    report = strategy_report(result)

    assert "fee sensitivity" in report
    assert "baseline" in report
    assert "1.50x fee" in report
    assert "slippage" not in report.lower()


def test_strategy_report_includes_regime_breakdown_with_pair_name_when_given():
    result = _core_result(
        regime_breakdown={
            "Bull/High": {
                "n_windows": 2,
                "n_trades": 14,
                "mean_test_sharpe": 0.42,
                "total_return": 0.0012,
            },
            "Bear/Low": {
                "n_windows": 1,
                "n_trades": 6,
                "mean_test_sharpe": -1.10,
                "total_return": -0.0034,
            },
        }
    )

    report = strategy_report(result, pair="BTC/USDT")

    assert "regime breakdown" in report
    assert "BTC/USDT" in report
    assert "Bull/High" in report
    assert "Bear/Low" in report


def test_strategy_report_omits_pair_name_when_not_given():
    result = _core_result(
        regime_breakdown={
            "Bull/High": {
                "n_windows": 2,
                "n_trades": 14,
                "mean_test_sharpe": 0.42,
                "total_return": 0.0012,
            },
        }
    )

    report = strategy_report(result, pair=None)

    assert "regime breakdown" in report
    assert "Bull/High" in report
    assert "(, " not in report
