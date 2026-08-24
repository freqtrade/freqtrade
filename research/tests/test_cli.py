from datetime import UTC

from research.cli import main
from research.gate import GateResult


def test_gate_command_parses_start_and_end_as_utc_aware_datetimes(mocker, capsys):
    """Regression test: freqtrade's Backtesting.backtest() internally compares
    against tz-aware (UTC) pandas Timestamps. A naive datetime passed as
    start_date/end_date crashes deep inside freqtrade with
    "can't compare offset-naive and offset-aware datetimes" -- caught only when
    running against real data, since research/tests' own fixtures always derive
    start/end from an already tz-aware loaded dataframe."""
    canned = GateResult(
        strategy_id="StrategyTestV3",
        passed=True,
        deflated_sharpe=0.97,
        permutation_p=0.01,
        pbo=0.1,
        mean_test_sharpe=1.2,
        n_trials=12,
        reasons=[],
    )
    mock_gate = mocker.patch("research.cli.run_promotion_gate", return_value=canned)
    mocker.patch(
        "research.cli.Configuration.from_files", return_value={"datadir": "user_data/data"}
    )

    main(
        [
            "gate",
            "--strategy",
            "StrategyTestV3",
            "--config",
            "config.json",
            "--pairs",
            "BTC/USDT",
            "--timeframe",
            "1h",
            "--start",
            "2024-01-01",
            "--end",
            "2024-06-01",
            "--train-days",
            "60",
            "--test-days",
            "20",
            "--param-grid",
            '[{"buy_rsi": 30}]',
        ]
    )

    _, kwargs = mock_gate.call_args
    assert kwargs["start"].utcoffset() == UTC.utcoffset(None)
    assert kwargs["end"].utcoffset() == UTC.utcoffset(None)


def test_gate_command_prints_verdict_and_returns_pass_exit_code(mocker, capsys):
    canned = GateResult(
        strategy_id="StrategyTestV3",
        passed=True,
        deflated_sharpe=0.97,
        permutation_p=0.01,
        pbo=0.1,
        mean_test_sharpe=1.2,
        n_trials=12,
        reasons=[],
    )
    mocker.patch("research.cli.run_promotion_gate", return_value=canned)
    mocker.patch(
        "research.cli.Configuration.from_files", return_value={"datadir": "user_data/data"}
    )

    exit_code = main(
        [
            "gate",
            "--strategy",
            "StrategyTestV3",
            "--config",
            "config.json",
            "--pairs",
            "BTC/USDT,ETH/USDT",
            "--timeframe",
            "1h",
            "--start",
            "2024-01-01",
            "--end",
            "2024-06-01",
            "--train-days",
            "60",
            "--test-days",
            "20",
            "--param-grid",
            '[{"buy_rsi": 30}]',
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS" in captured.out


def test_gate_command_returns_nonzero_exit_code_on_fail(mocker, capsys):
    canned = GateResult(
        strategy_id="StrategyTestV3",
        passed=False,
        deflated_sharpe=0.4,
        permutation_p=0.3,
        pbo=0.7,
        mean_test_sharpe=0.1,
        n_trials=12,
        reasons=["deflated_sharpe 0.400 below threshold 0.95"],
    )
    mocker.patch("research.cli.run_promotion_gate", return_value=canned)
    mocker.patch(
        "research.cli.Configuration.from_files", return_value={"datadir": "user_data/data"}
    )

    exit_code = main(
        [
            "gate",
            "--strategy",
            "StrategyTestV3",
            "--config",
            "config.json",
            "--pairs",
            "BTC/USDT",
            "--timeframe",
            "1h",
            "--start",
            "2024-01-01",
            "--end",
            "2024-06-01",
            "--train-days",
            "60",
            "--test-days",
            "20",
            "--param-grid",
            '[{"buy_rsi": 30}]',
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL" in captured.out


def test_gate_command_prints_fee_sensitivity_table_when_present(mocker, capsys):
    canned = GateResult(
        strategy_id="StrategyTestV3",
        passed=True,
        deflated_sharpe=0.97,
        permutation_p=0.01,
        pbo=0.1,
        mean_test_sharpe=1.2,
        n_trials=12,
        reasons=[],
        fee_sensitivity={
            1.0: {"mean_test_sharpe": 0.87, "deflated_sharpe": 0.91, "n_windows": 5},
            1.5: {"mean_test_sharpe": 0.33, "deflated_sharpe": 0.52, "n_windows": 5},
        },
    )
    mocker.patch("research.cli.run_promotion_gate", return_value=canned)
    mocker.patch(
        "research.cli.Configuration.from_files", return_value={"datadir": "user_data/data"}
    )

    exit_code = main(
        [
            "gate",
            "--strategy",
            "StrategyTestV3",
            "--config",
            "config.json",
            "--pairs",
            "BTC/USDT",
            "--timeframe",
            "1h",
            "--start",
            "2024-01-01",
            "--end",
            "2024-06-01",
            "--train-days",
            "60",
            "--test-days",
            "20",
            "--param-grid",
            '[{"buy_rsi": 30}]',
            "--fee-sensitivity",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "fee sensitivity" in captured.out
    assert "baseline" in captured.out
    assert "1.50x fee" in captured.out
    assert "slippage" not in captured.out.lower()


def test_gate_command_prints_regime_breakdown_table_when_present(mocker, capsys):
    canned = GateResult(
        strategy_id="StrategyTestV3",
        passed=False,
        deflated_sharpe=0.4,
        permutation_p=0.3,
        pbo=0.7,
        mean_test_sharpe=0.1,
        n_trials=12,
        reasons=["deflated_sharpe 0.400 below threshold 0.95"],
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
        },
    )
    mock_gate = mocker.patch("research.cli.run_promotion_gate", return_value=canned)
    mocker.patch(
        "research.cli.Configuration.from_files", return_value={"datadir": "user_data/data"}
    )

    exit_code = main(
        [
            "gate",
            "--strategy",
            "StrategyTestV3",
            "--config",
            "config.json",
            "--pairs",
            "BTC/USDT",
            "--timeframe",
            "1h",
            "--start",
            "2024-01-01",
            "--end",
            "2024-06-01",
            "--train-days",
            "60",
            "--test-days",
            "20",
            "--param-grid",
            '[{"buy_rsi": 30}]',
            "--regime-breakdown",
        ]
    )

    _, kwargs = mock_gate.call_args
    assert kwargs["include_regime_breakdown"] is True

    captured = capsys.readouterr()
    assert exit_code == 1  # this GateResult failed -- regime breakdown must print regardless
    assert "regime breakdown" in captured.out
    assert "Bull/High" in captured.out
    assert "Bear/Low" in captured.out
