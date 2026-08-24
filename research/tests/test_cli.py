from research.cli import main
from research.gate import GateResult


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
