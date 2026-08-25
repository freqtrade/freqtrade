from datetime import UTC

import pytest

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


def test_gate_command_threads_parameter_stability_flag_and_prints_report_line(mocker, capsys):
    canned = GateResult(
        strategy_id="StrategyTestV3",
        passed=True,
        deflated_sharpe=0.97,
        permutation_p=0.01,
        pbo=0.1,
        mean_test_sharpe=1.2,
        n_trials=12,
        reasons=[],
        parameter_stability=0.75,
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
            "--parameter-stability",
        ]
    )

    _, kwargs = mock_gate.call_args
    assert kwargs["include_parameter_stability"] is True

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "parameter stability  0.750" in captured.out


def test_trader_import_command_forwards_args_and_prints_result(mocker, capsys):
    from research.trader_mining.ingestion import IngestResult, LedgerIngestResult

    mock_ingest = mocker.patch(
        "research.cli.ingest_hyperliquid_fills",
        return_value=IngestResult(n_fetched=5, n_new=3, history_completeness="complete"),
    )
    mocker.patch(
        "research.cli.ingest_hyperliquid_ledger",
        return_value=LedgerIngestResult(n_fetched=0, n_new=0),
    )
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session")

    exit_code = main(
        [
            "trader-import",
            "--trader",
            "0x0000000000000000000000000000000000000000",
            "--since",
            "2026-01-01",
            "--db-path",
            "user_data/research.sqlite",
        ]
    )

    _, kwargs = mock_ingest.call_args
    assert kwargs["trader"] == "0x0000000000000000000000000000000000000000"
    assert kwargs["since"].year == 2026

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "n_fetched: 5" in captured.out
    assert "n_new: 3" in captured.out
    assert "complete" in captured.out


def test_trader_import_command_warns_on_truncated_history(mocker, capsys):
    from research.trader_mining.ingestion import IngestResult, LedgerIngestResult

    mocker.patch(
        "research.cli.ingest_hyperliquid_fills",
        return_value=IngestResult(
            n_fetched=10_000, n_new=10_000, history_completeness="truncated_by_provider_limit"
        ),
    )
    mocker.patch(
        "research.cli.ingest_hyperliquid_ledger",
        return_value=LedgerIngestResult(n_fetched=0, n_new=0),
    )
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session")

    exit_code = main(["trader-import", "--trader", "0x0000000000000000000000000000000000000000"])

    captured = capsys.readouterr()
    assert exit_code == 0  # not a failure -- an honest, informational result
    assert "truncated_by_provider_limit" in captured.out
    assert "WARNING" in captured.out


def test_trader_analyze_command_forwards_args_and_prints_result(mocker, capsys):
    from research.trader_mining.engine import ReconstructResult

    mock_reconstruct = mocker.patch(
        "research.cli.reconstruct_and_persist_trades",
        return_value=ReconstructResult(n_trades=7, symbols=["BTC/USDC:USDC", "ETH/USDC:USDC"]),
    )
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session")

    exit_code = main(
        [
            "trader-analyze",
            "--trader",
            "0x0000000000000000000000000000000000000000",
            "--symbol",
            "BTC/USDC:USDC",
            "--db-path",
            "user_data/research.sqlite",
        ]
    )

    _, kwargs = mock_reconstruct.call_args
    assert kwargs["trader"] == "0x0000000000000000000000000000000000000000"
    assert kwargs["symbol"] == "BTC/USDC:USDC"

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "n_trades: 7" in captured.out
    assert "BTC/USDC:USDC" in captured.out
    assert "ETH/USDC:USDC" in captured.out


def test_trader_report_command_prints_formatted_report(mocker, capsys):
    from research.trader_mining.metrics import WalletMetrics

    canned = WalletMetrics(
        trade_count=2,
        total_volume=300.0,
        gross_pnl=71.0,
        fees=1.0,
        net_pnl=70.0,
        win_rate=0.5,
        avg_win=100.0,
        avg_loss=-30.0,
        profit_factor=100.0 / 30.0,
        expectancy=35.0,
        payoff_ratio=100.0 / 30.0,
        median_trade_return=0.15,
        avg_holding_period_seconds=3600.0,
        median_holding_period_seconds=3600.0,
        long_count=2,
        short_count=0,
        long_pct=1.0,
        symbol_concentration=1.0,
        max_drawdown=30.0,
        max_losing_streak=1,
        pnl_concentration_top_5=1.0,
        trade_consistency_score=1.2,
        return_to_drawdown_ratio=70.0 / 30.0,
    )
    mock_query = mocker.MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = []  # irrelevant -- compute_metrics is mocked below
    mock_session = mocker.MagicMock()
    mock_session.query.return_value = mock_query
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session", return_value=mock_session)
    mocker.patch("research.cli.compute_metrics", return_value=canned)

    exit_code = main(["trader-report", "--trader", "0x0000000000000000000000000000000000000000"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "0x0000000000000000000000000000000000000000" in captured.out
    assert "70.00" in captured.out


def test_trader_import_command_also_ingests_ledger(mocker, capsys):
    from research.trader_mining.ingestion import IngestResult, LedgerIngestResult

    mocker.patch(
        "research.cli.ingest_hyperliquid_fills",
        return_value=IngestResult(n_fetched=5, n_new=3, history_completeness="complete"),
    )
    mock_ledger = mocker.patch(
        "research.cli.ingest_hyperliquid_ledger",
        return_value=LedgerIngestResult(n_fetched=2, n_new=2),
    )
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session")

    exit_code = main(["trader-import", "--trader", "0x0000000000000000000000000000000000000000"])

    assert mock_ledger.called
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "n_ledger_events_new: 2" in captured.out


def test_trader_analyze_command_prints_reconciled_gaps_when_present(mocker, capsys):
    from research.trader_mining.engine import ReconstructResult

    mocker.patch(
        "research.cli.reconstruct_and_persist_trades",
        return_value=ReconstructResult(
            n_trades=1,
            symbols=["HYPE/USDC"],
            reconciled_gaps=["HYPE/USDC: reconciled a -62264.0 position gap ..."],
        ),
    )
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session")

    exit_code = main(["trader-analyze", "--trader", "0x0000000000000000000000000000000000000000"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "reconciled_gaps" in captured.out
    assert "HYPE/USDC" in captured.out


def test_trader_report_rejects_partial_split_flags(mocker, capsys):
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session")

    with pytest.raises(SystemExit):
        main(
            [
                "trader-report",
                "--trader",
                "0xAAA",
                "--train-end",
                "2025-01-01",
                "--validation-end",
                "2025-07-01",
                # --test-end deliberately omitted
            ]
        )

    assert "must be given together" in capsys.readouterr().err


def test_trader_report_prints_split_report_when_all_three_flags_given(mocker, capsys):
    from datetime import UTC, datetime

    from research.models import ReconstructedTrade

    trade = ReconstructedTrade(
        trader="0xAAA",
        symbol="BTC/USDC:USDC",
        direction="long",
        entry_timestamp=datetime(2024, 6, 1, tzinfo=UTC),
        entry_price=100.0,
        exit_timestamp=datetime(2024, 6, 1, tzinfo=UTC),
        exit_price=100.0,
        quantity=1.0,
        gross_pnl=10.0,
        fees=0.0,
        net_pnl=10.0,
        holding_time_seconds=3600.0,
        n_fills=2,
        is_truncated_start=False,
        was_liquidated=False,
    )
    mock_query = mocker.MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = [trade]
    mock_session = mocker.MagicMock()
    mock_session.query.return_value = mock_query
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session", return_value=mock_session)

    exit_code = main(
        [
            "trader-report",
            "--trader",
            "0xAAA",
            "--train-end",
            "2025-01-01",
            "--validation-end",
            "2025-07-01",
            "--test-end",
            "2026-01-01",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "TRAIN" in captured.out and "VALIDATION" in captured.out
    assert "whole-history" in captured.out.lower()


def test_trader_report_unchanged_when_no_split_flags_given(mocker, capsys):
    """Regression guard: today's plain trader-report output must be the same code path
    (compute_metrics + format_report) when no split flags are passed."""
    mock_query = mocker.MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.all.return_value = []
    mock_session = mocker.MagicMock()
    mock_session.query.return_value = mock_query
    mocker.patch("research.cli.get_engine")
    mocker.patch("research.cli.get_session", return_value=mock_session)

    exit_code = main(["trader-report", "--trader", "0xAAA"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Chronological Split Report" not in captured.out
    assert "## Wallet Report: 0xAAA" in captured.out
