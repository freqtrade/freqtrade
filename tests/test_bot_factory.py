import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from freqtrade_ext.bot_factory.backtest_results import (
    GateThresholds,
    evaluate_initial_gate,
    load_backtest_result,
    load_gate_thresholds,
    summarize,
)
from freqtrade_ext.bot_factory.bybit_open_interest import (
    BybitOpenInterestDownloadInputs,
    default_bybit_open_interest_path,
    download_bybit_open_interest,
)
from freqtrade_ext.bot_factory.bybit_long_short_ratio import (
    BybitLongShortRatioDownloadInputs,
    default_bybit_long_short_ratio_path,
    download_bybit_long_short_ratio,
)
from freqtrade_ext.bot_factory.data_quality import (
    check_funding_rate_parquet,
    check_long_short_ratio_parquet,
    check_liquidation_parquet,
    check_mark_price_parquet,
    check_ohlcv_parquet,
    check_open_interest_parquet,
    check_order_book_parquet,
    pair_to_ohlcv_filename,
)
from freqtrade_ext.bot_factory.cost_model import (
    CostModelContext,
    cost_scenarios_from_spec,
    default_cost_scenarios,
)
from freqtrade_ext.bot_factory.cost_calibration import (
    CostCalibrationInputs,
    build_cost_calibration,
    write_cost_calibration_artifacts,
)
from freqtrade_ext.bot_factory.edge_discovery import (
    EdgeDiscoveryInputs,
    _control_events,
    _event_level_post_cost_report,
    _negative_control_summary,
    _price_frame_for_label,
    build_edge_discovery,
    write_edge_discovery_artifacts,
)
from freqtrade_ext.bot_factory.freqai_backtest import (
    build_freqai_metadata,
    candidate_freqai_identifier,
    freqai_input_pairs,
    freqai_input_timeframes,
    freqai_model_name,
    resolve_ohlcv_input_paths,
    write_freqai_identifier_override_config,
)
from freqtrade_ext.bot_factory.freqai_checks import (
    FREQAI_LABEL_NOTICE,
    DependencySpec,
    check_freqai_dependencies,
    validate_freqai_strategy_paths,
)
from freqtrade_ext.bot_factory.freqai_training import (
    TrainingStageResult,
    build_checked_freqai_backtest_command,
    build_checked_walk_forward_command,
    build_training_manifest,
    training_child_run_id,
)
from freqtrade_ext.bot_factory.local_falsification import (
    LocalFalsificationInputs,
    build_local_falsification,
    write_local_falsification_artifacts,
)
from freqtrade_ext.bot_factory.local_events import (
    LocalEventBuildInputs,
    _feature_series,
    build_local_events,
    write_local_event_artifacts,
)
from freqtrade_ext.bot_factory.research_selection_template import (
    ResearchSelectionTemplateInputs,
    build_research_selection_template,
    write_research_selection_template_artifacts,
)
from freqtrade_ext.bot_factory.structural_data_capabilities import (
    StructuralDataCapabilityInputs,
    build_structural_data_capability_report,
)
from freqtrade_ext.bot_factory.paper import (
    PaperReadinessInputs,
    evaluate_config_safety,
    evaluate_paper_readiness,
    evaluate_strategy_long_only,
)
from freqtrade_ext.bot_factory.paper_execution import (
    PaperExecutionRequestInputs,
    build_paper_execution_request,
    write_paper_execution_request_artifacts,
)
from freqtrade_ext.bot_factory.paper_executor import (
    PaperProcessExecutorPlanInputs,
    build_paper_process_executor_plan,
    write_paper_process_executor_plan_artifacts,
)
from freqtrade_ext.bot_factory.paper_drift import (
    PaperDriftReportInputs,
    build_paper_drift_report,
    write_paper_drift_report_artifacts,
)
from freqtrade_ext.bot_factory.paper_runtime import (
    PaperRuntimeValidationInputs,
    build_paper_runtime_validation,
    write_paper_runtime_validation_artifacts,
)
from freqtrade_ext.bot_factory.paper_plan import (
    PaperRunPlanInputs,
    build_paper_run_plan,
    write_paper_run_plan_artifacts,
)
from freqtrade_ext.bot_factory.paper_monitoring import (
    PaperMonitoringPlanInputs,
    build_paper_monitoring_plan,
    write_paper_monitoring_plan_artifacts,
)
from freqtrade_ext.bot_factory.paper_startup import (
    PaperStartupPreflightInputs,
    build_paper_startup_preflight,
    write_paper_startup_preflight_artifacts,
)
from freqtrade_ext.bot_factory.paper_stop_cleanup import (
    PaperStopCleanupPlanInputs,
    build_paper_stop_cleanup_plan,
    write_paper_stop_cleanup_plan_artifacts,
)
from freqtrade_ext.bot_factory.research_selection import (
    ResearchSelectionInputs,
    select_research_thesis,
    write_research_selection_artifacts,
)
from freqtrade_ext.bot_factory.strategy_proposals import (
    REQUIRED_PROPOSAL_SECTIONS,
    StrategyProposalEvidenceInput,
    StrategyProposalInputs,
    StrategyProposalResearchReference,
    build_strategy_proposal,
    write_strategy_proposal_artifacts,
)
from freqtrade_ext.bot_factory.strategy_code import (
    PARAMETER_OPTIMIZATION_POLICY,
    StrategyCodeInputs,
    build_strategy_code,
    write_strategy_code_artifacts,
)
from freqtrade_ext.bot_factory.signal_diagnostics import (
    CandidateSignalDiagnosticsInputs,
    diagnose_candidate_signals,
    write_signal_diagnostics_artifacts,
)
from freqtrade_ext.bot_factory.safety import scan_paths
from freqtrade_ext.bot_factory.walk_forward import (
    WalkForwardRules,
    aggregate_walk_forward_results,
    generate_rolling_windows,
    parse_window_specs,
    window_run_id,
)


def test_load_backtest_result_resolves_latest_zip(tmp_path):
    result = {
        "strategy": {
            "SampleStrategy": {
                "trades": [],
                "total_trades": 0,
                "profit_total": 0.0,
                "profit_total_pct": 0.0,
            }
        },
        "strategy_comparison": [],
    }
    zip_path = tmp_path / "backtest-result.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("backtest-result_config.json", "{}")
        archive.writestr("backtest-result.json", json.dumps(result))

    pointer_path = tmp_path / "result.json"
    pointer_path.write_text(json.dumps({"latest_backtest": zip_path.name}), encoding="utf-8")

    loaded = load_backtest_result(pointer_path)

    assert loaded == result


def test_summarize_backtest_result():
    result = {
        "strategy": {
            "SampleStrategy": {
                "trades": [
                    {"profit_ratio": 0.02, "fee_open_cost": 0.1, "fee_close_cost": 0.1},
                    {"profit_ratio": -0.01, "fee_open_cost": 0.1, "fee_close_cost": 0.1},
                ],
                "total_trades": 2,
                "profit_total": 0.01,
                "profit_total_pct": 1.0,
                "max_drawdown_account": 0.05,
                "profit_factor": 2.0,
                "sortino": 1.5,
            }
        }
    }

    metrics = summarize(result, "SampleStrategy")

    assert metrics.trade_count == 2
    assert metrics.win_rate == 0.5
    assert metrics.max_drawdown_pct == 5.0
    assert metrics.fee_paid == 0.4
    assert metrics.sortino == 1.5


def test_custom_gate_thresholds_can_pass_small_backtest():
    result = {
        "strategy": {
            "SampleStrategy": {
                "trades": [{"profit_ratio": 0.02}, {"profit_ratio": -0.01}],
                "total_trades": 2,
                "profit_total": 0.01,
                "profit_total_pct": 1.0,
                "max_drawdown_account": 0.05,
                "profit_factor": 2.0,
                "sortino": 1.5,
            }
        }
    }
    metrics = summarize(result, "SampleStrategy")

    gate = evaluate_initial_gate(
        metrics,
        GateThresholds(
            min_trades=2,
            min_profit_factor=1.5,
            max_drawdown_pct=10,
            min_sortino=1.0,
        ),
    )

    assert gate["recommendation"] == "pass"


def test_load_gate_thresholds_accepts_backtest_rules(tmp_path):
    config_path = tmp_path / "gate.json"
    config_path.write_text(
        json.dumps(
            {
                "backtest_rules": {
                    "min_trades": 10,
                    "min_profit_factor": 1.1,
                    "max_drawdown_pct": 12.5,
                    "min_sortino": None,
                }
            }
        ),
        encoding="utf-8",
    )

    thresholds = load_gate_thresholds(config_path)

    assert thresholds.min_trades == 10
    assert thresholds.min_sortino is None


def test_static_check_detects_shift_minus_one(tmp_path):
    strategy_path = tmp_path / "BadStrategy.py"
    strategy_path.write_text(
        "def populate_indicators(dataframe, metadata):\n"
        "    dataframe['future_close'] = dataframe['close'].shift(-1)\n"
        "    return dataframe\n",
        encoding="utf-8",
    )

    report = scan_paths([strategy_path])

    assert not report.ok
    assert [finding.rule for finding in report.findings] == ["no_shift_minus_one"]


def test_static_check_detects_keyword_shift_minus_one(tmp_path):
    strategy_path = tmp_path / "BadStrategy.py"
    strategy_path.write_text(
        "def populate_entry_trend(dataframe, metadata):\n"
        "    dataframe['future_close'] = dataframe['close'].shift(periods=-1)\n"
        "    return dataframe\n",
        encoding="utf-8",
    )

    report = scan_paths([strategy_path])

    assert not report.ok
    assert [finding.rule for finding in report.findings] == ["no_shift_minus_one"]


def test_static_check_detects_tuple_iloc_minus_one_in_signal_generation(tmp_path):
    strategy_path = tmp_path / "BadStrategy.py"
    strategy_path.write_text(
        "def populate_exit_trend(dataframe, metadata):\n"
        "    last_close = dataframe.iloc[-1, 3]\n"
        "    dataframe['exit_long'] = dataframe['close'] > last_close\n"
        "    return dataframe\n",
        encoding="utf-8",
    )

    report = scan_paths([strategy_path])

    assert not report.ok
    assert [finding.rule for finding in report.findings] == [
        "no_iloc_minus_one_in_signal_generation"
    ]


def test_static_check_allows_iloc_slice_excluding_last_row(tmp_path):
    strategy_path = tmp_path / "Strategy.py"
    strategy_path.write_text(
        "def populate_indicators(dataframe, metadata):\n"
        "    historical = dataframe.iloc[:-1]\n"
        "    dataframe['sample'] = historical['close'].mean()\n"
        "    return dataframe\n",
        encoding="utf-8",
    )

    report = scan_paths([strategy_path])

    assert report.ok
    assert report.findings == []


def test_static_check_allows_target_generation_shift_minus_one(tmp_path):
    strategy_path = tmp_path / "FreqAIStrategy.py"
    strategy_path.write_text(
        "def set_freqai_targets(dataframe, metadata):\n"
        "    dataframe['&-return'] = dataframe['close'].shift(-1) / dataframe['close'] - 1\n"
        "    return dataframe\n",
        encoding="utf-8",
    )

    report = scan_paths([strategy_path])

    assert report.ok
    assert report.findings == []


def test_ohlcv_quality_check_detects_invalid_price_bounds(tmp_path):
    data_path = tmp_path / "BTC_USDT-5m.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=2, freq="5min", tz="UTC"),
            "open": [100.0, 101.0],
            "high": [99.0, 102.0],
            "low": [98.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1.0, 2.0],
        }
    ).to_parquet(data_path)

    report = check_ohlcv_parquet(data_path, "5m")

    assert not report.ok
    assert "high_bounds" in {finding.rule for finding in report.findings}


def test_funding_rate_quality_check_allows_negative_rates(tmp_path):
    data_path = tmp_path / "BTC_USDT-8h-funding_rate.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=4, freq="8h", tz="UTC"),
            "open": [-0.0002, -0.0001, 0.0, 0.0001],
            "high": [0.0, 0.0, 0.0, 0.0],
            "low": [0.0, 0.0, 0.0, 0.0],
            "close": [0.0, 0.0, 0.0, 0.0],
            "volume": [0.0, 0.0, 0.0, 0.0],
        }
    ).to_parquet(data_path)

    report = check_funding_rate_parquet(data_path, "8h")

    assert report.ok
    assert report.rows == 4
    assert report.missing_intervals == 0
    assert report.findings == []


def test_mark_price_quality_check_allows_null_volume(tmp_path):
    data_path = tmp_path / "BTC_USDT_USDT-4h-mark.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=4, freq="4h", tz="UTC"),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [None, None, None, None],
        }
    ).to_parquet(data_path)

    report = check_mark_price_parquet(data_path, "4h")

    assert report.ok
    assert report.rows == 4
    assert report.missing_intervals == 0
    assert report.findings == []


def test_open_interest_quality_check_accepts_open_interest_column(tmp_path):
    data_path = tmp_path / "BTC_USDT_USDT-1h-open_interest.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=4, freq="1h", tz="UTC"),
            "open_interest": [1000.0, 1005.0, 1010.0, 1008.0],
        }
    ).to_parquet(data_path)

    report = check_open_interest_parquet(data_path, "1h")

    assert report.ok
    assert report.rows == 4
    assert report.missing_intervals == 0
    assert report.findings == []


def test_open_interest_quality_check_rejects_negative_values(tmp_path):
    data_path = tmp_path / "BTC_USDT_USDT-1h-open_interest.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC"),
            "open": [1000.0, -1.0, 1010.0],
        }
    ).to_parquet(data_path)

    report = check_open_interest_parquet(data_path, "1h")

    assert not report.ok
    assert any(finding.rule == "open_interest_non_negative" for finding in report.findings)


def test_bybit_open_interest_default_path_uses_futures_symbol_style():
    assert default_bybit_open_interest_path("BTCUSDT", interval_time="1h") == Path(
        "data/market_structure/bybit/futures/BTC_USDT_USDT-1h-open_interest.parquet"
    )


def test_bybit_open_interest_downloader_writes_sorted_parquet(tmp_path):
    calls: list[dict[str, Any]] = []

    def fake_request_json(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        assert url == "https://api.bybit.com/v5/market/open-interest"
        assert timeout == 20.0
        calls.append(dict(params))
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {"symbol": "BTCUSDT", "openInterest": "102.0", "timestamp": "1735693200000"},
                    {"symbol": "BTCUSDT", "openInterest": "100.0", "timestamp": "1735689600000"},
                ],
                "nextPageCursor": "",
            },
        }

    output_path = tmp_path / "oi.parquet"
    artifact = download_bybit_open_interest(
        BybitOpenInterestDownloadInputs(
            root_dir=tmp_path,
            symbol="btcusdt",
            category="linear",
            interval_time="1h",
            start_time=datetime(2025, 1, 1, tzinfo=UTC),
            end_time=datetime(2025, 1, 2, tzinfo=UTC),
            output_path=output_path,
        ),
        request_json=fake_request_json,
    )

    frame = pd.read_parquet(output_path)

    assert artifact["status"] == "completed"
    assert artifact["row_count"] == 2
    assert artifact["safety_scope"]["api_key_used"] is False
    assert artifact["safety_scope"]["order_endpoint_used"] is False
    assert calls[0]["category"] == "linear"
    assert calls[0]["symbol"] == "BTCUSDT"
    assert calls[0]["intervalTime"] == "1h"
    assert list(frame["open_interest"]) == [100.0, 102.0]
    assert list(frame["symbol"]) == ["BTCUSDT", "BTCUSDT"]


def test_bybit_open_interest_and_long_short_ratio_downloaders_block_request_errors(tmp_path):
    def failing_request_json(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        raise TimeoutError("public market data request timed out")

    open_interest_output = tmp_path / "oi.parquet"
    open_interest_artifact = download_bybit_open_interest(
        BybitOpenInterestDownloadInputs(
            root_dir=tmp_path,
            symbol="btcusdt",
            category="linear",
            interval_time="1h",
            start_time=datetime(2025, 1, 1, tzinfo=UTC),
            end_time=datetime(2025, 1, 2, tzinfo=UTC),
            output_path=open_interest_output,
        ),
        request_json=failing_request_json,
    )
    long_short_output = tmp_path / "long_short.parquet"
    long_short_artifact = download_bybit_long_short_ratio(
        BybitLongShortRatioDownloadInputs(
            root_dir=tmp_path,
            symbol="btcusdt",
            category="linear",
            period="1h",
            start_time=datetime(2025, 1, 1, tzinfo=UTC),
            end_time=datetime(2025, 1, 2, tzinfo=UTC),
            output_path=long_short_output,
        ),
        request_json=failing_request_json,
    )

    assert open_interest_artifact["status"] == "blocked"
    assert open_interest_artifact["output_written"] is False
    assert open_interest_artifact["page_count"] == 0
    assert open_interest_artifact["row_count"] == 0
    assert open_interest_artifact["blockers"] == [
        "request_failed:TimeoutError:public market data request timed out"
    ]
    assert not open_interest_output.exists()
    assert long_short_artifact["status"] == "blocked"
    assert long_short_artifact["output_written"] is False
    assert long_short_artifact["page_count"] == 0
    assert long_short_artifact["row_count"] == 0
    assert long_short_artifact["blockers"] == [
        "request_failed:TimeoutError:public market data request timed out"
    ]
    assert not long_short_output.exists()


def test_long_short_ratio_quality_check_accepts_ratio_columns(tmp_path):
    data_path = tmp_path / "BTC_USDT_USDT-1h-long_short_ratio.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=4, freq="1h", tz="UTC"),
            "long_account_ratio": [0.55, 0.58, 0.61, 0.57],
            "short_account_ratio": [0.45, 0.42, 0.39, 0.43],
            "long_short_ratio": [1.222222, 1.380952, 1.564103, 1.325581],
        }
    ).to_parquet(data_path)

    report = check_long_short_ratio_parquet(data_path, "1h")

    assert report.ok
    assert report.rows == 4
    assert report.missing_intervals == 0
    assert report.findings == []


def test_long_short_ratio_quality_check_rejects_out_of_bounds_values(tmp_path):
    data_path = tmp_path / "BTC_USDT_USDT-1h-long_short_ratio.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC"),
            "buyRatio": [0.55, 1.25, 0.50],
            "sellRatio": [0.45, -0.25, 0.50],
        }
    ).to_parquet(data_path)

    report = check_long_short_ratio_parquet(data_path, "1h")

    assert not report.ok
    assert any(
        finding.rule == "long_account_ratio_between_zero_and_one"
        for finding in report.findings
    )
    assert any(
        finding.rule == "short_account_ratio_between_zero_and_one"
        for finding in report.findings
    )


def test_order_book_quality_check_accepts_top_of_book_snapshot_columns(tmp_path):
    data_path = tmp_path / "BTC_USDT_USDT-1m-order_book.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3, freq="1min", tz="UTC"),
            "best_bid": [100.0, 100.5, 101.0],
            "best_ask": [100.1, 100.6, 101.2],
            "bid_size": [2.0, 1.5, 1.0],
            "ask_size": [1.8, 1.4, 1.2],
            "depth_imbalance": [0.05, 0.03, -0.09],
        }
    ).to_parquet(data_path)

    report = check_order_book_parquet(data_path, "1m")

    assert report.ok
    assert report.rows == 3
    assert report.expected_interval_seconds == 60


def test_order_book_quality_check_rejects_crossed_book(tmp_path):
    data_path = tmp_path / "BTC_USDT_USDT-1m-order_book.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=2, freq="1min", tz="UTC"),
            "best_bid": [100.0, 101.5],
            "best_ask": [100.1, 101.4],
            "bid_size": [2.0, 1.5],
            "ask_size": [1.8, 1.4],
        }
    ).to_parquet(data_path)

    report = check_order_book_parquet(data_path, "1m")

    assert not report.ok
    assert any(finding.rule == "best_bid_not_above_best_ask" for finding in report.findings)


def test_liquidation_quality_check_accepts_bybit_all_liquidation_columns(tmp_path):
    data_path = tmp_path / "BTC_USDT_USDT-liquidation.parquet"
    pd.DataFrame(
        {
            "T": [1735689600000, 1735689660000, 1735689720000],
            "S": ["Buy", "Sell", "buy"],
            "v": ["0.25", "1.5", "0.75"],
            "p": ["93500.5", "93420.0", "93610.0"],
            "s": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
        }
    ).to_parquet(data_path)

    report = check_liquidation_parquet(data_path)

    assert report.ok
    assert report.rows == 3
    assert report.start == "2025-01-01T00:00:00+00:00"
    assert report.end == "2025-01-01T00:02:00+00:00"


def test_liquidation_quality_check_rejects_bad_side_and_non_positive_values(tmp_path):
    data_path = tmp_path / "BTC_USDT_USDT-liquidation.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=2, freq="1min", tz="UTC"),
            "side": ["Buy", "Hold"],
            "quantity": [0.5, 0.0],
            "price": [93500.0, -1.0],
        }
    ).to_parquet(data_path)

    report = check_liquidation_parquet(data_path)

    assert not report.ok
    assert {finding.rule for finding in report.findings} >= {
        "side_buy_or_sell",
        "liquidation_size_positive",
        "liquidation_price_positive",
    }


def test_bybit_long_short_ratio_default_path_uses_futures_symbol_style():
    assert default_bybit_long_short_ratio_path("BTCUSDT", period="1h") == Path(
        "data/market_structure/bybit/futures/BTC_USDT_USDT-1h-long_short_ratio.parquet"
    )


def test_bybit_long_short_ratio_downloader_writes_sorted_parquet(tmp_path):
    calls: list[dict[str, Any]] = []

    def fake_request_json(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        assert url == "https://api.bybit.com/v5/market/account-ratio"
        assert timeout == 20.0
        calls.append(dict(params))
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {"symbol": "BTCUSDT", "buyRatio": "0.61", "sellRatio": "0.39", "timestamp": "1735693200000"},
                    {"symbol": "BTCUSDT", "buyRatio": "0.55", "sellRatio": "0.45", "timestamp": "1735689600000"},
                ],
                "nextPageCursor": "",
            },
        }

    output_path = tmp_path / "long_short.parquet"
    artifact = download_bybit_long_short_ratio(
        BybitLongShortRatioDownloadInputs(
            root_dir=tmp_path,
            symbol="btcusdt",
            category="linear",
            period="1h",
            start_time=datetime(2025, 1, 1, tzinfo=UTC),
            end_time=datetime(2025, 1, 2, tzinfo=UTC),
            output_path=output_path,
        ),
        request_json=fake_request_json,
    )

    frame = pd.read_parquet(output_path)

    assert artifact["status"] == "completed"
    assert artifact["row_count"] == 2
    assert artifact["safety_scope"]["api_key_used"] is False
    assert artifact["safety_scope"]["order_endpoint_used"] is False
    assert calls[0]["category"] == "linear"
    assert calls[0]["symbol"] == "BTCUSDT"
    assert calls[0]["period"] == "1h"
    assert list(frame["long_account_ratio"]) == [0.55, 0.61]
    assert list(frame["short_account_ratio"]) == [0.45, 0.39]
    assert list(frame["symbol"]) == ["BTCUSDT", "BTCUSDT"]


def test_bybit_long_short_ratio_downloader_drops_zero_short_ratio_rows(tmp_path):
    def fake_request_json(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {"symbol": "BTCUSDT", "buyRatio": "1.0", "sellRatio": "0.0", "timestamp": "1735689600000"},
                    {"symbol": "BTCUSDT", "buyRatio": "0.55", "sellRatio": "0.45", "timestamp": "1735693200000"},
                ],
                "nextPageCursor": "",
            },
        }

    output_path = tmp_path / "long_short.parquet"
    artifact = download_bybit_long_short_ratio(
        BybitLongShortRatioDownloadInputs(
            root_dir=tmp_path,
            symbol="btcusdt",
            category="linear",
            period="1h",
            start_time=datetime(2025, 1, 1, tzinfo=UTC),
            end_time=datetime(2025, 1, 2, tzinfo=UTC),
            output_path=output_path,
        ),
        request_json=fake_request_json,
    )

    frame = pd.read_parquet(output_path)

    assert artifact["status"] == "completed"
    assert artifact["row_count"] == 1
    assert list(frame["short_account_ratio"]) == [0.45]
    assert list(frame["long_short_ratio"]) == [0.55 / 0.45]
    assert not frame["long_short_ratio"].isna().any()


def test_structural_data_capability_report_marks_open_interest_codegen_supported_when_quality_passes(tmp_path):
    open_interest_path = (
        tmp_path
        / "user_data"
        / "data"
        / "bybit"
        / "futures"
        / "BTC_USDT_USDT-1h-open_interest.parquet"
    )
    open_interest_path.parent.mkdir(parents=True)
    open_interest_path.write_text("placeholder", encoding="utf-8")
    quality_path = tmp_path / "registry" / "strategies" / "checks" / "open_interest_quality.json"
    quality_path.parent.mkdir(parents=True)
    quality_path.write_text(
        json.dumps({"ok": True, "reports": [{"ok": True, "rows": 100}]}),
        encoding="utf-8",
    )

    artifact = build_structural_data_capability_report(
        StructuralDataCapabilityInputs(
            root_dir=tmp_path,
            open_interest_path=open_interest_path,
            open_interest_quality_report_paths=[quality_path],
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    open_interest = artifact["capabilities"]["open_interest"]
    assert open_interest["local_data_present"] is True
    assert open_interest["local_quality_ok"] is True
    assert open_interest["local_event_supported"] is True
    assert open_interest["strategy_codegen_supported"] is True
    assert artifact["proposal_guidance"]["local_research_usable"] == ["open_interest"]
    assert set(artifact["proposal_guidance"]["must_not_codegen"]) == {
        "long_short_ratio",
        "liquidation",
        "order_book",
    }


def test_structural_data_capability_report_marks_long_short_ratio_codegen_supported_when_quality_passes(tmp_path):
    ratio_path = (
        tmp_path
        / "user_data"
        / "data"
        / "bybit"
        / "futures"
        / "BTC_USDT_USDT-1h-long_short_ratio.parquet"
    )
    ratio_path.parent.mkdir(parents=True)
    ratio_path.write_text("placeholder", encoding="utf-8")
    quality_path = tmp_path / "registry" / "strategies" / "checks" / "long_short_ratio_quality.json"
    quality_path.parent.mkdir(parents=True)
    quality_path.write_text(
        json.dumps({"ok": True, "reports": [{"ok": True, "rows": 100}]}),
        encoding="utf-8",
    )

    artifact = build_structural_data_capability_report(
        StructuralDataCapabilityInputs(
            root_dir=tmp_path,
            long_short_ratio_path=ratio_path,
            long_short_ratio_quality_report_paths=[quality_path],
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    long_short_ratio = artifact["capabilities"]["long_short_ratio"]
    assert long_short_ratio["local_data_present"] is True
    assert long_short_ratio["local_quality_ok"] is True
    assert long_short_ratio["local_event_supported"] is True
    assert long_short_ratio["strategy_codegen_supported"] is True
    assert artifact["proposal_guidance"]["local_research_usable"] == [
        "long_short_ratio"
    ]
    assert "long_short_ratio" not in artifact["proposal_guidance"]["must_not_codegen"]


def test_structural_data_capability_report_blocks_missing_liquidation_and_order_book(tmp_path):
    artifact = build_structural_data_capability_report(
        StructuralDataCapabilityInputs(
            root_dir=tmp_path,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["proposal_guidance"]["local_research_usable"] == []
    assert "long_short_ratio" in artifact["proposal_guidance"]["blocked_without_new_data"]
    assert "liquidation" in artifact["proposal_guidance"]["blocked_without_new_data"]
    assert "order_book" in artifact["proposal_guidance"]["blocked_without_new_data"]
    assert artifact["capabilities"]["liquidation"]["historical_download_supported"] is False
    assert artifact["capabilities"]["liquidation"]["collection_mode"] == (
        "public_websocket_realtime_only"
    )
    assert artifact["capabilities"]["order_book"]["historical_download_supported"] is False
    assert artifact["capabilities"]["order_book"]["collection_mode"] == (
        "current_snapshot_or_user_supplied_historical_snapshots"
    )


def _write_passing_structural_quality_report(path: Path, *, rows: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"ok": True, "reports": [{"ok": True, "rows": rows}]}),
        encoding="utf-8",
    )


def test_structural_data_capability_report_accepts_order_book_quality_and_local_events_but_blocks_codegen(tmp_path):
    order_book_path = (
        tmp_path
        / "data"
        / "market_structure"
        / "bybit"
        / "futures"
        / "BTC_USDT_USDT-1m-order_book.parquet"
    )
    order_book_path.parent.mkdir(parents=True)
    order_book_path.write_text("placeholder", encoding="utf-8")
    quality_path = tmp_path / "registry" / "strategies" / "checks" / "order_book_quality.json"
    _write_passing_structural_quality_report(quality_path)

    artifact = build_structural_data_capability_report(
        StructuralDataCapabilityInputs(
            root_dir=tmp_path,
            order_book_paths=[order_book_path],
            order_book_quality_report_paths=[quality_path],
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    order_book = artifact["capabilities"]["order_book"]
    assert order_book["local_data_present"] is True
    assert order_book["local_quality_ok"] is True
    assert order_book["research_selection_quality_gate_supported"] is True
    assert order_book["local_event_supported"] is True
    assert order_book["strategy_codegen_supported"] is False
    assert "order_book" not in artifact["proposal_guidance"]["blocked_without_new_data"]
    assert "order_book" in artifact["proposal_guidance"]["local_research_usable"]
    assert "order_book" in artifact["proposal_guidance"]["must_not_codegen"]
    assert (
        "order_book_strategy_codegen_variant_before_promotion"
        in artifact["proposal_guidance"]["next_data_needed"]
    )


def test_structural_data_capability_report_accepts_liquidation_quality_and_local_events_but_blocks_codegen(tmp_path):
    liquidation_path = (
        tmp_path
        / "data"
        / "market_structure"
        / "bybit"
        / "futures"
        / "BTC_USDT_USDT-liquidation.parquet"
    )
    liquidation_path.parent.mkdir(parents=True)
    liquidation_path.write_text("placeholder", encoding="utf-8")
    quality_path = tmp_path / "registry" / "strategies" / "checks" / "liquidation_quality.json"
    _write_passing_structural_quality_report(quality_path)

    artifact = build_structural_data_capability_report(
        StructuralDataCapabilityInputs(
            root_dir=tmp_path,
            liquidation_paths=[liquidation_path],
            liquidation_quality_report_paths=[quality_path],
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    liquidation = artifact["capabilities"]["liquidation"]
    assert liquidation["local_data_present"] is True
    assert liquidation["local_quality_ok"] is True
    assert liquidation["research_selection_quality_gate_supported"] is True
    assert liquidation["local_event_supported"] is True
    assert liquidation["strategy_codegen_supported"] is False
    assert "liquidation" not in artifact["proposal_guidance"]["blocked_without_new_data"]
    assert "liquidation" in artifact["proposal_guidance"]["local_research_usable"]
    assert "liquidation" in artifact["proposal_guidance"]["must_not_codegen"]
    assert (
        "liquidation_strategy_codegen_variant_before_promotion"
        in artifact["proposal_guidance"]["next_data_needed"]
    )


def test_pair_to_ohlcv_filename_uses_futures_suffix():
    assert pair_to_ohlcv_filename("BTC/USDT:USDT", "5m", "futures") == (
        "BTC_USDT_USDT-5m-futures.parquet"
    )


def test_freqai_dependency_check_reports_installed_dependency():
    def fake_import_module(name):
        assert name == "lightgbm"
        return SimpleNamespace(__version__="1.2.3")

    def fake_package_version(name):
        assert name == "lightgbm"
        return "4.6.0"

    report = check_freqai_dependencies(
        [DependencySpec("lightgbm")],
        import_module=fake_import_module,
        package_version=fake_package_version,
    )

    assert report.ok
    assert report.dependencies[0].installed
    assert report.dependencies[0].version == "4.6.0"
    assert report.dependencies[0].error is None


def test_freqai_dependency_check_reports_missing_dependency():
    def fake_import_module(name):
        raise ModuleNotFoundError(f"No module named '{name}'")

    report = check_freqai_dependencies(
        [DependencySpec("xgboost")],
        import_module=fake_import_module,
    )

    assert not report.ok
    assert not report.dependencies[0].installed
    assert report.dependencies[0].version is None
    assert "ModuleNotFoundError" in report.dependencies[0].error


def test_freqai_validation_accepts_prefixed_features_targets_and_target_shift(tmp_path):
    strategy_path = tmp_path / "GoodFreqAIStrategy.py"
    strategy_path.write_text(
        "def populate_indicators(self, dataframe, metadata):\n"
        "    dataframe = self.freqai.start(dataframe, metadata, self)\n"
        "    return dataframe\n"
        "\n"
        "def feature_engineering_expand_all(dataframe, period, metadata):\n"
        "    dataframe['%-rsi-period'] = 50\n"
        "    return dataframe\n"
        "\n"
        "def set_freqai_targets(dataframe, metadata):\n"
        "    label_period = 12\n"
        "    dataframe['&-future_return'] = dataframe['close'].shift(-label_period)\n"
        "    return dataframe\n",
        encoding="utf-8",
    )

    report = validate_freqai_strategy_paths([strategy_path])

    assert report.ok
    assert [column.column for column in report.feature_columns] == ["%-rsi-period"]
    assert [column.column for column in report.target_columns] == ["&-future_return"]
    assert report.allowed_target_shift_lines[0]["function"] == "set_freqai_targets"
    assert report.label_notice == FREQAI_LABEL_NOTICE


def test_freqai_validation_rejects_missing_freqai_start(tmp_path):
    strategy_path = tmp_path / "NoStartFreqAIStrategy.py"
    strategy_path.write_text(
        "def feature_engineering_expand_all(dataframe, period, metadata):\n"
        "    dataframe['%-rsi-period'] = 50\n"
        "    return dataframe\n"
        "\n"
        "def set_freqai_targets(dataframe, metadata):\n"
        "    dataframe['&-future_return'] = dataframe['close'].shift(-12)\n"
        "    return dataframe\n",
        encoding="utf-8",
    )

    report = validate_freqai_strategy_paths([strategy_path])

    assert not report.ok
    assert "freqai_start_required" in {finding.rule for finding in report.findings}


def test_freqai_validation_rejects_unprefixed_features_and_targets(tmp_path):
    strategy_path = tmp_path / "BadFreqAIStrategy.py"
    strategy_path.write_text(
        "def feature_engineering_expand_all(dataframe, period, metadata):\n"
        "    dataframe['rsi'] = 50\n"
        "    return dataframe\n"
        "\n"
        "def set_freqai_targets(dataframe, metadata):\n"
        "    dataframe['future_return'] = dataframe['close'].pct_change()\n"
        "    return dataframe\n",
        encoding="utf-8",
    )

    report = validate_freqai_strategy_paths([strategy_path])

    assert not report.ok
    assert {finding.rule for finding in report.findings} >= {
        "freqai_feature_prefix",
        "freqai_target_prefix",
    }


def test_freqai_validation_rejects_negative_shift_outside_targets(tmp_path):
    strategy_path = tmp_path / "BadFreqAIStrategy.py"
    strategy_path.write_text(
        "def populate_entry_trend(dataframe, metadata):\n"
        "    dataframe['future_close'] = dataframe['close'].shift(-2)\n"
        "    return dataframe\n",
        encoding="utf-8",
    )

    report = validate_freqai_strategy_paths([strategy_path])

    assert not report.ok
    assert [finding.rule for finding in report.findings] == ["freqai_shift_outside_targets"]


def test_freqai_model_name_prefers_explicit_then_config():
    config = {"freqaimodel": "ConfigModel", "freqai": {"freqaimodel": "NestedModel"}}

    assert freqai_model_name(config, "ExplicitModel") == "ExplicitModel"
    assert freqai_model_name(config) == "ConfigModel"
    assert freqai_model_name({"freqai": {"freqaimodel": "NestedModel"}}) == "NestedModel"


def test_candidate_freqai_identifier_is_stable_and_candidate_scoped():
    identifier = candidate_freqai_identifier(
        "Hybrid Strategy",
        "20260506T094000JST_hybrid_ml_smoke",
        "future_return",
    )

    assert identifier == candidate_freqai_identifier(
        "Hybrid Strategy",
        "20260506T094000JST_hybrid_ml_smoke",
        "future_return",
    )
    assert identifier != candidate_freqai_identifier(
        "Hybrid Strategy",
        "other_candidate",
        "future_return",
    )
    assert identifier.startswith("bf_hybrid_strategy_")
    assert len(identifier) <= 96
    assert all(ch.isalnum() or ch in {"_", "-"} for ch in identifier)


def test_freqai_identifier_override_config_contains_only_identifier(tmp_path):
    path = tmp_path / "freqai_identifier_override.json"

    write_freqai_identifier_override_config("bf_candidate_123", path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "freqai": {"identifier": "bf_candidate_123"}
    }


def test_freqai_input_ohlcv_paths_include_corr_pairs_and_timeframes(tmp_path):
    config = {
        "timeframe": "5m",
        "trading_mode": "futures",
        "exchange": {
            "name": "bybit",
            "pair_whitelist": ["BTC/USDT:USDT"],
        },
        "freqai": {
            "feature_parameters": {
                "include_corr_pairlist": ["ETH/USDT:USDT"],
                "include_timeframes": ["15m", "5m"],
            }
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    pairs = freqai_input_pairs(config)
    timeframes = freqai_input_timeframes(config)
    paths = resolve_ohlcv_input_paths(
        config_path=config_path,
        config=config,
        userdir=tmp_path / "user_data",
        pairs=pairs,
        timeframes=timeframes,
    )

    assert pairs == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert timeframes == ["5m", "15m"]
    assert [path.name for path in paths] == [
        "BTC_USDT_USDT-5m-futures.parquet",
        "BTC_USDT_USDT-15m-futures.parquet",
        "ETH_USDT_USDT-5m-futures.parquet",
        "ETH_USDT_USDT-15m-futures.parquet",
    ]


def test_freqai_metadata_uses_relative_paths(tmp_path):
    run_dir = tmp_path / "data" / "freqai" / "SampleStrategy" / "run1"
    metadata = build_freqai_metadata(
        root_dir=tmp_path,
        strategy="SampleStrategy",
        run_id="run1",
        status="completed",
        config_paths=[tmp_path / "user_data" / "config.json"],
        freqaimodel="CatboostRegressor",
        freqai_id="sample_freqai",
        timeframe="5m",
        timerange="20250101-20250103",
        pairs=["BTC/USDT:USDT"],
        dependency_status={"ok": True},
        artifact_paths={"result": run_dir / "result.json", "skipped": None},
        notes=["backtest only"],
    )

    assert metadata["config_paths"] == [str(Path("user_data") / "config.json")]
    assert metadata["artifact_paths"]["result"] == str(
        Path("data") / "freqai" / "SampleStrategy" / "run1" / "result.json"
    )
    assert "skipped" not in metadata["artifact_paths"]
    assert metadata["safety_scope"]["live_trading"] is False
    assert metadata["safety_scope"]["metadata_contains_secrets"] is False


def test_walk_forward_generates_rolling_windows():
    windows = generate_rolling_windows(
        start="20250101",
        end="20250108",
        train_days=2,
        test_days=1,
        step_days=2,
    )

    assert [window.timerange for window in windows] == [
        "20250101-20250104",
        "20250103-20250106",
        "20250105-20250108",
    ]
    assert windows[0].train_start == "20250101"
    assert windows[0].test_start == "20250103"


def test_walk_forward_parses_fixed_and_train_test_windows():
    windows = parse_window_specs(
        ["20250105-20250107", "20250101:20250103:20250103:20250105"]
    )

    assert windows[0].timerange == "20250105-20250107"
    assert windows[0].test_start == "20250105"
    assert windows[1].timerange == "20250101-20250105"
    assert window_run_id("wf", windows[1]) == (
        "wf_02_train_20250101_20250103_test_20250103_20250105"
    )


def test_walk_forward_aggregates_window_metrics():
    window_results = [
        {
            "status": "completed",
            "gate_recommendation": "pass",
            "window": {"index": 1, "timerange": "20250101-20250103"},
            "metrics": {
                "total_return": 0.02,
                "total_return_pct": 2.0,
                "max_drawdown_pct": 4.0,
            },
        },
        {
            "status": "completed",
            "gate_recommendation": "pass",
            "window": {"index": 2, "timerange": "20250103-20250105"},
            "metrics": {
                "total_return": 0.015,
                "total_return_pct": 1.5,
                "max_drawdown_pct": 6.0,
            },
        },
    ]

    metrics = aggregate_walk_forward_results(
        window_results,
        WalkForwardRules(
            min_pass_rate=1.0,
            min_profitable_windows_ratio=1.0,
            max_drawdown_pct_any_window=10.0,
            max_single_window_profit_dependency=0.6,
        ),
    )

    assert metrics["recommendation"] == "pass"
    assert metrics["summary"]["window_count"] == 2
    assert metrics["summary"]["total_return"] == 0.035
    assert metrics["summary"]["max_drawdown_pct_any_window"] == 6.0


def test_walk_forward_aggregates_candidate_identity_lineage():
    from freqtrade_ext.bot_factory.candidate_identity import build_strategy_candidate_identity

    identity = build_strategy_candidate_identity(
        candidate_id="cand-wf",
        strategy_id="strategy-wf",
        strategy_class_name="WalkForwardStrategy",
        strategy_source_path="user_data/strategies/WalkForwardStrategy.py",
        strategy_version="strategy_wf_v1",
        signal_version="signal_wf_v1",
        risk_policy_version="risk_wf_v1",
        regime_classifier_version="regime_wf_v1",
        cost_model_id="cost_wf_v1",
        allowed_pairs=["BTC/USDT:USDT"],
        allowed_timeframes=["5m"],
        created_at="2026-05-21T00:00:00+00:00",
        source_artifacts={"strategy_source": "user_data/strategies/WalkForwardStrategy.py"},
    )
    window_results = [
        {
            "status": "completed",
            "gate_recommendation": "pass",
            "run_id": "wf_01",
            "window": {"index": 1, "timerange": "20250101-20250103"},
            "metrics": {
                "total_return": 0.02,
                "total_return_pct": 2.0,
                "max_drawdown_pct": 4.0,
                "candidate_identity": identity,
            },
        },
        {
            "status": "completed",
            "gate_recommendation": "pass",
            "run_id": "wf_02",
            "window": {"index": 2, "timerange": "20250103-20250105"},
            "metrics": {
                "total_return": 0.015,
                "total_return_pct": 1.5,
                "max_drawdown_pct": 6.0,
                "candidate_identity": identity,
            },
        },
    ]

    metrics = aggregate_walk_forward_results(
        window_results,
        WalkForwardRules(
            min_pass_rate=1.0,
            min_profitable_windows_ratio=1.0,
            max_drawdown_pct_any_window=10.0,
            max_single_window_profit_dependency=0.6,
        ),
        candidate_identity=identity,
    )

    identity_check = next(
        check for check in metrics["checks"] if check["name"] == "candidate_identity_lineage"
    )
    assert metrics["recommendation"] == "pass"
    assert metrics["candidate_identity"] == identity
    assert metrics["identity_lineage_validation"]["ok"] is True
    assert identity_check["pass"] is True


def test_walk_forward_identity_mismatch_fails_recommendation():
    from freqtrade_ext.bot_factory.candidate_identity import build_strategy_candidate_identity

    identity = build_strategy_candidate_identity(
        candidate_id="cand-wf",
        strategy_id="strategy-wf",
        strategy_class_name="WalkForwardStrategy",
        strategy_source_path="user_data/strategies/WalkForwardStrategy.py",
        strategy_version="strategy_wf_v1",
        signal_version="signal_wf_v1",
        risk_policy_version="risk_wf_v1",
        regime_classifier_version="regime_wf_v1",
        cost_model_id="cost_wf_v1",
        allowed_pairs=["BTC/USDT:USDT"],
        allowed_timeframes=["5m"],
        created_at="2026-05-21T00:00:00+00:00",
        source_artifacts={"strategy_source": "user_data/strategies/WalkForwardStrategy.py"},
    )
    wrong_identity = dict(identity)
    wrong_identity["cost_model_id"] = "cost_wf_v2"

    metrics = aggregate_walk_forward_results(
        [
            {
                "status": "completed",
                "gate_recommendation": "pass",
                "run_id": "wf_01",
                "window": {"index": 1, "timerange": "20250101-20250103"},
                "metrics": {
                    "total_return": 0.02,
                    "total_return_pct": 2.0,
                    "max_drawdown_pct": 4.0,
                    "candidate_identity": wrong_identity,
                },
            }
        ],
        WalkForwardRules(
            min_pass_rate=1.0,
            min_profitable_windows_ratio=1.0,
            max_drawdown_pct_any_window=10.0,
            max_single_window_profit_dependency=1.0,
        ),
        candidate_identity=identity,
    )

    identity_check = next(
        check for check in metrics["checks"] if check["name"] == "candidate_identity_lineage"
    )
    assert metrics["recommendation"] == "fail"
    assert metrics["identity_lineage_validation"]["ok"] is False
    assert identity_check["pass"] is False


def test_training_child_run_id_sanitizes_timerange():
    assert training_child_run_id("train", "20250105-20250107") == (
        "train_20250105_20250107"
    )


def test_training_backtest_command_uses_checked_wrapper_only(tmp_path):
    identity_path = tmp_path / "candidate_identity.json"
    cmd = build_checked_freqai_backtest_command(
        python_executable=".venv/Scripts/python.exe",
        runner_script="scripts/bot_factory_run_freqai_backtest.py",
        config="user_data/config_freqai_phase2_safe.json",
        strategy="LongOnlyFreqAIStrategy",
        strategy_path="user_data/strategies",
        output_root=tmp_path / "freqai_backtests",
        run_id="train_20250105_20250107",
        timerange="20250105-20250107",
        timeframe="5m",
        pairs=["BTC/USDT:USDT"],
        freqai_identifier="bf_longonly_train",
        candidate_identity_json=identity_path,
        reviewer_notes=["training factory test"],
    )

    assert cmd[:2] == [
        ".venv/Scripts/python.exe",
        "scripts/bot_factory_run_freqai_backtest.py",
    ]
    assert "backtesting" not in cmd
    assert "trade" not in cmd
    assert cmd[cmd.index("--timerange") + 1] == "20250105-20250107"
    assert cmd[cmd.index("--pairs") + 1] == "BTC/USDT:USDT"
    assert cmd[cmd.index("--freqai-identifier") + 1] == "bf_longonly_train"
    assert cmd[cmd.index("--candidate-identity-json") + 1] == str(identity_path)


def test_training_walk_forward_command_accepts_windows_and_rules(tmp_path):
    identity_path = tmp_path / "candidate_identity.json"
    cmd = build_checked_walk_forward_command(
        python_executable=".venv/Scripts/python.exe",
        runner_script="scripts/bot_factory_run_walk_forward.py",
        config="user_data/config_freqai_phase2_safe.json",
        strategy="LongOnlyFreqAIStrategy",
        strategy_path="user_data/strategies",
        output_root=tmp_path / "walk_forward",
        run_id="wf_run",
        window_specs=["20250105-20250107", "20250107-20250109"],
        timeframe="5m",
        pairs=["BTC/USDT:USDT"],
        freqai_identifier="bf_longonly_wf",
        candidate_id="cand-wf",
        candidate_identity_json=identity_path,
        min_pass_rate=0.5,
    )

    assert cmd[:2] == [
        ".venv/Scripts/python.exe",
        "scripts/bot_factory_run_walk_forward.py",
    ]
    assert cmd.count("--window") == 2
    assert "20250105-20250107" in cmd
    assert cmd[cmd.index("--min-pass-rate") + 1] == "0.5"
    assert cmd[cmd.index("--freqai-identifier") + 1] == "bf_longonly_wf"
    assert cmd[cmd.index("--candidate-id") + 1] == "cand-wf"
    assert cmd[cmd.index("--candidate-identity-json") + 1] == str(identity_path)


def test_walk_forward_child_command_forwards_full_candidate_identity(tmp_path):
    import runpy

    module = runpy.run_path(str(Path("scripts/bot_factory_run_walk_forward.py")))
    window = parse_window_specs(["20250105-20250107"])[0]
    identity_path = tmp_path / "candidate_identity.json"
    args = SimpleNamespace(
        python=".venv/Scripts/python.exe",
        runner_script="scripts/bot_factory_run_freqai_backtest.py",
        config="user_data/config.json",
        strategy="FreqaiS",
        strategy_path="user_data/strategies",
        data_format_ohlcv="parquet",
        reviewer_note=[],
        freqaimodel=None,
        freqaimodel_path=None,
        freqai_identifier=None,
        timeframe="5m",
        pairs=["BTC/USDT:USDT"],
        userdir=None,
        datadir=None,
        trading_mode=None,
        ohlcv_file=None,
        gate_config=None,
        mlflow=False,
        mlflow_tracking_uri=None,
        mlflow_experiment="bot_factory_freqai_walk_forward",
        candidate_identity={"candidate_id": "cand-wf"},
        candidate_identity_path=identity_path,
    )

    cmd = module["_build_window_command"](
        args,
        window,
        "wf_20250105_20250107",
        tmp_path / "windows",
    )

    assert cmd[cmd.index("--candidate-id") + 1] == "cand-wf"
    assert cmd[cmd.index("--candidate-identity-json") + 1] == str(identity_path)


def test_walk_forward_fallback_identity_uses_freqai_child_timeframes(tmp_path):
    import runpy

    module = runpy.run_path(str(Path("scripts/bot_factory_run_walk_forward.py")))
    strategy_file = tmp_path / "FreqaiS.py"
    strategy_file.write_text("class FreqaiS:\n    pass\n", encoding="utf-8")
    config = {
        "timeframe": "5m",
        "exchange": {"pair_whitelist": ["BTC/USDT:USDT"]},
        "freqai": {"feature_parameters": {"include_timeframes": ["15m", "5m"]}},
    }
    args = SimpleNamespace(
        strategy="FreqaiS",
        candidate_id="cand-wf",
        timeframe="5m",
        pairs=None,
        runner_script="scripts/bot_factory_run_freqai_backtest.py",
        freqaimodel=None,
        freqaimodel_path=None,
        freqai_identifier=None,
    )

    identity = module["_resolve_candidate_identity"](
        args,
        config,
        strategy_file,
        "wf_run",
    )

    assert identity["candidate_id"] == "cand-wf"
    assert identity["allowed_timeframes"] == ["15m", "5m"]


def test_checked_wrappers_accept_full_candidate_identity_json(tmp_path):
    import runpy

    from freqtrade_ext.bot_factory.candidate_identity import build_strategy_candidate_identity

    strategy_file = tmp_path / "FreqaiS.py"
    strategy_file.write_text("class FreqaiS:\n    pass\n", encoding="utf-8")
    identity = build_strategy_candidate_identity(
        candidate_id="cand-wf",
        strategy_id="FreqaiS",
        strategy_class_name="FreqaiS",
        strategy_source_path=strategy_file,
        strategy_version="FreqaiS_v1",
        signal_version="signal_v1",
        risk_policy_version="risk_v1",
        regime_classifier_version="regime_v1",
        cost_model_id="cost_v1",
        allowed_pairs=["BTC/USDT:USDT"],
        allowed_timeframes=["15m", "5m"],
        created_at="2026-05-23T00:00:00+00:00",
        source_artifacts={"strategy_source": strategy_file},
        root_dir=tmp_path,
    )
    identity_path = tmp_path / "candidate_identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    args = SimpleNamespace(
        candidate_identity_json=str(identity_path),
        candidate_id="cand-wf",
        strategy="FreqaiS",
    )
    backtest = runpy.run_path(str(Path("scripts/bot_factory_run_backtest.py")))
    freqai_backtest = runpy.run_path(
        str(Path("scripts/bot_factory_run_freqai_backtest.py"))
    )
    walk_forward = runpy.run_path(str(Path("scripts/bot_factory_run_walk_forward.py")))
    training = runpy.run_path(str(Path("scripts/bot_factory_run_freqai_training.py")))

    backtest_identity = backtest["_resolve_candidate_identity"](
        args,
        strategy_file,
        "child_run",
    )
    freqai_identity = freqai_backtest["_resolve_candidate_identity"](
        args,
        {"timeframe": "5m"},
        strategy_file,
        "child_run",
    )
    walk_forward_identity = walk_forward["_resolve_candidate_identity"](
        args,
        {"timeframe": "5m"},
        strategy_file,
        "child_run",
    )
    training_identity = training["_resolve_candidate_identity"](
        args,
        {"timeframe": "5m"},
        "child_run",
    )

    assert backtest_identity == identity
    assert freqai_identity == identity
    assert walk_forward_identity == identity
    assert training_identity == identity


def test_training_manifest_keeps_local_artifacts_as_source_of_truth(tmp_path):
    run_dir = tmp_path / "data" / "freqai_training" / "LongOnlyFreqAIStrategy" / "run1"
    stage = TrainingStageResult(
        name="freqai_backtest",
        run_id="train_20250105_20250107",
        status="completed",
        returncode=0,
        output_dir=run_dir / "freqai_backtests" / "LongOnlyFreqAIStrategy",
        recommendation="fail",
        artifacts={"metrics": run_dir / "freqai_backtests" / "metrics.json"},
        command=[".venv/Scripts/python.exe", "scripts/bot_factory_run_freqai_backtest.py"],
    )

    manifest = build_training_manifest(
        root_dir=tmp_path,
        strategy="LongOnlyFreqAIStrategy",
        run_id="run1",
        config_path=tmp_path / "user_data" / "config_freqai_phase2_safe.json",
        timeframe="5m",
        timerange="20250105-20250107",
        pairs=["BTC/USDT:USDT"],
        freqaimodel="LightGBMRegressor",
        freqai_identifier="phase2_safe_long_only",
        dependency_status={"ok": True},
        stages=[stage],
        artifact_paths={"training_manifest": run_dir / "training_manifest.json"},
        notes=[FREQAI_LABEL_NOTICE],
    )

    assert manifest["status"] == "completed"
    assert manifest["recommendation"] == "fail"
    assert manifest["config_path"] == str(Path("user_data") / "config_freqai_phase2_safe.json")
    assert manifest["safety_scope"]["local_artifacts_source_of_truth"] is True
    assert manifest["safety_scope"]["live_trading"] is False
    assert manifest["stages"][0]["artifacts"]["metrics"] == str(
        Path("data")
        / "freqai_training"
        / "LongOnlyFreqAIStrategy"
        / "run1"
        / "freqai_backtests"
        / "metrics.json"
    )


def test_strategy_proposal_generator_writes_safe_markdown_and_metadata(tmp_path):
    evidence_path = (
        tmp_path / "registry" / "strategies" / "checks" / "ohlcv_quality.json"
    )
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(json.dumps({"ok": True, "rows": 1000}), encoding="utf-8")
    inputs = _strategy_proposal_inputs(
        tmp_path,
        evidence_paths=[
            StrategyProposalEvidenceInput("ohlcv_quality", evidence_path)
        ],
    )

    artifacts = build_strategy_proposal(inputs)
    write_strategy_proposal_artifacts(artifacts)

    assert artifacts.metadata["status"] == "accepted"
    assert artifacts.metadata["code_generation_eligible"] is True
    assert artifacts.metadata["edge_discovery_handoff"]["passed"] is True
    assert artifacts.metadata["edge_discovery_handoff"]["passing_edge_artifact_count"] == 1
    assert artifacts.proposal_path.is_file()
    assert artifacts.metadata_path.is_file()
    for section in REQUIRED_PROPOSAL_SECTIONS:
        assert f"## {section}" in artifacts.proposal_markdown
    assert len(artifacts.metadata["proposal_content_hash"]) == 64
    assert artifacts.metadata["strategy_name"] == "LongOnlyRsiPullbackCandidate"
    assert artifacts.metadata["created_by_agent"] == "codex-test"
    assert artifacts.metadata["source_input_paths"]["ohlcv_quality"] == [
        str(Path("registry") / "strategies" / "checks" / "ohlcv_quality.json")
    ]
    assert artifacts.metadata["safety_scope"]["historical_evaluation_only"] is True
    assert artifacts.metadata["safety_scope"]["shorting"] is False
    assert artifacts.metadata["allowed_data_classes"]
    assert artifacts.metadata["research_references"][0]["relevance"]
    assert artifacts.metadata["research_references"][0]["motivated_thesis_ids"] == [
        artifacts.metadata["thesis_id"]
    ]
    assert artifacts.metadata["research_brief"]["research_references"] == (
        artifacts.metadata["research_references"]
    )


def test_strategy_proposal_generator_blocks_forbidden_dependencies(tmp_path):
    inputs = _strategy_proposal_inputs(
        tmp_path,
        summary="Uses future close and live data to decide entries.",
        required_data=[
            "Account balance, position data, and exchange order endpoint data."
        ],
        risk_logic=(
            "Use 2x leverage with short entry signals and api_key = "
            "'abcdef1234567890'."
        ),
    )

    artifacts = build_strategy_proposal(inputs)

    assert artifacts.metadata["status"] == "blocked"
    assert artifacts.metadata["code_generation_eligible"] is False
    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert {
        "no_future_data_dependency",
        "no_live_only_data_dependency",
        "no_account_or_position_data_dependency",
        "no_order_endpoint_dependency",
        "no_api_key_or_secret_dependency",
        "no_leverage_above_one_dependency",
        "no_shorting_dependency",
    }.issubset(blocker_names)
    assert "abcdef1234567890" not in artifacts.proposal_markdown
    assert "abcdef1234567890" not in json.dumps(artifacts.metadata)
    assert "[REDACTED]" in artifacts.proposal_markdown


def test_strategy_proposal_generator_blocks_evidence_outside_workspace(tmp_path):
    outside_path = tmp_path.parent / f"{tmp_path.name}_outside_evidence.json"
    outside_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    inputs = _strategy_proposal_inputs(
        tmp_path,
        evidence_paths=[
            StrategyProposalEvidenceInput("outside_metrics", outside_path)
        ],
    )

    artifacts = build_strategy_proposal(inputs)

    assert artifacts.metadata["status"] == "blocked"
    assert artifacts.metadata["code_generation_eligible"] is False
    assert artifacts.metadata["rejected_or_blocked_evidence"][0]["label"] == (
        "outside_metrics"
    )
    assert artifacts.metadata["rejected_or_blocked_evidence"][0]["status"] == "blocked"
    assert "evidence_outside_metrics_within_workspace" in {
        check["name"] for check in artifacts.metadata["blockers"]
    }


def test_strategy_proposal_generator_blocks_without_passing_edge_discovery(tmp_path):
    inputs = _strategy_proposal_inputs(tmp_path, include_edge_discovery=False)

    artifacts = build_strategy_proposal(inputs)

    assert artifacts.metadata["status"] == "blocked"
    assert artifacts.metadata["code_generation_eligible"] is False
    handoff = artifacts.metadata["edge_discovery_handoff"]
    assert handoff["required"] is True
    assert handoff["passed"] is False
    assert handoff["artifact_count"] == 0
    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert "edge_discovery_handoff_artifact_present" in blocker_names
    assert "edge_discovery_handoff_passed" in blocker_names


def test_strategy_proposal_generator_blocks_failed_synthesis_repeats(tmp_path):
    synthesis_path = tmp_path / "registry" / "strategies" / "synthesis" / "s1.json"
    synthesis_path.parent.mkdir(parents=True)
    synthesis_path.write_text(
        json.dumps(
                {
                    "factory": "candidate_failure_synthesis",
                    "status": "completed",
                    "synthesis_id": "synth-test",
                    "ranking_path": "registry/strategies/candidates/rankings/ranking.json",
                    "next_research_brief": {
                    "requires_new_thesis_id": True,
                    "requires_new_research_references": True,
                    "minimum_research_reference_count": 2,
                    "parameter_only_retry_allowed": False,
                    "prior_hypothesis_families_to_avoid_as_default": [
                        "trend_continuation"
                    ],
                    "failed_thesis_ids": ["TH-FAILED"],
                    "blocked_next_actions": ["parameter_only_threshold_loosen"],
                },
            }
        ),
        encoding="utf-8",
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            thesis_id="TH-FAILED",
            thesis_type="trend_continuation",
            strategy_logic_variant="trend_continuation",
            parameter_only_retry_count=1,
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                )
            ],
        )
    )

    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert artifacts.metadata["status"] == "blocked"
    assert "failure_synthesis_1_blocks_parameter_only_retry" in blocker_names
    assert "failure_synthesis_1_requires_new_thesis_id" in blocker_names
    assert "failure_synthesis_1_requires_new_hypothesis_family" in blocker_names
    assert "failure_synthesis_1_minimum_research_references" in blocker_names
    constraints = artifacts.metadata["failure_synthesis_constraints"][0]
    assert constraints["failed_thesis_id_match"] is True
    assert constraints["repeated_family_matches"] == ["trend_continuation"]


def test_strategy_proposal_generator_accepts_distinct_synthesis_thesis(tmp_path):
    synthesis_path = tmp_path / "registry" / "strategies" / "synthesis" / "s1.json"
    synthesis_path.parent.mkdir(parents=True)
    synthesis_path.write_text(
        json.dumps(
            {
                "factory": "candidate_failure_synthesis",
                "status": "completed",
                "synthesis_id": "synth-test",
                "ranking_path": "registry/strategies/candidates/rankings/ranking.json",
                "next_research_brief": {
                    "requires_new_thesis_id": True,
                    "requires_new_research_references": True,
                    "minimum_research_reference_count": 2,
                    "parameter_only_retry_allowed": False,
                    "prior_hypothesis_families_to_avoid_as_default": [
                        "trend_continuation"
                    ],
                    "failed_thesis_ids": ["TH-FAILED"],
                    "blocked_next_actions": ["parameter_only_threshold_loosen"],
                },
            }
        ),
        encoding="utf-8",
    )
    references = [
        StrategyProposalResearchReference(
            reference_id="paper:fractal-long-memory-1",
            title="Long-Range Correlations in Cryptocurrency Markets",
            source="Local bibliography",
            published_at="2024",
            relevance="Motivates a distinct Hurst and long-memory thesis.",
            motivated_thesis_ids=["TH-NEW"],
        ),
        StrategyProposalResearchReference(
            reference_id="paper:fractal-long-memory-2",
            title="Hurst Exponents and Dynamic Bitcoin Market Efficiency",
            source="Local bibliography",
            published_at="2025",
            relevance="Supports falsifying long-memory regimes with OHLCV features.",
            motivated_thesis_ids=["TH-NEW"],
        ),
    ]
    question_handoff = {
        "required": True,
        "passed": False,
        "computed_missing_research_question_response_indexes": [2],
    }
    handoff_summaries = [
        {
            "candidate_id": "cand-handoff",
            "research_handoff_summary": {
                "research_decision_question_handoff": question_handoff,
            },
        }
    ]
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id="TH-NEW",
        research_handoff_summaries=handoff_summaries,
        blocked_next_actions=[
            "retry_validated_local_rejection_by_parameter_tuning",
        ],
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_type="fractal_long_memory",
            thesis_id="TH-NEW",
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=references,
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    assert artifacts.metadata["status"] == "accepted"
    assert artifacts.metadata["strategy_logic_variant"] == "fractal_long_memory_regime"
    constraints = artifacts.metadata["failure_synthesis_constraints"][0]
    assert constraints["failed_thesis_id_match"] is False
    assert constraints["repeated_family_matches"] == []
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["thesis_id_match"] is True
    assert decision_constraints["proposal_generation_allowed"] is True
    assert decision_constraints["research_handoff_summaries"] == handoff_summaries
    assert decision_constraints["blocked_next_actions"] == [
        "retry_validated_local_rejection_by_parameter_tuning"
    ]
    assert artifacts.metadata["research_brief"]["research_handoff_summaries"] == (
        handoff_summaries
    )
    assert artifacts.metadata["research_brief"]["blocked_next_actions"] == [
        "parameter_only_threshold_loosen",
        "retry_validated_local_rejection_by_parameter_tuning",
    ]


def test_strategy_proposal_generator_blocks_stale_failure_synthesis(tmp_path):
    synthesis_root = tmp_path / "registry" / "strategies" / "synthesis"
    old_path = synthesis_root / "old" / "candidate_failure_synthesis.json"
    new_path = synthesis_root / "new" / "candidate_failure_synthesis.json"

    def write_synthesis(path: Path, *, synthesis_id: str, generated_at: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "factory": "candidate_failure_synthesis",
                    "status": "completed",
                    "synthesis_id": synthesis_id,
                    "generated_at": generated_at,
                    "ranking_path": (
                        "registry/strategies/candidates/rankings/ranking.json"
                    ),
                    "next_research_brief": {
                        "requires_new_thesis_id": True,
                        "requires_new_research_references": True,
                        "minimum_research_reference_count": 2,
                        "parameter_only_retry_allowed": False,
                        "prior_hypothesis_families_to_avoid_as_default": [
                            "trend_continuation"
                        ],
                        "failed_thesis_ids": ["TH-FAILED"],
                    },
                }
            ),
            encoding="utf-8",
        )

    write_synthesis(
        old_path,
        synthesis_id="synth-test",
        generated_at="2026-05-04T00:00:00+00:00",
    )
    write_synthesis(
        new_path,
        synthesis_id="new-synth",
        generated_at="2026-05-05T00:00:00+00:00",
    )
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id="TH-NEW",
        novelty_assessment={
            "repeated_failed_family_matches": [],
            "failed_thesis_id_match": False,
            "failure_synthesis_latest_checked": True,
            "failure_synthesis_is_latest": False,
            "latest_failure_synthesis_path": str(
                new_path.resolve().relative_to(tmp_path.resolve())
            ),
            "latest_failure_synthesis_id": "new-synth",
            "latest_failure_synthesis_generated_at": "2026-05-05T00:00:00+00:00",
        },
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_type="fractal_long_memory",
            thesis_id="TH-NEW",
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=_strategy_proposal_fractal_references("TH-NEW"),
            evidence_paths=[
                StrategyProposalEvidenceInput("candidate_failure_synthesis", old_path),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert artifacts.metadata["status"] == "blocked"
    assert "failure_synthesis_1_is_latest" in blocker_names
    assert "research_decision_1_failure_synthesis_was_latest" in blocker_names
    constraints = artifacts.metadata["failure_synthesis_constraints"][0]
    assert constraints["failure_synthesis_latest_checked"] is True
    assert constraints["failure_synthesis_is_latest"] is False
    assert constraints["latest_failure_synthesis_id"] == "new-synth"
    assert constraints["latest_failure_synthesis_path"].replace("\\", "/").endswith(
        "registry/strategies/synthesis/new/candidate_failure_synthesis.json"
    )
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["failure_synthesis_latest_checked"] is True
    assert decision_constraints["failure_synthesis_is_latest"] is False


def test_strategy_proposal_generator_blocks_structural_data_without_quality_reported_research_decision(
    tmp_path,
):
    thesis_id = "TH-OPEN-INTEREST-QUALITY-001"
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id=thesis_id,
        thesis_family="open_interest_resilience",
        mechanism_class="open_interest_resilience",
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyOpenInterestQualityCandidate",
            strategy_type="open_interest_resilience",
            thesis_id=thesis_id,
            thesis_type="open_interest_resilience",
            strategy_logic_variant="fractal_long_memory_regime",
            summary="Long-only proposal using open-interest structural context.",
            hypothesis=(
                "BTC open-interest contractions may identify positioning stress "
                "only when local quality-checked structural data supports it."
            ),
            market_condition="BTC/USDT futures with local open-interest context.",
            entry_logic="Enter long only after a closed-candle open-interest stress recovery.",
            required_data=[
                "BTC/USDT:USDT 5m OHLCV closed candles",
                "BTCUSDT open-interest parquet with local quality report",
            ],
            research_references=_strategy_proposal_fractal_references(thesis_id),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    assert artifacts.metadata["status"] == "blocked"
    assert "research_decision_1_structural_data_quality_report_present" in {
        check["name"] for check in artifacts.metadata["blockers"]
    }
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["proposal_structural_data_required"] is True
    assert decision_constraints["structural_data_quality_report_paths"] == []
    assert decision_constraints["structural_data_quality_report_gate_passed"] is False


def test_strategy_proposal_generator_accepts_structural_data_with_quality_reported_research_decision(
    tmp_path,
):
    thesis_id = "TH-OPEN-INTEREST-QUALITY-001"
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    quality_path = (
        tmp_path
        / "registry"
        / "strategies"
        / "checks"
        / "open_interest_quality.json"
    )
    quality_path.parent.mkdir(parents=True)
    quality_path.write_text(
        json.dumps(
            {
                "ok": True,
                "reports": [
                    {
                        "path": "user_data/data/bybit/futures/BTC_USDT_USDT-1h-open_interest.parquet",
                        "ok": True,
                        "rows": 100,
                        "findings": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    capability_path = _write_structural_capability_report(
        tmp_path,
        local_research_usable=["open_interest"],
        blocked_without_new_data=["liquidation", "order_book"],
        must_not_codegen=["open_interest", "liquidation", "order_book"],
    )
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id=thesis_id,
        thesis_family="open_interest_resilience",
        mechanism_class="open_interest_resilience",
        local_data_quality_report_paths=[quality_path],
        structural_data_capability_report_paths=[capability_path],
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyOpenInterestQualityCandidate",
            strategy_type="open_interest_resilience",
            thesis_id=thesis_id,
            thesis_type="open_interest_resilience",
            strategy_logic_variant="fractal_long_memory_regime",
            summary="Long-only proposal using open-interest structural context.",
            hypothesis=(
                "BTC open-interest contractions may identify positioning stress "
                "only when local quality-checked structural data supports it."
            ),
            market_condition="BTC/USDT futures with local open-interest context.",
            entry_logic="Enter long only after a closed-candle open-interest stress recovery.",
            required_data=[
                "BTC/USDT:USDT 5m OHLCV closed candles",
                "BTCUSDT open-interest parquet with local quality report",
            ],
            research_references=_strategy_proposal_fractal_references(thesis_id),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
                StrategyProposalEvidenceInput("local_data_quality", quality_path),
            ],
        )
    )

    assert artifacts.metadata["status"] == "accepted"
    assert artifacts.metadata["structural_data_requirement"]["required"] is True
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["structural_data_quality_report_paths"] == [
        "registry\\strategies\\checks\\open_interest_quality.json"
    ]
    assert decision_constraints["structural_data_quality_report_gate_passed"] is True
    assert decision_constraints["structural_data_capability_report_paths"] == [
        "registry\\strategies\\checks\\structural_capability.json"
    ]
    assert decision_constraints["structural_data_capability_report_gate_passed"] is True
    assert (
        decision_constraints["structural_data_capability_required_classes_supported"]
        is True
    )


def test_strategy_proposal_generator_blocks_structural_data_without_capability_reported_research_decision(
    tmp_path,
):
    thesis_id = "TH-OPEN-INTEREST-CAPABILITY-001"
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    quality_path = tmp_path / "registry" / "strategies" / "checks" / "open_interest_quality.json"
    quality_path.parent.mkdir(parents=True)
    quality_path.write_text(
        json.dumps({"ok": True, "reports": [{"ok": True, "rows": 100}]}),
        encoding="utf-8",
    )
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id=thesis_id,
        thesis_family="open_interest_resilience",
        mechanism_class="open_interest_resilience",
        local_data_quality_report_paths=[quality_path],
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyOpenInterestCapabilityCandidate",
            strategy_type="open_interest_resilience",
            thesis_id=thesis_id,
            thesis_type="open_interest_resilience",
            strategy_logic_variant="fractal_long_memory_regime",
            summary="Long-only proposal using open-interest structural context.",
            hypothesis="BTC open-interest context requires capability continuity.",
            market_condition="BTC/USDT futures with local open-interest context.",
            entry_logic="Enter long only after open-interest evidence passes the research gate.",
            required_data=[
                "BTC/USDT:USDT 5m OHLCV closed candles",
                "BTCUSDT open-interest parquet with local quality report",
            ],
            research_references=_strategy_proposal_fractal_references(thesis_id),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
                StrategyProposalEvidenceInput("local_data_quality", quality_path),
            ],
        )
    )

    assert artifacts.metadata["status"] == "blocked"
    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert (
        "research_decision_1_structural_data_capability_report_present"
        in blocker_names
    )
    assert (
        "research_decision_1_structural_data_capability_supports_required_classes"
        in blocker_names
    )


def test_strategy_code_generator_blocks_structural_data_without_quality_handoff(
    tmp_path,
):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    metadata = json.loads(proposal_artifacts.metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "strategy_type": "open_interest_resilience",
            "thesis_type": "open_interest_resilience",
            "thesis_statement": (
                "Open interest structural positioning should support long-only "
                "entries only after a verified quality handoff."
            ),
            "structural_data_requirement": {
                "required": True,
                "terms": ["open interest"],
            },
            "research_decision_constraints": [],
        }
    )
    proposal_artifacts.metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )

    assert artifacts.metadata["status"] == "blocked"
    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert "structural_data_quality_handoff_passed" in blocker_names
    assert "structural_data_capability_handoff_passed" in blocker_names
    assert "structural_data_code_generation_supported" in blocker_names


def test_strategy_code_generator_blocks_high_risk_proposal_without_local_falsification_handoff(
    tmp_path,
):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    metadata = json.loads(proposal_artifacts.metadata_path.read_text(encoding="utf-8"))
    metadata["research_decision_constraints"] = [
        {
            "path": "registry\\strategies\\research_decisions\\rd.json",
            "causal_risk_weights_present": True,
            "causal_required_categories_to_address": [
                "cost_sensitive_mechanism",
                "walk_forward_fragility",
            ],
            "local_falsification_handoff_required": True,
            "local_falsification_handoff_passed": False,
            "local_falsification_artifact_count": 0,
            "local_falsification_parseable_artifact_count": 0,
            "local_falsification_matching_thesis_artifact_count": 0,
            "local_falsification_passing_cost_edge_artifact_count": 0,
            "local_falsification_paths_valid": False,
            "local_falsification_factory_valid": False,
            "local_falsification_safety_scope_valid": False,
            "local_falsification_event_source_valid": False,
            "local_falsification_event_source_context_alignment_valid": False,
            "local_falsification_event_source_failure_synthesis_guard_valid": False,
            "local_falsification_artifact_paths": [],
            "local_falsification_blocker_names": [],
        }
    ]
    proposal_artifacts.metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )

    assert artifacts.metadata["status"] == "blocked"
    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert "local_falsification_handoff_passed" in blocker_names
    assert artifacts.metadata["local_falsification_handoff"]["required"] is True
    assert artifacts.metadata["local_falsification_handoff"]["passed"] is False


def test_strategy_code_generator_blocks_research_decision_local_rejection_novelty_handoff(
    tmp_path,
):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    metadata = json.loads(proposal_artifacts.metadata_path.read_text(encoding="utf-8"))
    metadata["research_decision_constraints"] = [
        {
            "path": "registry\\strategies\\research_decisions\\rd.json",
            "failed_thesis_id_match": False,
            "repeated_failed_family_matches": [],
            "local_falsification_failed_thesis_ids": [
                "TH-VALIDATED-LOCAL-REJECTED"
            ],
            "local_falsification_failed_thesis_id_match": False,
            "local_falsification_failed_mechanism_tokens": [
                "validated_local_rejection_mechanism"
            ],
            "local_falsification_failed_mechanism_class_matches": [
                "validated_local_rejection_mechanism"
            ],
        }
    ]
    proposal_artifacts.metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )

    assert artifacts.metadata["status"] == "blocked"
    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert "research_decision_novelty_handoff_passed" in blocker_names
    assert "local_falsification_handoff_passed" not in blocker_names
    handoff = artifacts.metadata["research_decision_novelty_handoff"]
    assert handoff["required"] is True
    assert handoff["passed"] is False
    assert handoff["failed_candidate_count"] == 1
    assert handoff["failed_paths"] == [
        "registry\\strategies\\research_decisions\\rd.json"
    ]
    assert (
        artifacts.metadata["research_brief"]["research_decision_novelty_handoff"]
        == handoff
    )


def test_strategy_code_generator_blocks_research_decision_question_handoff(
    tmp_path,
):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    metadata = json.loads(proposal_artifacts.metadata_path.read_text(encoding="utf-8"))
    handoff_summaries = [
        {
            "candidate_id": "cand-handoff",
            "research_handoff_summary": {
                "research_decision_question_handoff": {
                    "required": True,
                    "passed": False,
                }
            },
        }
    ]
    metadata["research_decision_constraints"] = [
        {
            "path": "registry\\strategies\\research_decisions\\rd.json",
            "requires_research_question_responses": True,
            "required_research_questions": [
                "What mechanism survives after failed families are excluded?",
                "Why should expected edge exceed fee and turnover costs?",
            ],
            "research_question_response_indexes": [1],
            "missing_research_question_response_indexes": [],
            "computed_missing_research_question_response_indexes": [],
            "weak_research_question_response_indexes": [],
            "blocked_next_actions": [
                "retry_validated_local_rejection_by_parameter_tuning",
            ],
            "research_handoff_summaries": handoff_summaries,
        }
    ]
    proposal_artifacts.metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )

    assert artifacts.metadata["status"] == "blocked"
    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert "research_decision_question_handoff_passed" in blocker_names
    assert "research_decision_novelty_handoff_passed" not in blocker_names
    handoff = artifacts.metadata["research_decision_question_handoff"]
    assert handoff["required"] is True
    assert handoff["passed"] is False
    assert handoff["failed_candidate_count"] == 1
    assert handoff["failed_paths"] == [
        "registry\\strategies\\research_decisions\\rd.json"
    ]
    candidate = handoff["candidates"][0]
    assert candidate["reported_missing_research_question_response_indexes"] == []
    assert (
        candidate["recomputed_missing_research_question_response_indexes"]
        == [2]
    )
    assert candidate["missing_research_question_response_indexes"] == [2]
    assert (
        artifacts.metadata["research_brief"]["research_decision_question_handoff"]
        == handoff
    )
    assert artifacts.metadata["blocked_next_actions"] == [
        "retry_validated_local_rejection_by_parameter_tuning"
    ]
    assert artifacts.metadata["research_handoff_summaries"] == handoff_summaries
    assert artifacts.metadata["research_brief"]["blocked_next_actions"] == [
        "retry_validated_local_rejection_by_parameter_tuning"
    ]
    assert artifacts.metadata["research_brief"]["research_handoff_summaries"] == (
        handoff_summaries
    )


def test_strategy_code_generator_blocks_structural_data_without_capability_handoff(
    tmp_path,
):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    metadata = json.loads(proposal_artifacts.metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "strategy_type": "open_interest_resilience",
            "thesis_type": "open_interest_resilience",
            "thesis_statement": (
                "Open interest structural positioning should support long-only "
                "entries only after a verified capability handoff."
            ),
            "structural_data_requirement": {
                "required": True,
                "terms": ["open interest"],
            },
            "research_decision_constraints": [
                {
                    "path": "registry\\strategies\\research_decisions\\rd.json",
                    "structural_data_quality_report_paths": [
                        "registry\\strategies\\checks\\open_interest_quality.json"
                    ],
                    "structural_data_quality_reports_exist": True,
                    "structural_data_quality_reports_valid_check_passed": True,
                    "structural_data_quality_check_passed": True,
                    "structural_data_quality_report_gate_passed": True,
                }
            ],
        }
    )
    proposal_artifacts.metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )

    assert artifacts.metadata["status"] == "blocked"
    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert "structural_data_quality_handoff_passed" not in blocker_names
    assert "structural_data_capability_handoff_passed" in blocker_names
    assert "structural_data_code_generation_supported" in blocker_names
    assert artifacts.metadata["structural_data_quality_handoff"]["passed"] is True
    assert (
        artifacts.metadata["structural_data_capability_handoff"]["passed"] is False
    )


def test_strategy_code_generator_blocks_structural_data_until_logic_supported(
    tmp_path,
):
    thesis_id = "TH-OPEN-INTEREST-CODEGEN-001"
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    quality_path = (
        tmp_path
        / "registry"
        / "strategies"
        / "checks"
        / "open_interest_quality.json"
    )
    quality_path.parent.mkdir(parents=True)
    quality_path.write_text(
        json.dumps({"ok": True, "reports": [{"ok": True, "rows": 100}]}),
        encoding="utf-8",
    )
    capability_path = _write_structural_capability_report(
        tmp_path,
        local_research_usable=["open_interest"],
        blocked_without_new_data=["liquidation", "order_book"],
        must_not_codegen=["open_interest", "liquidation", "order_book"],
    )
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id=thesis_id,
        thesis_family="open_interest_resilience",
        mechanism_class="open_interest_resilience",
        local_data_quality_report_paths=[quality_path],
        structural_data_capability_report_paths=[capability_path],
    )
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyOpenInterestCodegenCandidate",
            strategy_type="open_interest_resilience",
            thesis_id=thesis_id,
            thesis_type="open_interest_resilience",
            strategy_logic_variant="fractal_long_memory_regime",
            summary="Long-only proposal using open-interest structural context.",
            hypothesis=(
                "BTC open-interest contractions may identify positioning stress "
                "only when local quality-checked structural data supports it."
            ),
            market_condition="BTC/USDT futures with local open-interest context.",
            entry_logic="Enter long only after a closed-candle open-interest stress recovery.",
            required_data=[
                "BTC/USDT:USDT 5m OHLCV closed candles",
                "BTCUSDT open-interest parquet with local quality report",
            ],
            research_references=_strategy_proposal_fractal_references(thesis_id),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
                StrategyProposalEvidenceInput("local_data_quality", quality_path),
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(
            tmp_path,
            proposal_artifacts.metadata_path,
            candidate_id="open_interest_codegen_guard",
        )
    )

    assert proposal_artifacts.metadata["status"] == "accepted"
    assert artifacts.metadata["status"] == "blocked"
    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert "structural_data_quality_handoff_passed" not in blocker_names
    assert "structural_data_capability_handoff_passed" not in blocker_names
    assert "structural_data_code_generation_supported" in blocker_names
    assert artifacts.metadata["structural_data_quality_handoff"]["passed"] is True
    assert artifacts.metadata["structural_data_capability_handoff"]["passed"] is True


def test_strategy_code_generator_supports_crowding_unwind_local_structural_parquet(
    tmp_path,
):
    thesis_id = "TH-CROWDING-UNWIND-CODEGEN-001"
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    quality_root = tmp_path / "registry" / "strategies" / "checks"
    open_interest_quality_path = quality_root / "open_interest_quality.json"
    long_short_quality_path = quality_root / "long_short_ratio_quality.json"
    open_interest_quality_path.parent.mkdir(parents=True, exist_ok=True)
    open_interest_quality_path.write_text(
        json.dumps({"ok": True, "reports": [{"ok": True, "rows": 1000}]}),
        encoding="utf-8",
    )
    long_short_quality_path.write_text(
        json.dumps({"ok": True, "reports": [{"ok": True, "rows": 1000}]}),
        encoding="utf-8",
    )
    capability_path = _write_structural_capability_report(
        tmp_path,
        local_research_usable=["open_interest", "long_short_ratio"],
        blocked_without_new_data=["liquidation", "order_book"],
        must_not_codegen=["liquidation", "order_book"],
    )
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id=thesis_id,
        thesis_family="crowding_unwind_reaccumulation",
        mechanism_class="crowding_unwind_reaccumulation",
        local_data_quality_report_paths=[
            open_interest_quality_path,
            long_short_quality_path,
        ],
        structural_data_capability_report_paths=[capability_path],
    )
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyCrowdingUnwindCandidate",
            strategy_type="crowding_unwind_reaccumulation",
            thesis_id=thesis_id,
            thesis_type="crowding_unwind_reaccumulation",
            strategy_logic_variant="crowding_unwind_reaccumulation",
            summary=(
                "Long-only proposal using open-interest unwind and long/short "
                "account-ratio reaccumulation context."
            ),
            hypothesis=(
                "BTC positioning unwind in open interest plus depressed long/short "
                "account ratio can identify reaccumulation after crowding clears."
            ),
            market_condition=(
                "BTC/USDT futures with local open-interest and long/short "
                "account-ratio parquet context."
            ),
            entry_logic=(
                "Enter long only after closed-candle open-interest unwind, "
                "long/short account-ratio reaccumulation, positive SMA location, "
                "and adequate volume participation."
            ),
            exit_logic=(
                "Exit when open interest re-expands, account ratio recrowds, "
                "SMA support fails, participation disappears, or RSI target is reached."
            ),
            required_data=[
                "BTC/USDT:USDT 5m OHLCV closed candles",
                "BTCUSDT open-interest parquet with local quality report",
                "BTCUSDT long/short account-ratio parquet with local quality report",
            ],
            parameters=[
                "open_interest_delta_pct_288 <= -0.75",
                "long_short_ratio_zscore_864 <= -0.75",
                "sma_distance_bps_144 >= 0",
                "volume_zscore_288 >= -0.25",
                "sell_timeout_candles=72",
            ],
            research_references=_strategy_proposal_fractal_references(thesis_id),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
                StrategyProposalEvidenceInput("open_interest_quality", open_interest_quality_path),
                StrategyProposalEvidenceInput("long_short_ratio_quality", long_short_quality_path),
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(
            tmp_path,
            proposal_artifacts.metadata_path,
            candidate_id="crowding_unwind_codegen",
        )
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert proposal_artifacts.metadata["status"] == "accepted"
    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "crowding_unwind_reaccumulation"
    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert "structural_data_code_generation_supported" not in blocker_names
    assert "_attach_local_crowding_features" in generated_code
    assert "pd.read_parquet" in generated_code
    assert "open_interest_delta_pct_288" in generated_code
    assert "long_short_ratio_zscore_864" in generated_code
    assert "sma_distance_bps_144" in generated_code
    assert "volume_zscore_288" in generated_code
    assert "can_short = False" in generated_code


def test_strategy_proposal_generator_accepts_bipower_jump_decay_after_research_gate(
    tmp_path,
):
    thesis_id = "TH-BIPOWER-JUMP-DECAY-001"
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id=thesis_id,
        thesis_family="bipower_jump_decay",
        mechanism_class="realized_multipower_jump_decay",
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyBipowerJumpDecayCandidate",
            strategy_type="bipower_jump_decay",
            strategy_logic_variant="bipower_jump_decay",
            thesis_id=thesis_id,
            thesis_type="realized_multipower_jump_decay",
            summary="Long-only proposal for post-jump continuous-variance decay.",
            hypothesis=(
                "Large positive BTC jump events may have continuation edge only "
                "when bipower variation implies continuous variance is decaying."
            ),
            market_condition="BTC/USDT futures 5m closed-candle OHLCV only.",
            entry_logic=(
                "Enter after a positive jump event when jump variation dominates "
                "continuous variation and post-jump drift remains positive."
            ),
            exit_logic=(
                "Exit when continuous variance expands, jump edge fades, "
                "post-jump drift fails, or timeout risk appears."
            ),
            required_data=["BTC/USDT:USDT 5m OHLCV closed candles"],
            parameters=[
                "Jump lookback",
                "jump variation ratio floor",
                "continuous variance decay ceiling",
                "post-jump drift window",
            ],
            rejection_conditions=[
                "Post-jump entries have negative expectancy after costs.",
                "No walk-forward window is profitable.",
                "Events are too sparse for local falsification.",
            ],
            research_references=_strategy_proposal_bipower_references(thesis_id),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    assert artifacts.metadata["status"] == "accepted"
    assert artifacts.metadata["strategy_logic_variant"] == "bipower_jump_decay"
    assert "bipower_variation" in artifacts.metadata["feature_list"]
    assert "jump_variation_ratio" in artifacts.metadata["feature_list"]
    assert "continuous_variance_decaying" in artifacts.metadata["rule_filters"]
    constraints = artifacts.metadata["failure_synthesis_constraints"][0]
    assert constraints["failed_thesis_id_match"] is False
    assert constraints["repeated_family_matches"] == []
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["thesis_id_match"] is True
    assert decision_constraints["proposal_generation_allowed"] is True


def test_strategy_proposal_generator_accepts_directional_change_overshoot_after_research_gate(
    tmp_path,
):
    thesis_id = "TH-DIRECTIONAL-CHANGE-OVERSHOOT-001"
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id=thesis_id,
        thesis_family="directional_change_overshoot",
        mechanism_class="event_time_overshoot_continuation_reversal",
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyDirectionalChangeOvershootCandidate",
            strategy_type="directional_change_overshoot",
            strategy_logic_variant="directional_change_overshoot",
            thesis_id=thesis_id,
            thesis_type="directional_change_overshoot",
            summary="Long-only proposal for event-time directional-change overshoot states.",
            hypothesis=(
                "BTC futures may have a closed-candle event-time edge after "
                "completed directional-change reversal and persistent overshoot."
            ),
            market_condition="BTC/USDT futures 5m closed-candle OHLCV only.",
            entry_logic=(
                "Enter only after a directional-change event is confirmed, "
                "overshoot persists, and adverse reversal remains absent."
            ),
            exit_logic=(
                "Exit when overshoot persistence fails, adverse reversal appears, "
                "or event-time trend no longer supports long exposure."
            ),
            required_data=["BTC/USDT:USDT 5m OHLCV closed candles"],
            parameters=[
                "Directional-change event threshold",
                "overshoot persistence window",
                "adverse reversal distance",
                "event-time turnover guard",
            ],
            rejection_conditions=[
                "Directional-change entries have negative expectancy after costs.",
                "No walk-forward window is profitable.",
                "Event-time entries are too sparse for local falsification.",
            ],
            research_references=_strategy_proposal_directional_change_references(thesis_id),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    assert artifacts.metadata["status"] == "accepted"
    assert artifacts.metadata["strategy_logic_variant"] == "directional_change_overshoot"
    assert "directional_change_state" in artifacts.metadata["feature_list"]
    assert "overshoot_ratio" in artifacts.metadata["feature_list"]
    assert "overshoot_persisted" in artifacts.metadata["rule_filters"]
    assert "adverse_reversal_absent" in artifacts.metadata["rule_filters"]
    constraints = artifacts.metadata["failure_synthesis_constraints"][0]
    assert constraints["failed_thesis_id_match"] is False
    assert constraints["repeated_family_matches"] == []
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["thesis_id_match"] is True
    assert decision_constraints["proposal_generation_allowed"] is True


def test_strategy_proposal_generator_accepts_range_quarticity_after_research_gate(
    tmp_path,
):
    thesis_id = "TH-RANGE-QUARTICITY-VOL-OF-VOL-001"
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id=thesis_id,
        thesis_family="range_quarticity_vol_of_vol_state",
        mechanism_class="ohlc_quarticity_volatility_state_transition",
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyRangeQuarticityVolOfVolCandidate",
            strategy_type="range_quarticity_vol_of_vol_state",
            strategy_logic_variant="range_quarticity_vol_of_vol_state",
            thesis_id=thesis_id,
            thesis_type="range_quarticity_vol_of_vol_state",
            summary="Long-only proposal for OHLC range-quarticity volatility-state decay.",
            hypothesis=(
                "BTC futures may have a closed-candle edge when range-based "
                "quarticity and volatility-of-volatility decay after local stress."
            ),
            market_condition="BTC/USDT futures 5m closed-candle OHLCV only.",
            entry_logic=(
                "Enter only after range-quarticity stress decays, realized range "
                "stabilizes, and participation remains present."
            ),
            exit_logic=(
                "Exit when range-quarticity stress expands again, realized range "
                "destabilizes, or stabilization drift fails."
            ),
            required_data=["BTC/USDT:USDT 5m OHLCV closed candles"],
            parameters=[
                "Range window",
                "quarticity state window",
                "volatility-of-volatility decay window",
                "participation recovery floor",
            ],
            rejection_conditions=[
                "Range-quarticity entries have negative expectancy after costs.",
                "No walk-forward window is profitable.",
                "One component eliminates almost all entry rows.",
            ],
            research_references=_strategy_proposal_range_quarticity_references(thesis_id),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    assert artifacts.metadata["status"] == "accepted"
    assert artifacts.metadata["strategy_logic_variant"] == "range_quarticity_vol_of_vol_state"
    assert "range_quarticity_proxy" in artifacts.metadata["feature_list"]
    assert "range_vol_of_vol_state" in artifacts.metadata["feature_list"]
    assert "range_quarticity_state_decay" in artifacts.metadata["rule_filters"]
    assert "participation_present" in artifacts.metadata["rule_filters"]
    constraints = artifacts.metadata["failure_synthesis_constraints"][0]
    assert constraints["failed_thesis_id_match"] is False
    assert constraints["repeated_family_matches"] == []
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["thesis_id_match"] is True
    assert decision_constraints["proposal_generation_allowed"] is True


def test_strategy_code_generator_supports_directional_change_overshoot(
    tmp_path,
):
    thesis_id = "TH-DIRECTIONAL-CHANGE-OVERSHOOT-001"
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyDirectionalChangeOvershootCandidate",
            strategy_type="directional_change_overshoot",
            strategy_logic_variant="directional_change_overshoot",
            thesis_id=thesis_id,
            thesis_type="directional_change_overshoot",
            research_references=_strategy_proposal_directional_change_references(thesis_id),
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "directional_change_overshoot"
    assert artifacts.metadata["strategy_code_generated"] is True
    assert artifacts.metadata["candidate_evaluation_eligible"] is True
    assert artifacts.metadata["parameter_defaults"]["buy_pullback_lookback"] == 18
    assert "directional_change_state" in generated_code
    assert "directional_change_event_age" in generated_code
    assert "overshoot_ratio" in generated_code
    assert "event_time_trend" in generated_code
    assert "adverse_reversal_distance" in generated_code
    assert "directional_change_overshoot" in generated_code
    assert "overshoot_failed_or_reversal_exit" in generated_code
    assert "rsi_pullback_recovery" not in generated_code
    assert "shift(-1" not in generated_code
    assert scan_paths([artifacts.strategy_path]).ok


def test_strategy_code_generator_supports_range_quarticity_vol_of_vol(
    tmp_path,
):
    thesis_id = "TH-RANGE-QUARTICITY-VOL-OF-VOL-001"
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyRangeQuarticityVolOfVolCandidate",
            strategy_type="range_quarticity_vol_of_vol_state",
            strategy_logic_variant="range_quarticity_vol_of_vol_state",
            thesis_id=thesis_id,
            thesis_type="range_quarticity_vol_of_vol_state",
            research_references=_strategy_proposal_range_quarticity_references(thesis_id),
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "range_quarticity_vol_of_vol_state"
    assert artifacts.metadata["strategy_code_generated"] is True
    assert artifacts.metadata["candidate_evaluation_eligible"] is True
    assert artifacts.metadata["parameter_defaults"]["buy_pullback_lookback"] == 24
    assert "range_quarticity_proxy" in generated_code
    assert "range_vol_of_vol_state" in generated_code
    assert "range_state_decay" in generated_code
    assert "participation_recovery" in generated_code
    assert "range_quarticity_vol_of_vol_state" in generated_code
    assert "range_quarticity_stress_exit" in generated_code
    assert "rsi_pullback_recovery" not in generated_code
    assert "shift(-1" not in generated_code
    assert scan_paths([artifacts.strategy_path]).ok


def test_strategy_code_generator_blocks_unknown_strategy_logic_variant(tmp_path):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    metadata = json.loads(proposal_artifacts.metadata_path.read_text(encoding="utf-8"))
    metadata["strategy_logic_variant"] = "unsupported_new_family"
    proposal_artifacts.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)

    assert artifacts.metadata["status"] == "blocked"
    assert artifacts.metadata["strategy_code_generated"] is False
    assert "strategy_logic_variant_supported" in {
        check["name"] for check in artifacts.metadata["blockers"]
    }
    assert artifacts.strategy_code is None


def test_strategy_proposal_generator_requires_research_decision_after_synthesis(tmp_path):
    synthesis_path = tmp_path / "registry" / "strategies" / "synthesis" / "s1.json"
    synthesis_path.parent.mkdir(parents=True)
    synthesis_path.write_text(
        json.dumps(
            {
                "factory": "candidate_failure_synthesis",
                "status": "completed",
                "ranking_path": "registry/strategies/candidates/rankings/ranking.json",
                "next_research_brief": {
                    "requires_new_thesis_id": True,
                    "requires_new_research_references": True,
                    "minimum_research_reference_count": 2,
                    "parameter_only_retry_allowed": False,
                    "prior_hypothesis_families_to_avoid_as_default": [
                        "trend_continuation"
                    ],
                    "failed_thesis_ids": ["TH-FAILED"],
                },
            }
        ),
        encoding="utf-8",
    )
    references = [
        StrategyProposalResearchReference(
            reference_id="paper:fractal-long-memory-1",
            title="Long-Range Correlations in Cryptocurrency Markets",
            source="Local bibliography",
            published_at="2024",
            relevance="Motivates a distinct Hurst and long-memory thesis.",
            motivated_thesis_ids=["TH-NEW"],
        ),
        StrategyProposalResearchReference(
            reference_id="paper:fractal-long-memory-2",
            title="Hurst Exponents and Dynamic Bitcoin Market Efficiency",
            source="Local bibliography",
            published_at="2025",
            relevance="Supports falsifying long-memory regimes with OHLCV features.",
            motivated_thesis_ids=["TH-NEW"],
        ),
    ]

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_type="fractal_long_memory",
            thesis_id="TH-NEW",
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=references,
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                )
            ],
        )
    )

    assert artifacts.metadata["status"] == "blocked"
    assert "research_decision_required_for_failure_synthesis" in {
        check["name"] for check in artifacts.metadata["blockers"]
    }


def test_strategy_proposal_generator_blocks_unapproved_research_decision(tmp_path):
    synthesis_path = tmp_path / "registry" / "strategies" / "synthesis" / "s1.json"
    synthesis_path.parent.mkdir(parents=True)
    synthesis_path.write_text(
        json.dumps(
            {
                "factory": "candidate_failure_synthesis",
                "status": "completed",
                "ranking_path": "registry/strategies/candidates/rankings/ranking.json",
                "next_research_brief": {
                    "requires_new_thesis_id": True,
                    "requires_new_research_references": True,
                    "minimum_research_reference_count": 2,
                    "parameter_only_retry_allowed": False,
                    "prior_hypothesis_families_to_avoid_as_default": [
                        "trend_continuation"
                    ],
                    "failed_thesis_ids": ["TH-FAILED"],
                },
            }
        ),
        encoding="utf-8",
    )
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id="TH-NEW",
        status="blocked",
        proposal_generation_allowed=False,
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_type="fractal_long_memory",
            thesis_id="TH-NEW",
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="paper:fractal-long-memory-1",
                    title="Long-Range Correlations in Cryptocurrency Markets",
                    source="Local bibliography",
                    published_at="2024",
                    relevance="Motivates a distinct Hurst and long-memory thesis.",
                    motivated_thesis_ids=["TH-NEW"],
                ),
                StrategyProposalResearchReference(
                    reference_id="paper:fractal-long-memory-2",
                    title="Hurst Exponents and Dynamic Bitcoin Market Efficiency",
                    source="Local bibliography",
                    published_at="2025",
                    relevance="Supports falsifying long-memory regimes with OHLCV features.",
                    motivated_thesis_ids=["TH-NEW"],
                ),
            ],
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert artifacts.metadata["status"] == "blocked"
    assert "research_decision_1_approved_for_proposal_generation" in blocker_names


def test_strategy_proposal_generator_blocks_legacy_proposal_flag_when_research_gate_fails(
    tmp_path,
):
    thesis_id = "TH-EDGE-RESEARCH-GATE-BLOCKED-001"
    edge_path = _write_strategy_proposal_edge_discovery(
        tmp_path,
        thesis_id=thesis_id,
        mechanism_class="mean_reversion",
        candidate_generation_allowed=False,
    )
    payload = json.loads(edge_path.read_text(encoding="utf-8"))
    payload["proposal_generation_allowed"] = True
    payload["promotion_gate"]["proposal_generation_allowed"] = True
    edge_path.write_text(json.dumps(payload), encoding="utf-8")

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            thesis_id=thesis_id,
            include_edge_discovery=False,
            evidence_paths=[StrategyProposalEvidenceInput("edge_discovery", edge_path)],
        )
    )

    handoff = artifacts.metadata["edge_discovery_handoff"]
    edge_candidate = handoff["artifacts"][0]
    assert artifacts.metadata["status"] == "blocked"
    assert edge_candidate["status_passed"] is True
    assert edge_candidate["candidate_generation_allowed"] is False
    assert edge_candidate["proposal_generation_allowed"] is False
    assert "edge_discovery_handoff_passed" in {
        check["name"] for check in artifacts.metadata["blockers"]
    }


def test_strategy_proposal_generator_blocks_research_decision_with_local_rejection_novelty(
    tmp_path,
):
    thesis_id = "TH-NEW"
    mechanism_class = "validated_local_rejection_mechanism"
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id=thesis_id,
        novelty_assessment={
            "repeated_failed_family_matches": [mechanism_class],
            "failed_thesis_id_match": False,
            "local_falsification_failed_thesis_ids": [
                "TH-VALIDATED-LOCAL-REJECTED"
            ],
            "local_falsification_failed_thesis_id_match": False,
            "local_falsification_failed_mechanism_tokens": [mechanism_class],
            "local_falsification_failed_mechanism_class_matches": [
                mechanism_class
            ],
        },
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_type="fractal_long_memory",
            thesis_id=thesis_id,
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=_strategy_proposal_fractal_references(thesis_id),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert artifacts.metadata["status"] == "blocked"
    assert "research_decision_1_novelty_assessment_passed" in blocker_names
    assert (
        "research_decision_1_outside_failure_synthesis_local_rejections"
        in blocker_names
    )
    assert decision_constraints["repeated_failed_family_matches"] == [
        mechanism_class
    ]
    assert decision_constraints[
        "local_falsification_failed_mechanism_class_matches"
    ] == [mechanism_class]
    assert decision_constraints["local_falsification_failed_thesis_ids"] == [
        "TH-VALIDATED-LOCAL-REJECTED"
    ]


def test_strategy_proposal_generator_blocks_research_decision_without_causal_map(tmp_path):
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id="TH-NEW",
        include_causal_failure_map=False,
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_type="fractal_long_memory",
            thesis_id="TH-NEW",
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=_strategy_proposal_fractal_references("TH-NEW"),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert artifacts.metadata["status"] == "blocked"
    assert "research_decision_1_uses_causal_failure_map" in blocker_names
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["causal_failure_map_used"] is None


def test_strategy_proposal_generator_blocks_research_decision_with_stale_causal_map(
    tmp_path,
):
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    stale_causal_map = {
        "used": True,
        "available": True,
        "map_id": "old-map",
        "status": "completed",
        "source_synthesis_id": "older-synthesis",
        "candidate_count": 22,
        "category_count": 4,
        "requires_research_decision_before_proposal": True,
        "required_categories_to_address": [
            "regime_fragile_mechanism",
            "walk_forward_fragility",
            "cost_sensitive_mechanism",
        ],
        "response_categories": [
            "cost_sensitive_mechanism",
            "regime_fragile_mechanism",
            "walk_forward_fragility",
        ],
        "missing_response_categories": [],
        "weak_response_categories": [],
        "category_evidence_gaps": [],
        "parameter_only_response_categories": [],
    }
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id="TH-NEW",
        causal_failure_map=stale_causal_map,
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            thesis_id="TH-NEW",
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=_strategy_proposal_fractal_references("TH-NEW"),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert artifacts.metadata["status"] == "blocked"
    assert (
        "research_decision_1_causal_map_matches_failure_synthesis" in blocker_names
    )
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["causal_source_synthesis_id"] == "older-synthesis"
    assert decision_constraints["supplied_failure_synthesis_ids"] == ["synth-test"]
    assert decision_constraints["causal_map_matches_failure_synthesis"] is False


def test_strategy_proposal_generator_blocks_research_decision_missing_material_causal_categories(
    tmp_path,
):
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    material_causal_map = {
        "used": True,
        "available": True,
        "map_id": "map-test",
        "status": "completed",
        "source_synthesis_id": "synth-test",
        "candidate_count": 24,
        "category_count": 6,
        "requires_research_decision_before_proposal": True,
        "material_category_min_share": 0.70,
        "dominant_failure_categories": [
            {"category": "regime_fragile_mechanism", "candidate_count": 24},
            {"category": "walk_forward_fragility", "candidate_count": 24},
            {"category": "cost_sensitive_mechanism", "candidate_count": 23},
            {"category": "no_profitable_walk_forward_windows", "candidate_count": 18},
            {"category": "entry_exists_negative_edge", "candidate_count": 17},
            {"category": "overfit_or_window_dependency", "candidate_count": 14},
        ],
        "required_categories_to_address": [
            "regime_fragile_mechanism",
            "walk_forward_fragility",
            "cost_sensitive_mechanism",
        ],
        "response_categories": [
            "cost_sensitive_mechanism",
            "regime_fragile_mechanism",
            "walk_forward_fragility",
        ],
        "missing_response_categories": [],
        "weak_response_categories": [],
        "category_evidence_gaps": [],
        "parameter_only_response_categories": [],
    }
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id="TH-NEW",
        causal_failure_map=material_causal_map,
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            thesis_id="TH-NEW",
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=_strategy_proposal_fractal_references("TH-NEW"),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert artifacts.metadata["status"] == "blocked"
    assert (
        "research_decision_1_causal_required_categories_match_current_policy"
        in blocker_names
    )
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["causal_expected_required_categories"] == [
        "regime_fragile_mechanism",
        "walk_forward_fragility",
        "cost_sensitive_mechanism",
        "no_profitable_walk_forward_windows",
        "entry_exists_negative_edge",
    ]
    assert decision_constraints["missing_current_required_categories"] == [
        "no_profitable_walk_forward_windows",
        "entry_exists_negative_edge",
    ]
    assert decision_constraints["missing_current_response_categories"] == [
        "no_profitable_walk_forward_windows",
        "entry_exists_negative_edge",
    ]


def test_strategy_proposal_generator_blocks_research_decision_with_weak_causal_quality(tmp_path):
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    weak_causal_map = {
        "used": True,
        "available": True,
        "required_categories_to_address": [
            "regime_fragile_mechanism",
            "walk_forward_fragility",
            "cost_sensitive_mechanism",
        ],
        "missing_response_categories": [],
        "weak_response_categories": ["walk_forward_fragility"],
        "category_evidence_gaps": [
            {
                "category": "cost_sensitive_mechanism",
                "missing_requirement_groups": ["edge_terms"],
            }
        ],
        "parameter_only_response_categories": ["regime_fragile_mechanism"],
    }
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id="TH-NEW",
        causal_failure_map=weak_causal_map,
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_type="fractal_long_memory",
            thesis_id="TH-NEW",
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=_strategy_proposal_fractal_references("TH-NEW"),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert artifacts.metadata["status"] == "blocked"
    assert "research_decision_1_causal_response_quality_passed" in blocker_names
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["weak_response_categories"] == ["walk_forward_fragility"]
    assert decision_constraints["parameter_only_response_categories"] == [
        "regime_fragile_mechanism"
    ]


def test_strategy_proposal_generator_blocks_research_decision_below_selection_score(
    tmp_path,
):
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    causal_map = {
        "used": True,
        "available": True,
        "map_id": "map-test",
        "status": "completed",
        "source_synthesis_id": "synth-test",
        "candidate_count": 22,
        "category_count": 4,
        "requires_research_decision_before_proposal": True,
        "material_category_min_share": 0.70,
        "minimum_research_selection_score": 80.0,
        "dominant_failure_categories": [
            {"category": "regime_fragile_mechanism", "candidate_count": 22},
            {"category": "walk_forward_fragility", "candidate_count": 22},
            {"category": "cost_sensitive_mechanism", "candidate_count": 21},
        ],
        "required_categories_to_address": [
            "regime_fragile_mechanism",
            "walk_forward_fragility",
            "cost_sensitive_mechanism",
        ],
        "response_categories": [
            "cost_sensitive_mechanism",
            "regime_fragile_mechanism",
            "walk_forward_fragility",
        ],
        "missing_response_categories": [],
        "weak_response_categories": [],
        "category_evidence_gaps": [],
        "parameter_only_response_categories": [],
    }
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id="TH-NEW",
        causal_failure_map=causal_map,
        research_selection_score={
            "version": "research_selection_score_v1",
            "score": 79.0,
            "maximum_score": 100.0,
            "minimum_score_required": 80.0,
            "passes_minimum": False,
            "failed_components": ["local_historical_falsification"],
        },
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_type="fractal_long_memory",
            thesis_id="TH-NEW",
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=_strategy_proposal_fractal_references("TH-NEW"),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert artifacts.metadata["status"] == "blocked"
    assert "research_decision_1_research_selection_score_passed" in blocker_names
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["research_selection_score"] == 79.0
    assert decision_constraints["minimum_research_selection_score"] == 80.0
    assert decision_constraints["research_selection_score_passes_minimum"] is False
    assert decision_constraints["research_selection_failed_components"] == [
        "local_historical_falsification"
    ]


def test_strategy_proposal_generator_blocks_risk_weighted_decision_without_weighted_score(
    tmp_path,
):
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    causal_map = {
        "used": True,
        "available": True,
        "map_id": "map-test",
        "status": "completed",
        "source_synthesis_id": "synth-test",
        "candidate_count": 30,
        "category_count": 4,
        "requires_research_decision_before_proposal": True,
        "requires_research_question_responses": True,
        "material_category_min_share": 0.70,
        "minimum_research_selection_score": 80.0,
        "dominant_failure_categories": [
            {"category": "cost_sensitive_mechanism", "candidate_count": 29},
            {"category": "regime_fragile_mechanism", "candidate_count": 30},
            {"category": "walk_forward_fragility", "candidate_count": 30},
            {"category": "no_profitable_walk_forward_windows", "candidate_count": 24},
        ],
        "causal_risk_weights": [
            {"category": "cost_sensitive_mechanism", "risk_score": 100.0},
            {"category": "regime_fragile_mechanism", "risk_score": 100.0},
            {"category": "walk_forward_fragility", "risk_score": 100.0},
            {
                "category": "no_profitable_walk_forward_windows",
                "risk_score": 100.0,
            },
        ],
        "required_categories_to_address": [
            "cost_sensitive_mechanism",
            "regime_fragile_mechanism",
            "walk_forward_fragility",
            "no_profitable_walk_forward_windows",
        ],
        "response_categories": [
            "cost_sensitive_mechanism",
            "no_profitable_walk_forward_windows",
            "regime_fragile_mechanism",
            "walk_forward_fragility",
        ],
        "missing_response_categories": [],
        "weak_response_categories": [],
        "category_evidence_gaps": [],
        "parameter_only_response_categories": [],
        "required_research_questions": [
            "What mechanism survives after failed families are excluded?",
        ],
        "research_question_response_indexes": [1],
        "missing_research_question_response_indexes": [],
        "weak_research_question_response_indexes": [],
    }
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id="TH-NEW",
        causal_failure_map=causal_map,
        research_selection_score={
            "version": "research_selection_score_v1",
            "score": 100.0,
            "maximum_score": 100.0,
            "minimum_score_required": 80.0,
            "passes_minimum": True,
            "failed_components": [],
        },
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_type="fractal_long_memory",
            thesis_id="TH-NEW",
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=_strategy_proposal_fractal_references("TH-NEW"),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert artifacts.metadata["status"] == "blocked"
    assert (
        "research_decision_1_risk_weighted_selection_score_present" in blocker_names
    )
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["causal_risk_weights_present"] is True
    assert decision_constraints["research_selection_score_version"] == (
        "research_selection_score_v1"
    )
    assert decision_constraints["weighted_causal_score_available"] is False


def test_strategy_proposal_generator_blocks_high_risk_decision_without_local_falsification_handoff(
    tmp_path,
):
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id="TH-NEW",
        causal_failure_map=_strategy_proposal_high_risk_cost_causal_map(),
        research_selection_score=_strategy_proposal_weighted_research_selection_score(),
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_type="fractal_long_memory",
            thesis_id="TH-NEW",
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=_strategy_proposal_fractal_references("TH-NEW"),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert artifacts.metadata["status"] == "blocked"
    assert (
        "research_decision_1_local_falsification_handoff_passed"
        in blocker_names
    )
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["local_falsification_handoff_required"] is True
    assert decision_constraints["local_falsification_handoff_passed"] is False
    assert decision_constraints["local_falsification_artifact_count"] == 0
    assert decision_constraints["weighted_causal_score_available"] is True


def test_strategy_proposal_generator_accepts_high_risk_decision_with_local_falsification_handoff(
    tmp_path,
):
    thesis_id = "TH-NEW"
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id=thesis_id,
        causal_failure_map=_strategy_proposal_high_risk_cost_causal_map(),
        research_selection_score=_strategy_proposal_weighted_research_selection_score(),
        local_falsification_evidence=(
            _strategy_proposal_passing_local_falsification_evidence(thesis_id)
        ),
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_type="fractal_long_memory",
            thesis_id=thesis_id,
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=_strategy_proposal_fractal_references(thesis_id),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert artifacts.metadata["status"] == "accepted"
    assert (
        "research_decision_1_local_falsification_handoff_passed"
        not in blocker_names
    )
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["local_falsification_handoff_required"] is True
    assert decision_constraints["local_falsification_handoff_passed"] is True
    assert decision_constraints["local_falsification_passing_cost_edge_artifact_count"] == 1


def test_strategy_proposal_generator_blocks_missing_research_question_responses(
    tmp_path,
):
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    causal_map = {
        "used": True,
        "available": True,
        "map_id": "map-test",
        "status": "completed",
        "source_synthesis_id": "synth-test",
        "candidate_count": 22,
        "category_count": 4,
        "requires_research_decision_before_proposal": True,
        "requires_research_question_responses": True,
        "material_category_min_share": 0.70,
        "dominant_failure_categories": [
            {"category": "regime_fragile_mechanism", "candidate_count": 22},
            {"category": "walk_forward_fragility", "candidate_count": 22},
            {"category": "cost_sensitive_mechanism", "candidate_count": 21},
        ],
        "required_categories_to_address": [
            "regime_fragile_mechanism",
            "walk_forward_fragility",
            "cost_sensitive_mechanism",
        ],
        "response_categories": [
            "cost_sensitive_mechanism",
            "regime_fragile_mechanism",
            "walk_forward_fragility",
        ],
        "missing_response_categories": [],
        "weak_response_categories": [],
        "category_evidence_gaps": [],
        "parameter_only_response_categories": [],
        "required_research_questions": [
            "What mechanism survives after failed families are excluded?",
            "Why should expected edge exceed fee and turnover costs?",
        ],
        "research_question_response_indexes": [1],
        "missing_research_question_response_indexes": [2],
        "weak_research_question_response_indexes": [],
    }
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id="TH-NEW",
        causal_failure_map=causal_map,
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_type="fractal_long_memory",
            thesis_id="TH-NEW",
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=_strategy_proposal_fractal_references("TH-NEW"),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert artifacts.metadata["status"] == "blocked"
    assert (
        "research_decision_1_research_question_responses_complete"
        in blocker_names
    )
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["requires_research_question_responses"] is True
    assert decision_constraints["missing_research_question_response_indexes"] == [2]


def test_strategy_proposal_generator_recomputes_missing_research_question_responses(
    tmp_path,
):
    synthesis_path = _write_strategy_proposal_requiring_decision_synthesis(tmp_path)
    causal_map = {
        "used": True,
        "available": True,
        "map_id": "map-test",
        "status": "completed",
        "source_synthesis_id": "synth-test",
        "candidate_count": 22,
        "category_count": 4,
        "requires_research_decision_before_proposal": True,
        "requires_research_question_responses": True,
        "material_category_min_share": 0.70,
        "dominant_failure_categories": [
            {"category": "regime_fragile_mechanism", "candidate_count": 22},
            {"category": "walk_forward_fragility", "candidate_count": 22},
            {"category": "cost_sensitive_mechanism", "candidate_count": 21},
        ],
        "required_categories_to_address": [
            "regime_fragile_mechanism",
            "walk_forward_fragility",
            "cost_sensitive_mechanism",
        ],
        "response_categories": [
            "cost_sensitive_mechanism",
            "regime_fragile_mechanism",
            "walk_forward_fragility",
        ],
        "missing_response_categories": [],
        "weak_response_categories": [],
        "category_evidence_gaps": [],
        "parameter_only_response_categories": [],
        "required_research_questions": [
            "What mechanism survives after failed families are excluded?",
            "Why should expected edge exceed fee and turnover costs?",
        ],
        "research_question_response_indexes": [1],
        "missing_research_question_response_indexes": [],
        "weak_research_question_response_indexes": [],
    }
    research_decision_path = _write_strategy_proposal_research_decision(
        tmp_path,
        thesis_id="TH-NEW",
        causal_failure_map=causal_map,
    )

    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_type="fractal_long_memory",
            thesis_id="TH-NEW",
            thesis_type="fractal_long_memory",
            strategy_logic_variant="fractal_long_memory_regime",
            research_references=_strategy_proposal_fractal_references("TH-NEW"),
            evidence_paths=[
                StrategyProposalEvidenceInput(
                    "candidate_failure_synthesis", synthesis_path
                ),
                StrategyProposalEvidenceInput("research_decision", research_decision_path),
            ],
        )
    )

    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert artifacts.metadata["status"] == "blocked"
    assert (
        "research_decision_1_research_question_responses_complete"
        in blocker_names
    )
    decision_constraints = artifacts.metadata["research_decision_constraints"][0]
    assert decision_constraints["required_research_questions"] == [
        "What mechanism survives after failed families are excluded?",
        "Why should expected edge exceed fee and turnover costs?",
    ]
    assert decision_constraints["research_question_response_indexes"] == [1]
    assert (
        decision_constraints["reported_missing_research_question_response_indexes"]
        == []
    )
    assert (
        decision_constraints["computed_missing_research_question_response_indexes"]
        == [2]
    )
    assert decision_constraints["missing_research_question_response_indexes"] == [2]


def test_strategy_proposal_generator_emits_candidate_diversity_metadata(tmp_path):
    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            generator_mode="hybrid_ml",
            thesis_id="THESIS-TREND-001",
            thesis_type="trend_continuation",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE"],
            strategy_logic_variant="trend_continuation",
            feature_list=["ema_fast", "ema_slow", "atr"],
            target_definition="future_return",
            label_horizon=18,
            prediction_threshold=0.01,
            rule_filters=["trend_filter", "atr_floor"],
        )
    )

    metadata = artifacts.metadata
    assert metadata["status"] == "accepted"
    assert metadata["generator_mode"] == "hybrid_ml"
    assert metadata["strategy_logic_variant"] == "trend_continuation"
    assert metadata["thesis_id"] == "THESIS-TREND-001"
    assert "pullbacks in liquid BTC futures" in metadata["thesis_statement"]
    assert metadata["feature_list"] == ["ema_fast", "ema_slow", "atr"]
    assert metadata["failure_taxonomy_codes"] == ["FAIL_COST_SENSITIVE"]
    assert metadata["retry_budget_per_thesis"] == 3


def test_strategy_proposal_generator_blocks_unexplained_research_reference(tmp_path):
    artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="paper:missing-relevance",
                    title="Unexplained Market Reference",
                    source="Local bibliography",
                    relevance="",
                )
            ],
        )
    )

    assert artifacts.metadata["status"] == "blocked"
    assert "research_references_have_relevance" in {
        check["name"] for check in artifacts.metadata["blockers"]
    }


def test_strategy_proposal_cli_accepts_research_reference_file(tmp_path):
    from scripts.bot_factory_generate_strategy_proposal import _research_reference_inputs

    reference_path = tmp_path / "research_reference.json"
    reference_path.write_text(
        json.dumps(
            {
                "reference_id": "ssrn:4825389",
                "title": "Cryptocurrency Volume-Weighted Time Series Momentum",
                "source": "SSRN",
                "published_at": "2024-12-05",
                "relevance": "Motivates volume-confirmed crypto momentum candidates.",
                "motivated_thesis_ids": ["TH-CRYPTO-TSMOM-VOL-001"],
            }
        ),
        encoding="utf-8",
    )

    references = _research_reference_inputs([f"@{reference_path.name}"], root_dir=tmp_path)

    assert len(references) == 1
    assert references[0].reference_id == "ssrn:4825389"
    assert references[0].motivated_thesis_ids == ["TH-CRYPTO-TSMOM-VOL-001"]


def test_strategy_proposal_cli_maps_failure_synthesis_to_evidence(tmp_path):
    from types import SimpleNamespace

    from scripts.bot_factory_generate_strategy_proposal import _evidence_inputs

    synthesis_path = tmp_path / "candidate_failure_synthesis.json"
    research_decision_path = tmp_path / "research_decision.json"
    args = SimpleNamespace(
        ohlcv_quality_json=None,
        previous_metrics_json=None,
        walk_forward_metrics_json=None,
        training_manifest_json=None,
        failure_synthesis_json=[str(synthesis_path)],
        research_decision_json=[str(research_decision_path)],
        reviewer_notes_path=None,
        evidence_path=None,
    )

    evidence = _evidence_inputs(args)

    assert evidence == [
        StrategyProposalEvidenceInput(
            label="candidate_failure_synthesis",
            path=Path(str(synthesis_path)),
        ),
        StrategyProposalEvidenceInput(
            label="research_decision",
            path=Path(str(research_decision_path)),
        ),
    ]


def test_strategy_proposal_cli_blocks_research_reference_file_outside_root(tmp_path):
    import pytest

    from scripts.bot_factory_generate_strategy_proposal import _research_reference_inputs

    outside = tmp_path.parent / "outside_research_reference.json"

    with pytest.raises(SystemExit, match="inside the workspace"):
        _research_reference_inputs([f"@{outside}"], root_dir=tmp_path)


def test_strategy_code_generator_writes_long_only_strategy_and_metadata(tmp_path):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    inputs = _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)

    artifacts = build_strategy_code(inputs)
    write_strategy_code_artifacts(artifacts)

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_code_generated"] is True
    assert artifacts.metadata["candidate_evaluation_eligible"] is True
    assert artifacts.strategy_path.is_file()
    assert artifacts.metadata_path.is_file()
    assert artifacts.static_check_path.is_file()
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")
    assert "can_short = False" in generated_code
    assert "optimize=True" not in generated_code
    assert generated_code.count("optimize=False") == 10
    assert "enter_short" not in generated_code
    assert "exit_short" not in generated_code
    assert "def leverage" not in generated_code
    assert "shift(-1" not in generated_code
    assert ".iloc[-1" not in generated_code
    assert artifacts.metadata["source_proposal_content_hash"] == (
        proposal_artifacts.metadata["proposal_content_hash"]
    )
    assert artifacts.metadata["edge_discovery_handoff"]["passed"] is True
    assert artifacts.metadata["parameter_optimization_enabled"] is False
    assert artifacts.metadata["parameter_optimization_policy"] == PARAMETER_OPTIMIZATION_POLICY
    assert (
        artifacts.metadata["safety_scope"]["freqtrade_hyperopt_parameter_optimization"]
        is False
    )
    assert {
        check["name"]: check["status"] for check in artifacts.metadata["checks"]
    }["generated_code_freqtrade_hyperopt_disabled"] == "pass"
    assert artifacts.metadata["parameter_defaults"]["buy_rsi_window"] == 14
    assert artifacts.metadata["static_check"]["ran"] is True
    assert artifacts.metadata["static_check"]["ok"] is True
    assert scan_paths([artifacts.strategy_path]).ok


def test_strategy_code_generator_blocks_accepted_metadata_without_edge_discovery_handoff(
    tmp_path,
):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    metadata = json.loads(proposal_artifacts.metadata_path.read_text(encoding="utf-8"))
    metadata.pop("edge_discovery_handoff", None)
    metadata["status"] = "accepted"
    metadata["proposal_status"] = "accepted"
    metadata["code_generation_eligible"] = True
    metadata["blockers"] = []
    metadata["rejection_reasons"] = []
    proposal_artifacts.metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )

    assert artifacts.metadata["status"] == "blocked"
    assert artifacts.metadata["strategy_code_generated"] is False
    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert "edge_discovery_handoff_passed" in blocker_names


def test_strategy_code_generator_varies_logic_by_hypothesis_family(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            thesis_type="trend_continuation",
            strategy_logic_variant="trend_continuation",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE"],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "trend_continuation"
    assert artifacts.metadata["parameter_defaults"]["buy_ema_slow"] == 64
    assert "trend_continuation" in generated_code
    assert 'rsi_target = dataframe["rsi"] >= self.sell_rsi_exit.value' in generated_code
    assert 'rsi_cooldown = dataframe["rsi"] < self.sell_rsi_exit.value' not in generated_code
    assert "rsi_pullback_recovery" not in generated_code


def test_strategy_code_generator_supports_downside_liquidity_shock_reversal(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyDownsideLiquidityShockCandidate",
            strategy_type="downside_liquidity_shock",
            thesis_id="TH-DOWNSIDE-SHOCK-001",
            thesis_type="downside_liquidity_shock_reversal",
            thesis_statement=(
                "Short-horizon downside shocks can create a local liquidity-provision "
                "reversal setup when price reclaims the local low without requiring "
                "EMA trend alignment."
            ),
            hypothesis=(
                "Long-only entries after downside shock, RSI washout/recovery, quiet "
                "volume, and local-low reclaim should avoid the failed trend-filter "
                "conflict."
            ),
            strategy_logic_variant="downside_liquidity_shock_reversal",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "lookback_return",
                "rsi_washout",
                "atr_normalized_drop",
                "quiet_volume",
                "local_low_reclaim",
            ],
            rule_filters=[
                "downside_shock",
                "rsi_washout_recovery",
                "quiet_volume",
                "local_low_reclaim",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="paper:crypto-liquidity-provision",
                    title="Trading volume and liquidity provision in cryptocurrency markets",
                    source="Journal of Banking and Finance",
                    published_at="2022",
                    relevance=(
                        "Motivates short-term reversal as compensation for liquidity "
                        "provision in cryptocurrency markets."
                    ),
                    motivated_thesis_ids=["TH-DOWNSIDE-SHOCK-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert (
        artifacts.metadata["strategy_logic_variant"]
        == "downside_liquidity_shock_reversal"
    )
    assert artifacts.metadata["parameter_defaults"]["buy_volume_factor"] == 0.95
    assert "downside_shock" in generated_code
    assert "quiet_volume" in generated_code
    assert "local_low_reclaim" in generated_code
    assert 'trend_filter = dataframe["ema_fast"]' not in generated_code


def test_strategy_code_generator_supports_intraday_session_liquidity_reclaim(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyIntradaySessionLiquidityCandidate",
            strategy_type="intraday_liquidity_timing",
            thesis_id="TH-SESSION-LIQ-001",
            thesis_type="intraday_session_liquidity",
            thesis_statement=(
                "Bitcoin intraday liquidity and price discovery vary by UTC session, "
                "so a long-only VWAP reclaim during the London-New York overlap can "
                "test a session-timing mechanism without repeating failed trend, "
                "breakout, or shock-reversal families."
            ),
            hypothesis=(
                "Enter only during high-liquidity weekday UTC session windows when "
                "price reclaims same-day VWAP with confirming volume and controlled "
                "ATR."
            ),
            strategy_logic_variant="intraday_session_liquidity_reclaim",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "hour_utc",
                "weekday",
                "session_vwap",
                "session_vwap_distance",
                "volume_mean",
                "atr_regime",
            ],
            rule_filters=[
                "session_window",
                "weekday_liquidity",
                "vwap_reclaim",
                "volume_filter",
                "controlled_atr",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="paper:bitcoin-intraday-price-discovery",
                    title="The intraday dynamics and intraday price discovery of bitcoin",
                    source="Research in International Business and Finance",
                    published_at="2022",
                    relevance=(
                        "Motivates testing London-New York overlap timing as a "
                        "distinct intraday liquidity and price-discovery mechanism."
                    ),
                    motivated_thesis_ids=["TH-SESSION-LIQ-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert (
        artifacts.metadata["strategy_logic_variant"]
        == "intraday_session_liquidity_reclaim"
    )
    assert artifacts.metadata["parameter_defaults"]["buy_volume_factor"] == 1.05
    assert "session_vwap" in generated_code
    assert "session_window" in generated_code
    assert "vwap_reclaim" in generated_code
    assert "controlled_atr" in generated_code
    assert 'trend_filter = dataframe["ema_fast"]' not in generated_code


def test_strategy_code_generator_supports_liquidity_recovery_horizon(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyLiquidityRecoveryHorizonCandidate",
            strategy_type="liquidity_recovery_horizon",
            thesis_id="TH-LIQUIDITY-RECOVERY-HORIZON-001",
            thesis_type="liquidity_recovery_horizon",
            thesis_statement=(
                "BTC futures may show a delayed long-only edge after local stress "
                "when closed-candle liquidity proxies normalize and participation "
                "recovers toward baseline."
            ),
            hypothesis=(
                "Enter only after a recent liquidity stress episode when illiquidity "
                "and range proxies normalize, participation recovers, and price has "
                "not yet reached the recovery anchor."
            ),
            strategy_logic_variant="liquidity_recovery_horizon",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "liquidity_stress_recent",
                "liquidity_recovery_score",
                "liquidity_recovery_anchor",
                "volume_recovery_ratio",
                "amihud_illiquidity_ratio",
                "range_recovery_ratio",
                "recovery_horizon_return",
            ],
            rule_filters=[
                "recent_liquidity_stress",
                "liquidity_normalizing",
                "participation_recovered",
                "below_recovery_anchor",
                "recovery_turn",
                "controlled_cost_proxy",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="paper:liquidity-recovery-horizon",
                    title="Liquidity Recovery Dynamics after Volatility Shocks",
                    source="Local bibliography",
                    published_at="2026",
                    relevance=(
                        "Motivates a distinct recovery-horizon mechanism after "
                        "volatility stress using falsifiable local liquidity proxies."
                    ),
                    motivated_thesis_ids=["TH-LIQUIDITY-RECOVERY-HORIZON-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert proposal_artifacts.metadata["status"] == "accepted"
    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "liquidity_recovery_horizon"
    assert artifacts.metadata["parameter_defaults"]["buy_volume_window"] == 72
    assert "liquidity_recovery_score" in generated_code
    assert "liquidity_stress_recent" in generated_code
    assert "controlled_cost_proxy" in generated_code
    assert "liquidity_recovery_or_stress_exit" in generated_code
    assert 'trend_filter = dataframe["ema_fast"]' not in generated_code


def test_strategy_code_generator_supports_signed_volume_imbalance_accumulation(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlySignedVolumeImbalanceCandidate",
            strategy_type="order_flow_imbalance",
            thesis_id="TH-SIGNED-IMBALANCE-001",
            thesis_type="signed_volume_imbalance",
            thesis_statement=(
                "Candle-direction signed volume imbalance and repeated upper-range "
                "closes can proxy accumulation pressure without repeating failed "
                "trend, breakout, pullback, ML, shock, or session families."
            ),
            hypothesis=(
                "Enter long when rolling signed-volume imbalance is positive, "
                "closes persist in the upper part of candle ranges, price reclaims "
                "the rolling midpoint, and the setup is not a rolling-high breakout."
            ),
            strategy_logic_variant="signed_volume_imbalance_accumulation",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "signed_volume",
                "signed_volume_imbalance",
                "close_location_value",
                "close_location_mean",
                "rolling_mid",
                "range_pct",
            ],
            rule_filters=[
                "positive_signed_imbalance",
                "close_location_accumulation",
                "mid_reclaim",
                "not_breakout_chase",
                "controlled_range",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="paper:bitcoin-tick-rule",
                    title="The Accuracy of the Tick Rule in the Bitcoin Market",
                    source="Sage Open",
                    published_at="2021",
                    relevance=(
                        "Motivates using signed trade proxies carefully when true "
                        "trade initiator labels are unavailable."
                    ),
                    motivated_thesis_ids=["TH-SIGNED-IMBALANCE-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert (
        artifacts.metadata["strategy_logic_variant"]
        == "signed_volume_imbalance_accumulation"
    )
    assert artifacts.metadata["parameter_defaults"]["buy_pullback_lookback"] == 24
    assert "signed_volume_imbalance" in generated_code
    assert "close_location_value" in generated_code
    assert "mid_reclaim" in generated_code
    assert "not_breakout_chase" in generated_code
    assert 'trend_filter = dataframe["ema_fast"]' not in generated_code
    assert "breakout_filter" not in generated_code


def test_strategy_code_generator_supports_entropy_regime_transition(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyEntropyRegimeCandidate",
            strategy_type="entropy_regime",
            thesis_id="TH-ENTROPY-REGIME-001",
            thesis_type="entropy_regime",
            thesis_statement=(
                "Directional entropy and range efficiency can identify a distinct "
                "information-regime transition in closed-candle BTC futures data."
            ),
            hypothesis=(
                "Enter long only when directional entropy is compressed, range "
                "efficiency is expanding, drift is positive, price holds the "
                "rolling midpoint, and the setup is not a rolling-high chase."
            ),
            strategy_logic_variant="entropy_regime_transition",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "direction_entropy",
                "direction_entropy_baseline",
                "range_efficiency",
                "range_efficiency_mean",
                "entropy_drift",
                "rolling_mid",
            ],
            rule_filters=[
                "low_directional_entropy",
                "efficiency_expanding",
                "positive_entropy_drift",
                "midline_hold",
                "range_not_extended",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="paper:crypto-permutation-entropy",
                    title=(
                        "Clustering patterns in efficiency and the coming-of-age "
                        "of the cryptocurrency market"
                    ),
                    source="Scientific Reports",
                    published_at="2019-02-05",
                    relevance=(
                        "Motivates entropy-based crypto market efficiency regimes "
                        "as a distinct hypothesis family."
                    ),
                    motivated_thesis_ids=["TH-ENTROPY-REGIME-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "entropy_regime_transition"
    assert artifacts.metadata["parameter_defaults"]["buy_volume_window"] == 48
    assert "import numpy as np" in generated_code
    assert "direction_entropy" in generated_code
    assert "range_efficiency" in generated_code
    assert "low_directional_entropy" in generated_code
    assert "range_not_extended" in generated_code
    assert 'trend_filter = dataframe["ema_fast"]' not in generated_code
    assert "breakout_filter" not in generated_code


def test_strategy_code_generator_supports_fractal_long_memory_regime(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyFractalMemoryCandidate",
            strategy_type="fractal_long_memory",
            thesis_id="TH-FRACTAL-MEMORY-001",
            thesis_type="fractal_long_memory",
            thesis_statement=(
                "Rolling Hurst behavior and fractal path efficiency can identify "
                "a distinct long-memory regime in closed-candle BTC futures data."
            ),
            hypothesis=(
                "Enter long only when the Hurst proxy shows persistence, the "
                "price path is efficient, drift is positive, price holds the "
                "rolling midpoint, and the setup is not a rolling-high extension."
            ),
            strategy_logic_variant="fractal_long_memory_regime",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "log_return",
                "hurst_proxy",
                "fractal_efficiency",
                "fractal_efficiency_mean",
                "fractal_drift",
                "rolling_mid",
            ],
            rule_filters=[
                "persistent_memory_regime",
                "efficient_path",
                "positive_fractal_drift",
                "midline_hold",
                "not_range_extension",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="paper:bitcoin-long-range-correlations",
                    title="Long-range correlations and asymmetry in the Bitcoin market",
                    source="Physica A",
                    published_at="2018-02-15",
                    relevance=(
                        "Motivates Hurst and long-range-correlation regimes as "
                        "a distinct hypothesis family for Bitcoin."
                    ),
                    motivated_thesis_ids=["TH-FRACTAL-MEMORY-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "fractal_long_memory_regime"
    assert artifacts.metadata["parameter_defaults"]["buy_pullback_lookback"] == 36
    assert "hurst_proxy" in generated_code
    assert "fractal_efficiency" in generated_code
    assert "persistent_memory_regime" in generated_code
    assert "not_range_extension" in generated_code
    assert 'trend_filter = dataframe["ema_fast"]' not in generated_code
    assert "breakout_filter" not in generated_code


def test_strategy_code_generator_supports_semivariance_asymmetry_regime(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlySemivarianceAsymmetryCandidate",
            strategy_type="semivariance_asymmetry",
            thesis_id="TH-SEMIVARIANCE-ASYMMETRY-001",
            thesis_type="semivariance_asymmetry",
            thesis_statement=(
                "Upside and downside realized semivariance asymmetry can identify "
                "a distinct good-volatility regime in closed-candle BTC futures data."
            ),
            hypothesis=(
                "Enter long only when upside semivariance dominates downside "
                "semivariance, downside risk is decaying, drift is positive, "
                "price holds the rolling midpoint, and range expansion is controlled."
            ),
            strategy_logic_variant="semivariance_asymmetry_regime",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "upside_semivariance",
                "downside_semivariance",
                "downside_semivariance_mean",
                "semivariance_balance",
                "semivariance_drift",
                "range_pct",
            ],
            rule_filters=[
                "good_volatility_dominance",
                "bad_volatility_decay",
                "positive_semivariance_drift",
                "midline_hold",
                "controlled_range",
                "not_range_extension",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="paper:realized-semivariance",
                    title="Measuring downside risk: realised semivariance",
                    source="Oxford University Press",
                    published_at="2010-03-01",
                    relevance=(
                        "Motivates realized semivariance as a distinct downside-risk "
                        "feature family for closed-candle code generation."
                    ),
                    motivated_thesis_ids=["TH-SEMIVARIANCE-ASYMMETRY-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "semivariance_asymmetry_regime"
    assert artifacts.metadata["parameter_defaults"]["buy_volume_factor"] == 0.85
    assert "upside_semivariance" in generated_code
    assert "downside_semivariance" in generated_code
    assert "good_volatility_dominance" in generated_code
    assert "bad_volatility_decay" in generated_code
    assert 'trend_filter = dataframe["ema_fast"]' not in generated_code
    assert "breakout_filter" not in generated_code


def test_strategy_code_generator_supports_funding_pressure_carry(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyFundingPressureCarryCandidate",
            strategy_type="funding_pressure_carry",
            thesis_id="TH-FUNDING-PRESSURE-CARRY-001",
            thesis_type="funding_pressure_carry",
            thesis_statement=(
                "Perpetual funding pressure can identify a distinct futures carry "
                "regime when negative funding is releasing and price remains resilient."
            ),
            hypothesis=(
                "Enter long only when funding pressure is negative but improving, "
                "price holds the rolling midpoint, funding has not turned into "
                "positive crowding, candle range is controlled, and volume confirms."
            ),
            strategy_logic_variant="funding_pressure_carry",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "funding_rate",
                "funding_rate_mean",
                "funding_rate_abs_mean",
                "funding_pressure",
                "funding_pressure_delta",
                "rolling_mid",
            ],
            rule_filters=[
                "negative_funding_pressure",
                "funding_pressure_releasing",
                "price_resilience",
                "not_positive_crowding",
                "controlled_range",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="paper:perpetual-futures-pricing",
                    title="Perpetual Futures Pricing",
                    source="NBER Working Paper 32936",
                    published_at="2024-09-01",
                    relevance=(
                        "Motivates funding payments as the anchoring mechanism "
                        "that distinguishes perpetual futures from spot OHLCV regimes."
                    ),
                    motivated_thesis_ids=["TH-FUNDING-PRESSURE-CARRY-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "funding_pressure_carry"
    assert artifacts.metadata["parameter_defaults"]["buy_volume_factor"] == 0.90
    assert "informative_pairs" in generated_code
    assert "funding_rate" in generated_code
    assert "funding_pressure_delta" in generated_code
    assert "negative_funding_pressure" in generated_code
    assert "funding_pressure_releasing" in generated_code
    assert 'trend_filter = dataframe["ema_fast"]' not in generated_code
    assert "breakout_filter" not in generated_code


def test_strategy_code_generator_supports_realized_skewness_tail_shape(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyRealizedSkewnessTailCandidate",
            strategy_type="realized_skewness_tail",
            thesis_id="TH-REALIZED-SKEWNESS-TAIL-001",
            thesis_type="realized_skewness_tail",
            thesis_statement=(
                "Realized skewness and kurtosis can identify a distinct "
                "higher-moment tail-shape regime in closed-candle BTC data."
            ),
            hypothesis=(
                "Enter long only when realized skewness is low relative to its "
                "baseline, realized kurtosis is elevated, lottery-like positive "
                "tail returns are cooling, drift is positive, and range is controlled."
            ),
            strategy_logic_variant="realized_skewness_tail_shape",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "realized_skewness",
                "realized_skewness_mean",
                "realized_kurtosis",
                "realized_kurtosis_mean",
                "max_return",
                "max_return_mean",
                "tail_shape_drift",
            ],
            rule_filters=[
                "low_realized_skewness",
                "kurtosis_risk_premium",
                "lottery_tail_cooling",
                "positive_tail_shape_drift",
                "midline_hold",
                "controlled_range",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="paper:crypto-higher-moments",
                    title="Higher moments, extreme returns, and cross-section of cryptocurrency returns",
                    source="Finance Research Letters",
                    published_at="2021-03-01",
                    relevance=(
                        "Motivates realized skewness, kurtosis, and extreme "
                        "positive returns as a distinct crypto return-predictability family."
                    ),
                    motivated_thesis_ids=["TH-REALIZED-SKEWNESS-TAIL-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "realized_skewness_tail_shape"
    assert artifacts.metadata["parameter_defaults"]["buy_volume_factor"] == 0.90
    assert "realized_skewness" in generated_code
    assert "realized_kurtosis" in generated_code
    assert "lottery_tail_cooling" in generated_code
    assert "kurtosis_risk_premium" in generated_code
    assert 'trend_filter = dataframe["ema_fast"]' not in generated_code
    assert "breakout_filter" not in generated_code


def test_strategy_code_generator_supports_calendar_turnover_seasonality(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyCalendarTurnoverCandidate",
            strategy_type="calendar_turnover",
            thesis_id="TH-CALENDAR-TURNOVER-001",
            thesis_type="calendar_turnover",
            thesis_statement=(
                "Bitcoin's continuous market can exhibit day-of-week turnover "
                "and risk windows that differ from intraday session liquidity."
            ),
            hypothesis=(
                "Enter long only during Monday or Thursday UTC risk windows when "
                "weekend turnover has been discounted, turnover recovers, drift is "
                "positive, and the move is not an extended breakout chase."
            ),
            strategy_logic_variant="calendar_turnover_seasonality",
            failure_taxonomy_codes=["FAIL_OVERFIT_WF_GAP", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "weekday",
                "hour_utc",
                "calendar_turnover_ratio",
                "calendar_turnover_ratio_mean",
                "weekend_turnover_baseline",
                "weekday_turnover_baseline",
                "calendar_drift",
            ],
            rule_filters=[
                "calendar_risk_window",
                "weekend_discount_context",
                "turnover_recovery",
                "positive_calendar_drift",
                "midline_hold",
                "controlled_range",
                "not_breakout_chase",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="paper:bitcoin-calendar-turnover",
                    title="Bitcoin time-of-day, day-of-week and month-of-year effects in returns and trading volume",
                    source="Finance Research Letters",
                    published_at="2019-12-01",
                    relevance=(
                        "Motivates calendar turnover features and the weekend "
                        "lower-volume context in Bitcoin's continuous market."
                    ),
                    motivated_thesis_ids=["TH-CALENDAR-TURNOVER-001"],
                ),
                StrategyProposalResearchReference(
                    reference_id="paper:bitcoin-day-of-week",
                    title="Bitcoin and the day-of-the-week effect",
                    source="Finance Research Letters",
                    published_at="2019-12-01",
                    relevance=(
                        "Motivates Monday/weekday return and volatility windows "
                        "as a separate calendar anomaly hypothesis."
                    ),
                    motivated_thesis_ids=["TH-CALENDAR-TURNOVER-001"],
                ),
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "calendar_turnover_seasonality"
    assert artifacts.metadata["parameter_defaults"]["buy_volume_factor"] == 1.00
    assert "calendar_turnover_ratio" in generated_code
    assert "weekend_turnover_baseline" in generated_code
    assert "calendar_risk_window" in generated_code
    assert "weekend_discount_context" in generated_code
    assert 'trend_filter = dataframe["ema_fast"]' not in generated_code
    assert "breakout_filter" not in generated_code


def test_strategy_code_generator_supports_amihud_illiquidity_premium(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyAmihudIlliquidityCandidate",
            strategy_type="amihud_illiquidity",
            thesis_id="TH-AMIHUD-ILLIQUIDITY-001",
            thesis_type="amihud_illiquidity",
            thesis_statement=(
                "Amihud-style price impact from closed-candle returns and "
                "dollar volume can identify a distinct illiquidity premium regime."
            ),
            hypothesis=(
                "Enter long only when price impact is elevated but improving, "
                "not in an extreme stress tail, price is resilient, drift is "
                "positive, and participation is above a minimum volume floor."
            ),
            strategy_logic_variant="amihud_illiquidity_premium",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "amihud_illiquidity",
                "amihud_illiquidity_mean",
                "amihud_illiquidity_delta",
                "dollar_volume",
                "illiquidity_drift",
                "range_pct",
                "rolling_mid",
            ],
            rule_filters=[
                "price_impact_premium",
                "illiquidity_releasing",
                "not_extreme_impact",
                "price_resilience",
                "positive_illiquidity_drift",
                "controlled_range",
                "volume_floor",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="paper:amihud-illiquidity",
                    title="Illiquidity and stock returns: Cross-section and time-series effects",
                    source="Journal of Financial Markets",
                    published_at="2002-01-01",
                    relevance=(
                        "Defines the absolute-return to dollar-volume price-impact "
                        "measure used as the generated candidate's core feature."
                    ),
                    motivated_thesis_ids=["TH-AMIHUD-ILLIQUIDITY-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "amihud_illiquidity_premium"
    assert artifacts.metadata["parameter_defaults"]["buy_volume_factor"] == 0.90
    assert "amihud_illiquidity" in generated_code
    assert "dollar_volume" in generated_code
    assert "price_impact_premium" in generated_code
    assert "illiquidity_releasing" in generated_code
    assert "not_extreme_impact" in generated_code
    assert 'trend_filter = dataframe["ema_fast"]' not in generated_code
    assert "breakout_filter" not in generated_code


def test_strategy_code_generator_supports_cross_asset_lead_lag(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyCrossAssetLeadLagCandidate",
            strategy_type="cross_asset_lead_lag",
            thesis_id="TH-CROSS-ASSET-LEAD-LAG-001",
            thesis_type="btc_eth_lead_lag",
            thesis_statement=(
                "ETH closed-candle returns can lead short-horizon BTC drift in "
                "a distinct inter-crypto spillover regime."
            ),
            hypothesis=(
                "Enter long only when ETH's prior closed-candle return is above "
                "its local baseline, BTC has not caught up to the ETH-BTC return "
                "spread, BTC holds the rolling midpoint, drift is positive, and "
                "range expansion is controlled."
            ),
            strategy_logic_variant="cross_asset_lead_lag",
            failure_taxonomy_codes=["FAIL_REGIME_FRAGILE", "FAIL_COST_SENSITIVE"],
            feature_list=[
                "eth_log_return",
                "eth_lead_return",
                "eth_lead_return_mean",
                "btc_log_return",
                "eth_btc_return_spread",
                "eth_btc_spread_mean",
                "cross_asset_drift",
            ],
            rule_filters=[
                "eth_positive_lead",
                "btc_lag_discount",
                "spread_not_extreme",
                "btc_resilience",
                "positive_cross_asset_drift",
                "controlled_range",
                "volume_filter",
            ],
            required_data=[
                "BTC/USDT:USDT 5m closed-candle OHLCV",
                "ETH/USDT:USDT 5m closed-candle OHLCV as an informative series",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="doi:10.1186/s40854-023-00460-6",
                    title="Intraday lead/lag relationships between Bitcoin and Ethereum",
                    source="Journal of Capital Markets Studies",
                    published_at="2023-01-01",
                    relevance=(
                        "Directly motivates testing BTC/ETH intraday lead-lag "
                        "features instead of parameter-only retrying."
                    ),
                ),
                StrategyProposalResearchReference(
                    reference_id="doi:10.1016/j.irfa.2019.101371",
                    title="Intraday return predictability in the cryptocurrency markets",
                    source="International Review of Financial Analysis",
                    published_at="2019-01-01",
                    relevance=(
                        "Supports closed-candle intraday crypto return-predictability "
                        "features used by the generated candidate."
                    ),
                ),
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "cross_asset_lead_lag"
    assert artifacts.metadata["parameter_defaults"]["buy_pullback_lookback"] == 24
    assert '("ETH/USDT:USDT", self.timeframe)' in generated_code
    assert "eth_log_return" in generated_code
    assert "eth_lead_return" in generated_code
    assert "eth_btc_return_spread" in generated_code
    assert "cross_asset_drift" in generated_code
    assert "eth_positive_lead" in generated_code
    assert "btc_lag_discount" in generated_code
    assert 'trend_filter = dataframe["ema_fast"]' not in generated_code
    assert "breakout_filter" not in generated_code


def test_strategy_code_generator_supports_cross_asset_cointegration_spread(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyCrossAssetCointegrationCandidate",
            strategy_type="cross_asset_cointegration",
            thesis_id="TH-CROSS-ASSET-COINTEGRATION-001",
            thesis_type="btc_eth_cointegration",
            thesis_statement=(
                "BTC and ETH can share a local equilibrium relationship where "
                "BTC discounts against ETH may mean-revert without relying on "
                "lead-lag timing."
            ),
            hypothesis=(
                "Enter long BTC only when the BTC/ETH log-price ratio is below "
                "its rolling equilibrium, the spread has started reverting, "
                "ETH drift is positive, BTC has reclaimed local resilience, "
                "and range plus volume filters confirm tradable conditions."
            ),
            strategy_logic_variant="cross_asset_cointegration_spread",
            failure_taxonomy_codes=["FAIL_REGIME_FRAGILE", "FAIL_COST_SENSITIVE"],
            feature_list=[
                "eth_close",
                "btc_eth_log_ratio",
                "btc_eth_ratio_mean",
                "btc_eth_ratio_zscore",
                "btc_eth_ratio_zscore_delta",
                "eth_regime_drift",
                "range_pct",
            ],
            rule_filters=[
                "btc_discount_to_eth",
                "spread_reversion_turn",
                "eth_market_support",
                "btc_resilience",
                "cointegration_spread_not_extreme",
                "controlled_range",
                "volume_filter",
            ],
            required_data=[
                "BTC/USDT:USDT 5m closed-candle OHLCV",
                "ETH/USDT:USDT 5m closed-candle OHLCV as an informative series",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="doi:10.1108/SEF-08-2018-0264",
                    title="Constructing cointegrated cryptocurrency portfolios for statistical arbitrage",
                    source="Studies in Economics and Finance",
                    published_at="2019-01-01",
                    relevance=(
                        "Motivates testing cointegrated cryptocurrency spread "
                        "reversion as a distinct family from lead-lag timing."
                    ),
                ),
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "cross_asset_cointegration_spread"
    assert artifacts.metadata["parameter_defaults"]["buy_pullback_lookback"] == 36
    assert '("ETH/USDT:USDT", self.timeframe)' in generated_code
    assert "btc_eth_log_ratio" in generated_code
    assert "btc_eth_ratio_zscore" in generated_code
    assert "btc_discount_to_eth" in generated_code
    assert "spread_reversion_turn" in generated_code
    assert "cointegration_spread_not_extreme" in generated_code
    assert "eth_positive_lead" not in generated_code
    assert "variance_ratio_expansion" not in generated_code


def test_strategy_code_generator_supports_cross_asset_correlation_recovery(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyCrossAssetCorrelationCandidate",
            strategy_type="cross_asset_correlation",
            thesis_id="TH-CROSS-ASSET-CORRELATION-001",
            thesis_type="btc_eth_correlation_recovery",
            thesis_statement=(
                "BTC and ETH return correlation can disconnect and then recover "
                "as market integration returns, without relying on lead-lag or "
                "cointegration spread timing."
            ),
            hypothesis=(
                "Enter long BTC only when BTC/ETH rolling return correlation "
                "has been locally depressed, the correlation is recovering, "
                "BTC relative return improves versus ETH, ETH drift is positive, "
                "and BTC range plus volume filters remain tradable."
            ),
            strategy_logic_variant="cross_asset_correlation_recovery",
            failure_taxonomy_codes=["FAIL_REGIME_FRAGILE", "FAIL_COST_SENSITIVE"],
            feature_list=[
                "eth_close",
                "btc_log_return",
                "eth_log_return",
                "btc_eth_return_corr",
                "btc_eth_corr_mean",
                "btc_eth_corr_delta",
                "btc_eth_relative_return",
                "btc_eth_relative_return_mean",
                "eth_regime_drift",
                "range_pct",
            ],
            rule_filters=[
                "correlation_breakdown",
                "correlation_recovery",
                "btc_relative_recovery",
                "eth_market_support",
                "btc_resilience",
                "controlled_range",
                "volume_filter",
            ],
            required_data=[
                "BTC/USDT:USDT 5m closed-candle OHLCV",
                "ETH/USDT:USDT 5m closed-candle OHLCV as an informative series",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="doi:10.1016/j.qref.2021.04.002",
                    title="Dynamic connectedness between cryptocurrency and commodity markets",
                    source="Quarterly Review of Economics and Finance",
                    published_at="2021-01-01",
                    relevance=(
                        "Motivates testing dynamic cross-market correlation "
                        "regimes as a distinct family from spread cointegration."
                    ),
                ),
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "cross_asset_correlation_recovery"
    assert artifacts.metadata["parameter_defaults"]["buy_pullback_lookback"] == 36
    assert '("ETH/USDT:USDT", self.timeframe)' in generated_code
    assert "btc_eth_return_corr" in generated_code
    assert "btc_eth_corr_delta" in generated_code
    assert "btc_eth_relative_return" in generated_code
    assert "correlation_breakdown" in generated_code
    assert "correlation_recovery" in generated_code
    assert "btc_relative_recovery" in generated_code
    assert "btc_discount_to_eth" not in generated_code
    assert "eth_positive_lead" not in generated_code


def test_strategy_code_generator_supports_market_beta_drawdown_carry(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyMarketBetaDrawdownCarryCandidate",
            strategy_type="market_beta_carry",
            thesis_id="TH-MARKET-BETA-DRAWDOWN-CARRY-001",
            thesis_type="market_beta_drawdown_carry",
            thesis_statement=(
                "Bitcoin long beta can earn a risk premium after moderate "
                "drawdowns when realized volatility remains inside a local "
                "risk budget, without repeating trend, funding, or pullback "
                "families."
            ),
            hypothesis=(
                "Enter long BTC only after a moderate recent-high drawdown, "
                "when realized volatility is inside budget, the closed candle "
                "recovers above its open, price holds a local midpoint, and "
                "participation remains adequate."
            ),
            strategy_logic_variant="market_beta_drawdown_carry",
            failure_taxonomy_codes=["FAIL_REGIME_FRAGILE", "FAIL_COST_SENSITIVE"],
            feature_list=[
                "log_return",
                "realized_volatility",
                "realized_volatility_mean",
                "market_beta_high",
                "market_beta_drawdown",
                "market_beta_drift",
                "rolling_mid",
                "volume_mean",
            ],
            rule_filters=[
                "moderate_drawdown",
                "volatility_budget",
                "positive_candle_reentry",
                "beta_resilience",
                "participation_floor",
                "not_overheated",
            ],
            required_data=["BTC/USDT:USDT 5m closed-candle OHLCV"],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="doi:10.1093/rfs/hhaa113",
                    title="Risks and Returns of Cryptocurrency",
                    source="The Review of Financial Studies",
                    published_at="2021-01-01",
                    relevance=(
                        "Motivates treating long cryptocurrency beta as a "
                        "distinct compensated risk-premium exposure."
                    ),
                    motivated_thesis_ids=["TH-MARKET-BETA-DRAWDOWN-CARRY-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "market_beta_drawdown_carry"
    assert artifacts.metadata["parameter_defaults"]["buy_volume_window"] == 72
    assert artifacts.metadata["parameter_defaults"]["sell_rsi_exit"] == 78
    assert "realized_volatility" in generated_code
    assert "market_beta_drawdown" in generated_code
    assert "moderate_drawdown" in generated_code
    assert "volatility_budget" in generated_code
    assert "positive_candle_reentry" in generated_code
    assert "participation_floor" in generated_code
    assert "correlation_breakdown" not in generated_code
    assert "negative_funding_pressure" not in generated_code


def test_strategy_code_generator_supports_regime_state_reentry(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyRegimeStateReentryCandidate",
            strategy_type="regime_state_reentry",
            thesis_id="TH-REGIME-STATE-REENTRY-001",
            thesis_type="regime_state_reentry",
            thesis_statement=(
                "Bitcoin long entries are only allowed inside a positive "
                "bull-state proxy where multi-horizon drift is positive, "
                "negative-return frequency is stable, volatility remains "
                "inside budget, and drawdown has not broken the state."
            ),
            hypothesis=(
                "Enter long BTC on closed-candle reentry during a positive "
                "state-dependent drift regime, and exit when the fast regime "
                "state, drawdown boundary, or volatility budget fails."
            ),
            strategy_logic_variant="regime_state_reentry",
            failure_taxonomy_codes=["FAIL_REGIME_FRAGILE", "FAIL_COST_SENSITIVE"],
            feature_list=[
                "log_return",
                "regime_return_fast",
                "regime_return_slow",
                "regime_negative_frequency",
                "regime_negative_frequency_mean",
                "regime_volatility",
                "regime_volatility_mean",
                "regime_drawdown",
                "regime_trendline",
                "rolling_mid",
                "volume_mean",
            ],
            rule_filters=[
                "positive_regime_drift",
                "state_stability",
                "volatility_state_budget",
                "trendline_support",
                "closed_candle_reentry",
                "drawdown_state_intact",
                "participation_floor",
                "not_overheated",
            ],
            required_data=["BTC/USDT:USDT 5m closed-candle OHLCV"],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="doi:10.2307/1912559",
                    title=(
                        "A New Approach to the Economic Analysis of "
                        "Nonstationary Time Series and the Business Cycle"
                    ),
                    source="Econometrica",
                    published_at="1989-03-01",
                    relevance=(
                        "Motivates explicit regime-state modeling rather than "
                        "treating all candles as one stationary return process."
                    ),
                    motivated_thesis_ids=["TH-REGIME-STATE-REENTRY-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "regime_state_reentry"
    assert artifacts.metadata["parameter_defaults"]["buy_volume_factor"] == 0.75
    assert artifacts.metadata["parameter_defaults"]["sell_timeout_candles"] == 48
    assert "regime_return_fast" in generated_code
    assert "regime_negative_frequency" in generated_code
    assert "regime_volatility" in generated_code
    assert "regime_drawdown" in generated_code
    assert "state_stability" in generated_code
    assert "drawdown_state_intact" in generated_code
    assert "correlation_breakdown" not in generated_code
    assert "negative_funding_pressure" not in generated_code


def test_strategy_code_generator_supports_mark_price_dislocation_reclaim(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyMarkPriceDislocationCandidate",
            strategy_type="mark_price_dislocation",
            thesis_id="TH-MARK-PRICE-DISLOCATION-001",
            thesis_type="mark_price_dislocation_reclaim",
            thesis_statement=(
                "Bitcoin perpetual futures can mean-revert after traded "
                "price falls below mark-price fair value, but only when the "
                "discount is reclaiming and mark price support is intact."
            ),
            hypothesis=(
                "Enter long BTC only when the local last-vs-mark discount is "
                "large, contracting, not extreme, and supported by closed "
                "mark-price trend; exit when fair value is reclaimed or the "
                "gap deteriorates."
            ),
            strategy_logic_variant="mark_price_dislocation_reclaim",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "mark_close",
                "mark_log_return",
                "mark_price_gap",
                "mark_price_gap_delta",
                "mark_price_gap_mean",
                "mark_price_gap_abs_mean",
                "mark_price_trend",
                "rolling_mid",
                "range_pct",
                "range_pct_mean",
                "volume_mean",
            ],
            rule_filters=[
                "mark_discount_pressure",
                "mark_gap_reclaiming",
                "mark_price_support",
                "discount_not_extreme",
                "price_resilience",
                "controlled_range",
                "participation_floor",
            ],
            required_data=[
                "BTC/USDT:USDT 5m closed-candle OHLCV",
                "BTC/USDT:USDT 4h mark-price candles",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="bybit:mark-price-docs",
                    title="Mark Price",
                    source="Bybit Help Center",
                    published_at="2026-03-24",
                    relevance=(
                        "Defines mark price as the fair-value anchor used by "
                        "perpetual contracts, motivating last-vs-mark "
                        "dislocation features."
                    ),
                    motivated_thesis_ids=["TH-MARK-PRICE-DISLOCATION-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "mark_price_dislocation_reclaim"
    assert artifacts.metadata["parameter_defaults"]["buy_volume_factor"] == 0.50
    assert artifacts.metadata["parameter_defaults"]["sell_timeout_candles"] == 36
    assert '(pair, "4h", "mark")' in generated_code
    assert "mark_price_gap" in generated_code
    assert "mark_discount_pressure" in generated_code
    assert "mark_gap_reclaiming" in generated_code
    assert "fair_value_reclaimed" in generated_code
    assert "negative_funding_pressure" not in generated_code


def test_strategy_code_generator_supports_mark_discount_reclaim_continuation(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyMarkDiscountReclaimCandidate",
            strategy_type="mark_discount_reclaim",
            thesis_id="TH-MARK-DISCOUNT-RECLAIM-001",
            thesis_type="mark_discount_reclaim",
            thesis_statement=(
                "Bitcoin perpetual futures can continue higher when traded "
                "price is below mark fair value and the discount is closing "
                "on closed candles."
            ),
            hypothesis=(
                "Enter long BTC only when mark-price discount is at least "
                "5 bps, six-candle discount reclaim is positive, and the "
                "three-candle return is non-negative."
            ),
            strategy_logic_variant="mark_discount_reclaim_continuation",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "mark_close",
                "mark_price_gap",
                "mark_price_gap_delta_6",
                "return_3",
            ],
            rule_filters=[
                "mark_discount_pressure",
                "six_candle_discount_reclaim",
                "short_return_nonnegative",
            ],
            required_data=[
                "BTC/USDT:USDT 5m closed-candle OHLCV",
                "BTC/USDT:USDT 4h mark-price candles",
            ],
            parameters=[
                "mark_price_gap_bps_max=-5.0; mark_price_gap_delta_6_bps_min=1.0",
                "return_3_bps_min=0.0; local_falsification_hold_candles=6",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="bybit:mark-price-docs",
                    title="Mark Price",
                    source="Bybit Help Center",
                    published_at="2026-03-24",
                    relevance=(
                        "Defines mark price as the fair-value anchor used by "
                        "perpetual contracts, motivating last-vs-mark "
                        "discount reclaim features."
                    ),
                    motivated_thesis_ids=["TH-MARK-DISCOUNT-RECLAIM-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert (
        artifacts.metadata["strategy_logic_variant"]
        == "mark_discount_reclaim_continuation"
    )
    assert proposal_artifacts.metadata["parameter_overrides"][
        "local_falsification_hold_candles"
    ] == 6
    assert artifacts.metadata["parameter_defaults"]["buy_pullback_lookback"] == 6
    assert artifacts.metadata["parameter_defaults"]["sell_timeout_candles"] == 6
    assert artifacts.metadata["parameter_defaults"]["buy_volume_factor"] == 0.0
    assert '(pair, "4h", "mark")' in generated_code
    assert "6, 288, default=6" in generated_code
    assert "mark_price_gap_delta_6" in generated_code
    assert "return_3" in generated_code
    assert "six_candle_discount_reclaim" in generated_code
    assert "short_return_nonnegative" in generated_code
    assert "discount_reclaimed" in generated_code
    assert "negative_funding_pressure" not in generated_code


def test_strategy_code_generator_supports_mark_fair_value_momentum_lag(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyMarkFairValueMomentumLagCandidate",
            strategy_type="mark_fair_value_momentum_lag",
            thesis_id="TH-MARK-FAIR-VALUE-MOMENTUM-LAG-001",
            thesis_type="mark_fair_value_momentum_lag",
            thesis_statement=(
                "Bitcoin perpetual mark price can lead traded price when the "
                "4h fair-value anchor rises while the last twelve 5m traded "
                "candles have not advanced."
            ),
            hypothesis=(
                "Enter long BTC only when mark-price return is at least 25 "
                "bps, traded twelve-candle return is non-positive, range is "
                "inside budget, and participation is not absent."
            ),
            strategy_logic_variant="mark_fair_value_momentum_lag",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "mark_close",
                "mark_price_return_bps",
                "traded_lag_return_bps",
                "range_pct",
                "volume_zscore",
            ],
            rule_filters=[
                "mark_fair_value_momentum",
                "traded_price_lag",
                "range_budget",
                "participation_floor",
                "event_cooldown",
            ],
            required_data=[
                "BTC/USDT:USDT 5m closed-candle OHLCV",
                "BTC/USDT:USDT 4h mark-price candles",
            ],
            parameters=[
                "mark_price_return_bps_min=25.0; traded_lag_return_bps_max=0.0",
                "range_pct_max=0.8; volume_zscore_min=-1.0",
                "local_falsification_hold_candles=12",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="bybit:mark-price-fair-value",
                    title="Mark Price",
                    source="Bybit Help Center",
                    published_at="2026-03-24",
                    relevance=(
                        "Defines mark price as the fair-value anchor used to "
                        "test whether fair-value momentum can lead traded "
                        "perpetual price."
                    ),
                    motivated_thesis_ids=["TH-MARK-FAIR-VALUE-MOMENTUM-LAG-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "mark_fair_value_momentum_lag"
    assert artifacts.metadata["parameter_defaults"]["buy_pullback_lookback"] == 12
    assert artifacts.metadata["parameter_defaults"]["buy_volume_window"] == 48
    assert artifacts.metadata["parameter_defaults"]["sell_timeout_candles"] == 12
    assert '(pair, "4h", "mark")' in generated_code
    assert "mark_price_return_bps" in generated_code
    assert "traded_lag_return_bps" in generated_code
    assert "mark_fair_value_momentum" in generated_code
    assert "traded_price_lag" in generated_code
    assert "mark_fair_value_event_cooldown" in generated_code
    assert "traded_lag_resolved" in generated_code
    assert "mark_discount_pressure" not in generated_code
    assert "negative_funding_pressure" not in generated_code


def test_strategy_code_generator_supports_microstructure_spread_reversion(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyMicrostructureSpreadCandidate",
            strategy_type="microstructure_spread",
            thesis_id="TH-MICROSTRUCTURE-SPREAD-001",
            thesis_type="microstructure_spread_reversion",
            thesis_statement=(
                "Short-horizon negative return autocovariance can proxy an "
                "elevated bid-ask spread regime; long entries should wait "
                "for the spread proxy to compress and price to recover."
            ),
            hypothesis=(
                "Enter long BTC only when Roll-style spread pressure is "
                "elevated but compressing, high-low spread is normalizing, "
                "price has recovered above the rolling midpoint, and "
                "participation remains present."
            ),
            strategy_logic_variant="microstructure_spread_reversion",
            failure_taxonomy_codes=["FAIL_COST_SENSITIVE", "FAIL_REGIME_FRAGILE"],
            feature_list=[
                "log_return",
                "roll_spread_proxy",
                "roll_spread_mean",
                "roll_spread_delta",
                "hl_spread_proxy",
                "hl_spread_mean",
                "microstructure_noise_ratio",
                "rolling_mid",
                "range_pct",
                "range_pct_mean",
                "volume_mean",
            ],
            rule_filters=[
                "spread_pressure",
                "spread_compressing",
                "hl_spread_normalizing",
                "price_resilience",
                "positive_recovery",
                "controlled_range",
                "participation_floor",
            ],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="doi:10.1111/j.1540-6261.1984.tb03897.x",
                    title="A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market",
                    source="The Journal of Finance",
                    published_at="1984",
                    relevance=(
                        "Defines the Roll spread estimator from return "
                        "autocovariance, motivating the spread proxy."
                    ),
                    motivated_thesis_ids=["TH-MICROSTRUCTURE-SPREAD-001"],
                )
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "microstructure_spread_reversion"
    assert artifacts.metadata["parameter_defaults"]["buy_volume_factor"] == 0.50
    assert artifacts.metadata["parameter_defaults"]["sell_timeout_candles"] == 48
    assert "roll_spread_proxy" in generated_code
    assert "return_autocovariance" in generated_code
    assert "spread_pressure" in generated_code
    assert "spread_compressing" in generated_code
    assert "spread_normalized" in generated_code
    assert "mark_price_gap" not in generated_code
    assert "negative_funding_pressure" not in generated_code


def test_strategy_code_generator_supports_variance_ratio_regime_switch(tmp_path):
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyVarianceRatioRegimeCandidate",
            strategy_type="variance_ratio_regime",
            thesis_id="TH-VARIANCE-RATIO-REGIME-001",
            thesis_type="variance_ratio_regime",
            thesis_statement=(
                "Closed-candle variance-ratio and autocorrelation features can "
                "separate persistent drift regimes from random-walk noise."
            ),
            hypothesis=(
                "Enter long only when the rolling variance ratio is at or above "
                "its local baseline, first-lag autocorrelation is positive, "
                "lookback drift is positive but ATR-normalized extension is "
                "controlled, and price holds the range midpoint."
            ),
            strategy_logic_variant="variance_ratio_regime_switch",
            failure_taxonomy_codes=["FAIL_REGIME_FRAGILE"],
            feature_list=[
                "variance_ratio",
                "variance_ratio_mean",
                "variance_ratio_delta",
                "return_autocorr",
                "autocorr_mean",
                "regime_drift",
                "normalized_regime_return",
            ],
            rule_filters=[
                "variance_ratio_expansion",
                "positive_autocorr_regime",
                "positive_regime_drift",
                "controlled_regime_return",
                "midline_resilience",
                "controlled_range",
                "volume_filter",
            ],
            required_data=["BTC/USDT:USDT 5m closed-candle OHLCV"],
            research_references=[
                StrategyProposalResearchReference(
                    reference_id="doi:10.1093/rfs/1.1.41",
                    title=(
                        "Stock Market Prices Do Not Follow Random Walks: "
                        "Evidence from a Simple Specification Test"
                    ),
                    source="The Review of Financial Studies",
                    published_at="1988-01-01",
                    relevance=(
                        "Introduces the variance-ratio random-walk diagnostic "
                        "that motivates this distinct regime family."
                    ),
                ),
            ],
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "variance_ratio_regime_switch"
    assert artifacts.metadata["parameter_defaults"]["buy_pullback_lookback"] == 24
    assert "variance_ratio" in generated_code
    assert "return_autocorr" in generated_code
    assert "normalized_regime_return" in generated_code
    assert "variance_ratio_expansion" in generated_code
    assert "positive_autocorr_regime" in generated_code
    assert "persistent_memory_regime" not in generated_code
    assert "eth_positive_lead" not in generated_code


def test_strategy_code_generator_supports_bipower_jump_decay(tmp_path):
    thesis_id = "TH-BIPOWER-JUMP-DECAY-001"
    proposal_artifacts = build_strategy_proposal(
        _strategy_proposal_inputs(
            tmp_path,
            strategy_name="LongOnlyBipowerJumpDecayCandidate",
            strategy_type="bipower_jump_decay",
            strategy_logic_variant="bipower_jump_decay",
            thesis_id=thesis_id,
            thesis_type="realized_multipower_jump_decay",
            research_references=_strategy_proposal_bipower_references(thesis_id),
        )
    )
    write_strategy_proposal_artifacts(proposal_artifacts)
    _write_hypothesis_metadata(proposal_artifacts.metadata_path)

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)
    generated_code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["strategy_logic_variant"] == "bipower_jump_decay"
    assert artifacts.metadata["parameter_defaults"]["buy_pullback_lookback"] == 24
    assert "bipower_variation" in generated_code
    assert "jump_variation_ratio" in generated_code
    assert "continuous_variance_decay" in generated_code
    assert "positive_jump_detected" in generated_code
    assert "jump_dominates_continuous_variance" in generated_code
    assert "jump_decay_failed" in generated_code
    assert "volatility_breakout" not in generated_code


def test_strategy_code_generator_blocks_tampered_proposal_hash(tmp_path):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    _write_hypothesis_metadata(proposal_artifacts.metadata_path)
    proposal_artifacts.proposal_path.write_text(
        proposal_artifacts.proposal_markdown + "\nTampered after metadata write.\n",
        encoding="utf-8",
    )

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)

    assert artifacts.metadata["status"] == "blocked"
    assert artifacts.metadata["strategy_code_generated"] is False
    assert artifacts.metadata["candidate_evaluation_eligible"] is False
    assert not artifacts.strategy_path.exists()
    assert "proposal_content_hash_matches" in {
        check["name"] for check in artifacts.metadata["blockers"]
    }


def test_strategy_code_generator_blocks_missing_required_proposal_section(tmp_path):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    _write_hypothesis_metadata(proposal_artifacts.metadata_path)
    proposal_markdown = proposal_artifacts.proposal_markdown.replace(
        "## Entry Logic\n\n", "", 1
    )
    proposal_artifacts.proposal_path.write_text(proposal_markdown, encoding="utf-8")
    metadata = json.loads(proposal_artifacts.metadata_path.read_text(encoding="utf-8"))
    metadata["proposal_content_hash"] = hashlib.sha256(
        proposal_markdown.encode("utf-8")
    ).hexdigest()
    proposal_artifacts.metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)

    assert artifacts.metadata["status"] == "blocked"
    assert not artifacts.strategy_path.exists()
    assert "proposal_markdown_section_entry_logic_present" in {
        check["name"] for check in artifacts.metadata["blockers"]
    }


def test_strategy_code_generator_blocks_unsafe_proposal_scope(tmp_path):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    _write_hypothesis_metadata(proposal_artifacts.metadata_path)
    metadata = json.loads(proposal_artifacts.metadata_path.read_text(encoding="utf-8"))
    metadata["safety_scope"]["shorting"] = True
    metadata["safety_scope"]["leverage"] = 2.0
    proposal_artifacts.metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    artifacts = build_strategy_code(
        _strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path)
    )
    write_strategy_code_artifacts(artifacts)

    assert artifacts.metadata["status"] == "blocked"
    assert artifacts.metadata["strategy_code_generated"] is False
    assert not artifacts.strategy_path.exists()
    blocker_names = {check["name"] for check in artifacts.metadata["blockers"]}
    assert "proposal_safety_shorting_false" in blocker_names
    assert "proposal_safety_leverage_capped_at_one" in blocker_names


def test_strategy_code_generator_freqai_mode_emits_freqai_methods(tmp_path):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    _write_hypothesis_metadata(proposal_artifacts.metadata_path)
    metadata = json.loads(proposal_artifacts.metadata_path.read_text(encoding="utf-8"))
    metadata["generator_mode"] = "freqai"
    metadata["feature_list"] = ["rsi", "ema", "atr"]
    metadata["target_definition"] = "future_return"
    metadata["label_horizon"] = 12
    metadata["prediction_threshold"] = 0.01
    metadata["rule_filters"] = ["trend_filter", "volume_filter"]
    metadata["risk_policy"] = "long_only_leverage_1"
    proposal_artifacts.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    artifacts = build_strategy_code(_strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path))
    write_strategy_code_artifacts(artifacts)
    code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["generator_mode"] == "freqai"
    assert "def feature_engineering_expand_all" in code
    assert "def feature_engineering_expand_basic" in code
    assert "def feature_engineering_standard" in code
    assert "def set_freqai_targets" in code
    assert "self.freqai.start(dataframe, metadata, self)" in code
    assert 'shift(-12)' in code
    assert artifacts.metadata["label_horizon"] == 12
    assert artifacts.metadata["freqai_expected_target_column"] == "&-future_return"
    assert artifacts.metadata["freqai_identifier"] == candidate_freqai_identifier(
        artifacts.metadata["strategy_name"],
        artifacts.metadata["candidate_id"],
        "future_return",
    )
    assert artifacts.metadata["freqai_identifier_policy"] == "candidate_specific"
    assert artifacts.metadata["freqai_cache_policy"][
        "reuse_existing_predictions_allowed"
    ] is False


def test_strategy_code_generator_rule_based_mode_does_not_require_ml_threshold(tmp_path):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    _write_hypothesis_metadata(proposal_artifacts.metadata_path)
    metadata = json.loads(proposal_artifacts.metadata_path.read_text(encoding="utf-8"))
    metadata["generator_mode"] = "rule_based"
    proposal_artifacts.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    artifacts = build_strategy_code(_strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path))
    write_strategy_code_artifacts(artifacts)
    code = artifacts.strategy_path.read_text(encoding="utf-8")

    assert artifacts.metadata["status"] == "generated"
    assert artifacts.metadata["generator_mode"] == "rule_based"
    assert "def set_freqai_targets" not in code


def test_paper_config_safety_accepts_dry_run_sanitized_config():
    config = _paper_config("PaperStrategy")

    result = evaluate_config_safety(config, strategy="PaperStrategy")

    assert result.ok
    assert result.sanitized_summary["dry_run"] is True
    assert result.to_dict()["metadata_contains_secrets"] is False


def test_paper_config_safety_rejects_credentials_and_live_mode():
    config = _paper_config("PaperStrategy")
    config["dry_run"] = False
    config["exchange"]["key"] = "not-a-real-key-but-nonempty"

    result = evaluate_config_safety(config, strategy="PaperStrategy")

    assert not result.ok
    assert {check.name for check in result.checks if check.status == "blocked"} >= {
        "dry_run_true",
        "no_credential_values",
    }


def test_paper_config_safety_rejects_forced_entry_and_unsafe_startup():
    config = _paper_config("PaperStrategy")
    config["force_entry_enable"] = True
    config["initial_state"] = "running"
    config.pop("cancel_open_orders_on_exit")

    result = evaluate_config_safety(config, strategy="PaperStrategy")

    assert not result.ok
    assert {check.name for check in result.checks if check.status == "blocked"} >= {
        "force_entry_disabled",
        "initial_state_stopped",
        "cancel_open_orders_on_exit_explicit",
    }


def test_paper_config_safety_rejects_oversized_simulation_limits():
    config = _paper_config("PaperStrategy")
    config["max_open_trades"] = 4
    config["stake_amount"] = 2000
    config["dry_run_wallet"] = 20000

    result = evaluate_config_safety(config, strategy="PaperStrategy")

    assert not result.ok
    assert {check.name for check in result.checks if check.status == "blocked"} >= {
        "max_open_trades_conservative",
        "stake_amount_conservative",
        "dry_run_wallet_conservative",
    }


def test_paper_strategy_long_only_rejects_short_signals_and_high_leverage(tmp_path):
    strategy_path = tmp_path / "PaperStrategy.py"
    strategy_path.write_text(
        "class PaperStrategy:\n"
        "    can_short = False\n"
        "    def leverage(self, *args, **kwargs):\n"
        "        return 2.0\n"
        "    def populate_entry_trend(self, dataframe, metadata):\n"
        "        dataframe.loc[:, 'enter_short'] = 1\n"
        "        return dataframe\n",
        encoding="utf-8",
    )

    result = evaluate_strategy_long_only(strategy_path, "PaperStrategy")

    assert not result.ok
    blocked_names = {check.name for check in result.checks if check.status == "blocked"}
    assert "no_short_signals" in blocked_names
    assert "leverage_hook_no_constant_above_one" in blocked_names


def test_paper_readiness_fails_when_phase2_gates_fail(tmp_path):
    strategy_path = _write_paper_strategy(tmp_path, "PaperStrategy")
    historical_dir, walk_forward_dir, training_dir = _write_paper_evidence(
        tmp_path,
        historical_gate_pass=False,
        walk_forward_recommendation="pass",
        training_recommendation="pass",
    )
    config = _paper_config("PaperStrategy")
    static_report = scan_paths([strategy_path])
    inputs = PaperReadinessInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_check",
        config_path=tmp_path / "config.json",
        strategy_path=strategy_path,
        historical_dir=historical_dir,
        walk_forward_dir=walk_forward_dir,
        training_dir=training_dir,
        reviewer_notes=["Reviewed for no-startup paper readiness."],
    )

    readiness, _, _ = evaluate_paper_readiness(
        inputs,
        static_report=static_report,
        config=config,
        strategy_file=strategy_path,
    )

    assert readiness["readiness"] == "fail"
    assert readiness["blockers"] == []
    assert "historical_backtest_gate" in {
        check["name"] for check in readiness["failures"]
    }


def test_paper_readiness_passes_with_required_local_evidence(tmp_path):
    strategy_path = _write_paper_strategy(tmp_path, "PaperStrategy")
    historical_dir, walk_forward_dir, training_dir = _write_paper_evidence(
        tmp_path,
        historical_gate_pass=True,
        walk_forward_recommendation="pass",
        training_recommendation="pass",
    )
    config = _paper_config("PaperStrategy")
    static_report = scan_paths([strategy_path])
    inputs = PaperReadinessInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_check",
        config_path=tmp_path / "config.json",
        strategy_path=strategy_path,
        historical_dir=historical_dir,
        walk_forward_dir=walk_forward_dir,
        training_dir=training_dir,
        reviewer_notes=["Reviewed for no-startup paper readiness."],
    )

    readiness, candidate_artifacts, config_safety = evaluate_paper_readiness(
        inputs,
        static_report=static_report,
        config=config,
        strategy_file=strategy_path,
    )

    assert readiness["readiness"] == "pass"
    assert all(info["exists"] for info in candidate_artifacts["artifacts"].values())
    assert config_safety.ok
    assert readiness["safety_scope"]["bot_startup"] is False


def test_paper_readiness_blocks_short_trade_artifact(tmp_path):
    strategy_path = _write_paper_strategy(tmp_path, "PaperStrategy")
    historical_dir, walk_forward_dir, training_dir = _write_paper_evidence(
        tmp_path,
        historical_gate_pass=True,
        walk_forward_recommendation="pass",
        training_recommendation="pass",
    )
    (historical_dir / "trades.csv").write_text(
        "is_short,leverage\nTrue,1.0\n", encoding="utf-8"
    )
    config = _paper_config("PaperStrategy")
    static_report = scan_paths([strategy_path])
    inputs = PaperReadinessInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_check",
        config_path=tmp_path / "config.json",
        strategy_path=strategy_path,
        historical_dir=historical_dir,
        walk_forward_dir=walk_forward_dir,
        training_dir=training_dir,
        reviewer_notes=["Reviewed for no-startup paper readiness."],
    )

    readiness, _, _ = evaluate_paper_readiness(
        inputs,
        static_report=static_report,
        config=config,
        strategy_file=strategy_path,
    )

    assert readiness["readiness"] == "blocked"
    assert "historical_trades_no_shorts" in {
        check["name"] for check in readiness["blockers"]
    }


def test_paper_readiness_blocks_missing_training_child_evidence(tmp_path):
    strategy_path = _write_paper_strategy(tmp_path, "PaperStrategy")
    historical_dir, walk_forward_dir, training_dir = _write_paper_evidence(
        tmp_path,
        historical_gate_pass=True,
        walk_forward_recommendation="pass",
        training_recommendation="pass",
    )
    child_trades = (
        training_dir
        / "freqai_backtests"
        / "PaperStrategy"
        / "train_20250101_20250201"
        / "trades.csv"
    )
    child_trades.unlink()
    config = _paper_config("PaperStrategy")
    static_report = scan_paths([strategy_path])
    inputs = PaperReadinessInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_check",
        config_path=tmp_path / "config.json",
        strategy_path=strategy_path,
        historical_dir=historical_dir,
        walk_forward_dir=walk_forward_dir,
        training_dir=training_dir,
        reviewer_notes=["Reviewed for no-startup paper readiness."],
    )

    readiness, _, _ = evaluate_paper_readiness(
        inputs,
        static_report=static_report,
        config=config,
        strategy_file=strategy_path,
    )

    assert readiness["readiness"] == "blocked"
    assert "training_train_20250101_20250201_trades_present" in {
        check["name"] for check in readiness["blockers"]
    }


def test_paper_readiness_blocks_walk_forward_child_high_leverage(tmp_path):
    strategy_path = _write_paper_strategy(tmp_path, "PaperStrategy")
    historical_dir, walk_forward_dir, training_dir = _write_paper_evidence(
        tmp_path,
        historical_gate_pass=True,
        walk_forward_recommendation="pass",
        training_recommendation="pass",
    )
    window_trades = (
        walk_forward_dir
        / "windows"
        / "PaperStrategy"
        / "wf_01_20250101_20250115"
        / "trades.csv"
    )
    window_trades.write_text("is_short,leverage\nFalse,1.5\n", encoding="utf-8")
    config = _paper_config("PaperStrategy")
    static_report = scan_paths([strategy_path])
    inputs = PaperReadinessInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_check",
        config_path=tmp_path / "config.json",
        strategy_path=strategy_path,
        historical_dir=historical_dir,
        walk_forward_dir=walk_forward_dir,
        training_dir=training_dir,
        reviewer_notes=["Reviewed for no-startup paper readiness."],
    )

    readiness, _, _ = evaluate_paper_readiness(
        inputs,
        static_report=static_report,
        config=config,
        strategy_file=strategy_path,
    )

    assert readiness["readiness"] == "blocked"
    assert "walk_forward_wf_01_20250101_20250115_trades_no_leverage_above_one" in {
        check["name"] for check in readiness["blockers"]
    }


def test_paper_run_plan_ready_requires_passed_readiness_and_acknowledgement(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_paper_config("PaperStrategy")), encoding="utf-8")
    readiness_path = tmp_path / "paper_readiness.json"
    readiness = _paper_readiness_payload(
        strategy="PaperStrategy",
        status="pass",
        config_path=config_path,
    )
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    inputs = PaperRunPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_plan",
        readiness_path=readiness_path,
        config_path=config_path,
        strategy_path=tmp_path / "strategies",
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed for future paper planning only."],
        confirm_paper=True,
        command=["python", "scripts/bot_factory_plan_paper_run.py"],
    )

    plan = build_paper_run_plan(inputs, readiness)
    write_paper_run_plan_artifacts(inputs, plan)

    assert plan["status"] == "ready"
    assert plan["future_startup"]["eligible"] is True
    assert plan["future_startup"]["startup_authorized_by_this_command"] is False
    assert plan["safety_scope"]["bot_startup"] is False
    assert plan["future_startup"]["command_preview"][:2] == ["freqtrade", "trade"]
    assert (inputs.output_dir / "paper_run_plan.json").is_file()
    assert (inputs.output_dir / "stop_cleanup.md").is_file()


def test_paper_run_plan_blocks_failed_readiness_without_start_command(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_paper_config("PaperStrategy")), encoding="utf-8")
    readiness = _paper_readiness_payload(
        strategy="PaperStrategy",
        status="fail",
        config_path=config_path,
        failures=[{"name": "historical_backtest_gate"}],
    )
    inputs = PaperRunPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_plan",
        readiness_path=tmp_path / "paper_readiness.json",
        config_path=config_path,
        reviewer_notes=["Reviewed for future paper planning only."],
        confirm_paper=True,
    )

    plan = build_paper_run_plan(inputs, readiness)

    assert plan["status"] == "blocked"
    assert plan["future_startup"]["eligible"] is False
    assert plan["future_startup"]["command_preview"] == []
    assert {"readiness_passed", "readiness_has_no_failures"} <= {
        check["name"] for check in plan["blockers"]
    }


def test_paper_run_plan_blocks_missing_confirmation_and_reviewer_note(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_paper_config("PaperStrategy")), encoding="utf-8")
    readiness = _paper_readiness_payload(
        strategy="PaperStrategy",
        status="pass",
        config_path=config_path,
    )
    inputs = PaperRunPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_plan",
        readiness_path=tmp_path / "paper_readiness.json",
        config_path=config_path,
    )

    plan = build_paper_run_plan(inputs, readiness)

    assert plan["status"] == "blocked"
    assert {"confirm_paper_acknowledged", "reviewer_note_present"} <= {
        check["name"] for check in plan["blockers"]
    }


def test_paper_run_plan_blocks_unsafe_readiness_scope(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_paper_config("PaperStrategy")), encoding="utf-8")
    readiness = _paper_readiness_payload(
        strategy="PaperStrategy",
        status="pass",
        config_path=config_path,
    )
    readiness["safety_scope"]["uses_api_keys_or_secrets"] = True
    readiness["safety_scope"]["local_artifacts_source_of_truth"] = False
    inputs = PaperRunPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_plan",
        readiness_path=tmp_path / "paper_readiness.json",
        config_path=config_path,
        reviewer_notes=["Reviewed for future paper planning only."],
        confirm_paper=True,
    )

    plan = build_paper_run_plan(inputs, readiness)

    assert plan["status"] == "blocked"
    assert {
        "readiness_metadata_sanitized",
        "readiness_local_artifacts_source_of_truth",
    } <= {check["name"] for check in plan["blockers"]}


def test_paper_startup_preflight_ready_records_templates_without_starting(tmp_path):
    plan, plan_path = _write_ready_paper_run_plan(tmp_path)
    start_command = " ".join(plan["future_startup"]["command_preview"])
    inputs = PaperStartupPreflightInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_start_preflight",
        plan_path=plan_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed startup preflight only; do not start paper trading."],
        confirm_paper_start=True,
        requested_start_command=start_command,
        command=["python", "scripts/bot_factory_prepare_paper_start.py"],
    )

    preflight = build_paper_startup_preflight(inputs, plan)
    write_paper_startup_preflight_artifacts(inputs, preflight)

    assert preflight["status"] == "ready"
    assert preflight["startup"]["eligible"] is True
    assert preflight["startup"]["startup_executed"] is False
    assert preflight["startup"]["startup_authorized_by_this_command"] is False
    assert preflight["safety_scope"]["bot_startup"] is False
    assert preflight["process_metadata"]["process_started"] is False
    assert preflight["process_metadata"]["pid"] is None
    assert preflight["startup"]["command_preview"][:2] == ["freqtrade", "trade"]
    assert (inputs.output_dir / "paper_startup_preflight.json").is_file()
    assert (inputs.output_dir / "process_metadata_template.json").is_file()
    assert (inputs.output_dir / "status_snapshot_template.json").is_file()
    assert (inputs.output_dir / "start_command_preview.txt").read_text(
        encoding="utf-8"
    ).startswith("freqtrade trade")


def test_paper_startup_preflight_blocks_failed_plan_without_start_command(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_paper_config("PaperStrategy")), encoding="utf-8")
    readiness = _paper_readiness_payload(
        strategy="PaperStrategy",
        status="fail",
        config_path=config_path,
        failures=[{"name": "historical_backtest_gate"}],
    )
    plan_inputs = PaperRunPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_plan",
        readiness_path=tmp_path / "paper_readiness.json",
        config_path=config_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed for future paper planning only."],
        confirm_paper=True,
    )
    plan = build_paper_run_plan(plan_inputs, readiness)
    write_paper_run_plan_artifacts(plan_inputs, plan)
    inputs = PaperStartupPreflightInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_start_preflight",
        plan_path=plan_inputs.output_dir / "paper_run_plan.json",
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed startup preflight only."],
        confirm_paper_start=True,
        requested_start_command="freqtrade trade --config config.json",
    )

    preflight = build_paper_startup_preflight(inputs, plan)
    write_paper_startup_preflight_artifacts(inputs, preflight)

    assert preflight["status"] == "blocked"
    assert preflight["startup"]["eligible"] is False
    assert preflight["startup"]["command_preview"] == []
    assert (inputs.output_dir / "start_command_preview.txt").read_text(
        encoding="utf-8"
    ) == ""
    assert {"paper_plan_ready", "paper_plan_future_startup_eligible"} <= {
        check["name"] for check in preflight["blockers"]
    }


def test_paper_startup_preflight_blocks_missing_confirmation_command_and_note(tmp_path):
    plan, plan_path = _write_ready_paper_run_plan(tmp_path)
    inputs = PaperStartupPreflightInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_start_preflight",
        plan_path=plan_path,
        output_root=tmp_path / "data" / "paper",
    )

    preflight = build_paper_startup_preflight(inputs, plan)

    assert preflight["status"] == "blocked"
    assert {
        "confirm_paper_start_acknowledged",
        "requested_start_command_present",
        "reviewer_note_present",
    } <= {check["name"] for check in preflight["blockers"]}


def test_paper_startup_preflight_blocks_tampered_start_command_preview(tmp_path):
    plan, plan_path = _write_ready_paper_run_plan(tmp_path)
    plan["future_startup"]["command_preview"] = [
        "python",
        "trade",
        "--config",
        plan["config_path"],
        "--strategy",
        "OtherStrategy",
        "--strategy-path",
        plan["strategy_path"],
    ]
    inputs = PaperStartupPreflightInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_start_preflight",
        plan_path=plan_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed startup preflight only."],
        confirm_paper_start=True,
        requested_start_command=" ".join(plan["future_startup"]["command_preview"]),
    )

    preflight = build_paper_startup_preflight(inputs, plan)

    assert preflight["status"] == "blocked"
    assert {
        "paper_plan_start_command_uses_freqtrade_trade",
        "paper_plan_start_command_strategy_matches_candidate",
    } <= {check["name"] for check in preflight["blockers"]}


def test_paper_monitoring_plan_ready_records_schemas_without_process_control(tmp_path):
    preflight, preflight_path = _write_ready_paper_startup_preflight(tmp_path)
    inputs = PaperMonitoringPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_monitoring",
        startup_preflight_path=preflight_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed monitoring schema planning only."],
        command=["python", "scripts/bot_factory_plan_paper_monitoring.py"],
    )

    plan = build_paper_monitoring_plan(inputs, preflight)
    write_paper_monitoring_plan_artifacts(inputs, plan)

    assert plan["status"] == "ready"
    assert plan["monitoring"]["eligible"] is True
    assert plan["monitoring"]["monitoring_started"] is False
    assert plan["monitoring"]["status_polling_started"] is False
    assert plan["monitoring"]["process_control"] is False
    assert plan["safety_scope"]["bot_startup"] is False
    assert "status" in plan["schemas"]["status_snapshot"]["required"]
    assert "trade_counts" in plan["schemas"]["paper_metrics"]["required"]
    assert "stdout_log" in plan["schemas"]["process_metadata"]["required"]
    assert (inputs.output_dir / "paper_monitoring_plan.json").is_file()
    assert (inputs.output_dir / "status_snapshot_schema.json").is_file()
    assert (inputs.output_dir / "paper_metrics_schema.json").is_file()
    assert (inputs.output_dir / "process_metadata_schema.json").is_file()


def test_paper_monitoring_plan_blocks_failed_startup_preflight(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_paper_config("PaperStrategy")), encoding="utf-8")
    readiness = _paper_readiness_payload(
        strategy="PaperStrategy",
        status="fail",
        config_path=config_path,
        failures=[{"name": "historical_backtest_gate"}],
    )
    plan_inputs = PaperRunPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_plan",
        readiness_path=tmp_path / "paper_readiness.json",
        config_path=config_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed for future paper planning only."],
        confirm_paper=True,
    )
    paper_plan = build_paper_run_plan(plan_inputs, readiness)
    write_paper_run_plan_artifacts(plan_inputs, paper_plan)
    preflight_inputs = PaperStartupPreflightInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_start_preflight",
        plan_path=plan_inputs.output_dir / "paper_run_plan.json",
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed startup preflight only."],
        confirm_paper_start=True,
        requested_start_command="freqtrade trade --config config.json",
    )
    preflight = build_paper_startup_preflight(preflight_inputs, paper_plan)
    write_paper_startup_preflight_artifacts(preflight_inputs, preflight)
    inputs = PaperMonitoringPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_monitoring",
        startup_preflight_path=preflight_inputs.output_dir / "paper_startup_preflight.json",
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed monitoring schema planning only."],
    )

    plan = build_paper_monitoring_plan(inputs, preflight)
    write_paper_monitoring_plan_artifacts(inputs, plan)

    assert plan["status"] == "blocked"
    assert plan["monitoring"]["eligible"] is False
    assert plan["monitoring"]["process_control"] is False
    assert {"startup_preflight_ready", "startup_preflight_startup_eligible"} <= {
        check["name"] for check in plan["blockers"]
    }
    assert (inputs.output_dir / "status_snapshot_schema.json").is_file()


def test_paper_monitoring_plan_blocks_missing_note_and_unsafe_scope(tmp_path):
    preflight, preflight_path = _write_ready_paper_startup_preflight(tmp_path)
    preflight["safety_scope"]["metadata_contains_secrets"] = True
    preflight["status_snapshot"]["paper_trading_started"] = True
    inputs = PaperMonitoringPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_monitoring",
        startup_preflight_path=preflight_path,
        output_root=tmp_path / "data" / "paper",
    )

    plan = build_paper_monitoring_plan(inputs, preflight)

    assert plan["status"] == "blocked"
    assert {
        "startup_preflight_no_secrets_leverage_or_shorting_scope",
        "status_snapshot_template_records_no_startup",
        "reviewer_note_present",
    } <= {check["name"] for check in plan["blockers"]}


def test_paper_stop_cleanup_plan_ready_records_artifacts_without_process_control(tmp_path):
    monitoring_plan, monitoring_plan_path = _write_ready_paper_monitoring_plan(tmp_path)
    inputs = PaperStopCleanupPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_stop_cleanup",
        monitoring_plan_path=monitoring_plan_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed stop and cleanup planning only."],
        command=["python", "scripts/bot_factory_plan_paper_stop_cleanup.py"],
    )

    plan = build_paper_stop_cleanup_plan(inputs, monitoring_plan)
    write_paper_stop_cleanup_plan_artifacts(inputs, plan)

    assert plan["status"] == "ready"
    assert plan["stop_cleanup"]["eligible"] is True
    assert plan["stop_cleanup"]["stop_executed"] is False
    assert plan["stop_cleanup"]["cleanup_executed"] is False
    assert plan["stop_cleanup"]["process_control"] is False
    assert plan["stop_cleanup"]["stop_authorized_by_this_command"] is False
    assert plan["safety_scope"]["process_stop_started"] is False
    assert "process_metadata_path" in plan["schemas"]["stop_request"]["required"]
    assert (inputs.output_dir / "paper_stop_cleanup_plan.json").is_file()
    assert (inputs.output_dir / "paper_stop_cleanup_report.md").is_file()
    assert (inputs.output_dir / "stop_request_schema.json").is_file()
    assert (inputs.output_dir / "cleanup_checklist.md").is_file()


def test_paper_stop_cleanup_plan_blocks_failed_monitoring_plan(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_paper_config("PaperStrategy")), encoding="utf-8")
    readiness = _paper_readiness_payload(
        strategy="PaperStrategy",
        status="fail",
        config_path=config_path,
        failures=[{"name": "historical_backtest_gate"}],
    )
    plan_inputs = PaperRunPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_plan",
        readiness_path=tmp_path / "paper_readiness.json",
        config_path=config_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed for future paper planning only."],
        confirm_paper=True,
    )
    paper_plan = build_paper_run_plan(plan_inputs, readiness)
    write_paper_run_plan_artifacts(plan_inputs, paper_plan)
    preflight_inputs = PaperStartupPreflightInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_start_preflight",
        plan_path=plan_inputs.output_dir / "paper_run_plan.json",
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed startup preflight only."],
        confirm_paper_start=True,
        requested_start_command="freqtrade trade --config config.json",
    )
    preflight = build_paper_startup_preflight(preflight_inputs, paper_plan)
    write_paper_startup_preflight_artifacts(preflight_inputs, preflight)
    monitoring_inputs = PaperMonitoringPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_monitoring",
        startup_preflight_path=preflight_inputs.output_dir / "paper_startup_preflight.json",
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed monitoring schema planning only."],
    )
    monitoring_plan = build_paper_monitoring_plan(monitoring_inputs, preflight)
    write_paper_monitoring_plan_artifacts(monitoring_inputs, monitoring_plan)
    inputs = PaperStopCleanupPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_stop_cleanup",
        monitoring_plan_path=monitoring_inputs.output_dir / "paper_monitoring_plan.json",
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed stop and cleanup planning only."],
    )

    plan = build_paper_stop_cleanup_plan(inputs, monitoring_plan)
    write_paper_stop_cleanup_plan_artifacts(inputs, plan)

    assert plan["status"] == "blocked"
    assert plan["stop_cleanup"]["eligible"] is False
    assert plan["stop_cleanup"]["process_control"] is False
    assert {"monitoring_plan_ready", "monitoring_plan_eligible"} <= {
        check["name"] for check in plan["blockers"]
    }
    assert (inputs.output_dir / "stop_request_schema.json").is_file()


def test_paper_stop_cleanup_plan_blocks_missing_note_and_unsafe_monitoring_scope(tmp_path):
    monitoring_plan, monitoring_plan_path = _write_ready_paper_monitoring_plan(tmp_path)
    monitoring_plan["monitoring"]["process_control"] = True
    monitoring_plan["safety_scope"]["metadata_contains_secrets"] = True
    inputs = PaperStopCleanupPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_stop_cleanup",
        monitoring_plan_path=monitoring_plan_path,
        output_root=tmp_path / "data" / "paper",
    )

    plan = build_paper_stop_cleanup_plan(inputs, monitoring_plan)

    assert plan["status"] == "blocked"
    assert {
        "monitoring_plan_no_process_control",
        "monitoring_plan_no_secrets_leverage_or_shorting_scope",
        "reviewer_note_present",
    } <= {check["name"] for check in plan["blockers"]}


def test_paper_execution_request_ready_records_manifest_without_starting(tmp_path):
    chain = _write_ready_paper_execution_chain(tmp_path)
    start_command = " ".join(chain["startup_preflight"]["startup"]["command_preview"])
    inputs = PaperExecutionRequestInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_execution_request",
        readiness_path=chain["readiness_path"],
        plan_path=chain["plan_path"],
        startup_preflight_path=chain["startup_preflight_path"],
        monitoring_plan_path=chain["monitoring_plan_path"],
        stop_cleanup_plan_path=chain["stop_cleanup_plan_path"],
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed execution request only; do not start paper trading."],
        confirm_paper_execution=True,
        requested_start_command=start_command,
        command=["python", "scripts/bot_factory_request_paper_start.py"],
    )

    request = build_paper_execution_request(
        inputs,
        chain["readiness"],
        chain["paper_plan"],
        chain["startup_preflight"],
        chain["monitoring_plan"],
        chain["stop_cleanup_plan"],
    )
    write_paper_execution_request_artifacts(inputs, request)

    assert request["status"] == "ready"
    assert request["execution_request"]["eligible"] is True
    assert request["execution_request"]["startup_executed"] is False
    assert request["execution_request"]["process_control"] is False
    assert request["execution_request"]["startup_authorized_by_this_command"] is False
    assert request["safety_scope"]["bot_startup"] is False
    assert request["execution_request"]["command_preview"][:2] == ["freqtrade", "trade"]
    assert (inputs.output_dir / "paper_execution_request.json").is_file()
    assert (inputs.output_dir / "paper_execution_request_report.md").is_file()
    assert (inputs.output_dir / "execution_manifest_template.json").is_file()
    assert (inputs.output_dir / "start_command_request.txt").read_text(
        encoding="utf-8"
    ).startswith("freqtrade trade")


def test_paper_execution_request_blocks_failed_stop_cleanup_plan(tmp_path):
    chain = _write_ready_paper_execution_chain(tmp_path)
    chain["stop_cleanup_plan"]["status"] = "blocked"
    chain["stop_cleanup_plan"]["blockers"] = [{"name": "stop_cleanup_plan_ready"}]
    chain["stop_cleanup_plan"]["stop_cleanup"]["eligible"] = False
    start_command = " ".join(chain["startup_preflight"]["startup"]["command_preview"])
    inputs = PaperExecutionRequestInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_execution_request",
        readiness_path=chain["readiness_path"],
        plan_path=chain["plan_path"],
        startup_preflight_path=chain["startup_preflight_path"],
        monitoring_plan_path=chain["monitoring_plan_path"],
        stop_cleanup_plan_path=chain["stop_cleanup_plan_path"],
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed execution request only."],
        confirm_paper_execution=True,
        requested_start_command=start_command,
    )

    request = build_paper_execution_request(
        inputs,
        chain["readiness"],
        chain["paper_plan"],
        chain["startup_preflight"],
        chain["monitoring_plan"],
        chain["stop_cleanup_plan"],
    )
    write_paper_execution_request_artifacts(inputs, request)

    assert request["status"] == "blocked"
    assert request["execution_request"]["eligible"] is False
    assert request["execution_request"]["command_preview"] == []
    assert {
        "stop_cleanup_plan_ready",
        "stop_cleanup_plan_has_no_blockers",
        "stop_cleanup_plan_eligible",
    } <= {check["name"] for check in request["blockers"]}
    assert (inputs.output_dir / "start_command_request.txt").read_text(
        encoding="utf-8"
    ) == ""


def test_paper_execution_request_blocks_missing_note_confirmation_and_unsafe_scope(tmp_path):
    chain = _write_ready_paper_execution_chain(tmp_path)
    chain["startup_preflight"]["safety_scope"]["metadata_contains_secrets"] = True
    chain["monitoring_plan"]["monitoring"]["process_control"] = True
    chain["stop_cleanup_plan"]["stop_cleanup"]["stop_executed"] = True
    inputs = PaperExecutionRequestInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_execution_request",
        readiness_path=chain["readiness_path"],
        plan_path=chain["plan_path"],
        startup_preflight_path=chain["startup_preflight_path"],
        monitoring_plan_path=chain["monitoring_plan_path"],
        stop_cleanup_plan_path=chain["stop_cleanup_plan_path"],
        output_root=tmp_path / "data" / "paper",
    )

    request = build_paper_execution_request(
        inputs,
        chain["readiness"],
        chain["paper_plan"],
        chain["startup_preflight"],
        chain["monitoring_plan"],
        chain["stop_cleanup_plan"],
    )

    assert request["status"] == "blocked"
    assert {
        "startup_preflight_no_secrets_leverage_or_shorting_scope",
        "monitoring_plan_no_process_control",
        "stop_cleanup_plan_no_process_control",
        "confirm_paper_execution_acknowledged",
        "requested_start_command_present",
        "reviewer_note_present",
    } <= {check["name"] for check in request["blockers"]}


def test_paper_process_executor_plan_ready_records_manifest_without_starting(tmp_path):
    execution_request, execution_request_path = _write_ready_paper_execution_request(tmp_path)
    start_command = " ".join(execution_request["execution_request"]["command_preview"])
    inputs = PaperProcessExecutorPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_executor_plan",
        execution_request_path=execution_request_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed process executor planning only; do not start paper trading."],
        confirm_process_executor_plan=True,
        requested_start_command=start_command,
        command=["python", "scripts/bot_factory_plan_paper_executor.py"],
    )

    plan = build_paper_process_executor_plan(inputs, execution_request)
    write_paper_process_executor_plan_artifacts(inputs, plan)

    assert plan["status"] == "ready"
    assert plan["executor_plan"]["eligible"] is True
    assert plan["executor_plan"]["startup_executed"] is False
    assert plan["executor_plan"]["process_started"] is False
    assert plan["executor_plan"]["process_control"] is False
    assert plan["executor_plan"]["start_authorized_by_this_command"] is False
    assert plan["safety_scope"]["bot_startup"] is False
    assert plan["executor_plan"]["command_preview"][:2] == ["freqtrade", "trade"]
    assert (inputs.output_dir / "paper_process_executor_plan.json").is_file()
    assert (inputs.output_dir / "paper_process_executor_report.md").is_file()
    assert (inputs.output_dir / "process_executor_manifest.json").is_file()
    assert (inputs.output_dir / "operator_start_checklist.md").is_file()
    assert (inputs.output_dir / "start_command_review.txt").read_text(
        encoding="utf-8"
    ).startswith("freqtrade trade")


def test_paper_process_executor_plan_blocks_failed_execution_request(tmp_path):
    execution_request, execution_request_path = _write_ready_paper_execution_request(tmp_path)
    execution_request["status"] = "blocked"
    execution_request["blockers"] = [{"name": "execution_request_ready"}]
    execution_request["execution_request"]["eligible"] = False
    start_command = " ".join(execution_request["execution_request"]["command_preview"])
    inputs = PaperProcessExecutorPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_executor_plan",
        execution_request_path=execution_request_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed process executor planning only."],
        confirm_process_executor_plan=True,
        requested_start_command=start_command,
    )

    plan = build_paper_process_executor_plan(inputs, execution_request)
    write_paper_process_executor_plan_artifacts(inputs, plan)

    assert plan["status"] == "blocked"
    assert plan["executor_plan"]["eligible"] is False
    assert plan["executor_plan"]["command_preview"] == []
    assert {
        "execution_request_ready",
        "execution_request_has_no_blockers",
        "execution_request_eligible",
    } <= {check["name"] for check in plan["blockers"]}
    assert (inputs.output_dir / "start_command_review.txt").read_text(
        encoding="utf-8"
    ) == ""


def test_paper_process_executor_plan_blocks_missing_note_confirmation_and_unsafe_scope(tmp_path):
    execution_request, execution_request_path = _write_ready_paper_execution_request(tmp_path)
    execution_request["execution_request"]["process_control"] = True
    execution_request["execution_manifest"]["startup_executed"] = True
    execution_request["safety_scope"]["metadata_contains_secrets"] = True
    inputs = PaperProcessExecutorPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_executor_plan",
        execution_request_path=execution_request_path,
        output_root=tmp_path / "data" / "paper",
    )

    plan = build_paper_process_executor_plan(inputs, execution_request)

    assert plan["status"] == "blocked"
    assert {
        "execution_request_did_not_start_or_manage_process",
        "execution_manifest_no_startup_or_process_control",
        "execution_request_no_secrets_leverage_or_shorting_scope",
        "confirm_process_executor_plan_acknowledged",
        "requested_start_command_present",
        "reviewer_note_present",
    } <= {check["name"] for check in plan["blockers"]}


def test_paper_runtime_validation_accepts_local_artifacts_without_process_control(tmp_path):
    process_executor_plan, process_executor_plan_path = (
        _write_ready_paper_process_executor_plan(tmp_path)
    )
    artifacts = _write_paper_runtime_artifacts(tmp_path, process_executor_plan)
    inputs = PaperRuntimeValidationInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_runtime_validation",
        process_executor_plan_path=process_executor_plan_path,
        process_metadata_path=artifacts["process_metadata_path"],
        status_snapshot_path=artifacts["status_snapshot_path"],
        stdout_path=artifacts["stdout_path"],
        stderr_path=artifacts["stderr_path"],
        paper_metrics_path=artifacts["paper_metrics_path"],
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed runtime artifacts only; do not control the process."],
        command=["python", "scripts/bot_factory_validate_paper_runtime.py"],
    )

    validation = build_paper_runtime_validation(
        inputs,
        process_executor_plan,
        artifacts["process_metadata"],
        artifacts["status_snapshot"],
        artifacts["paper_metrics"],
    )
    write_paper_runtime_validation_artifacts(inputs, validation)

    assert validation["status"] == "pass"
    assert validation["runtime_validation"]["valid"] is True
    assert validation["runtime_validation"]["process_control"] is False
    assert validation["runtime_validation"]["status_polling_started"] is False
    assert validation["safety_scope"]["bot_startup_by_validator"] is False
    assert validation["safety_scope"]["process_control"] is False
    assert (inputs.output_dir / "paper_runtime_validation.json").is_file()
    assert (inputs.output_dir / "paper_runtime_validation_report.md").is_file()
    assert (inputs.output_dir / "runtime_artifacts_manifest.json").is_file()


def test_paper_runtime_validation_blocks_blocked_plan_and_missing_artifacts(tmp_path):
    process_executor_plan, process_executor_plan_path = (
        _write_ready_paper_process_executor_plan(tmp_path)
    )
    process_executor_plan["status"] = "blocked"
    process_executor_plan["blockers"] = [{"name": "process_executor_plan_ready"}]
    process_executor_plan["executor_plan"]["eligible"] = False
    inputs = PaperRuntimeValidationInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_runtime_validation",
        process_executor_plan_path=process_executor_plan_path,
        process_metadata_path=tmp_path / "missing_process_metadata.json",
        status_snapshot_path=tmp_path / "missing_status_snapshot.json",
        stdout_path=tmp_path / "missing_stdout.log",
        stderr_path=tmp_path / "missing_stderr.log",
        paper_metrics_path=tmp_path / "missing_paper_metrics.json",
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed runtime artifacts only."],
    )

    validation = build_paper_runtime_validation(
        inputs,
        process_executor_plan,
        {},
        {},
        {},
    )

    assert validation["status"] == "blocked"
    assert validation["runtime_validation"]["valid"] is False
    assert {
        "process_executor_plan_ready",
        "process_executor_plan_has_no_blockers",
        "process_executor_plan_eligible",
        "process_metadata_within_workspace_and_present",
        "status_snapshot_within_workspace_and_present",
        "stdout_log_within_workspace_and_present",
        "stderr_log_within_workspace_and_present",
        "paper_metrics_within_workspace_and_present",
    } <= {check["name"] for check in validation["blockers"]}


def test_paper_runtime_validation_blocks_secret_leverage_short_and_path_mismatch(tmp_path):
    process_executor_plan, process_executor_plan_path = (
        _write_ready_paper_process_executor_plan(tmp_path)
    )
    artifacts = _write_paper_runtime_artifacts(tmp_path, process_executor_plan)
    process_metadata = dict(artifacts["process_metadata"])
    status_snapshot = dict(artifacts["status_snapshot"])
    paper_metrics = dict(artifacts["paper_metrics"])
    paper_metrics["safety_scope"] = dict(paper_metrics["safety_scope"])
    paper_metrics["risk"] = dict(paper_metrics["risk"])

    process_metadata["api_key"] = "not-written-to-report"
    process_metadata["status_snapshot"] = "data/paper/other_status_snapshot.json"
    status_snapshot["exchange_order_placement"] = True
    paper_metrics["safety_scope"]["shorting"] = True
    paper_metrics["risk"]["max_leverage"] = 2.0
    inputs = PaperRuntimeValidationInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_runtime_validation",
        process_executor_plan_path=process_executor_plan_path,
        process_metadata_path=artifacts["process_metadata_path"],
        status_snapshot_path=artifacts["status_snapshot_path"],
        stdout_path=artifacts["stdout_path"],
        stderr_path=artifacts["stderr_path"],
        paper_metrics_path=artifacts["paper_metrics_path"],
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed runtime artifacts only."],
    )

    validation = build_paper_runtime_validation(
        inputs,
        process_executor_plan,
        process_metadata,
        status_snapshot,
        paper_metrics,
    )

    assert validation["status"] == "blocked"
    assert {
        "process_metadata_status_snapshot_path_matches_input",
        "runtime_no_live_or_exchange_order_scope",
        "runtime_metadata_no_credential_values",
        "runtime_no_leverage_above_one",
        "runtime_no_shorting",
    } <= {check["name"] for check in validation["blockers"]}


def test_paper_drift_report_accepts_local_artifacts_without_process_control(tmp_path):
    runtime_validation, artifacts = _write_passed_paper_runtime_validation(tmp_path)
    _add_paper_drift_metrics(artifacts["paper_metrics_path"], total_return_pct=1.1)
    historical_metrics_path = _write_historical_metrics(tmp_path)
    walk_forward_metrics_path = _write_walk_forward_metrics(tmp_path)
    training_manifest_path = _write_training_manifest(tmp_path)
    inputs = PaperDriftReportInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_drift_report",
        historical_metrics_path=historical_metrics_path,
        walk_forward_metrics_path=walk_forward_metrics_path,
        training_manifest_path=training_manifest_path,
        paper_runtime_validation_path=artifacts["runtime_validation_path"],
        paper_metrics_path=artifacts["paper_metrics_path"],
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed paper/backtest drift only."],
        command=["python", "scripts/bot_factory_report_paper_drift.py"],
    )

    report = build_paper_drift_report(
        inputs,
        json.loads(historical_metrics_path.read_text(encoding="utf-8")),
        json.loads(walk_forward_metrics_path.read_text(encoding="utf-8")),
        json.loads(training_manifest_path.read_text(encoding="utf-8")),
        runtime_validation,
        json.loads(artifacts["paper_metrics_path"].read_text(encoding="utf-8")),
    )
    write_paper_drift_report_artifacts(inputs, report)

    assert report["status"] == "pass"
    assert report["drift_report"]["valid"] is True
    assert report["drift_report"]["process_control"] is False
    assert report["drift_report"]["promotion_authorized_by_this_command"] is False
    assert report["safety_scope"]["bot_startup_by_reporter"] is False
    assert report["drift"]["return_vs_historical_pct_points"] == 0.10000000000000009
    assert (inputs.output_dir / "paper_drift_report.json").is_file()
    assert (inputs.output_dir / "paper_drift_report.md").is_file()
    assert (inputs.output_dir / "drift_metrics.json").is_file()


def test_paper_drift_report_blocks_blocked_runtime_and_missing_paper_metrics(tmp_path):
    runtime_validation, artifacts = _write_passed_paper_runtime_validation(tmp_path)
    runtime_validation["status"] = "blocked"
    runtime_validation["runtime_validation"]["valid"] = False
    historical_metrics_path = _write_historical_metrics(tmp_path)
    walk_forward_metrics_path = _write_walk_forward_metrics(tmp_path)
    training_manifest_path = _write_training_manifest(tmp_path)
    missing_paper_metrics_path = tmp_path / "missing_paper_metrics.json"
    inputs = PaperDriftReportInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_drift_report",
        historical_metrics_path=historical_metrics_path,
        walk_forward_metrics_path=walk_forward_metrics_path,
        training_manifest_path=training_manifest_path,
        paper_runtime_validation_path=artifacts["runtime_validation_path"],
        paper_metrics_path=missing_paper_metrics_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed paper/backtest drift only."],
    )

    report = build_paper_drift_report(
        inputs,
        json.loads(historical_metrics_path.read_text(encoding="utf-8")),
        json.loads(walk_forward_metrics_path.read_text(encoding="utf-8")),
        json.loads(training_manifest_path.read_text(encoding="utf-8")),
        runtime_validation,
        {},
    )

    assert report["status"] == "blocked"
    assert {
        "paper_metrics_within_workspace_and_present",
        "runtime_validation_passed",
        "paper_metrics_source_is_local",
        "paper_return_metric_present",
        "paper_drawdown_metric_present",
    } <= {check["name"] for check in report["blockers"]}


def test_paper_drift_report_blocks_metrics_path_not_runtime_validated(tmp_path):
    runtime_validation, artifacts = _write_passed_paper_runtime_validation(tmp_path)
    alternate_paper_metrics_path = (
        tmp_path
        / "data"
        / "paper"
        / "PaperStrategy"
        / "alternate_runtime"
        / "paper_metrics.json"
    )
    alternate_paper_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    alternate_paper_metrics_path.write_text(
        artifacts["paper_metrics_path"].read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    historical_metrics_path = _write_historical_metrics(tmp_path)
    walk_forward_metrics_path = _write_walk_forward_metrics(tmp_path)
    training_manifest_path = _write_training_manifest(tmp_path)
    inputs = PaperDriftReportInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_drift_report",
        historical_metrics_path=historical_metrics_path,
        walk_forward_metrics_path=walk_forward_metrics_path,
        training_manifest_path=training_manifest_path,
        paper_runtime_validation_path=artifacts["runtime_validation_path"],
        paper_metrics_path=alternate_paper_metrics_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed paper/backtest drift only."],
    )

    report = build_paper_drift_report(
        inputs,
        json.loads(historical_metrics_path.read_text(encoding="utf-8")),
        json.loads(walk_forward_metrics_path.read_text(encoding="utf-8")),
        json.loads(training_manifest_path.read_text(encoding="utf-8")),
        runtime_validation,
        json.loads(alternate_paper_metrics_path.read_text(encoding="utf-8")),
    )

    assert report["status"] == "blocked"
    assert "paper_metrics_path_matches_runtime_validation" in {
        check["name"] for check in report["blockers"]
    }


def test_paper_drift_report_blocks_reference_artifact_secret_metadata(tmp_path):
    runtime_validation, artifacts = _write_passed_paper_runtime_validation(tmp_path)
    historical_metrics_path = _write_historical_metrics(tmp_path)
    walk_forward_metrics_path = _write_walk_forward_metrics(tmp_path)
    training_manifest_path = _write_training_manifest(tmp_path)
    historical_metrics = json.loads(historical_metrics_path.read_text(encoding="utf-8"))
    walk_forward_metrics = json.loads(
        walk_forward_metrics_path.read_text(encoding="utf-8")
    )
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    historical_metrics["unsafe_metadata"] = {"api_key": "not-safe"}
    walk_forward_metrics["unsafe_env_reference"] = "${PAPER_SECRET}"
    inputs = PaperDriftReportInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_drift_report",
        historical_metrics_path=historical_metrics_path,
        walk_forward_metrics_path=walk_forward_metrics_path,
        training_manifest_path=training_manifest_path,
        paper_runtime_validation_path=artifacts["runtime_validation_path"],
        paper_metrics_path=artifacts["paper_metrics_path"],
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed paper/backtest drift only."],
    )

    report = build_paper_drift_report(
        inputs,
        historical_metrics,
        walk_forward_metrics,
        training_manifest,
        runtime_validation,
        json.loads(artifacts["paper_metrics_path"].read_text(encoding="utf-8")),
    )

    assert report["status"] == "blocked"
    blockers = {check["name"]: check for check in report["blockers"]}
    assert "drift_inputs_no_credential_values" in blockers
    assert "drift_inputs_no_private_env_references" in blockers
    assert blockers["drift_inputs_no_credential_values"]["details"] == {
        "credential_key_paths": ["historical_metrics.unsafe_metadata.api_key"]
    }
    assert blockers["drift_inputs_no_private_env_references"]["details"] == {
        "env_reference_paths": ["walk_forward_metrics.unsafe_env_reference"]
    }


def test_paper_drift_report_fails_prior_failed_gates_and_metric_drift(tmp_path):
    runtime_validation, artifacts = _write_passed_paper_runtime_validation(tmp_path)
    _add_paper_drift_metrics(
        artifacts["paper_metrics_path"], total_return_pct=-12.0, max_drawdown_pct=12.0
    )
    historical_metrics_path = _write_historical_metrics(tmp_path)
    walk_forward_metrics_path = _write_walk_forward_metrics(
        tmp_path, recommendation="fail"
    )
    training_manifest_path = _write_training_manifest(tmp_path, recommendation="fail")
    inputs = PaperDriftReportInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_drift_report",
        historical_metrics_path=historical_metrics_path,
        walk_forward_metrics_path=walk_forward_metrics_path,
        training_manifest_path=training_manifest_path,
        paper_runtime_validation_path=artifacts["runtime_validation_path"],
        paper_metrics_path=artifacts["paper_metrics_path"],
        output_root=tmp_path / "data" / "paper",
        max_return_drift_pct=5.0,
        max_drawdown_drift_pct=5.0,
        reviewer_notes=["Reviewed paper/backtest drift only."],
    )

    report = build_paper_drift_report(
        inputs,
        json.loads(historical_metrics_path.read_text(encoding="utf-8")),
        json.loads(walk_forward_metrics_path.read_text(encoding="utf-8")),
        json.loads(training_manifest_path.read_text(encoding="utf-8")),
        runtime_validation,
        json.loads(artifacts["paper_metrics_path"].read_text(encoding="utf-8")),
    )

    assert report["status"] == "fail"
    assert report["drift_report"]["paper_promotion_eligible"] is False
    assert {
        "walk_forward_recommendation_passed",
        "training_recommendation_passed",
        "paper_return_not_worse_than_historical_threshold",
        "paper_drawdown_not_worse_than_historical_threshold",
    } <= {check["name"] for check in report["failures"]}


def _strategy_proposal_inputs(tmp_path, **overrides) -> StrategyProposalInputs:
    include_edge_discovery = overrides.pop("include_edge_discovery", True)
    data = {
        "root_dir": tmp_path,
        "strategy_name": "LongOnlyRsiPullbackCandidate",
        "strategy_type": "mean_reversion",
        "target_exchange": "bybit",
        "target_symbols": ["BTC/USDT:USDT"],
        "timeframe": "5m",
        "spot_or_futures": "futures",
        "long_short": "long-only",
        "summary": "Long-only RSI pullback candidate for historical evaluation.",
        "hypothesis": (
            "After sharp short-term pullbacks in liquid BTC futures, mean "
            "reversion may occur when volume and volatility filters confirm liquidity."
        ),
        "market_condition": "Liquid BTC/USDT futures, historical OHLCV only.",
        "entry_logic": (
            "Enter long after RSI pullback and recovery confirmation using "
            "closed candles only."
        ),
        "exit_logic": (
            "Exit on mean-reversion target, momentum failure, or timeout using "
            "closed candles only."
        ),
        "risk_logic": "Use strategy stoploss, leverage 1.0, and no shorting.",
        "required_data": ["OHLCV closed candles only"],
        "parameters": ["RSI window, recovery threshold, stoploss, timeout candles"],
        "expected_failure_cases": ["Trend continuation after pullback"],
        "backtest_plan": (
            "Run static checks, OHLCV quality check, historical backtest, "
            "walk-forward validation, and training factory if FreqAI is added later."
        ),
        "rejection_conditions": [
            "Future data is required.",
            "Trade count is too low.",
            "Profit depends on one narrow period.",
        ],
        "reviewer_notes": [
            "Strategy proposal generation test only; do not generate code or start paper trading."
        ],
        "research_references": [
            StrategyProposalResearchReference(
                reference_id="paper:10.2139/ssrn.1968356",
                title="The Behavior of Individual Investors",
                source="SSRN",
                published_at="2011",
                relevance=(
                    "Documents behaviorally motivated trading frictions that "
                    "make pullback and overreaction hypotheses falsifiable."
                ),
            )
        ],
        "thesis_id": "TH-DEFAULT-PROPOSAL-001",
        "evidence_paths": [],
        "output_root": tmp_path / "registry" / "strategies" / "proposals",
        "created_by_agent": "codex-test",
        "created_at": "2026-05-04T00:00:00+00:00",
        "command": ["python", "scripts/bot_factory_generate_strategy_proposal.py"],
    }
    data.update(overrides)
    if include_edge_discovery:
        edge_path = _write_strategy_proposal_edge_discovery(
            tmp_path,
            thesis_id=str(data["thesis_id"]),
            mechanism_class=str(data.get("strategy_type") or "mean_reversion"),
        )
        data["evidence_paths"] = [
            *list(data.get("evidence_paths") or []),
            StrategyProposalEvidenceInput("edge_discovery", edge_path),
        ]
    return StrategyProposalInputs(**data)


def _write_strategy_proposal_edge_discovery(
    tmp_path,
    *,
    thesis_id: str,
    mechanism_class: str,
    status: str = "passed",
    net_edge_bps: float = 5.0,
    candidate_generation_allowed: bool | None = None,
) -> Path:
    candidate_allowed = (
        status == "passed"
        if candidate_generation_allowed is None
        else candidate_generation_allowed
    )
    edge_path = (
        tmp_path
        / "registry"
        / "strategies"
        / "research_decisions"
        / f"{thesis_id}_edge_discovery.json"
    )
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    edge_path.write_text(
        json.dumps(
            {
                "factory": "research_edge_discovery",
                "status": status,
                "edge_discovery_id": f"ED-{thesis_id}",
                "thesis_id": thesis_id,
                "mechanism_class": mechanism_class,
                "candidate_generation_allowed": candidate_allowed,
                "candidate_generation_result": (
                    "candidate generation allowed"
                    if candidate_allowed
                    else "no candidate generated"
                ),
                "proposal_generation_allowed": candidate_allowed,
                "strategy_codegen_allowed": False,
                "passing_horizon_count": 1 if status == "passed" else 0,
                "best_horizon_by_net_edge": {
                    "hold_candles": 3,
                    "status": status,
                    "net_edge_bps": net_edge_bps,
                    "sample_count": 64,
                },
                "horizon_results": [
                    {
                        "hold_candles": 3,
                        "status": status,
                        "net_edge_bps": net_edge_bps,
                        "sample_count": 64,
                    }
                ],
                "anti_parameter_search": {"valid": True},
                "research_gate": {
                    "passes_research_gate": candidate_allowed,
                    "rejection_reason": None
                    if candidate_allowed
                    else "research_gate_failed",
                    "blockers": []
                    if candidate_allowed
                    else [{"name": "research_gate_failed"}],
                },
                "promotion_gate": {
                    "proposal_generation_allowed": candidate_allowed,
                    "strategy_codegen_allowed": False,
                    "candidate_generation_allowed": candidate_allowed,
                },
                "blocked_next_actions": [
                    "strategy_codegen_directly_from_edge_discovery"
                ],
                "blockers": [],
                "safety_scope": {
                    "historical_only": True,
                    "local_artifacts_source_of_truth": True,
                    "future_data": False,
                    "backtest_started": False,
                    "strategy_code_generated": False,
                    "paper_trading_started": False,
                    "dry_run_trading_started": False,
                    "live_trading": False,
                    "exchange_order_placement": False,
                    "shorting": False,
                    "leverage": 1.0,
                    "process_control": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return edge_path


def _write_strategy_proposal_requiring_decision_synthesis(tmp_path) -> Path:
    synthesis_path = tmp_path / "registry" / "strategies" / "synthesis" / "s1.json"
    synthesis_path.parent.mkdir(parents=True, exist_ok=True)
    synthesis_path.write_text(
        json.dumps(
            {
                "factory": "candidate_failure_synthesis",
                "status": "completed",
                "synthesis_id": "synth-test",
                "ranking_path": "registry/strategies/candidates/rankings/ranking.json",
                "next_research_brief": {
                    "requires_new_thesis_id": True,
                    "requires_new_research_references": True,
                    "minimum_research_reference_count": 2,
                    "parameter_only_retry_allowed": False,
                    "prior_hypothesis_families_to_avoid_as_default": [
                        "trend_continuation"
                    ],
                    "failed_thesis_ids": ["TH-FAILED"],
                },
            }
        ),
        encoding="utf-8",
    )
    return synthesis_path


def _strategy_proposal_fractal_references(
    thesis_id: str = "TH-NEW",
) -> list[StrategyProposalResearchReference]:
    return [
        StrategyProposalResearchReference(
            reference_id="paper:fractal-long-memory-1",
            title="Long-Range Correlations in Cryptocurrency Markets",
            source="Local bibliography",
            published_at="2024",
            relevance="Motivates a distinct Hurst and long-memory thesis.",
            motivated_thesis_ids=[thesis_id],
        ),
        StrategyProposalResearchReference(
            reference_id="paper:fractal-long-memory-2",
            title="Hurst Exponents and Dynamic Bitcoin Market Efficiency",
            source="Local bibliography",
            published_at="2025",
            relevance="Supports falsifying long-memory regimes with OHLCV features.",
            motivated_thesis_ids=[thesis_id],
        ),
    ]


def _strategy_proposal_bipower_references(
    thesis_id: str,
) -> list[StrategyProposalResearchReference]:
    return [
        StrategyProposalResearchReference(
            reference_id="paper:bipower-variation",
            title="Power and Bipower Variation with Stochastic Volatility and Jumps",
            source="Journal of Financial Econometrics",
            published_at="2004",
            relevance=(
                "Motivates separating realized variance into jump and continuous "
                "variation components before proposing a post-jump strategy."
            ),
            motivated_thesis_ids=[thesis_id],
        ),
        StrategyProposalResearchReference(
            reference_id="paper:crypto-jump-dynamics",
            title="Understanding Temporal Dynamics of Jumps in Cryptocurrency Markets",
            source="Digital Finance",
            published_at="2024",
            relevance=(
                "Supports treating cryptocurrency jump behavior as a falsifiable "
                "mechanism rather than a threshold-retuning exercise."
            ),
            motivated_thesis_ids=[thesis_id],
        ),
    ]


def _strategy_proposal_directional_change_references(
    thesis_id: str,
) -> list[StrategyProposalResearchReference]:
    return [
        StrategyProposalResearchReference(
            reference_id="doi:10.1080/14697688.2010.481632",
            title="Patterns in High-Frequency FX Data: Discovery of 12 Empirical Scaling Laws",
            source="Quantitative Finance",
            published_at="2010",
            relevance=(
                "Motivates directional-change event-time states as a falsifiable "
                "market mechanism rather than a fixed-time threshold retry."
            ),
            motivated_thesis_ids=[thesis_id],
        ),
        StrategyProposalResearchReference(
            reference_id="doi:10.1007/s10462-022-10307-0",
            title="Algorithmic Trading with Directional Changes",
            source="Artificial Intelligence Review",
            published_at="2022",
            relevance=(
                "Supports evaluating directional-change trading methods with "
                "local out-of-sample evidence before code generation."
            ),
            motivated_thesis_ids=[thesis_id],
        ),
    ]


def _strategy_proposal_range_quarticity_references(
    thesis_id: str,
) -> list[StrategyProposalResearchReference]:
    return [
        StrategyProposalResearchReference(
            reference_id="doi:10.1093/jjfinec/nbu016",
            title="Quarticity Estimation on OHLC Data",
            source="Journal of Financial Econometrics",
            published_at="2014",
            relevance=(
                "Motivates OHLC range-based quarticity as a distinct local "
                "volatility-state measure before proposing a trading rule."
            ),
            motivated_thesis_ids=[thesis_id],
        ),
        StrategyProposalResearchReference(
            reference_id="doi:10.3386/w8160",
            title="Modeling and Forecasting Realized Volatility",
            source="NBER",
            published_at="2001",
            relevance=(
                "Supports treating realized volatility dynamics as a "
                "falsifiable state process rather than a threshold retry."
            ),
            motivated_thesis_ids=[thesis_id],
        ),
        StrategyProposalResearchReference(
            reference_id="doi:10.3390/stats5030050",
            title="Modeling Realized Variance with Realized Quarticity",
            source="Stats",
            published_at="2022",
            relevance=(
                "Supports testing whether quarticity improves realized "
                "variance state modeling on local historical data."
            ),
            motivated_thesis_ids=[thesis_id],
        ),
    ]


def _research_selection_inputs(tmp_path, **overrides) -> ResearchSelectionInputs:
    thesis_id = overrides.get("thesis_id", "TH-ORDERBOOK-RESILIENCE-001")
    data = {
        "root_dir": tmp_path,
        "failure_synthesis_path": _write_research_selection_synthesis(tmp_path),
        "thesis_id": thesis_id,
        "thesis_family": "closed_candle_liquidity_resilience",
        "mechanism_class": "closed_candle_resilience_reclaim",
        "thesis_statement": (
            "Liquid BTC futures may show resilience after local liquidity stress "
            "when closed-candle recovery and participation improve together."
        ),
        "mechanism_summary": (
            "Measure local stress, recovery, and participation from historical "
            "closed-candle OHLCV before allowing a proposal."
        ),
        "novelty_rationale": (
            "This mechanism is outside the failed trend, spread, funding, "
            "calendar, entropy, cross-asset, and mark-price families."
        ),
        "required_data": ["Local BTC/USDT futures 5m closed-candle OHLCV"],
        "edge_rationale": (
            "Expected edge comes from resilience after local liquidity stress, "
            "not from threshold-only retuning of failed families."
        ),
        "transaction_cost_exposure": (
            "Frequent entries and maker/taker fee drag are explicit risks that "
            "must be rejected if walk-forward windows remain negative."
        ),
        "falsification_plan": (
            "Use local historical closed-candle OHLCV with walk-forward splits; "
            "reject the thesis if costs dominate or profitable windows are absent."
        ),
        "stop_conditions": [
            "Block proposal generation if the family repeats a failed synthesis family.",
            "Block proposal generation if structured references do not motivate this thesis.",
            "Defer proposal generation if local historical data artifacts are missing.",
        ],
        "research_references": [
            _research_selection_reference(thesis_id, "resilience-a"),
            _research_selection_reference(thesis_id, "resilience-b"),
        ],
        "research_question_responses": [
            (
                "1=Select a closed-candle resilience mechanism that is outside "
                "the failed families and reject it if local historical evidence fails."
            ),
            (
                "2=Use local BTC futures OHLCV artifacts and walk-forward splits "
                "to falsify the mechanism before proposal generation."
            ),
            (
                "3=Reject the thesis when fees, spread, slippage, or turnover "
                "costs dominate the expected edge after entries."
            ),
        ],
        "local_data_paths": [],
        "output_root": tmp_path / "registry" / "strategies" / "research_decisions",
        "decision_id": "research-selection-test",
        "reviewer_notes": ["research selection unit test only"],
        "created_by_agent": "codex-test",
        "created_at": "2026-05-07T00:00:00+00:00",
        "command": ["python", "scripts/bot_factory_select_research_thesis.py"],
    }
    data.update(overrides)
    return ResearchSelectionInputs(**data)


def _research_selection_reference(
    thesis_id: str, suffix: str
) -> StrategyProposalResearchReference:
    return StrategyProposalResearchReference(
        reference_id=f"paper:{suffix}",
        title=f"Research reference {suffix}",
        source="Local bibliography",
        published_at="2024",
        relevance=(
            "Supports a falsifiable closed-candle market-resilience mechanism "
            "and explains why local historical evidence can reject it."
        ),
        motivated_thesis_ids=[thesis_id],
    )


def _write_structural_capability_report(
    tmp_path,
    *,
    local_research_usable: list[str] | None = None,
    blocked_without_new_data: list[str] | None = None,
    must_not_codegen: list[str] | None = None,
) -> Path:
    path = tmp_path / "registry" / "strategies" / "checks" / "structural_capability.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "factory": "structural_data_capability_report",
                "proposal_guidance": {
                    "local_research_usable": local_research_usable or [],
                    "blocked_without_new_data": blocked_without_new_data or [],
                    "must_not_codegen": must_not_codegen or [],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_strategy_proposal_research_decision(
    tmp_path,
    *,
    thesis_id: str,
    thesis_family: str = "fractal_long_memory",
    mechanism_class: str = "fractal_long_memory_regime",
    status: str = "approved_for_proposal_generation",
    proposal_generation_allowed: bool = True,
    include_causal_failure_map: bool = True,
    causal_failure_map: dict[str, Any] | None = None,
    research_selection_score: dict[str, Any] | None = None,
    novelty_assessment: dict[str, Any] | None = None,
    local_falsification_evidence: dict[str, Any] | None = None,
    local_data_quality_report_paths: list[Path] | None = None,
    structural_data_capability_report_paths: list[Path] | None = None,
    research_handoff_summaries: list[dict[str, Any]] | None = None,
    blocked_next_actions: list[str] | None = None,
) -> Path:
    decision_path = tmp_path / "registry" / "strategies" / "research_decisions" / "rd.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    quality_report_paths = [
        str(path.resolve().relative_to(tmp_path.resolve()))
        for path in local_data_quality_report_paths or []
    ]
    capability_report_paths = [
        str(path.resolve().relative_to(tmp_path.resolve()))
        for path in structural_data_capability_report_paths or []
    ]
    payload = {
        "factory": "research_selection_gate",
        "status": status,
        "decision": status,
        "decision_id": "research-decision-test",
        "proposal_generation_allowed": proposal_generation_allowed,
        "code_generation_allowed": False,
        "thesis": {
            "thesis_id": thesis_id,
            "thesis_family": thesis_family,
            "mechanism_class": mechanism_class,
            "local_data_quality_report_paths": quality_report_paths,
            "structural_data_capability_report_paths": capability_report_paths,
            "local_falsification_paths": [
                str(artifact.get("path"))
                for artifact in (
                    local_falsification_evidence.get("artifacts", [])
                    if isinstance(local_falsification_evidence, dict)
                    else []
                )
                if isinstance(artifact, dict) and str(artifact.get("path") or "").strip()
            ],
        },
        "novelty_assessment": novelty_assessment
        or {
            "repeated_failed_family_matches": [],
            "failed_thesis_id_match": False,
        },
        "research_references": [
            {
                "reference_id": "paper:research-decision-reference",
                "title": "Research decision reference",
                "source": "Local bibliography",
                "published_at": "2025",
                "relevance": "Motivates this thesis before proposal generation.",
                "motivated_thesis_ids": [thesis_id],
            }
        ],
        "blockers": []
        if proposal_generation_allowed
        else [{"name": "blocked_for_test"}],
        "checks": [
            {
                "name": "local_data_quality_reports_valid",
                "status": "pass" if quality_report_paths else "blocked",
                "severity": "blocker",
                "message": "Local data quality reports must be valid.",
                "details": {"reports": quality_report_paths},
            },
            {
                "name": "structural_data_quality_report_present",
                "status": "pass" if quality_report_paths else "blocked",
                "severity": "blocker",
                "message": "Structural data requires quality reports.",
                "details": {"reports": quality_report_paths},
            },
            {
                "name": "structural_data_capability_reports_valid",
                "status": "pass" if capability_report_paths else "blocked",
                "severity": "blocker",
                "message": "Structural data capability reports must be valid.",
                "details": {"reports": capability_report_paths},
            },
            {
                "name": "structural_data_capability_report_present",
                "status": "pass" if capability_report_paths else "blocked",
                "severity": "blocker",
                "message": "Structural data requires capability reports.",
                "details": {"reports": capability_report_paths},
            },
            {
                "name": "structural_data_capability_supports_required_classes",
                "status": "pass" if capability_report_paths else "blocked",
                "severity": "blocker",
                "message": "Structural data required classes must be supported.",
                "details": {"reports": capability_report_paths},
            },
        ],
        "safety_scope": {
            "historical_only": True,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "process_control": False,
        },
        "research_selection_score": research_selection_score
        or {
            "version": "research_selection_score_v1",
            "score": 100.0,
            "maximum_score": 100.0,
            "minimum_score_required": 80.0,
            "passes_minimum": True,
            "failed_components": [],
        },
    }
    if local_falsification_evidence is not None:
        payload["local_falsification_evidence"] = local_falsification_evidence
    if include_causal_failure_map:
        payload["causal_failure_map"] = causal_failure_map or {
            "used": True,
            "available": True,
            "map_id": "map-test",
            "status": "completed",
            "source_synthesis_id": "synth-test",
            "candidate_count": 22,
            "category_count": 4,
            "requires_research_decision_before_proposal": True,
            "requires_research_question_responses": True,
            "material_category_min_share": 0.70,
            "dominant_failure_categories": [
                {"category": "regime_fragile_mechanism", "candidate_count": 22},
                {"category": "walk_forward_fragility", "candidate_count": 22},
                {"category": "cost_sensitive_mechanism", "candidate_count": 21},
                {"category": "entry_exists_negative_edge", "candidate_count": 15},
            ],
            "required_categories_to_address": [
                "regime_fragile_mechanism",
                "walk_forward_fragility",
                "cost_sensitive_mechanism",
            ],
            "response_categories": [
                "cost_sensitive_mechanism",
                "regime_fragile_mechanism",
                "walk_forward_fragility",
            ],
            "missing_response_categories": [],
            "weak_response_categories": [],
            "category_evidence_gaps": [],
            "parameter_only_response_categories": [],
            "required_research_questions": [
                "What mechanism survives after failed families are excluded?",
                "Why should expected edge exceed fee and turnover costs?",
                "Which walk-forward regimes should pass or fail?",
            ],
            "research_question_response_indexes": [1, 2, 3],
            "missing_research_question_response_indexes": [],
            "weak_research_question_response_indexes": [],
            "research_handoff_summaries": research_handoff_summaries or [],
            "blocked_next_actions": blocked_next_actions or [],
        }
    decision_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return decision_path


def _strategy_proposal_high_risk_cost_causal_map() -> dict[str, Any]:
    return {
        "used": True,
        "available": True,
        "map_id": "map-test",
        "status": "completed",
        "source_synthesis_id": "synth-test",
        "candidate_count": 30,
        "category_count": 4,
        "requires_research_decision_before_proposal": True,
        "requires_research_question_responses": True,
        "material_category_min_share": 0.70,
        "minimum_research_selection_score": 80.0,
        "dominant_failure_categories": [
            {"category": "cost_sensitive_mechanism", "candidate_count": 29},
            {"category": "regime_fragile_mechanism", "candidate_count": 30},
            {"category": "walk_forward_fragility", "candidate_count": 30},
        ],
        "causal_risk_weights": [
            {"category": "cost_sensitive_mechanism", "risk_score": 100.0},
            {"category": "regime_fragile_mechanism", "risk_score": 75.0},
            {"category": "walk_forward_fragility", "risk_score": 75.0},
        ],
        "required_categories_to_address": [
            "cost_sensitive_mechanism",
            "regime_fragile_mechanism",
            "walk_forward_fragility",
        ],
        "response_categories": [
            "cost_sensitive_mechanism",
            "regime_fragile_mechanism",
            "walk_forward_fragility",
        ],
        "missing_response_categories": [],
        "weak_response_categories": [],
        "category_evidence_gaps": [],
        "parameter_only_response_categories": [],
        "required_research_questions": [
            "What mechanism survives after failed families are excluded?",
        ],
        "research_question_response_indexes": [1],
        "missing_research_question_response_indexes": [],
        "weak_research_question_response_indexes": [],
    }


def _strategy_proposal_weighted_research_selection_score() -> dict[str, Any]:
    return {
        "version": "research_selection_score_v2",
        "score": 100.0,
        "maximum_score": 100.0,
        "minimum_score_required": 80.0,
        "passes_minimum": True,
        "failed_components": [],
        "components": [
            {
                "name": "causal_failure_response_quality",
                "score": 100.0,
                "passed": True,
                "details": {
                    "weighted_response_score": 100.0,
                    "unanswered_required_risk_weight": 0.0,
                    "category_scores": [
                        {
                            "category": "cost_sensitive_mechanism",
                            "risk_score": 100.0,
                            "quality_ratio": 1.0,
                        }
                    ],
                },
            }
        ],
    }


def _strategy_proposal_passing_local_falsification_evidence(
    thesis_id: str,
) -> dict[str, Any]:
    return {
        "high_risk_cost_evidence_required": True,
        "minimum_sample_count": 20,
        "minimum_data_span_days": 180.0,
        "artifact_count": 1,
        "parseable_artifact_count": 1,
        "matching_thesis_artifact_count": 1,
        "passing_cost_edge_artifact_count": 1,
        "failures": [],
        "artifacts": [
            {
                "path": (
                    "registry\\strategies\\research_decisions\\"
                    "cost_edge_falsification.json"
                ),
                "exists": True,
                "within_workspace": True,
                "parseable": True,
                "factory": "research_local_falsification",
                "factory_valid": True,
                "safety_scope_valid": True,
                "event_source_valid": True,
                "event_source_context_alignment_valid": True,
                "event_source_failure_synthesis_guard_valid": True,
                "thesis_id": thesis_id,
                "thesis_matches": True,
                "status": "completed",
                "expected_edge_bps": 18.0,
                "all_in_cost_bps": 12.0,
                "net_edge_bps": 6.0,
                "sample_count": 64,
                "sample_sufficient": True,
                "data_span_days": 365.0,
                "data_span_sufficient": True,
                "cost_edge_passes": True,
                "failure_reasons": [],
            }
        ],
    }


def _write_research_selection_synthesis(tmp_path) -> Path:
    synthesis_path = tmp_path / "registry" / "strategies" / "synthesis" / "synth.json"
    synthesis_path.parent.mkdir(parents=True, exist_ok=True)
    synthesis_path.write_text(
        json.dumps(
            {
                "factory": "candidate_failure_synthesis",
                "status": "completed",
                "synthesis_id": "synth-test",
                "candidate_count": 22,
                "aggregate_failure_summary": {
                    "paper_ready_candidate_ids": [],
                    "paper_ready_count": 0,
                    "all_candidates_failed_gates": True,
                    "hypothesis_families_tried": [
                        "trend_continuation",
                        "microstructure_spread_reversion",
                    ],
                    "thesis_ids_tried": ["TH-FAILED"],
                    "negative_return_candidate_ids": ["cand-a"],
                    "walk_forward_failed_candidate_ids": ["cand-a", "cand-b"],
                },
                "next_research_brief": {
                    "requires_new_thesis_id": True,
                    "requires_new_research_references": True,
                    "minimum_research_reference_count": 2,
                    "parameter_only_retry_allowed": False,
                    "paper_or_live_promotion_allowed": False,
                    "prior_hypothesis_families_to_avoid_as_default": [
                        "trend_continuation",
                        "microstructure_spread_reversion",
                    ],
                    "failed_thesis_ids": ["TH-FAILED"],
                },
            }
        ),
        encoding="utf-8",
    )
    return synthesis_path


def _write_research_selection_synthesis_with_local_rejection(
    tmp_path,
    *,
    thesis_id: str,
    mechanism_class: str,
    valid: bool,
) -> tuple[Path, dict[str, Any]]:
    from freqtrade_ext.bot_factory.candidate_failure_synthesis import (
        CandidateFailureSynthesisInputs,
        synthesize_candidate_failures,
    )

    ranking_path = tmp_path / "local_rejection_ranking.json"
    ranking_path.write_text(
        json.dumps(
            {
                "ranking_id": "rank-local-rejection-test",
                "paper_ready_candidate_ids": [],
                "ranked_candidates": [
                    {
                        "candidate_id": "cand-control",
                        "strategy_name": "ControlStrategy",
                        "recommendation": "retry",
                        "paper_ready_eligible": False,
                        "paper_ready_blockers": ["historical_backtest"],
                        "failure_taxonomy_codes": ["FAIL_COST_SENSITIVE"],
                        "hypothesis_family": "control_failed_family",
                        "thesis": {"thesis_id": "TH-CONTROL"},
                        "metrics": {
                            "historical_trade_count": 5,
                            "historical_total_return_pct": -0.8,
                            "walk_forward_pass_rate": 0.0,
                        },
                        "rank": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    local_falsification_payload: dict[str, Any] = {
        "factory": "research_local_falsification",
        "status": "failed",
        "thesis_id": thesis_id,
        "mechanism_class": mechanism_class,
        "expected_edge_bps": -3.0,
        "all_in_cost_bps": 12.0,
        "net_edge_bps": -15.0,
        "sample_count": 64,
        "data_span_days": 365.0,
        "profitable_windows_ratio": 0.0,
        "blockers": [{"name": "expected_edge_exceeds_all_in_cost"}],
    }
    if valid:
        local_falsification_payload.update(
            {
                "event_source": {
                    "used": True,
                    "factory": "research_local_event_builder",
                    "factory_valid": True,
                    "status": "completed",
                    "status_completed": True,
                    "thesis_id": thesis_id,
                    "thesis_matches": True,
                    "events_csv_path": (
                        "registry/strategies/research_decisions/events.csv"
                    ),
                    "event_path_matches": True,
                    "source_ohlcv_path": (
                        "user_data/data/bybit/futures/BTC_USDT-5m.parquet"
                    ),
                    "ohlcv_path_matches": True,
                    "event_count": 64,
                    "safety_scope_valid": True,
                    "context_features_used": False,
                    "required_contexts": [],
                    "failure_synthesis_used": True,
                    "failure_synthesis_parseable": True,
                    "failure_synthesis_path": (
                        "registry/strategies/synthesis/candidate_failure_synthesis.json"
                    ),
                    "failure_synthesis_thesis_repeats": False,
                    "failure_synthesis_mechanism_repeats": False,
                    "failure_synthesis_allow_failed_thesis_or_family": False,
                    "failure_synthesis_guard_valid": True,
                },
                "safety_scope": {
                    "historical_only": True,
                    "backtest_started": False,
                    "strategy_code_generated": False,
                    "paper_trading_started": False,
                    "dry_run_trading_started": False,
                    "live_trading": False,
                    "exchange_order_placement": False,
                    "shorting": False,
                    "leverage": 1.0,
                    "process_control": False,
                },
            }
        )
    local_falsification_path = tmp_path / "local_rejection_falsification.json"
    local_falsification_path.write_text(
        json.dumps(local_falsification_payload),
        encoding="utf-8",
    )

    synthesis = synthesize_candidate_failures(
        CandidateFailureSynthesisInputs(
            root_dir=tmp_path,
            ranking_path=ranking_path,
            local_falsification_paths=[local_falsification_path],
            synthesis_id="synth-local-rejection-test",
        )
    )
    synthesis_path = (
        tmp_path
        / "registry"
        / "strategies"
        / "synthesis"
        / "local_rejection_synthesis.json"
    )
    synthesis_path.parent.mkdir(parents=True, exist_ok=True)
    synthesis_path.write_text(json.dumps(synthesis), encoding="utf-8")
    return synthesis_path, synthesis


def _write_research_selection_synthesis_with_edge_rejection(
    tmp_path,
    *,
    thesis_id: str,
    mechanism_class: str,
    valid: bool,
) -> tuple[Path, dict[str, Any]]:
    from freqtrade_ext.bot_factory.candidate_failure_synthesis import (
        CandidateFailureSynthesisInputs,
        synthesize_candidate_failures,
    )

    ranking_path = tmp_path / "edge_rejection_ranking.json"
    ranking_path.write_text(
        json.dumps(
            {
                "ranking_id": "rank-edge-rejection-test",
                "paper_ready_candidate_ids": [],
                "ranked_candidates": [
                    {
                        "candidate_id": "cand-control",
                        "strategy_name": "ControlStrategy",
                        "recommendation": "retry",
                        "paper_ready_eligible": False,
                        "paper_ready_blockers": ["historical_backtest"],
                        "failure_taxonomy_codes": ["FAIL_COST_SENSITIVE"],
                        "hypothesis_family": "control_failed_family",
                        "thesis": {"thesis_id": "TH-CONTROL"},
                        "metrics": {
                            "historical_trade_count": 5,
                            "historical_total_return_pct": -0.8,
                            "walk_forward_pass_rate": 0.0,
                        },
                        "rank": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    edge_payload: dict[str, Any] = {
        "factory": "research_edge_discovery",
        "edge_discovery_id": "edge-rejection-test",
        "status": "failed",
        "thesis_id": thesis_id,
        "mechanism_class": mechanism_class,
        "source_ohlcv_path": "user_data/data/bybit/futures/BTC_USDT-5m.parquet",
        "edge_spec_path": "registry/strategies/research_decisions/edge_spec.json",
        "anti_parameter_search": {"valid": valid},
        "event_count": 64,
        "data_span_days": 365.0,
        "all_in_cost_bps": 12.0,
        "passing_horizon_count": 0,
        "horizon_results": [
            {
                "hold_candles": 3,
                "status": "failed",
                "sample_count": 64,
                "expected_edge_bps": -2.0,
                "all_in_cost_bps": 12.0,
                "net_edge_bps": -14.0,
                "profitable_windows_ratio": 0.0,
                "calendar_window_frequency": "quarter",
                "calendar_window_count": 4,
                "profitable_calendar_windows_ratio": 0.0,
            }
        ],
        "best_horizon_by_net_edge": {
            "hold_candles": 3,
            "status": "failed",
            "sample_count": 64,
            "expected_edge_bps": -2.0,
            "net_edge_bps": -14.0,
            "profitable_windows_ratio": 0.0,
            "calendar_window_frequency": "quarter",
            "calendar_window_count": 4,
            "profitable_calendar_windows_ratio": 0.0,
        },
        "blockers": [
            {"name": "passing_horizon_count_sufficient"},
            {"name": "horizon_3_edge_evidence_passed"},
        ],
        "safety_scope": {
            "historical_only": True,
            "backtest_started": False,
            "strategy_code_generated": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "shorting": False,
            "leverage": 1.0,
            "process_control": False,
        },
    }
    edge_path = tmp_path / "edge_discovery_failed.json"
    edge_path.write_text(json.dumps(edge_payload), encoding="utf-8")

    synthesis = synthesize_candidate_failures(
        CandidateFailureSynthesisInputs(
            root_dir=tmp_path,
            ranking_path=ranking_path,
            edge_discovery_paths=[edge_path],
            synthesis_id="synth-edge-rejection-test",
        )
    )
    synthesis_path = (
        tmp_path
        / "registry"
        / "strategies"
        / "synthesis"
        / "edge_rejection_synthesis.json"
    )
    synthesis_path.parent.mkdir(parents=True, exist_ok=True)
    synthesis_path.write_text(json.dumps(synthesis), encoding="utf-8")
    return synthesis_path, synthesis


def _write_research_selection_causal_failure_map(
    tmp_path,
    *,
    source_synthesis_id: str = "synth-test",
    candidate_count: int = 22,
    dominant_categories: list[dict[str, Any]] | None = None,
    causal_risk_weights: list[dict[str, Any]] | None = None,
    required_research_questions: list[str] | None = None,
    validated_local_falsification_rejections: list[dict[str, Any]] | None = None,
    research_handoff_summaries: list[dict[str, Any]] | None = None,
    blocked_next_actions: list[str] | None = None,
    minimum_research_selection_score: float | None = None,
) -> Path:
    map_path = (
        tmp_path
        / "registry"
        / "strategies"
        / "failure_maps"
        / "map-test"
        / "causal_failure_map.json"
    )
    map_path.parent.mkdir(parents=True, exist_ok=True)
    dominant_categories = dominant_categories or [
        {"category": "regime_fragile_mechanism", "candidate_count": 22},
        {"category": "walk_forward_fragility", "candidate_count": 22},
        {"category": "cost_sensitive_mechanism", "candidate_count": 21},
        {"category": "entry_exists_negative_edge", "candidate_count": 15},
    ]
    map_path.write_text(
        json.dumps(
            {
                "factory": "candidate_failure_map",
                "map_id": "map-test",
                "status": "completed",
                "source_synthesis_id": source_synthesis_id,
                "candidate_count": candidate_count,
                "causal_failure_categories": {
                    "category_count": 4,
                    "categories": {
                        item["category"]: {
                            "candidate_count": item["candidate_count"],
                            "candidate_ids": ["cand-a"],
                        }
                        for item in dominant_categories
                    },
                },
                "research_selection_guidance": {
                    "requires_research_decision_before_proposal": True,
                    "requires_research_question_responses": True,
                    "minimum_research_selection_score": (
                        80
                        if minimum_research_selection_score is None
                        else minimum_research_selection_score
                    ),
                    "research_selection_rubric": [
                        {
                            "component": "novelty_against_failure_set",
                            "max_points": 20,
                            "requirement": "Must not repeat failed families.",
                        }
                    ],
                    "dominant_failure_categories": dominant_categories,
                    "causal_risk_weights": causal_risk_weights or [],
                    "required_research_questions": required_research_questions or [
                        "What mechanism survives after failed families are excluded?",
                        "Why should expected edge exceed fee and turnover costs?",
                        "Which walk-forward regimes should pass or fail?",
                    ],
                    "validated_local_falsification_rejections": (
                        validated_local_falsification_rejections or []
                    ),
                    "research_handoff_summaries": research_handoff_summaries or [],
                    "blocked_next_actions": blocked_next_actions or [],
                },
            }
        ),
        encoding="utf-8",
    )
    return map_path




def _apply_hypothesis_metadata(metadata: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    defaults = {
        "thesis_id": "THESIS-MR-001",
        "thesis_type": "mean_reversion",
        "thesis_statement": "Pullback recoveries in liquid BTC futures revert on closed-candle confirmation.",
        "falsification_criteria": "Reject if walk-forward return remains negative with acceptable trade count.",
        "novelty_vs_previous": "Adds explicit liquidity filter and timeout risk guard versus prior baseline.",
        "evidence_refs": ["paper:10.2139/ssrn.1968356"],
        "research_references": [
            {
                "reference_id": "paper:10.2139/ssrn.1968356",
                "title": "The Behavior of Individual Investors",
                "source": "SSRN",
                "published_at": "2011",
                "relevance": (
                    "Motivates a falsifiable mean-reversion thesis through "
                    "behaviorally driven overreaction evidence."
                ),
                "motivated_thesis_ids": ["THESIS-MR-001"],
            }
        ],
        "retry_budget_per_thesis": 3,
        "thesis_retry_count": 1,
        "parameter_only_retry_limit": 1,
        "parameter_only_retry_count": 0,
        "force_distinct_hypothesis_family": False,
        "failure_taxonomy_codes": ["FAIL_OVERFIT_WF_GAP"],
    }
    defaults.update(overrides)
    metadata.update(defaults)
    return metadata


def _write_hypothesis_metadata(metadata_path: Path, **overrides: Any) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = _apply_hypothesis_metadata(metadata, **overrides)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def test_strategy_code_generator_blocks_invalid_failure_taxonomy_code(tmp_path):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    _write_hypothesis_metadata(
        proposal_artifacts.metadata_path,
        failure_taxonomy_codes=["FAIL_UNKNOWN"],
    )

    artifacts = build_strategy_code(_strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path))
    write_strategy_code_artifacts(artifacts)

    assert artifacts.metadata["status"] == "blocked"
    assert "failure_taxonomy_codes_normalized" in {
        check["name"] for check in artifacts.metadata["blockers"]
    }


def test_strategy_code_generator_writes_research_brief_artifact(tmp_path):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    _write_hypothesis_metadata(proposal_artifacts.metadata_path)

    artifacts = build_strategy_code(_strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path))
    write_strategy_code_artifacts(artifacts)

    assert artifacts.research_brief_path.is_file()
    brief = json.loads(artifacts.research_brief_path.read_text(encoding="utf-8"))
    assert brief["thesis_id"] == "THESIS-MR-001"
    assert brief["failure_taxonomy_codes"] == ["FAIL_OVERFIT_WF_GAP"]
    assert brief["research_references"][0]["relevance"]
    assert brief["research_references"][0]["motivated_thesis_ids"] == ["THESIS-MR-001"]


def test_strategy_code_generator_blocks_research_reference_without_thesis_mapping(tmp_path):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    _write_hypothesis_metadata(
        proposal_artifacts.metadata_path,
        research_references=[
            {
                "reference_id": "paper:unmapped",
                "title": "Unmapped Reference",
                "source": "Local bibliography",
                "relevance": "Relevant to a different hypothesis only.",
                "motivated_thesis_ids": ["OTHER-THESIS"],
            }
        ],
    )

    artifacts = build_strategy_code(_strategy_code_inputs(tmp_path, proposal_artifacts.metadata_path))
    write_strategy_code_artifacts(artifacts)

    assert artifacts.metadata["status"] == "blocked"
    assert "research_references_motivate_current_thesis" in {
        check["name"] for check in artifacts.metadata["blockers"]
    }


def _strategy_code_inputs(tmp_path, proposal_metadata_path, **overrides) -> StrategyCodeInputs:
    data = {
        "root_dir": tmp_path,
        "proposal_metadata_path": proposal_metadata_path,
        "candidate_id": "candidate_a",
        "output_root": tmp_path / "registry" / "strategies" / "generated",
        "created_by_agent": "codex-test",
        "created_at": "2026-05-04T01:00:00+00:00",
        "command": [
            "python",
            "scripts/bot_factory_generate_strategy_code.py",
            "--proposal-metadata-json",
            str(proposal_metadata_path),
        ],
    }
    data.update(overrides)
    return StrategyCodeInputs(**data)


def _paper_config(strategy: str) -> dict:
    return {
        "dry_run": True,
        "dry_run_wallet": 1000,
        "max_open_trades": 1,
        "cancel_open_orders_on_exit": False,
        "initial_state": "stopped",
        "force_entry_enable": False,
        "stake_currency": "USDT",
        "stake_amount": 100,
        "strategy": strategy,
        "timeframe": "5m",
        "exchange": {
            "name": "bybit",
            "key": "",
            "secret": "",
            "pair_whitelist": ["BTC/USDT:USDT"],
        },
    }


def _write_paper_strategy(tmp_path, strategy: str) -> Path:
    strategy_path = tmp_path / f"{strategy}.py"
    strategy_path.write_text(
        f"class {strategy}:\n"
        "    can_short = False\n"
        "    def populate_entry_trend(self, dataframe, metadata):\n"
        "        dataframe.loc[:, 'enter_long'] = 1\n"
        "        return dataframe\n",
        encoding="utf-8",
    )
    return strategy_path


def _paper_readiness_payload(
    *,
    strategy: str,
    status: str,
    config_path: Path,
    blockers: list[dict] | None = None,
    failures: list[dict] | None = None,
) -> dict:
    return {
        "generated_at": "2026-05-03T00:00:00+00:00",
        "phase": "3",
        "factory": "paper_readiness",
        "strategy": strategy,
        "run_id": "readiness_run",
        "status": status,
        "readiness": status,
        "config_path": str(config_path),
        "blockers": blockers or [],
        "failures": failures or [],
        "reviewer_notes": ["Reviewed for no-startup paper readiness."],
        "safety_scope": {
            "command": "paper readiness preflight only",
            "bot_startup": False,
            "freqtrade_trade": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "uses_api_keys_or_secrets": False,
            "leverage_above_one": False,
            "shorting": False,
            "metadata_contains_secrets": False,
            "local_artifacts_source_of_truth": True,
        },
    }


def _write_ready_paper_run_plan(tmp_path) -> tuple[dict, Path]:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_paper_config("PaperStrategy")), encoding="utf-8")
    (tmp_path / "strategies").mkdir()
    readiness_path = tmp_path / "paper_readiness.json"
    readiness = _paper_readiness_payload(
        strategy="PaperStrategy",
        status="pass",
        config_path=config_path,
    )
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    inputs = PaperRunPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_plan",
        readiness_path=readiness_path,
        config_path=config_path,
        strategy_path=tmp_path / "strategies",
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed for future paper planning only."],
        confirm_paper=True,
    )
    plan = build_paper_run_plan(inputs, readiness)
    write_paper_run_plan_artifacts(inputs, plan)
    return plan, inputs.output_dir / "paper_run_plan.json"


def _write_ready_paper_startup_preflight(tmp_path) -> tuple[dict, Path]:
    plan, plan_path = _write_ready_paper_run_plan(tmp_path)
    start_command = " ".join(plan["future_startup"]["command_preview"])
    inputs = PaperStartupPreflightInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_start_preflight",
        plan_path=plan_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed startup preflight only; do not start paper trading."],
        confirm_paper_start=True,
        requested_start_command=start_command,
        command=["python", "scripts/bot_factory_prepare_paper_start.py"],
    )
    preflight = build_paper_startup_preflight(inputs, plan)
    write_paper_startup_preflight_artifacts(inputs, preflight)
    return preflight, inputs.output_dir / "paper_startup_preflight.json"


def _write_ready_paper_monitoring_plan(tmp_path) -> tuple[dict, Path]:
    preflight, preflight_path = _write_ready_paper_startup_preflight(tmp_path)
    inputs = PaperMonitoringPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_monitoring",
        startup_preflight_path=preflight_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed monitoring schema planning only."],
        command=["python", "scripts/bot_factory_plan_paper_monitoring.py"],
    )
    plan = build_paper_monitoring_plan(inputs, preflight)
    write_paper_monitoring_plan_artifacts(inputs, plan)
    return plan, inputs.output_dir / "paper_monitoring_plan.json"


def _write_ready_paper_stop_cleanup_plan(tmp_path) -> tuple[dict, Path]:
    monitoring_plan, monitoring_plan_path = _write_ready_paper_monitoring_plan(tmp_path)
    inputs = PaperStopCleanupPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_stop_cleanup",
        monitoring_plan_path=monitoring_plan_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed stop and cleanup planning only."],
        command=["python", "scripts/bot_factory_plan_paper_stop_cleanup.py"],
    )
    plan = build_paper_stop_cleanup_plan(inputs, monitoring_plan)
    write_paper_stop_cleanup_plan_artifacts(inputs, plan)
    return plan, inputs.output_dir / "paper_stop_cleanup_plan.json"


def _write_ready_paper_execution_chain(tmp_path) -> dict:
    stop_cleanup_plan, stop_cleanup_plan_path = _write_ready_paper_stop_cleanup_plan(
        tmp_path
    )
    monitoring_plan_path = tmp_path / stop_cleanup_plan["monitoring_plan_path"]
    monitoring_plan = json.loads(monitoring_plan_path.read_text(encoding="utf-8"))
    startup_preflight_path = tmp_path / monitoring_plan["startup_preflight_path"]
    startup_preflight = json.loads(
        startup_preflight_path.read_text(encoding="utf-8")
    )
    plan_path = tmp_path / startup_preflight["plan_path"]
    paper_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    readiness_path = tmp_path / paper_plan["readiness_path"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))

    return {
        "readiness": readiness,
        "readiness_path": readiness_path,
        "paper_plan": paper_plan,
        "plan_path": plan_path,
        "startup_preflight": startup_preflight,
        "startup_preflight_path": startup_preflight_path,
        "monitoring_plan": monitoring_plan,
        "monitoring_plan_path": monitoring_plan_path,
        "stop_cleanup_plan": stop_cleanup_plan,
        "stop_cleanup_plan_path": stop_cleanup_plan_path,
    }


def _write_ready_paper_execution_request(tmp_path) -> tuple[dict, Path]:
    chain = _write_ready_paper_execution_chain(tmp_path)
    start_command = " ".join(chain["startup_preflight"]["startup"]["command_preview"])
    inputs = PaperExecutionRequestInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_execution_request",
        readiness_path=chain["readiness_path"],
        plan_path=chain["plan_path"],
        startup_preflight_path=chain["startup_preflight_path"],
        monitoring_plan_path=chain["monitoring_plan_path"],
        stop_cleanup_plan_path=chain["stop_cleanup_plan_path"],
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed execution request only; do not start paper trading."],
        confirm_paper_execution=True,
        requested_start_command=start_command,
        command=["python", "scripts/bot_factory_request_paper_start.py"],
    )
    request = build_paper_execution_request(
        inputs,
        chain["readiness"],
        chain["paper_plan"],
        chain["startup_preflight"],
        chain["monitoring_plan"],
        chain["stop_cleanup_plan"],
    )
    write_paper_execution_request_artifacts(inputs, request)
    return request, inputs.output_dir / "paper_execution_request.json"


def _write_ready_paper_process_executor_plan(tmp_path) -> tuple[dict, Path]:
    execution_request, execution_request_path = _write_ready_paper_execution_request(tmp_path)
    start_command = " ".join(execution_request["execution_request"]["command_preview"])
    inputs = PaperProcessExecutorPlanInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_executor_plan",
        execution_request_path=execution_request_path,
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed process executor planning only."],
        confirm_process_executor_plan=True,
        requested_start_command=start_command,
        command=["python", "scripts/bot_factory_plan_paper_executor.py"],
    )
    plan = build_paper_process_executor_plan(inputs, execution_request)
    write_paper_process_executor_plan_artifacts(inputs, plan)
    return plan, inputs.output_dir / "paper_process_executor_plan.json"


def _write_paper_runtime_artifacts(tmp_path, process_executor_plan: dict) -> dict:
    planned_paths = process_executor_plan["planned_paths"]
    strategy = process_executor_plan["strategy"]
    runtime_run_id = process_executor_plan["run_id"]

    process_metadata_path = _repo_path(tmp_path, planned_paths["process_metadata"])
    status_snapshot_path = _repo_path(tmp_path, planned_paths["status_snapshot"])
    stdout_path = _repo_path(tmp_path, planned_paths["stdout"])
    stderr_path = _repo_path(tmp_path, planned_paths["stderr"])
    paper_metrics_path = _repo_path(tmp_path, planned_paths["paper_metrics"])
    for path in [
        process_metadata_path,
        status_snapshot_path,
        stdout_path,
        stderr_path,
        paper_metrics_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    command = process_executor_plan["executor_plan"]["command_preview"]
    process_metadata = {
        "strategy": strategy,
        "run_id": runtime_run_id,
        "process_started": True,
        "startup_executed": True,
        "pid": 1234,
        "started_at": "2026-05-03T00:00:00+00:00",
        "ended_at": None,
        "command": command,
        "stdout_log": planned_paths["stdout"],
        "stderr_log": planned_paths["stderr"],
        "status_snapshot": planned_paths["status_snapshot"],
        "paper_metrics": planned_paths["paper_metrics"],
        "process_control": False,
        "notice": "Synthetic runtime metadata for artifact validation only.",
    }
    status_snapshot = {
        "generated_at": "2026-05-03T00:00:30+00:00",
        "strategy": strategy,
        "run_id": runtime_run_id,
        "status": "running",
        "pid": 1234,
        "startup_executed": True,
        "bot_startup": True,
        "freqtrade_trade_executed": True,
        "paper_trading_started": True,
        "dry_run_trading_started": True,
        "live_trading": False,
        "exchange_order_placement": False,
        "open_trade_count": 1,
        "closed_trade_count": 2,
        "last_heartbeat_at": "2026-05-03T00:00:30+00:00",
        "message": "Synthetic running snapshot.",
    }
    paper_metrics = {
        "generated_at": "2026-05-03T00:00:30+00:00",
        "strategy": strategy,
        "run_id": runtime_run_id,
        "source": "local_paper_artifacts",
        "status": "running",
        "trade_counts": {"open": 1, "closed": 2, "total": 3},
        "profit": {"realized": 1.25, "unrealized": 0.5, "currency": "USDT"},
        "risk": {"max_drawdown_pct": 0.2, "max_open_trades": 1},
        "safety_scope": {
            "metadata_contains_secrets": False,
            "uses_api_keys_or_secrets": False,
            "local_artifacts_source_of_truth": True,
            "live_trading": False,
            "canary_live_trading": False,
            "exchange_order_placement": False,
            "leverage_above_one": False,
            "shorting": False,
            "process_control": False,
            "status_polling_started": False,
            "process_stop_started": False,
            "cleanup_executed": False,
        },
    }

    process_metadata_path.write_text(json.dumps(process_metadata), encoding="utf-8")
    status_snapshot_path.write_text(json.dumps(status_snapshot), encoding="utf-8")
    paper_metrics_path.write_text(json.dumps(paper_metrics), encoding="utf-8")
    stdout_path.write_text("synthetic stdout\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    return {
        "process_metadata": process_metadata,
        "status_snapshot": status_snapshot,
        "paper_metrics": paper_metrics,
        "process_metadata_path": process_metadata_path,
        "status_snapshot_path": status_snapshot_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "paper_metrics_path": paper_metrics_path,
    }


def _write_passed_paper_runtime_validation(tmp_path) -> tuple[dict, dict]:
    process_executor_plan, process_executor_plan_path = (
        _write_ready_paper_process_executor_plan(tmp_path)
    )
    artifacts = _write_paper_runtime_artifacts(tmp_path, process_executor_plan)
    _add_paper_drift_metrics(artifacts["paper_metrics_path"])
    inputs = PaperRuntimeValidationInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_runtime_validation",
        process_executor_plan_path=process_executor_plan_path,
        process_metadata_path=artifacts["process_metadata_path"],
        status_snapshot_path=artifacts["status_snapshot_path"],
        stdout_path=artifacts["stdout_path"],
        stderr_path=artifacts["stderr_path"],
        paper_metrics_path=artifacts["paper_metrics_path"],
        output_root=tmp_path / "data" / "paper",
        reviewer_notes=["Reviewed runtime artifacts only."],
        command=["python", "scripts/bot_factory_validate_paper_runtime.py"],
    )
    validation = build_paper_runtime_validation(
        inputs,
        process_executor_plan,
        artifacts["process_metadata"],
        artifacts["status_snapshot"],
        json.loads(artifacts["paper_metrics_path"].read_text(encoding="utf-8")),
    )
    write_paper_runtime_validation_artifacts(inputs, validation)
    artifacts["runtime_validation_path"] = (
        inputs.output_dir / "paper_runtime_validation.json"
    )
    return validation, artifacts


def _add_paper_drift_metrics(
    paper_metrics_path: Path,
    *,
    total_return_pct: float = 1.1,
    max_drawdown_pct: float = 1.5,
) -> None:
    paper_metrics = json.loads(paper_metrics_path.read_text(encoding="utf-8"))
    paper_metrics["profit"] = dict(paper_metrics["profit"])
    paper_metrics["risk"] = dict(paper_metrics["risk"])
    paper_metrics["profit"]["total_return_pct"] = total_return_pct
    paper_metrics["risk"]["max_drawdown_pct"] = max_drawdown_pct
    paper_metrics_path.write_text(json.dumps(paper_metrics), encoding="utf-8")


def _write_historical_metrics(tmp_path) -> Path:
    path = tmp_path / "data" / "freqai" / "PaperStrategy" / "historical" / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "strategy_name": "PaperStrategy",
                "total_return_pct": 1.0,
                "max_drawdown_pct": 1.0,
                "profit_factor": 1.5,
                "trade_count": 10,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_walk_forward_metrics(tmp_path, *, recommendation: str = "pass") -> Path:
    path = (
        tmp_path
        / "data"
        / "walk_forward"
        / "PaperStrategy"
        / "walk_forward"
        / "walk_forward_metrics.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "phase": "2",
                "status": "completed",
                "recommendation": recommendation,
                "strategy": "PaperStrategy",
                "summary": {
                    "completed_windows": 2,
                    "total_return_pct": 0.9,
                    "max_drawdown_pct_any_window": 1.2,
                    "pass_rate": 1.0,
                    "profitable_windows_ratio": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_training_manifest(tmp_path, *, recommendation: str = "pass") -> Path:
    path = (
        tmp_path
        / "data"
        / "freqai_training"
        / "PaperStrategy"
        / "training"
        / "training_manifest.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "phase": "2",
                "factory": "freqai_training",
                "status": "completed",
                "recommendation": recommendation,
                "strategy": "PaperStrategy",
                "summary": {"completed_stages": 1},
                "safety_scope": {
                    "paper_trading": False,
                    "dry_run_trading": False,
                    "live_trading": False,
                    "exchange_order_placement": False,
                    "leverage_experiments": False,
                    "shorting": False,
                    "metadata_contains_secrets": False,
                    "local_artifacts_source_of_truth": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _repo_path(root: Path, path_value: str) -> Path:
    return root / Path(str(path_value).replace("\\", "/"))


def _write_paper_evidence(
    tmp_path,
    *,
    historical_gate_pass: bool,
    walk_forward_recommendation: str,
    training_recommendation: str,
) -> tuple[Path, Path, Path]:
    historical_dir = tmp_path / "data" / "freqai" / "PaperStrategy" / "historical"
    walk_forward_dir = tmp_path / "data" / "walk_forward" / "PaperStrategy" / "wf"
    training_dir = tmp_path / "data" / "freqai_training" / "PaperStrategy" / "training"
    window_dir = walk_forward_dir / "windows" / "PaperStrategy" / "wf_01_20250101_20250115"
    training_child_dir = (
        training_dir
        / "freqai_backtests"
        / "PaperStrategy"
        / "train_20250101_20250201"
    )
    historical_dir.mkdir(parents=True)
    walk_forward_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)
    window_dir.mkdir(parents=True)
    training_child_dir.mkdir(parents=True)

    metrics = {
        "strategy_name": "PaperStrategy",
        "total_return": 0.05 if historical_gate_pass else -0.01,
        "total_return_pct": 5.0 if historical_gate_pass else -1.0,
        "cagr": 0.2 if historical_gate_pass else -0.1,
        "sharpe": 2.0 if historical_gate_pass else -1.0,
        "sortino": 2.0 if historical_gate_pass else -1.0,
        "calmar": 2.0 if historical_gate_pass else -1.0,
        "max_drawdown_pct": 5.0,
        "profit_factor": 1.5 if historical_gate_pass else 0.5,
        "win_rate": 0.55 if historical_gate_pass else 0.2,
        "average_win": 0.02,
        "average_loss": -0.01,
        "trade_count": 250 if historical_gate_pass else 2,
        "expectancy": 0.001 if historical_gate_pass else -0.001,
        "fee_paid": None,
        "backtest_start": "2025-01-01 00:00:00",
        "backtest_end": "2025-02-01 00:00:00",
        "generated_at": "2026-05-03T00:00:00+00:00",
    }
    (historical_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (historical_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    (historical_dir / "freqai_metadata.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (historical_dir / "trades.csv").write_text("is_short,leverage\nFalse,1.0\n", encoding="utf-8")

    (walk_forward_dir / "walk_forward_metrics.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "recommendation": walk_forward_recommendation,
                "summary": {"window_count": 1, "completed_windows": 1},
                "windows": [
                    {
                        "run_id": "wf_01_20250101_20250115",
                        "run_dir": str(window_dir),
                        "status": "completed",
                        "artifacts": {
                            "metrics": str(window_dir / "metrics.json"),
                            "trades": str(window_dir / "trades.csv"),
                            "freqai_metadata": str(window_dir / "freqai_metadata.json"),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (walk_forward_dir / "walk_forward_report.md").write_text("# Walk Forward\n", encoding="utf-8")
    (window_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (window_dir / "trades.csv").write_text("is_short,leverage\nFalse,1.0\n", encoding="utf-8")
    (window_dir / "freqai_metadata.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )

    (training_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "recommendation": training_recommendation,
                "summary": {"stage_count": 1, "completed_stages": 1},
                "stages": [
                    {
                        "name": "freqai_backtest",
                        "run_id": "train_20250101_20250201",
                        "status": "completed",
                        "output_dir": str(training_child_dir),
                        "artifacts": {
                            "metrics": str(training_child_dir / "metrics.json"),
                            "trades": str(training_child_dir / "trades.csv"),
                            "freqai_metadata": str(
                                training_child_dir / "freqai_metadata.json"
                            ),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (training_dir / "training_report.md").write_text("# Training\n", encoding="utf-8")
    (training_child_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (training_child_dir / "trades.csv").write_text(
        "is_short,leverage\nFalse,1.0\n", encoding="utf-8"
    )
    (training_child_dir / "freqai_metadata.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    return historical_dir, walk_forward_dir, training_dir


def _candidate_research_metadata(thesis_id: str) -> dict[str, Any]:
    references = [
        {
            "reference_id": f"paper:research-{thesis_id}",
            "title": "Hypothesis-Driven Market Microstructure Review",
            "source": "Local research bibliography",
            "published_at": "2026-05-06",
            "relevance": "Maps the candidate thesis to falsifiable market behavior.",
            "motivated_thesis_ids": [thesis_id],
        }
    ]
    return {
        "research_references": references,
        "research_brief": {
            "thesis_id": thesis_id,
            "thesis_statement": "Candidate thesis under historical evaluation.",
            "research_references": references,
            "evidence_refs": [f"research:paper:research-{thesis_id}"],
            "failure_taxonomy_codes": ["FAIL_OVERFIT_WF_GAP"],
        },
    }


def test_candidate_evaluation_writes_manifest_and_index(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import (
        CandidateEvaluationInputs,
        evaluate_candidate,
        write_candidate_artifacts,
    )

    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "LongOnlyRsiPullbackCandidate",
        "code_generation_eligible": True,
        "thesis_id": "TH-1",
        "thesis_type": "mean_reversion",
        "falsification_criteria": "wf_gap",
        "failure_taxonomy_codes": ["FAIL_OVERFIT_WF_GAP"],
        "retry_budget_per_thesis": 3,
        "thesis_retry_count": 1,
        "parameter_only_retry_count": 0,
        "force_distinct_hypothesis_family": False,
        **_candidate_research_metadata("TH-1"),
    }), encoding="utf-8")
    generated = tmp_path / "generated.json"
    generated.write_text(json.dumps({"strategy_name": "LongOnlyRsiPullbackCandidate", "candidate_evaluation_eligible": True}), encoding="utf-8")
    static = tmp_path / "static.json"
    static.write_text(json.dumps({"ok": True}), encoding="utf-8")
    ohlcv = tmp_path / "ohlcv.json"
    ohlcv.write_text(json.dumps({"ok": True}), encoding="utf-8")
    backtest = tmp_path / "backtest.json"
    backtest.write_text(json.dumps({"strategy_name": "LongOnlyRsiPullbackCandidate", "recommendation": "pass", "total_return_pct": 4.2, "trade_count": 240, "max_drawdown_pct": 3.0, "profit_factor": 1.5}), encoding="utf-8")
    trades = tmp_path / "trades.csv"
    trades.write_text("is_short,leverage\nFalse,1.0\n", encoding="utf-8")
    backtest_report = tmp_path / "report.md"
    backtest_report.write_text("# Backtest Report\n", encoding="utf-8")
    walk = tmp_path / "walk.json"
    walk.write_text(json.dumps({"strategy": "LongOnlyRsiPullbackCandidate", "recommendation": "pass", "summary": {"pass_rate": 1.0, "profitable_windows_ratio": 1.0, "total_return_pct": 3.0, "max_single_window_profit_dependency": 0.3}}), encoding="utf-8")
    walk_report = tmp_path / "walk.md"
    walk_report.write_text("# Walk Forward Report\n", encoding="utf-8")
    training = tmp_path / "train.json"
    training.write_text(json.dumps({"strategy": "LongOnlyRsiPullbackCandidate", "recommendation": "pass", "summary": {"stage_count": 1, "failed_stages": 0}}), encoding="utf-8")
    training_report = tmp_path / "training.md"
    training_report.write_text("# Training Report\n", encoding="utf-8")

    manifest = evaluate_candidate(CandidateEvaluationInputs(
        root_dir=tmp_path,
        proposal_metadata_path=proposal,
        generated_metadata_path=generated,
        candidate_id="cand-1",
        static_check_path=static,
        ohlcv_quality_path=ohlcv,
        backtest_metrics_path=backtest,
        backtest_trades_path=trades,
        backtest_report_path=backtest_report,
        walk_forward_metrics_path=walk,
        walk_forward_report_path=walk_report,
        training_manifest_path=training,
        training_report_path=training_report,
    ))
    assert manifest["recommendation"] == "pass", [
        (check["name"], check["status"], check.get("path")) for check in manifest["checks"]
    ]
    research_check = next(c for c in manifest["checks"] if c["name"] == "research_brief")
    assert research_check["status"] == "pass"
    assert manifest["research_brief"]["research_references"][0]["reference_id"] == (
        "paper:research-TH-1"
    )
    manifest_path, index_path = write_candidate_artifacts(
        manifest, root_dir=tmp_path, output_root=Path("out"), index_path=Path("idx.jsonl")
    )
    assert manifest_path.is_file()
    assert (manifest_path.parent / "candidate_record.json").is_file()
    record = json.loads((manifest_path.parent / "candidate_record.json").read_text(encoding="utf-8"))
    assert record["research_brief"]["thesis_id"] == "TH-1"
    assert (manifest_path.parent / "candidate_report.md").is_file()
    assert (manifest_path.parent / "metrics_summary.json").is_file()
    assert (manifest_path.parent / "artifact_paths.json").is_file()
    lines = index_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    first = json.loads(lines[0])
    assert first["candidate_report_path"].endswith("candidate_report.md")


def test_candidate_evaluation_derives_backtest_gate_when_metrics_lack_recommendation(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import (
        CandidateEvaluationInputs,
        evaluate_candidate,
    )

    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "LongOnlyRsiPullbackCandidate",
        "code_generation_eligible": True,
        "thesis_id": "TH-1",
        "thesis_type": "mean_reversion",
        "falsification_criteria": "wf_gap",
        "failure_taxonomy_codes": ["FAIL_OVERFIT_WF_GAP"],
        "retry_budget_per_thesis": 3,
        "thesis_retry_count": 1,
        "parameter_only_retry_count": 0,
        "force_distinct_hypothesis_family": False,
        **_candidate_research_metadata("TH-1"),
    }), encoding="utf-8")
    generated = tmp_path / "generated.json"
    generated.write_text(
        json.dumps({
            "strategy_name": "LongOnlyRsiPullbackCandidate",
            "candidate_evaluation_eligible": True,
        }),
        encoding="utf-8",
    )
    static = tmp_path / "static.json"
    static.write_text(json.dumps({"ok": True}), encoding="utf-8")
    ohlcv = tmp_path / "ohlcv.json"
    ohlcv.write_text(json.dumps({"ok": True}), encoding="utf-8")
    backtest = tmp_path / "backtest.json"
    backtest.write_text(
        json.dumps({
            "strategy_name": "LongOnlyRsiPullbackCandidate",
            "total_return": -0.01,
            "total_return_pct": -1.0,
            "trade_count": 240,
            "max_drawdown_pct": 3.0,
            "profit_factor": 0.8,
            "sortino": -0.5,
        }),
        encoding="utf-8",
    )
    trades = tmp_path / "trades.csv"
    trades.write_text("is_short,leverage\nFalse,1.0\n", encoding="utf-8")
    backtest_report = tmp_path / "report.md"
    backtest_report.write_text("# Backtest Report\n", encoding="utf-8")
    walk = tmp_path / "walk.json"
    walk.write_text(
        json.dumps({
            "strategy": "LongOnlyRsiPullbackCandidate",
            "recommendation": "pass",
            "summary": {"pass_rate": 1.0, "profitable_windows_ratio": 1.0},
        }),
        encoding="utf-8",
    )
    walk_report = tmp_path / "walk.md"
    walk_report.write_text("# Walk Forward Report\n", encoding="utf-8")

    manifest = evaluate_candidate(CandidateEvaluationInputs(
        root_dir=tmp_path,
        proposal_metadata_path=proposal,
        generated_metadata_path=generated,
        candidate_id="cand-1",
        static_check_path=static,
        ohlcv_quality_path=ohlcv,
        backtest_metrics_path=backtest,
        backtest_trades_path=trades,
        backtest_report_path=backtest_report,
        walk_forward_metrics_path=walk,
        walk_forward_report_path=walk_report,
    ))

    historical_check = next(c for c in manifest["checks"] if c["name"] == "historical_backtest")
    assert historical_check["value"] == "fail"
    assert historical_check["status"] == "fail"
    assert manifest["recommendation"] == "retry"


def test_candidate_evaluation_carries_blocked_next_actions_to_iteration(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import (
        CandidateEvaluationInputs,
        evaluate_candidate,
    )
    from freqtrade_ext.bot_factory.candidate_iteration import (
        CandidateIterationInputs,
        build_candidate_iteration_plan,
    )

    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "S",
        "code_generation_eligible": True,
        "thesis_id": "TH-LOCAL-REJECTED",
        "thesis_type": "mean_reversion",
        "falsification_criteria": "Local falsification remains negative.",
        "failure_taxonomy_codes": ["FAIL_COST_SENSITIVE"],
        "retry_budget_per_thesis": 3,
        "thesis_retry_count": 1,
        "parameter_only_retry_count": 0,
        "force_distinct_hypothesis_family": False,
        "failure_synthesis_constraints": [
            {
                "path": "registry\\strategies\\synthesis\\s1.json",
                "blocked_next_actions": [
                    "retry_validated_local_rejection_by_parameter_tuning",
                ],
            }
        ],
        **_candidate_research_metadata("TH-LOCAL-REJECTED"),
    }), encoding="utf-8")
    generated = tmp_path / "generated.json"
    generated.write_text(json.dumps({
        "strategy_name": "S",
        "candidate_evaluation_eligible": True,
        "generator_mode": "rule_based",
    }), encoding="utf-8")
    static = tmp_path / "static.json"
    static.write_text(json.dumps({"ok": True}), encoding="utf-8")
    ohlcv = tmp_path / "ohlcv.json"
    ohlcv.write_text(json.dumps({"ok": True}), encoding="utf-8")
    backtest = tmp_path / "backtest.json"
    backtest.write_text(json.dumps({
        "strategy_name": "S",
        "recommendation": "pass",
        "total_return_pct": 1.2,
        "trade_count": 40,
        "max_drawdown_pct": 2.0,
        "profit_factor": 1.2,
    }), encoding="utf-8")
    trades = tmp_path / "trades.csv"
    trades.write_text("is_short,leverage\nFalse,1.0\n", encoding="utf-8")
    walk = tmp_path / "walk.json"
    walk.write_text(json.dumps({
        "strategy": "S",
        "recommendation": "retry",
        "summary": {
            "pass_rate": 0.25,
            "profitable_windows_ratio": 0.25,
            "total_return_pct": -0.8,
            "max_single_window_profit_dependency": 0.9,
        },
    }), encoding="utf-8")

    manifest = evaluate_candidate(CandidateEvaluationInputs(
        root_dir=tmp_path,
        proposal_metadata_path=proposal,
        generated_metadata_path=generated,
        candidate_id="cand-local-reject",
        static_check_path=static,
        ohlcv_quality_path=ohlcv,
        backtest_metrics_path=backtest,
        backtest_trades_path=trades,
        walk_forward_metrics_path=walk,
    ))

    assert manifest["research_brief"]["blocked_next_actions"] == [
        "retry_validated_local_rejection_by_parameter_tuning"
    ]
    assert manifest["next_candidate_input"]["blocked_next_actions"] == [
        "retry_validated_local_rejection_by_parameter_tuning"
    ]

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    plan = build_candidate_iteration_plan(CandidateIterationInputs(
        root_dir=tmp_path,
        candidate_manifest_path=manifest_path,
        proposal_metadata_path=proposal,
        reviewer_findings=["Walk-forward evidence remains fragile."],
        changed_assumptions=[
            "Retry validated local rejection by parameter tuning."
        ],
        unchanged_rejection_rules=[
            "Reject if local falsification still shows non-positive edge."
        ],
        prior_timerange="20250101-20250201",
        proposed_timerange="20250101-20250201",
    ))

    assert plan["action"] == "blocked"
    assert plan["blocked_next_action_matches"] == [
        "retry_validated_local_rejection_by_parameter_tuning"
    ]


def test_candidate_evaluation_preserves_research_handoff_summaries(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import (
        CandidateEvaluationInputs,
        evaluate_candidate,
    )

    question_handoff = {
        "required": True,
        "passed": False,
        "failed_paths": ["registry\\strategies\\research_decisions\\rd.json"],
        "candidates": [
            {
                "path": "registry\\strategies\\research_decisions\\rd.json",
                "missing_research_question_response_indexes": [2],
            }
        ],
    }
    handoff_summaries = [
        {
            "candidate_id": "cand-handoff",
            "research_handoff_summary": {
                "research_decision_question_handoff": question_handoff,
            },
        }
    ]
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "S",
        "code_generation_eligible": True,
        "thesis_id": "TH-HANDOFF",
        "thesis_type": "mean_reversion",
        "falsification_criteria": "Question handoff must remain visible.",
        **_candidate_research_metadata("TH-HANDOFF"),
    }), encoding="utf-8")
    generated = tmp_path / "generated.json"
    generated.write_text(json.dumps({
        "strategy_name": "S",
        "candidate_evaluation_eligible": True,
        "generator_mode": "rule_based",
        "research_brief": {
            "thesis_id": "TH-HANDOFF",
            "thesis_statement": "Question handoff preservation thesis.",
            "research_references": _candidate_research_metadata("TH-HANDOFF")[
                "research_references"
            ],
            "research_decision_question_handoff": question_handoff,
            "research_decision_novelty_handoff": {
                "required": True,
                "passed": True,
            },
            "research_handoff_summaries": handoff_summaries,
        },
    }), encoding="utf-8")
    static = tmp_path / "static.json"
    static.write_text(json.dumps({"ok": True}), encoding="utf-8")
    ohlcv = tmp_path / "ohlcv.json"
    ohlcv.write_text(json.dumps({"ok": True}), encoding="utf-8")
    backtest = tmp_path / "backtest.json"
    backtest.write_text(json.dumps({
        "strategy_name": "S",
        "recommendation": "pass",
        "total_return_pct": 1.2,
        "trade_count": 40,
        "max_drawdown_pct": 2.0,
        "profit_factor": 1.2,
    }), encoding="utf-8")
    trades = tmp_path / "trades.csv"
    trades.write_text("is_short,leverage\nFalse,1.0\n", encoding="utf-8")
    walk = tmp_path / "walk.json"
    walk.write_text(json.dumps({
        "strategy": "S",
        "recommendation": "pass",
        "summary": {
            "pass_rate": 1.0,
            "profitable_windows_ratio": 1.0,
            "total_return_pct": 1.0,
            "max_single_window_profit_dependency": 0.2,
        },
    }), encoding="utf-8")

    manifest = evaluate_candidate(CandidateEvaluationInputs(
        root_dir=tmp_path,
        proposal_metadata_path=proposal,
        generated_metadata_path=generated,
        candidate_id="cand-handoff",
        static_check_path=static,
        ohlcv_quality_path=ohlcv,
        backtest_metrics_path=backtest,
        backtest_trades_path=trades,
        walk_forward_metrics_path=walk,
    ))

    assert manifest["research_brief"]["research_decision_question_handoff"] == (
        question_handoff
    )
    assert manifest["research_brief"]["research_decision_novelty_handoff"] == {
        "required": True,
        "passed": True,
    }
    assert manifest["research_brief"]["research_handoff_summaries"] == (
        handoff_summaries
    )


def test_candidate_evaluation_rejects_ineligible_candidate(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import CandidateEvaluationInputs, evaluate_candidate

    p = tmp_path / "p.json"
    p.write_text(json.dumps({"strategy_name": "S", "code_generation_eligible": False}), encoding="utf-8")
    g = tmp_path / "g.json"
    g.write_text(json.dumps({"strategy_name": "S", "candidate_evaluation_eligible": False}), encoding="utf-8")
    manifest = evaluate_candidate(CandidateEvaluationInputs(root_dir=tmp_path, proposal_metadata_path=p, generated_metadata_path=g, candidate_id="c"))
    assert manifest["recommendation"] == "reject"


def test_candidate_evaluation_rejects_generated_hyperopt_parameter_surface(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import (
        CandidateEvaluationInputs,
        evaluate_candidate,
    )

    proposal = tmp_path / "proposal.json"
    proposal.write_text(
        json.dumps({"strategy_name": "S", "code_generation_eligible": True}),
        encoding="utf-8",
    )
    strategy = tmp_path / "S.py"
    strategy.write_text(
        (
            "from freqtrade.strategy import IntParameter\n\n"
            "class S:\n"
            "    buy_rsi_window = IntParameter(8, 30, default=14, "
            "space=\"buy\", optimize=True, load=True)\n"
        ),
        encoding="utf-8",
    )
    generated = tmp_path / "generated.json"
    generated.write_text(
        json.dumps({
            "factory": "strategy_code_generator",
            "strategy_name": "S",
            "candidate_evaluation_eligible": True,
            "strategy_code_generated": True,
            "generated_strategy_path": "S.py",
            "parameter_optimization_enabled": False,
            "parameter_optimization_policy": PARAMETER_OPTIMIZATION_POLICY,
            "safety_scope": {
                "freqtrade_hyperopt_parameter_optimization": False,
            },
            "checks": [
                {
                    "name": "generated_code_freqtrade_hyperopt_disabled",
                    "status": "pass",
                }
            ],
        }),
        encoding="utf-8",
    )

    manifest = evaluate_candidate(CandidateEvaluationInputs(
        root_dir=tmp_path,
        proposal_metadata_path=proposal,
        generated_metadata_path=generated,
        candidate_id="c-hyperopt",
    ))

    check = next(
        item for item in manifest["checks"]
        if item["name"] == "generated_parameter_optimization_policy"
    )
    assert manifest["recommendation"] == "reject"
    assert "parameter optimization" in manifest["recommendation_rationale"]
    assert check["status"] == "fail"
    assert check["code_contains_optimize_true"] is True
    assert check["code_contains_optimize_false"] is False


def test_candidate_evaluation_rule_based_does_not_require_training_manifest(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import CandidateEvaluationInputs, evaluate_candidate

    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "RuleS",
        "code_generation_eligible": True,
        "thesis_id": "TH-RULE",
        **_candidate_research_metadata("TH-RULE"),
    }), encoding="utf-8")
    generated = tmp_path / "generated.json"
    generated.write_text(json.dumps({"strategy_name": "RuleS", "candidate_evaluation_eligible": True, "generator_mode": "rule_based"}), encoding="utf-8")
    static = tmp_path / "static.json"
    static.write_text(json.dumps({"ok": True}), encoding="utf-8")
    ohlcv = tmp_path / "ohlcv.json"
    ohlcv.write_text(json.dumps({"ok": True}), encoding="utf-8")
    backtest = tmp_path / "backtest.json"
    backtest.write_text(json.dumps({"strategy_name": "RuleS", "recommendation": "pass"}), encoding="utf-8")
    walk = tmp_path / "walk.json"
    walk.write_text(json.dumps({"strategy": "RuleS", "recommendation": "pass"}), encoding="utf-8")
    trades = tmp_path / "trades.csv"
    trades.write_text("is_short,leverage\nFalse,1.0\n", encoding="utf-8")
    backtest_report = tmp_path / "report.md"
    backtest_report.write_text("# Backtest Report\n", encoding="utf-8")
    walk_report = tmp_path / "walk.md"
    walk_report.write_text("# Walk Forward Report\n", encoding="utf-8")

    manifest = evaluate_candidate(CandidateEvaluationInputs(
        root_dir=tmp_path,
        proposal_metadata_path=proposal,
        generated_metadata_path=generated,
        candidate_id="cand-rule",
        static_check_path=static,
        ohlcv_quality_path=ohlcv,
        backtest_metrics_path=backtest,
        backtest_trades_path=trades,
        backtest_report_path=backtest_report,
        walk_forward_metrics_path=walk,
        walk_forward_report_path=walk_report,
        training_manifest_path=None,
    ))
    training_check = next(c for c in manifest["checks"] if c["name"] == "training_factory")
    assert training_check["status"] == "skipped"
    assert manifest["evaluation_orchestration"]["steps"][0]["name"] == "static_strategy_check"
    assert manifest["recommendation"] == "pass", [
        (check["name"], check["status"], check.get("path")) for check in manifest["checks"]
    ]


def test_candidate_evaluation_freqai_requires_feature_label_validation(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import CandidateEvaluationInputs, evaluate_candidate

    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "FreqaiS",
        "code_generation_eligible": True,
        "generator_mode": "freqai",
        "thesis_id": "TH-FREQAI",
        **_candidate_research_metadata("TH-FREQAI"),
    }), encoding="utf-8")
    generated = tmp_path / "generated.json"
    generated.write_text(json.dumps({"strategy_name": "FreqaiS", "candidate_evaluation_eligible": True, "generator_mode": "freqai"}), encoding="utf-8")
    static = tmp_path / "static.json"
    static.write_text(json.dumps({"ok": True}), encoding="utf-8")
    ohlcv = tmp_path / "ohlcv.json"
    ohlcv.write_text(json.dumps({"ok": True}), encoding="utf-8")
    backtest = tmp_path / "backtest.json"
    backtest.write_text(json.dumps({"strategy_name": "FreqaiS", "recommendation": "pass"}), encoding="utf-8")
    trades = tmp_path / "trades.csv"
    trades.write_text("is_short,leverage\nFalse,1.0\n", encoding="utf-8")
    backtest_report = tmp_path / "report.md"
    backtest_report.write_text("# Backtest Report\n", encoding="utf-8")
    walk = tmp_path / "walk.json"
    walk.write_text(json.dumps({"strategy": "FreqaiS", "recommendation": "pass"}), encoding="utf-8")
    walk_report = tmp_path / "walk.md"
    walk_report.write_text("# Walk Forward Report\n", encoding="utf-8")
    training = tmp_path / "train.json"
    training.write_text(json.dumps({"strategy": "FreqaiS", "recommendation": "pass"}), encoding="utf-8")
    training_report = tmp_path / "training.md"
    training_report.write_text("# Training Report\n", encoding="utf-8")

    manifest = evaluate_candidate(CandidateEvaluationInputs(
        root_dir=tmp_path,
        proposal_metadata_path=proposal,
        generated_metadata_path=generated,
        candidate_id="cand-freqai",
        static_check_path=static,
        ohlcv_quality_path=ohlcv,
        backtest_metrics_path=backtest,
        backtest_trades_path=trades,
        backtest_report_path=backtest_report,
        walk_forward_metrics_path=walk,
        walk_forward_report_path=walk_report,
        training_manifest_path=training,
        training_report_path=training_report,
        freqai_validation_path=None,
    ))
    validation_check = next(c for c in manifest["checks"] if c["name"] == "freqai_feature_label_validation")
    assert validation_check["status"] == "missing"
    assert manifest["recommendation"] == "fail"


def test_candidate_evaluation_executes_checked_wrapper_chain_with_fake_runner(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import CandidateEvaluationInputs, evaluate_candidate
    from freqtrade_ext.bot_factory.candidate_identity import build_strategy_candidate_identity

    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "FreqaiS",
        "code_generation_eligible": True,
        "generator_mode": "freqai",
        "thesis_id": "TH-FREQAI",
        **_candidate_research_metadata("TH-FREQAI"),
    }), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    identity = build_strategy_candidate_identity(
        candidate_id="cand-exec",
        strategy_id="FreqaiS",
        strategy_class_name="FreqaiS",
        strategy_source_path=strategy_dir / "FreqaiS.py",
        strategy_version="FreqaiS_v1",
        signal_version="signal_v1",
        risk_policy_version="risk_v1",
        regime_classifier_version="regime_v1",
        cost_model_id="cost_v1",
        allowed_pairs=["BTC/USDT:USDT"],
        allowed_timeframes=["5m"],
        created_at="2026-05-23T00:00:00+00:00",
        source_artifacts={"generated_metadata": "generated.json"},
        root_dir=tmp_path,
    )
    generated = tmp_path / "generated.json"
    generated.write_text(json.dumps({
        "strategy_name": "FreqaiS",
        "candidate_id": "cand-exec",
        "candidate_evaluation_eligible": True,
        "generator_mode": "freqai",
        "target_definition": "future_return",
        "candidate_identity": identity,
    }), encoding="utf-8")
    ohlcv = tmp_path / "BTC_USDT-5m.parquet"
    ohlcv.write_text("fake parquet for command construction only", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_runner(command, cwd):
        commands.append(list(command))
        command_text = " ".join(command)
        if "bot_factory_static_check.py" in command_text:
            output = Path(cwd) / command[command.index("--output") + 1]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"ok": True}), encoding="utf-8")
        elif "bot_factory_validate_freqai_strategy.py" in command_text:
            output = Path(cwd) / command[command.index("--output") + 1]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"ok": True}), encoding="utf-8")
        elif "bot_factory_check_ohlcv.py" in command_text:
            output = Path(cwd) / command[command.index("--output") + 1]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"ok": True}), encoding="utf-8")
        elif "bot_factory_run_walk_forward.py" in command_text:
            out = Path(cwd) / command[command.index("--output-root") + 1] / "FreqaiS" / command[command.index("--run-id") + 1]
            out.mkdir(parents=True, exist_ok=True)
            (out / "walk_forward_metrics.json").write_text(json.dumps({"strategy": "FreqaiS", "recommendation": "pass", "candidate_identity": identity}), encoding="utf-8")
            (out / "walk_forward_report.md").write_text("# Walk-forward\n", encoding="utf-8")
        elif "bot_factory_run_freqai_backtest.py" in command_text:
            out = Path(cwd) / command[command.index("--output-root") + 1] / "FreqaiS" / command[command.index("--run-id") + 1]
            out.mkdir(parents=True, exist_ok=True)
            (out / "metrics.json").write_text(json.dumps({"strategy_name": "FreqaiS", "recommendation": "pass", "candidate_identity": identity}), encoding="utf-8")
            (out / "trades.csv").write_text("is_short,leverage\nFalse,1.0\n", encoding="utf-8")
            (out / "report.md").write_text("# Backtest\n", encoding="utf-8")
        elif "bot_factory_run_freqai_training.py" in command_text:
            out = Path(cwd) / command[command.index("--output-root") + 1] / "FreqaiS" / command[command.index("--run-id") + 1]
            out.mkdir(parents=True, exist_ok=True)
            (out / "training_manifest.json").write_text(json.dumps({"strategy": "FreqaiS", "recommendation": "pass", "candidate_identity": identity}), encoding="utf-8")
            (out / "training_report.md").write_text("# Training\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    manifest = evaluate_candidate(CandidateEvaluationInputs(
        root_dir=tmp_path,
        proposal_metadata_path=proposal,
        generated_metadata_path=generated,
        candidate_id="cand-exec",
        config_path=config,
        strategy_path=strategy_dir,
        ohlcv_parquet_paths=[ohlcv],
        execute_historical_chain=True,
        execution_run_id="exec1",
        python_executable="python",
        timeframe="5m",
        timerange="20250101-20250103",
        pairs=["BTC/USDT:USDT"],
        walk_forward_windows=["20250101-20250103", "20250103-20250105"],
        training_timerange="20250101-20250103",
        command_runner=fake_runner,
    ))

    assert manifest["recommendation"] == "pass", [
        (check["name"], check["status"], check.get("path")) for check in manifest["checks"]
    ]
    assert manifest["candidate_execution"]["status"] == "completed"
    assert [step["name"] for step in manifest["candidate_execution"]["steps"]] == [
        "static_strategy_check",
        "freqai_feature_label_validation",
        "ohlcv_quality_check",
        "historical_backtest",
        "walk_forward",
        "training_factory",
    ]
    assert [command[1] for command in commands] == [
        "scripts/bot_factory_static_check.py",
        "scripts/bot_factory_validate_freqai_strategy.py",
        "scripts/bot_factory_check_ohlcv.py",
        "scripts/bot_factory_run_freqai_backtest.py",
        "scripts/bot_factory_run_walk_forward.py",
        "scripts/bot_factory_run_freqai_training.py",
    ]
    expected_identifier = candidate_freqai_identifier(
        "FreqaiS",
        "cand-exec",
        "future_return",
    )
    assert manifest["candidate_execution"]["freqai"]["identifier"] == expected_identifier
    for command in commands[3:]:
        assert command[command.index("--freqai-identifier") + 1] == expected_identifier
        identity_arg = command[command.index("--candidate-identity-json") + 1]
        assert json.loads((tmp_path / identity_arg).read_text(encoding="utf-8")) == identity
    assert all(check["status"] in {"pass", "skipped"} for check in manifest["checks"])


def test_candidate_evaluation_execution_requires_safe_inputs(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import CandidateEvaluationInputs, evaluate_candidate

    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "S",
        "code_generation_eligible": True,
        "thesis_id": "TH-REQ",
        **_candidate_research_metadata("TH-REQ"),
    }), encoding="utf-8")
    generated = tmp_path / "generated.json"
    generated.write_text(json.dumps({
        "strategy_name": "S",
        "candidate_evaluation_eligible": True,
        "generator_mode": "rule_based",
    }), encoding="utf-8")
    calls: list[list[str]] = []

    manifest = evaluate_candidate(CandidateEvaluationInputs(
        root_dir=tmp_path,
        proposal_metadata_path=proposal,
        generated_metadata_path=generated,
        candidate_id="cand-blocked",
        execute_historical_chain=True,
        command_runner=lambda command, cwd: calls.append(list(command)),
    ))

    assert manifest["candidate_execution"]["status"] == "blocked"
    assert set(manifest["candidate_execution"]["blockers"]) == {
        "config_required_for_execution",
        "timerange_required_for_historical_backtest_execution",
        "strategy_path_required_for_execution",
    }
    assert calls == []


def test_candidate_evaluation_execution_stops_after_wrapper_failure(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import CandidateEvaluationInputs, evaluate_candidate

    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "S",
        "code_generation_eligible": True,
        "thesis_id": "TH-FAIL",
        **_candidate_research_metadata("TH-FAIL"),
    }), encoding="utf-8")
    generated = tmp_path / "generated.json"
    generated.write_text(json.dumps({
        "strategy_name": "S",
        "candidate_evaluation_eligible": True,
        "generator_mode": "rule_based",
    }), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir()
    calls: list[list[str]] = []

    def failing_runner(command, cwd):
        calls.append(list(command))
        if "bot_factory_static_check.py" in " ".join(command):
            output = Path(cwd) / command[command.index("--output") + 1]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"ok": False}), encoding="utf-8")
            return SimpleNamespace(returncode=1, stdout="static failed", stderr="")
        raise AssertionError("runner should stop after the first failed wrapper")

    manifest = evaluate_candidate(CandidateEvaluationInputs(
        root_dir=tmp_path,
        proposal_metadata_path=proposal,
        generated_metadata_path=generated,
        candidate_id="cand-failure",
        config_path=config,
        strategy_path=strategy_dir,
        execute_historical_chain=True,
        execution_run_id="exec-failure",
        python_executable="python",
        timerange="20250101-20250103",
        command_runner=failing_runner,
    ))

    assert manifest["candidate_execution"]["status"] == "failed"
    assert len(manifest["candidate_execution"]["results"]) == 1
    assert manifest["candidate_execution"]["results"][0]["returncode"] == 1
    assert len(calls) == 1
    assert manifest["recommendation"] == "fail"


def test_candidate_evaluation_static_check_uses_generated_strategy_file_path(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import CandidateEvaluationInputs, evaluate_candidate

    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "GeneratedRuleStrategy",
        "code_generation_eligible": True,
        "thesis_id": "TH-FILE-PATH",
        **_candidate_research_metadata("TH-FILE-PATH"),
    }), encoding="utf-8")
    generated = tmp_path / "generated.json"
    generated.write_text(json.dumps({
        "strategy_name": "GeneratedRuleStrategy",
        "candidate_evaluation_eligible": True,
        "generator_mode": "rule_based",
        "generated_strategy_path": "user_data/strategies/GeneratedRuleStrategy.py",
    }), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []

    def failing_runner(command, cwd):
        calls.append(list(command))
        return SimpleNamespace(returncode=1, stdout="static failed", stderr="")

    manifest = evaluate_candidate(CandidateEvaluationInputs(
        root_dir=tmp_path,
        proposal_metadata_path=proposal,
        generated_metadata_path=generated,
        candidate_id="cand-file-path",
        config_path=config,
        execute_historical_chain=True,
        execution_run_id="exec-file-path",
        python_executable="python",
        timerange="20250101-20250103",
        command_runner=failing_runner,
    ))

    assert len(calls) == 1
    assert calls[0][0:3] == [
        "python",
        "scripts/bot_factory_static_check.py",
        "user_data/strategies/GeneratedRuleStrategy.py",
    ]
    assert manifest["candidate_execution"]["status"] == "failed"


def test_candidate_evaluation_cli_maps_execution_flags(tmp_path):
    from scripts.bot_factory_evaluate_candidate import build_inputs_from_args

    args = SimpleNamespace(
        proposal_metadata_json="registry/strategies/proposals/p.metadata.json",
        generated_metadata_json="registry/strategies/generated/S/run/metadata.json",
        candidate_id="cand-cli",
        config="user_data/config_freqai_phase2_safe.json",
        strategy_path="user_data/strategies",
        ohlcv_parquet=["user_data/data/bybit/futures/BTC_USDT_USDT-5m-futures.parquet"],
        static_check_json="registry/strategies/checks/static.json",
        freqai_validation_json="registry/strategies/checks/freqai.json",
        ohlcv_quality_json="registry/strategies/checks/ohlcv.json",
        backtest_metrics_json="data/freqai/S/run/metrics.json",
        backtest_trades_csv="data/freqai/S/run/trades.csv",
        backtest_report_md="data/freqai/S/run/report.md",
        walk_forward_metrics_json="data/walk_forward/S/run/walk_forward_metrics.json",
        walk_forward_report_md="data/walk_forward/S/run/walk_forward_report.md",
        training_manifest_json="data/freqai_training/S/run/training_manifest.json",
        training_report_md="data/freqai_training/S/run/training_report.md",
        reviewer_note=["historical-safe CLI mapping test"],
        execute_historical_chain=True,
        execution_run_id="exec-cli",
        python=".venv/Scripts/python.exe",
        timeframe="5m",
        timerange="20250101-20250103",
        pairs=["BTC/USDT:USDT", "ETH/USDT:USDT"],
        walk_forward_window=["20250101-20250103", "20250103-20250105"],
        training_timerange="20250101-20250103",
        freqai_identifier="bf_cli_candidate",
        execution_output_root="registry/strategies/candidates/executions-cli",
        backtest_output_root="data/backtests-cli",
        freqai_output_root="data/freqai-cli",
        walk_forward_output_root="data/walk_forward-cli",
        training_output_root="data/freqai_training-cli",
    )

    inputs = build_inputs_from_args(args, root_dir=tmp_path)

    assert inputs.root_dir == tmp_path
    assert inputs.proposal_metadata_path == Path(args.proposal_metadata_json)
    assert inputs.generated_metadata_path == Path(args.generated_metadata_json)
    assert inputs.candidate_id == "cand-cli"
    assert inputs.config_path == Path(args.config)
    assert inputs.strategy_path == Path(args.strategy_path)
    assert inputs.ohlcv_parquet_paths == [Path(args.ohlcv_parquet[0])]
    assert inputs.static_check_path == Path(args.static_check_json)
    assert inputs.freqai_validation_path == Path(args.freqai_validation_json)
    assert inputs.ohlcv_quality_path == Path(args.ohlcv_quality_json)
    assert inputs.backtest_metrics_path == Path(args.backtest_metrics_json)
    assert inputs.backtest_trades_path == Path(args.backtest_trades_csv)
    assert inputs.backtest_report_path == Path(args.backtest_report_md)
    assert inputs.walk_forward_metrics_path == Path(args.walk_forward_metrics_json)
    assert inputs.walk_forward_report_path == Path(args.walk_forward_report_md)
    assert inputs.training_manifest_path == Path(args.training_manifest_json)
    assert inputs.training_report_path == Path(args.training_report_md)
    assert inputs.reviewer_notes == ["historical-safe CLI mapping test"]
    assert inputs.execute_historical_chain is True
    assert inputs.execution_run_id == "exec-cli"
    assert inputs.python_executable == ".venv/Scripts/python.exe"
    assert inputs.timeframe == "5m"
    assert inputs.timerange == "20250101-20250103"
    assert inputs.pairs == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert inputs.walk_forward_windows == ["20250101-20250103", "20250103-20250105"]
    assert inputs.training_timerange == "20250101-20250103"
    assert inputs.freqai_identifier == "bf_cli_candidate"
    assert inputs.execution_output_root == Path("registry/strategies/candidates/executions-cli")
    assert inputs.backtest_output_root == Path("data/backtests-cli")
    assert inputs.freqai_output_root == Path("data/freqai-cli")
    assert inputs.walk_forward_output_root == Path("data/walk_forward-cli")
    assert inputs.training_output_root == Path("data/freqai_training-cli")


def test_candidate_evaluation_missing_required_artifact_is_fail_not_retry(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import CandidateEvaluationInputs, evaluate_candidate

    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({"strategy_name": "RuleS", "code_generation_eligible": True, "failure_taxonomy_codes": ["FAIL_OVERFIT_WF_GAP"]}), encoding="utf-8")
    generated = tmp_path / "generated.json"
    generated.write_text(json.dumps({"strategy_name": "RuleS", "candidate_evaluation_eligible": True, "generator_mode": "rule_based"}), encoding="utf-8")

    manifest = evaluate_candidate(CandidateEvaluationInputs(
        root_dir=tmp_path,
        proposal_metadata_path=proposal,
        generated_metadata_path=generated,
        candidate_id="cand-missing",
        static_check_path=None,
        ohlcv_quality_path=None,
        backtest_metrics_path=None,
        walk_forward_metrics_path=None,
    ))
    assert manifest["recommendation"] == "fail"
    assert "missing" in manifest["recommendation_rationale"].lower()

def test_candidate_artifact_paths_are_sanitized(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import write_candidate_artifacts

    manifest = {
        "generated_at": "2026-05-05T00:00:00+00:00",
        "candidate_id": "../bad/candidate",
        "strategy_name": "../../bad strategy",
        "recommendation": "pass",
        "thesis": {},
        "failure_taxonomy_codes": [],
    }
    manifest_path, _ = write_candidate_artifacts(
        manifest,
        root_dir=tmp_path,
        output_root=Path("registry/strategies/candidates"),
        index_path=Path("registry/strategies/candidates/index.jsonl"),
    )
    assert ".." not in str(manifest_path)
    assert manifest_path.is_file()


def test_candidate_ranking_compares_candidates_and_gates_paper_ready(tmp_path):
    from freqtrade_ext.bot_factory.candidate_ranking import (
        CandidateRankingInputs,
        rank_candidates,
        write_candidate_ranking_artifacts,
    )

    full_manifest = tmp_path / "full.json"
    full_manifest.write_text(json.dumps({
        "candidate_id": "full-pass",
        "strategy_name": "S",
        "recommendation": "pass",
        "recommendation_rationale": "full chain passed",
        "thesis": {"thesis_id": "TH-TREND", "thesis_type": "trend_continuation"},
        "strategy_logic_variant": "trend_continuation",
        "checks": [
            {"name": "historical_backtest", "status": "pass", "path": "bt.json", "payload_summary": {"total_return_pct": 5.0, "trade_count": 250, "max_drawdown_pct": 4.0, "profit_factor": 1.6}},
            {"name": "historical_strategy_identity", "status": "pass", "path": "bt.json"},
            {"name": "historical_trades_export", "status": "pass", "path": "trades.csv"},
            {"name": "historical_markdown_report", "status": "pass", "path": "report.md"},
            {"name": "walk_forward", "status": "pass", "path": "wf.json", "payload_summary": {"summary": {"pass_rate": 1.0, "profitable_windows_ratio": 1.0, "total_return_pct": 4.0, "max_single_window_profit_dependency": 0.3}}},
            {"name": "walk_forward_strategy_identity", "status": "pass", "path": "wf.json"},
            {"name": "walk_forward_markdown_report", "status": "pass", "path": "wf.md"},
            {"name": "training_factory", "status": "pass", "path": "train.json", "payload_summary": {"summary": {"stage_count": 1, "failed_stages": 0}}},
            {"name": "training_strategy_identity", "status": "pass", "path": "train.json"},
            {"name": "training_markdown_report", "status": "pass", "path": "train.md"},
        ],
    }), encoding="utf-8")
    partial_manifest = tmp_path / "partial.json"
    partial_manifest.write_text(json.dumps({
        "candidate_id": "partial-pass",
        "strategy_name": "S",
        "recommendation": "pass",
        "recommendation_rationale": "rule chain passed without training",
        "thesis": {"thesis_id": "TH-MR", "thesis_type": "mean_reversion"},
        "strategy_logic_variant": "mean_reversion_pullback",
        "checks": [
            {"name": "historical_backtest", "status": "pass", "path": "bt.json", "payload_summary": {"total_return_pct": 6.0, "trade_count": 250, "max_drawdown_pct": 4.0}},
            {"name": "historical_strategy_identity", "status": "pass", "path": "bt.json"},
            {"name": "historical_trades_export", "status": "pass", "path": "trades.csv"},
            {"name": "historical_markdown_report", "status": "pass", "path": "report.md"},
            {"name": "walk_forward", "status": "pass", "path": "wf.json", "payload_summary": {"summary": {"pass_rate": 1.0, "profitable_windows_ratio": 1.0, "total_return_pct": 4.0}}},
            {"name": "walk_forward_strategy_identity", "status": "pass", "path": "wf.json"},
            {"name": "walk_forward_markdown_report", "status": "pass", "path": "wf.md"},
            {"name": "training_factory", "status": "skipped", "path": None},
            {"name": "training_strategy_identity", "status": "skipped", "path": None},
            {"name": "training_markdown_report", "status": "skipped", "path": None},
        ],
    }), encoding="utf-8")

    ranking = rank_candidates(CandidateRankingInputs(
        root_dir=tmp_path,
        candidate_manifest_paths=[partial_manifest, full_manifest],
        reviewer_notes=["Ranking test only; no paper process."],
    ))
    ranking_path, report_path = write_candidate_ranking_artifacts(
        ranking,
        root_dir=tmp_path,
        output_root=Path("rankings"),
    )

    assert ranking["hypothesis_diversity"]["passed"] is True
    assert set(ranking["paper_ready_candidate_ids"]) == {"full-pass", "partial-pass"}
    partial = next(
        item for item in ranking["ranked_candidates"] if item["candidate_id"] == "partial-pass"
    )
    assert partial["paper_ready_eligible"] is True
    assert partial["paper_ready_blockers"] == []
    assert ranking_path.is_file()
    assert report_path.is_file()


def test_candidate_ranking_blocks_paper_ready_without_hypothesis_diversity(tmp_path):
    from freqtrade_ext.bot_factory.candidate_ranking import (
        CandidateRankingInputs,
        rank_candidates,
    )

    manifest_path = tmp_path / "single_family.json"
    manifest_path.write_text(json.dumps({
        "candidate_id": "full-pass",
        "strategy_name": "S",
        "recommendation": "pass",
        "recommendation_rationale": "full chain passed",
        "thesis": {"thesis_id": "TH-MR-1", "thesis_type": "mean_reversion"},
        "strategy_logic_variant": "mean_reversion_pullback",
        "checks": [
            {"name": "historical_backtest", "status": "pass", "path": "bt.json", "payload_summary": {"total_return_pct": 5.0, "trade_count": 250, "max_drawdown_pct": 4.0, "profit_factor": 1.6}},
            {"name": "historical_strategy_identity", "status": "pass", "path": "bt.json"},
            {"name": "historical_trades_export", "status": "pass", "path": "trades.csv"},
            {"name": "historical_markdown_report", "status": "pass", "path": "report.md"},
            {"name": "walk_forward", "status": "pass", "path": "wf.json", "payload_summary": {"summary": {"pass_rate": 1.0, "profitable_windows_ratio": 1.0, "total_return_pct": 4.0, "max_single_window_profit_dependency": 0.3}}},
            {"name": "walk_forward_strategy_identity", "status": "pass", "path": "wf.json"},
            {"name": "walk_forward_markdown_report", "status": "pass", "path": "wf.md"},
            {"name": "training_factory", "status": "pass", "path": "train.json", "payload_summary": {"summary": {"stage_count": 1, "failed_stages": 0}}},
            {"name": "training_strategy_identity", "status": "pass", "path": "train.json"},
            {"name": "training_markdown_report", "status": "pass", "path": "train.md"},
        ],
    }), encoding="utf-8")

    ranking = rank_candidates(CandidateRankingInputs(
        root_dir=tmp_path,
        candidate_manifest_paths=[manifest_path],
    ))

    assert ranking["hypothesis_diversity"]["passed"] is False
    assert ranking["paper_ready_candidate_ids"] == []
    candidate = ranking["ranked_candidates"][0]
    assert candidate["paper_ready_eligible"] is False
    assert "hypothesis_diversity" in candidate["paper_ready_blockers"]


def test_candidate_ranking_preserves_research_handoff_context(tmp_path):
    from freqtrade_ext.bot_factory.candidate_ranking import (
        CandidateRankingInputs,
        rank_candidates,
    )

    question_handoff = {
        "required": True,
        "passed": False,
        "computed_missing_research_question_response_indexes": [2],
    }
    novelty_handoff = {
        "passed": False,
        "failed_thesis_ids": ["TH-LOCAL-REJECTED"],
        "blocked_next_actions": ["retry_validated_local_rejection_by_parameter_tuning"],
    }
    manifest_path = tmp_path / "handoff.json"
    manifest_path.write_text(json.dumps({
        "candidate_id": "handoff-candidate",
        "strategy_name": "S",
        "recommendation": "retry",
        "recommendation_rationale": "needs distinct local falsification response",
        "thesis": {"thesis_id": "TH-HANDOFF", "thesis_type": "mean_reversion"},
        "research_brief": {
            "thesis_id": "TH-HANDOFF",
            "research_references": [
                {
                    "reference_id": "paper:handoff",
                    "title": "Handoff paper",
                    "source": "journal",
                    "published_at": "2024-01-01",
                    "relevance": "tests preserved handoff context",
                }
            ],
            "blocked_next_actions": [
                "retry_validated_local_rejection_by_parameter_tuning",
            ],
            "research_decision_question_handoff": question_handoff,
            "research_decision_novelty_handoff": novelty_handoff,
        },
        "next_candidate_input": {
            "blocked_next_actions": ["parameter_only_threshold_loosen"],
        },
        "checks": [],
    }), encoding="utf-8")

    ranking = rank_candidates(CandidateRankingInputs(
        root_dir=tmp_path,
        candidate_manifest_paths=[manifest_path],
    ))

    candidate = ranking["ranked_candidates"][0]
    assert candidate["blocked_next_actions"] == [
        "retry_validated_local_rejection_by_parameter_tuning",
        "parameter_only_threshold_loosen",
    ]
    assert candidate["research_brief"]["research_decision_question_handoff"] == (
        question_handoff
    )
    assert candidate["research_handoff_summary"] == {
        "research_decision_question_handoff": question_handoff,
        "research_decision_novelty_handoff": novelty_handoff,
    }


def _regime_contract(
    *,
    intended_regimes: list[str] | None = None,
    excluded_regimes: list[str] | None = None,
    no_trade_conditions: list[str] | None = None,
    risk_policy_version: str = "risk_v1",
    maximum_drawdown_by_regime: dict[str, float] | None = None,
):
    from freqtrade_ext.bot_factory.regime_promotion import RegimeStrategyContract

    return RegimeStrategyContract(
        strategy_version="strategy_v1",
        signal_version="signal_v1",
        risk_policy_version=risk_policy_version,
        regime_classifier_version="regime_classifier_v1",
        cost_model_id="cost_model_v1",
        intended_regimes=intended_regimes or ["trend_up"],
        excluded_regimes=excluded_regimes if excluded_regimes is not None else ["unknown"],
        activation_conditions=["closed candle regime label matches intended scope"],
        no_trade_conditions=(
            no_trade_conditions
            if no_trade_conditions is not None
            else ["missing required feature", "excluded regime active"]
        ),
        regime_shift_stop_conditions=["regime label leaves intended scope"],
        required_features=["close", "volume"],
        minimum_evidence={"min_window_count": 2, "min_trade_count": 10},
        maximum_drawdown_by_regime=maximum_drawdown_by_regime or {"trend_up": 8.0},
        cost_sensitivity_limits={"normal_cost_bps": 10.0, "stress_cost_bps": 20.0},
        cooldown_after_regime_change=3,
        allowed_pairs=["BTC/USDT:USDT", "ETH/USDT:USDT"],
        allowed_timeframes=["5m"],
    )


def _regime_observation(
    observation_id: str,
    *,
    candidate_id: str = "candidate",
    strategy_id: str = "strategy",
    strategy_class_name: str = "FixtureStrategy",
    strategy_source_path: str = "tests/fixtures/FixtureStrategy.py",
    strategy_version: str = "strategy_v1",
    signal_version: str = "signal_v1",
    risk_policy_version: str = "risk_v1",
    regime_classifier_version: str = "regime_classifier_v1",
    cost_model_id: str = "cost_model_v1",
    allowed_pairs: list[str] | None = None,
    allowed_timeframes: list[str] | None = None,
    candidate_identity: dict[str, Any] | None = None,
    source_type: str = "walk_forward",
    regime: str = "trend_up",
    baseline_id: str = "candidate",
    pair: str = "BTC/USDT:USDT",
    timeframe: str = "5m",
    window_start: str = "2026-01-01T00:00:00+00:00",
    window_end: str = "2026-02-01T00:00:00+00:00",
    trade_count: int = 12,
    net_return_normal_cost: float = 2.0,
    net_return_stress_cost: float = 1.0,
    gross_return: float = 3.0,
    max_drawdown: float = 2.0,
    downside_deviation: float = 0.2,
    lower_confidence_bound: float = 0.3,
):
    from freqtrade_ext.bot_factory.candidate_identity import build_strategy_candidate_identity

    identity = candidate_identity or build_strategy_candidate_identity(
        candidate_id=candidate_id,
        strategy_id=strategy_id,
        strategy_class_name=strategy_class_name,
        strategy_source_path=strategy_source_path,
        strategy_version=strategy_version,
        signal_version=signal_version,
        risk_policy_version=risk_policy_version,
        regime_classifier_version=regime_classifier_version,
        cost_model_id=cost_model_id,
        allowed_pairs=allowed_pairs or ["BTC/USDT:USDT", "ETH/USDT:USDT"],
        allowed_timeframes=allowed_timeframes or ["5m"],
        created_at="2026-05-20T00:00:00+00:00",
        source_artifacts={"test_fixture": "tests/test_bot_factory.py"},
    )
    return {
        "observation_id": observation_id,
        "created_at": "2026-05-20T00:00:00+00:00",
        "source_type": source_type,
        "strategy_id": strategy_id,
        "strategy_version": strategy_version,
        "candidate_id": candidate_id,
        "candidate_identity": identity,
        "signal_version": signal_version,
        "risk_policy_version": risk_policy_version,
        "pair": pair,
        "timeframe": timeframe,
        "window_start": window_start,
        "window_end": window_end,
        "market_regime": regime,
        "regime_classifier_version": regime_classifier_version,
        "baseline_id": baseline_id,
        "cost_model_id": cost_model_id,
        "normal_cost_bps": 10.0,
        "stress_cost_bps": 20.0,
        "trade_count": trade_count,
        "exposure_ratio": 0.2,
        "gross_return": gross_return,
        "net_return_normal_cost": net_return_normal_cost,
        "net_return_stress_cost": net_return_stress_cost,
        "max_drawdown": max_drawdown,
        "downside_deviation": downside_deviation,
        "win_rate": 0.55,
        "profit_factor": 1.4,
        "no_trade_reason": "",
        "no_trade_opportunity_cost": max(net_return_normal_cost, 0.0),
        "data_quality_flags": [],
        "reason_codes": ["pass"],
        "lower_confidence_bound": lower_confidence_bound,
    }


def _check_by_name(checks: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(check for check in checks if check["name"] == name)


def _selector_candidate_for_logic(
    logic,
    *,
    candidate_id: str,
    regime: str,
    normal_returns: tuple[float, float] = (6.0, 5.0),
    stress_returns: tuple[float, float] = (4.0, 3.0),
    lower_confidence_bounds: tuple[float, float] = (0.8, 0.6),
):
    from freqtrade_ext.bot_factory.regime_promotion import (
        RegimePromotionThresholds,
        build_regime_fitness_scorecard,
        candidate_identity_from_logic_spec,
        contract_from_logic_spec,
        selection_candidate_from_scorecard,
    )

    contract = contract_from_logic_spec(logic)
    identity = candidate_identity_from_logic_spec(logic, candidate_id=candidate_id)
    observations = [
        _regime_observation(
            f"{candidate_id}-btc",
            candidate_id=candidate_id,
            strategy_id=logic.strategy_id,
            strategy_class_name=logic.strategy_class_name,
            strategy_source_path=logic.strategy_source_path,
            strategy_version=logic.strategy_version,
            signal_version=logic.signal_version,
            risk_policy_version=logic.risk_policy_version,
            regime_classifier_version=logic.regime_classifier_version,
            cost_model_id=logic.cost_model_id,
            allowed_pairs=list(logic.allowed_pairs),
            allowed_timeframes=list(logic.allowed_timeframes),
            candidate_identity=identity,
            regime=regime,
            pair="BTC/USDT:USDT",
            window_start="2026-01-01T00:00:00+00:00",
            window_end="2026-02-01T00:00:00+00:00",
            net_return_normal_cost=normal_returns[0],
            net_return_stress_cost=stress_returns[0],
            lower_confidence_bound=lower_confidence_bounds[0],
        ),
        _regime_observation(
            f"{candidate_id}-eth",
            candidate_id=candidate_id,
            strategy_id=logic.strategy_id,
            strategy_class_name=logic.strategy_class_name,
            strategy_source_path=logic.strategy_source_path,
            strategy_version=logic.strategy_version,
            signal_version=logic.signal_version,
            risk_policy_version=logic.risk_policy_version,
            regime_classifier_version=logic.regime_classifier_version,
            cost_model_id=logic.cost_model_id,
            allowed_pairs=list(logic.allowed_pairs),
            allowed_timeframes=list(logic.allowed_timeframes),
            candidate_identity=identity,
            regime=regime,
            pair="ETH/USDT:USDT",
            window_start="2026-02-01T00:00:00+00:00",
            window_end="2026-03-05T00:00:00+00:00",
            net_return_normal_cost=normal_returns[1],
            net_return_stress_cost=stress_returns[1],
            lower_confidence_bound=lower_confidence_bounds[1],
        ),
    ]
    scorecard = build_regime_fitness_scorecard(
        observations,
        contract=contract,
        baseline_observations=[],
        thresholds=RegimePromotionThresholds(max_calendar_concentration=0.5),
    )
    assert scorecard["decision"] == "REGIME_SCOPED_SELECTOR_ELIGIBLE"
    return selection_candidate_from_scorecard(
        logic=logic,
        scorecard=scorecard,
        candidate_id=candidate_id,
    )


def _passing_feature_quality_report(required_features):
    from freqtrade_ext.bot_factory.feature_quality import build_feature_quality_report

    data: dict[str, Any] = {
        "date": pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC"),
    }
    for feature in required_features:
        data[str(feature)] = [1.0, 1.0, 1.0, 1.0]
    return build_feature_quality_report(
        pd.DataFrame(data),
        required_features=required_features,
        now=datetime(2026, 1, 1, 0, 20, tzinfo=UTC),
        classifier_confidence=0.9,
        cost_model_updated_at="2026-01-01T00:15:00+00:00",
    )


def test_regime_observation_rejects_future_dry_run_in_current_scope():
    from freqtrade_ext.bot_factory.regime_promotion import validate_observation_record

    observation = _regime_observation(
        "future-dry-run",
        source_type="future_dry_run",
        regime="trend_up",
    )

    result = validate_observation_record(observation)

    assert result["ok"] is False
    assert _check_by_name(result["checks"], "source_type_current_scope_allowed")[
        "passed"
    ] is False
    assert result["safety_scope"]["dry_run_trading_started"] is False
    assert result["safety_scope"]["process_control"] is False


def test_regime_scorecard_scopes_range_strategy_without_global_eligibility():
    from freqtrade_ext.bot_factory.regime_promotion import (
        RegimePromotionThresholds,
        build_regime_fitness_scorecard,
    )

    contract = _regime_contract(
        intended_regimes=["range"],
        excluded_regimes=["trend_up", "high_volatility"],
    )
    observations = [
        _regime_observation(
            "range-btc",
            regime="range",
            pair="BTC/USDT:USDT",
            window_start="2026-01-01T00:00:00+00:00",
            window_end="2026-02-01T00:00:00+00:00",
            net_return_normal_cost=4.0,
            net_return_stress_cost=2.5,
            lower_confidence_bound=0.8,
        ),
        _regime_observation(
            "range-eth",
            regime="range",
            pair="ETH/USDT:USDT",
            window_start="2026-02-01T00:00:00+00:00",
            window_end="2026-03-05T00:00:00+00:00",
            net_return_normal_cost=3.0,
            net_return_stress_cost=1.5,
            lower_confidence_bound=0.5,
        ),
    ]
    baselines = [
        _regime_observation(
            "range-btc-no-trade",
            regime="range",
            baseline_id="no_trade",
            pair="BTC/USDT:USDT",
            window_start="2026-01-01T00:00:00+00:00",
            window_end="2026-02-01T00:00:00+00:00",
            trade_count=0,
            net_return_normal_cost=0.0,
            net_return_stress_cost=0.0,
        ),
        _regime_observation(
            "range-eth-no-trade",
            regime="range",
            baseline_id="no_trade",
            pair="ETH/USDT:USDT",
            window_start="2026-02-01T00:00:00+00:00",
            window_end="2026-03-05T00:00:00+00:00",
            trade_count=0,
            net_return_normal_cost=0.0,
            net_return_stress_cost=0.0,
        ),
    ]

    scorecard = build_regime_fitness_scorecard(
        observations,
        contract=contract,
        baseline_observations=baselines,
        thresholds=RegimePromotionThresholds(max_calendar_concentration=0.5),
    )

    assert scorecard["decision"] == "REGIME_SCOPED_SELECTOR_ELIGIBLE"
    assert scorecard["eligible_regimes"] == ["range"]
    assert scorecard["raw_aggregate_pnl_promotion_allowed"] is False
    assert scorecard["phase3_readiness_required_after_scorecard"] is True
    assert scorecard["safety_scope"]["promotion_authorized_by_this_command"] is False


def test_strong_uptrend_logic_selected_in_assumed_production_when_regime_matches():
    from freqtrade_ext.bot_factory.regime_promotion import (
        RegimePromotionThresholds,
        RuntimeRegimeSnapshot,
        build_regime_fitness_scorecard,
        candidate_identity_from_logic_spec,
        contract_from_logic_spec,
        evaluate_runtime_strategy_selection,
        selection_candidate_from_scorecard,
        strong_uptrend_momentum_logic_spec,
    )

    logic = strong_uptrend_momentum_logic_spec()
    contract = contract_from_logic_spec(logic)
    candidate_id = "strong-uptrend-candidate"
    identity = candidate_identity_from_logic_spec(logic, candidate_id=candidate_id)
    observations = [
        _regime_observation(
            "trend-up-btc",
            candidate_id=candidate_id,
            strategy_id=logic.strategy_id,
            strategy_class_name=logic.strategy_class_name,
            strategy_source_path=logic.strategy_source_path,
            strategy_version=logic.strategy_version,
            signal_version=logic.signal_version,
            risk_policy_version=logic.risk_policy_version,
            regime_classifier_version=logic.regime_classifier_version,
            cost_model_id=logic.cost_model_id,
            allowed_pairs=list(logic.allowed_pairs),
            allowed_timeframes=list(logic.allowed_timeframes),
            candidate_identity=identity,
            regime="trend_up",
            pair="BTC/USDT:USDT",
            window_start="2026-01-01T00:00:00+00:00",
            window_end="2026-02-01T00:00:00+00:00",
            net_return_normal_cost=9.0,
            net_return_stress_cost=6.0,
            lower_confidence_bound=1.5,
        ),
        _regime_observation(
            "trend-up-eth",
            candidate_id=candidate_id,
            strategy_id=logic.strategy_id,
            strategy_class_name=logic.strategy_class_name,
            strategy_source_path=logic.strategy_source_path,
            strategy_version=logic.strategy_version,
            signal_version=logic.signal_version,
            risk_policy_version=logic.risk_policy_version,
            regime_classifier_version=logic.regime_classifier_version,
            cost_model_id=logic.cost_model_id,
            allowed_pairs=list(logic.allowed_pairs),
            allowed_timeframes=list(logic.allowed_timeframes),
            candidate_identity=identity,
            regime="trend_up",
            pair="ETH/USDT:USDT",
            window_start="2026-02-01T00:00:00+00:00",
            window_end="2026-03-05T00:00:00+00:00",
            net_return_normal_cost=7.5,
            net_return_stress_cost=4.0,
            lower_confidence_bound=1.0,
        ),
    ]
    no_trade_baseline = [
        _regime_observation(
            "trend-up-btc-no-trade",
            candidate_id=candidate_id,
            strategy_id=logic.strategy_id,
            strategy_class_name=logic.strategy_class_name,
            strategy_source_path=logic.strategy_source_path,
            strategy_version=logic.strategy_version,
            signal_version=logic.signal_version,
            risk_policy_version=logic.risk_policy_version,
            regime_classifier_version=logic.regime_classifier_version,
            cost_model_id=logic.cost_model_id,
            allowed_pairs=list(logic.allowed_pairs),
            allowed_timeframes=list(logic.allowed_timeframes),
            candidate_identity=identity,
            regime="trend_up",
            baseline_id="no_trade",
            pair="BTC/USDT:USDT",
            window_start="2026-01-01T00:00:00+00:00",
            window_end="2026-02-01T00:00:00+00:00",
            trade_count=0,
            net_return_normal_cost=0.0,
            net_return_stress_cost=0.0,
        ),
        _regime_observation(
            "trend-up-eth-no-trade",
            candidate_id=candidate_id,
            strategy_id=logic.strategy_id,
            strategy_class_name=logic.strategy_class_name,
            strategy_source_path=logic.strategy_source_path,
            strategy_version=logic.strategy_version,
            signal_version=logic.signal_version,
            risk_policy_version=logic.risk_policy_version,
            regime_classifier_version=logic.regime_classifier_version,
            cost_model_id=logic.cost_model_id,
            allowed_pairs=list(logic.allowed_pairs),
            allowed_timeframes=list(logic.allowed_timeframes),
            candidate_identity=identity,
            regime="trend_up",
            baseline_id="no_trade",
            pair="ETH/USDT:USDT",
            window_start="2026-02-01T00:00:00+00:00",
            window_end="2026-03-05T00:00:00+00:00",
            trade_count=0,
            net_return_normal_cost=0.0,
            net_return_stress_cost=0.0,
        ),
    ]
    scorecard = build_regime_fitness_scorecard(
        observations,
        contract=contract,
        baseline_observations=no_trade_baseline,
        thresholds=RegimePromotionThresholds(max_calendar_concentration=0.5),
    )
    candidate = selection_candidate_from_scorecard(
        logic=logic,
        scorecard=scorecard,
        candidate_id=candidate_id,
    )
    runtime = RuntimeRegimeSnapshot(
        current_regime="trend_up",
        pair="BTC/USDT:USDT",
        timeframe="5m",
        regime_classifier_version="regime_classifier_v1",
        data_quality_pass=True,
        available_features=[
            "close",
            "volume",
            "moving_average_slope",
            "range_efficiency",
            "regime_label",
            "cost_model",
        ],
        feature_quality_report=_passing_feature_quality_report(
            [
                "close",
                "volume",
                "moving_average_slope",
                "range_efficiency",
                "regime_label",
                "cost_model",
            ]
        ),
        production_assumption=True,
    )

    selection = evaluate_runtime_strategy_selection(
        runtime=runtime,
        candidates=[candidate],
        selector_id="assumed_production_selector_test",
    )

    assert logic.logic_id == "strong_uptrend_momentum_v1"
    assert scorecard["decision"] == "REGIME_SCOPED_SELECTOR_ELIGIBLE"
    assert selection["action"] == "select"
    assert selection["selected_candidate_id"] == "strong-uptrend-candidate"
    assert selection["selected_logic_id"] == "strong_uptrend_momentum_v1"
    assert selection["would_select_in_production_assumption"] is True
    assert selection["safety_scope"]["process_control"] is False
    assert selection["safety_scope"]["dry_run_trading_started"] is False


def test_strong_uptrend_logic_not_selected_when_runtime_regime_changes_to_range():
    from freqtrade_ext.bot_factory.regime_promotion import (
        RegimePromotionThresholds,
        RuntimeRegimeSnapshot,
        build_regime_fitness_scorecard,
        candidate_identity_from_logic_spec,
        contract_from_logic_spec,
        evaluate_runtime_strategy_selection,
        selection_candidate_from_scorecard,
        strong_uptrend_momentum_logic_spec,
    )

    logic = strong_uptrend_momentum_logic_spec()
    contract = contract_from_logic_spec(logic)
    candidate_id = "strong-uptrend-candidate"
    identity = candidate_identity_from_logic_spec(logic, candidate_id=candidate_id)
    scorecard = build_regime_fitness_scorecard(
        [
            _regime_observation(
                "trend-up-btc",
                candidate_id=candidate_id,
                strategy_id=logic.strategy_id,
                strategy_class_name=logic.strategy_class_name,
                strategy_source_path=logic.strategy_source_path,
                strategy_version=logic.strategy_version,
                signal_version=logic.signal_version,
                risk_policy_version=logic.risk_policy_version,
                regime_classifier_version=logic.regime_classifier_version,
                cost_model_id=logic.cost_model_id,
                allowed_pairs=list(logic.allowed_pairs),
                allowed_timeframes=list(logic.allowed_timeframes),
                candidate_identity=identity,
                regime="trend_up",
                pair="BTC/USDT:USDT",
                window_start="2026-01-01T00:00:00+00:00",
                window_end="2026-02-01T00:00:00+00:00",
                net_return_normal_cost=9.0,
                net_return_stress_cost=6.0,
                lower_confidence_bound=1.5,
            ),
            _regime_observation(
                "trend-up-eth",
                candidate_id=candidate_id,
                strategy_id=logic.strategy_id,
                strategy_class_name=logic.strategy_class_name,
                strategy_source_path=logic.strategy_source_path,
                strategy_version=logic.strategy_version,
                signal_version=logic.signal_version,
                risk_policy_version=logic.risk_policy_version,
                regime_classifier_version=logic.regime_classifier_version,
                cost_model_id=logic.cost_model_id,
                allowed_pairs=list(logic.allowed_pairs),
                allowed_timeframes=list(logic.allowed_timeframes),
                candidate_identity=identity,
                regime="trend_up",
                pair="ETH/USDT:USDT",
                window_start="2026-02-01T00:00:00+00:00",
                window_end="2026-03-05T00:00:00+00:00",
                net_return_normal_cost=7.5,
                net_return_stress_cost=4.0,
                lower_confidence_bound=1.0,
            ),
        ],
        contract=contract,
        baseline_observations=[],
        thresholds=RegimePromotionThresholds(max_calendar_concentration=0.5),
    )
    candidate = selection_candidate_from_scorecard(
        logic=logic,
        scorecard=scorecard,
        candidate_id=candidate_id,
    )
    runtime = RuntimeRegimeSnapshot(
        current_regime="range",
        pair="BTC/USDT:USDT",
        timeframe="5m",
        regime_classifier_version="regime_classifier_v1",
        data_quality_pass=True,
        available_features=[
            "close",
            "volume",
            "moving_average_slope",
            "range_efficiency",
            "regime_label",
            "cost_model",
        ],
        production_assumption=True,
    )

    selection = evaluate_runtime_strategy_selection(runtime=runtime, candidates=[candidate])

    assert selection["action"] == "no_trade"
    assert selection["selected_candidate_id"] is None
    assert selection["would_select_in_production_assumption"] is False
    evaluated = selection["evaluated_candidates"][0]
    assert "runtime_regime_eligible" in evaluated["reason_codes"]


def test_selector_chooses_regime_matching_logic_from_multiple_candidates():
    from freqtrade_ext.bot_factory.regime_promotion import (
        RuntimeRegimeSnapshot,
        downtrend_defensive_rebound_logic_spec,
        evaluate_runtime_strategy_selection,
        range_mean_reversion_logic_spec,
        strong_uptrend_momentum_logic_spec,
    )

    uptrend_logic = strong_uptrend_momentum_logic_spec()
    downtrend_logic = downtrend_defensive_rebound_logic_spec()
    range_logic = range_mean_reversion_logic_spec()
    candidates = [
        _selector_candidate_for_logic(
            uptrend_logic,
            candidate_id="uptrend-candidate",
            regime="trend_up",
            normal_returns=(9.0, 7.0),
            stress_returns=(6.0, 4.0),
            lower_confidence_bounds=(1.5, 1.0),
        ),
        _selector_candidate_for_logic(
            downtrend_logic,
            candidate_id="downtrend-candidate",
            regime="trend_down",
            normal_returns=(5.5, 4.5),
            stress_returns=(3.5, 2.5),
            lower_confidence_bounds=(0.8, 0.6),
        ),
        _selector_candidate_for_logic(
            range_logic,
            candidate_id="range-candidate",
            regime="range",
            normal_returns=(4.5, 4.0),
            stress_returns=(3.0, 2.5),
            lower_confidence_bounds=(0.7, 0.5),
        ),
    ]
    all_features = sorted({
        feature
        for logic in (uptrend_logic, downtrend_logic, range_logic)
        for feature in logic.required_features
    })

    downtrend_selection = evaluate_runtime_strategy_selection(
        runtime=RuntimeRegimeSnapshot(
            current_regime="trend_down",
            pair="BTC/USDT:USDT",
            timeframe="5m",
            regime_classifier_version="regime_classifier_v1",
            data_quality_pass=True,
            available_features=all_features,
            feature_quality_report=_passing_feature_quality_report(all_features),
            production_assumption=True,
        ),
        candidates=candidates,
        selector_id="multi_regime_selector_test_downtrend",
    )
    range_selection = evaluate_runtime_strategy_selection(
        runtime=RuntimeRegimeSnapshot(
            current_regime="range",
            pair="BTC/USDT:USDT",
            timeframe="5m",
            regime_classifier_version="regime_classifier_v1",
            data_quality_pass=True,
            available_features=all_features,
            feature_quality_report=_passing_feature_quality_report(all_features),
            production_assumption=True,
        ),
        candidates=candidates,
        selector_id="multi_regime_selector_test_range",
    )
    uptrend_selection = evaluate_runtime_strategy_selection(
        runtime=RuntimeRegimeSnapshot(
            current_regime="trend_up",
            pair="BTC/USDT:USDT",
            timeframe="5m",
            regime_classifier_version="regime_classifier_v1",
            data_quality_pass=True,
            available_features=all_features,
            feature_quality_report=_passing_feature_quality_report(all_features),
            production_assumption=True,
        ),
        candidates=candidates,
        selector_id="multi_regime_selector_test_uptrend",
    )

    assert downtrend_logic.logic_id == "downtrend_defensive_rebound_v1"
    assert range_logic.logic_id == "range_mean_reversion_v1"
    assert downtrend_selection["action"] == "select"
    assert downtrend_selection["selected_candidate_id"] == "downtrend-candidate"
    assert downtrend_selection["selected_logic_id"] == "downtrend_defensive_rebound_v1"
    assert range_selection["action"] == "select"
    assert range_selection["selected_candidate_id"] == "range-candidate"
    assert range_selection["selected_logic_id"] == "range_mean_reversion_v1"
    assert uptrend_selection["action"] == "select"
    assert uptrend_selection["selected_candidate_id"] == "uptrend-candidate"
    assert uptrend_selection["selected_logic_id"] == "strong_uptrend_momentum_v1"
    rejected_uptrend = next(
        item
        for item in downtrend_selection["evaluated_candidates"]
        if item["candidate_id"] == "uptrend-candidate"
    )
    assert "runtime_regime_eligible" in rejected_uptrend["reason_codes"]


def test_selector_ranks_same_regime_candidates_by_stress_adjusted_score():
    from freqtrade_ext.bot_factory.regime_promotion import (
        RuntimeRegimeSnapshot,
        evaluate_runtime_strategy_selection,
        range_mean_reversion_logic_spec,
    )

    higher_normal_logic = range_mean_reversion_logic_spec(
        strategy_id="long_only_range_mean_reversion_higher_normal",
        strategy_version="range_mean_reversion_higher_normal_v1",
    )
    robust_logic = range_mean_reversion_logic_spec(
        strategy_id="long_only_range_mean_reversion_robust",
        strategy_version="range_mean_reversion_robust_v1",
    )
    higher_normal_candidate = _selector_candidate_for_logic(
        higher_normal_logic,
        candidate_id="range-higher-normal-candidate",
        regime="range",
        normal_returns=(10.0, 9.0),
        stress_returns=(1.5, 1.0),
        lower_confidence_bounds=(0.3, 0.2),
    )
    robust_candidate = _selector_candidate_for_logic(
        robust_logic,
        candidate_id="range-robust-candidate",
        regime="range",
        normal_returns=(8.0, 7.0),
        stress_returns=(5.0, 4.5),
        lower_confidence_bounds=(0.9, 0.8),
    )

    selection = evaluate_runtime_strategy_selection(
        runtime=RuntimeRegimeSnapshot(
            current_regime="range",
            pair="BTC/USDT:USDT",
            timeframe="5m",
            regime_classifier_version="regime_classifier_v1",
            data_quality_pass=True,
            available_features=sorted(robust_logic.required_features),
            feature_quality_report=_passing_feature_quality_report(
                sorted(robust_logic.required_features)
            ),
            production_assumption=True,
        ),
        candidates=[higher_normal_candidate, robust_candidate],
        selector_id="same_regime_selector_rank_test",
    )

    summaries = {
        item["candidate_id"]: item["scorecard_summary"]
        for item in selection["evaluated_candidates"]
    }
    assert selection["action"] == "select"
    assert selection["selected_candidate_id"] == "range-robust-candidate"
    assert "selected_highest_stress_adjusted_candidate" in selection["reason_codes"]
    assert (
        summaries["range-higher-normal-candidate"]["net_pnl_normal_cost"]
        > summaries["range-robust-candidate"]["net_pnl_normal_cost"]
    )
    assert (
        summaries["range-robust-candidate"]["net_pnl_stress_cost"]
        > summaries["range-higher-normal-candidate"]["net_pnl_stress_cost"]
    )


def test_regime_scorecard_blocks_global_when_high_volatility_crashes():
    from freqtrade_ext.bot_factory.regime_promotion import (
        RegimePromotionThresholds,
        build_regime_fitness_scorecard,
    )

    contract = _regime_contract(
        intended_regimes=["range", "high_volatility"],
        excluded_regimes=[],
        maximum_drawdown_by_regime={"range": 8.0, "high_volatility": 5.0},
    )
    observations = [
        _regime_observation(
            "range-btc",
            regime="range",
            pair="BTC/USDT:USDT",
            window_start="2026-01-01T00:00:00+00:00",
            window_end="2026-02-01T00:00:00+00:00",
            net_return_normal_cost=20.0,
            net_return_stress_cost=18.0,
            lower_confidence_bound=2.0,
            max_drawdown=3.0,
        ),
        _regime_observation(
            "range-eth",
            regime="range",
            pair="ETH/USDT:USDT",
            window_start="2026-02-01T00:00:00+00:00",
            window_end="2026-03-05T00:00:00+00:00",
            net_return_normal_cost=16.0,
            net_return_stress_cost=12.0,
            lower_confidence_bound=1.5,
            max_drawdown=3.0,
        ),
        _regime_observation(
            "high-vol-btc",
            regime="high_volatility",
            pair="BTC/USDT:USDT",
            window_start="2026-01-01T00:00:00+00:00",
            window_end="2026-02-01T00:00:00+00:00",
            net_return_normal_cost=-5.0,
            net_return_stress_cost=-8.0,
            lower_confidence_bound=-4.0,
            max_drawdown=18.0,
        ),
        _regime_observation(
            "high-vol-eth",
            regime="high_volatility",
            pair="ETH/USDT:USDT",
            window_start="2026-02-01T00:00:00+00:00",
            window_end="2026-03-05T00:00:00+00:00",
            net_return_normal_cost=-4.0,
            net_return_stress_cost=-7.0,
            lower_confidence_bound=-3.5,
            max_drawdown=16.0,
        ),
    ]

    scorecard = build_regime_fitness_scorecard(
        observations,
        contract=contract,
        baseline_observations=[],
        thresholds=RegimePromotionThresholds(max_calendar_concentration=0.5),
    )

    assert sum(item["net_pnl_normal_cost"] for item in scorecard["scorecard_by_regime"]) > 0
    assert scorecard["decision"] == "SHADOW_ONLY"
    assert "high_volatility" in scorecard["blocked_regimes"]
    assert scorecard["decision"] != "GLOBAL_SELECTOR_ELIGIBLE"


def test_regime_evidence_unit_segments_version_changes():
    from freqtrade_ext.bot_factory.regime_promotion import evidence_unit

    first = _regime_contract(risk_policy_version="risk_v1")
    second = _regime_contract(risk_policy_version="risk_v2")

    assert evidence_unit(first) != evidence_unit(second)
    assert evidence_unit(first)["risk_policy_version"] == "risk_v1"
    assert evidence_unit(second)["risk_policy_version"] == "risk_v2"


def test_candidate_identity_segments_signal_risk_regime_and_cost_versions():
    from freqtrade_ext.bot_factory.candidate_identity import (
        build_strategy_candidate_identity,
        compare_candidate_identities,
    )

    base = build_strategy_candidate_identity(
        candidate_id="cand-a",
        strategy_id="strategy-a",
        strategy_class_name="StrategyA",
        strategy_source_path="user_data/strategies/StrategyA.py",
        strategy_version="strategy_v1",
        signal_version="signal_v1",
        risk_policy_version="risk_v1",
        regime_classifier_version="regime_v1",
        cost_model_id="cost_v1",
        allowed_pairs=["BTC/USDT:USDT"],
        allowed_timeframes=["5m"],
        created_at="2026-05-21T00:00:00+00:00",
        source_artifacts={"strategy_source": "user_data/strategies/StrategyA.py"},
    )

    for field, changed in {
        "signal_version": "signal_v2",
        "risk_policy_version": "risk_v2",
        "regime_classifier_version": "regime_v2",
        "cost_model_id": "cost_v2",
    }.items():
        observed = dict(base)
        observed[field] = changed
        result = compare_candidate_identities(base, observed, observed_label="observed")
        assert result["ok"] is False
        assert {item["field"] for item in result["mismatches"]} == {field}


def _market_state_ohlcv_frame(dates, close_values):
    close = pd.Series(close_values, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(close), 100.0),
        }
    )


def test_deterministic_regime_classifier_labels_fixed_ohlcv_and_churn_is_bounded():
    from freqtrade_ext.bot_factory.market_regime import (
        RegimeClassifierConfig,
        classify_ohlcv_regimes,
        regime_churn_report,
    )

    dates = pd.date_range("2026-01-01", periods=40, freq="5min", tz="UTC")
    close = pd.Series(np.linspace(100.0, 120.0, 40))
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(40, 100.0),
        }
    )
    config = RegimeClassifierConfig(lookback=6, min_rows=12)

    first = classify_ohlcv_regimes(frame, pair="BTC/USDT:USDT", timeframe="5m", config=config)
    perturbed = frame.copy()
    perturbed["close"] = perturbed["close"] * 1.0001
    second = classify_ohlcv_regimes(
        perturbed, pair="BTC/USDT:USDT", timeframe="5m", config=config
    )
    churn = regime_churn_report(first, second, max_churn_ratio=0.1)

    assert first["regime_classifier_version"] == "deterministic_regime_classifier_v1"
    assert first["label_counts"]["trend_up"] > 0
    assert churn["ok"] is True


def test_market_state_snapshot_writes_multi_horizon_artifacts_and_current_report(tmp_path):
    from freqtrade_ext.bot_factory.market_regime import (
        MarketStateConfig,
        RegimeClassifierConfig,
        build_market_state_snapshot,
        write_market_state_artifacts,
    )

    dates = pd.date_range("2026-01-01", periods=96, freq="5min", tz="UTC")
    frame = _market_state_ohlcv_frame(dates, np.linspace(100.0, 130.0, len(dates)))
    config = MarketStateConfig(
        horizons=("5m", "15m"),
        min_horizon_rows=8,
        max_staleness_seconds=900,
        confidence_threshold=0.45,
        regime_classifier_config=RegimeClassifierConfig(
            lookback=4,
            min_rows=8,
            trend_return_threshold=0.005,
            trend_efficiency_threshold=0.2,
        ),
    )
    now = dates[-1].to_pydatetime() + pd.Timedelta(minutes=5)

    snapshot = build_market_state_snapshot(
        frame,
        pair="BTC/USDT:USDT",
        base_timeframe="5m",
        pair_group="btc_major",
        run_id="market_state_test",
        cost_model_id="cost_v1",
        config=config,
        generated_at="2026-01-01T08:00:00+00:00",
        now=now,
    )
    paths = write_market_state_artifacts(snapshot, output_root=tmp_path / "market_state")
    current = json.loads(paths["current_market_state"].read_text(encoding="utf-8"))
    current_report = paths["current_market_state_report"].read_text(encoding="utf-8")

    assert snapshot["schema_version"] == "market_state_snapshot_v1"
    assert {row["horizon"] for row in snapshot["horizons"]} == {"5m", "15m"}
    assert snapshot["aggregate_label"] == "trend_up"
    assert snapshot["no_trade_default"] is False
    assert snapshot["safety_scope"]["freqtrade_trade_started"] is False
    for row in snapshot["horizons"]:
        assert row["schema_version"] == "market_state_window_v1"
        assert row["future_data_used"] is False
        assert row["feature_cutoff_timestamp"] <= row["decision_window_end"]
        assert "rolling_return_bps" in row["state_vector"]
    assert current["schema_version"] == "current_market_state_v1"
    assert current["cost_model_id"] == "cost_v1"
    assert current["stale_data"] is False
    assert current["not_allowed_confirmation"]["paper_trading_started"] is True
    assert "Cost model: `cost_v1`" in current_report
    assert "Current means current as of local data timestamp" in current_report
    for path in paths.values():
        assert path.exists()


def test_market_state_snapshot_conflicting_horizons_default_to_no_trade():
    from freqtrade_ext.bot_factory.market_regime import (
        MarketStateConfig,
        RegimeClassifierConfig,
        build_market_state_snapshot,
    )

    latest = pd.Timestamp("2026-01-02T03:00:00Z")
    up_dates = pd.date_range(end=latest, periods=40, freq="5min")
    down_dates = pd.date_range(end=latest, periods=40, freq="1h")
    up_frame = _market_state_ohlcv_frame(up_dates, np.linspace(100.0, 125.0, len(up_dates)))
    down_frame = _market_state_ohlcv_frame(
        down_dates, np.linspace(125.0, 90.0, len(down_dates))
    )
    config = MarketStateConfig(
        horizons=("5m", "1h"),
        min_horizon_rows=8,
        max_staleness_seconds=900,
        confidence_threshold=0.45,
        regime_classifier_config=RegimeClassifierConfig(
            lookback=4,
            min_rows=8,
            trend_return_threshold=0.005,
            trend_efficiency_threshold=0.2,
        ),
    )

    snapshot = build_market_state_snapshot(
        up_frame,
        pair="BTC/USDT:USDT",
        base_timeframe="5m",
        run_id="market_state_conflict",
        horizon_frames={"1h": down_frame},
        config=config,
        now=latest.to_pydatetime() + pd.Timedelta(minutes=5),
    )

    assert {row["label"] for row in snapshot["horizons"]} == {"trend_up", "trend_down"}
    assert snapshot["aggregate_label"] == "mixed"
    assert snapshot["no_trade_default"] is True
    assert snapshot["horizon_conflict"]["conflict_detected"] is True
    assert "horizon_conflict" in snapshot["reason_codes"]


def test_market_state_snapshot_stale_local_candles_force_unknown_no_trade():
    from freqtrade_ext.bot_factory.market_regime import (
        MarketStateConfig,
        RegimeClassifierConfig,
        build_current_market_state,
        build_market_state_snapshot,
    )

    dates = pd.date_range("2026-01-01", periods=40, freq="5min", tz="UTC")
    frame = _market_state_ohlcv_frame(dates, np.linspace(100.0, 120.0, len(dates)))
    config = MarketStateConfig(
        horizons=("5m",),
        min_horizon_rows=8,
        max_staleness_seconds=900,
        confidence_threshold=0.45,
        regime_classifier_config=RegimeClassifierConfig(
            lookback=4,
            min_rows=8,
            trend_return_threshold=0.005,
            trend_efficiency_threshold=0.2,
        ),
    )

    snapshot = build_market_state_snapshot(
        frame,
        pair="BTC/USDT:USDT",
        base_timeframe="5m",
        run_id="market_state_stale",
        config=config,
        now=datetime(2026, 1, 2, tzinfo=UTC),
    )
    current = build_current_market_state(snapshot)

    assert snapshot["aggregate_label"] == "unknown"
    assert snapshot["unknown_reason"] == "stale_local_data"
    assert snapshot["no_trade_default"] is True
    assert snapshot["horizons"][0]["label"] == "unknown"
    assert "stale_local_data" in snapshot["horizons"][0]["data_quality_flags"]
    assert "stale_local_data" in snapshot["reason_codes"]
    assert current["stale_data"] is True


def test_market_state_snapshot_drops_incomplete_resampled_higher_timeframe():
    from freqtrade_ext.bot_factory.market_regime import (
        MarketStateConfig,
        RegimeClassifierConfig,
        build_market_state_snapshot,
    )

    dates = pd.date_range("2026-01-01T00:00:00Z", periods=12, freq="5min")
    frame = _market_state_ohlcv_frame(dates, np.linspace(100.0, 112.0, len(dates)))
    config = MarketStateConfig(
        horizons=("5m", "1h"),
        min_horizon_rows=1,
        max_staleness_seconds=900,
        confidence_threshold=0.1,
        regime_classifier_config=RegimeClassifierConfig(
            lookback=1,
            min_rows=1,
            trend_return_threshold=0.001,
            trend_efficiency_threshold=0.1,
        ),
    )

    incomplete = build_market_state_snapshot(
        frame,
        pair="BTC/USDT:USDT",
        base_timeframe="5m",
        run_id="market_state_incomplete_1h",
        config=config,
        now=datetime(2026, 1, 1, 0, 55, tzinfo=UTC),
    )
    closed = build_market_state_snapshot(
        frame,
        pair="BTC/USDT:USDT",
        base_timeframe="5m",
        run_id="market_state_closed_1h",
        config=config,
        now=datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
    )

    assert {row["horizon"] for row in incomplete["horizons"]} == {"5m"}
    closed_1h = next(row for row in closed["horizons"] if row["horizon"] == "1h")
    assert closed_1h["decision_window_end"] == "2026-01-01T01:00:00+00:00"
    assert closed_1h["feature_cutoff_timestamp"] == "2026-01-01T01:00:00+00:00"
    assert closed_1h["future_data_used"] is False


def test_market_state_weekly_resample_uses_monday_anchor():
    from freqtrade_ext.bot_factory.market_regime import _resample_ohlcv

    dates = pd.date_range("2026-01-05T00:00:00Z", periods=7 * 24, freq="1h")
    frame = _market_state_ohlcv_frame(dates, np.linspace(100.0, 120.0, len(dates)))

    resampled = _resample_ohlcv(
        frame,
        "1w",
        asof=datetime(2026, 1, 12, tzinfo=UTC),
    )

    assert [item.isoformat() for item in resampled["date"]] == [
        "2026-01-05T00:00:00+00:00"
    ]


def test_market_state_snapshot_staleness_uses_candle_close_time_for_long_base_timeframe():
    from freqtrade_ext.bot_factory.market_regime import (
        MarketStateConfig,
        RegimeClassifierConfig,
        build_market_state_snapshot,
    )

    dates = pd.date_range(end="2026-01-01T00:00:00Z", periods=24, freq="1h")
    frame = _market_state_ohlcv_frame(dates, np.linspace(100.0, 124.0, len(dates)))
    config = MarketStateConfig(
        horizons=("1h",),
        min_horizon_rows=8,
        max_staleness_seconds=900,
        confidence_threshold=0.45,
        regime_classifier_config=RegimeClassifierConfig(
            lookback=4,
            min_rows=8,
            trend_return_threshold=0.005,
            trend_efficiency_threshold=0.2,
        ),
    )

    snapshot = build_market_state_snapshot(
        frame,
        pair="BTC/USDT:USDT",
        base_timeframe="1h",
        run_id="market_state_1h_fresh_close",
        config=config,
        now=datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
    )

    assert snapshot["latest_local_candle_at"] == "2026-01-01T00:00:00+00:00"
    assert snapshot["latest_local_candle_close_at"] == "2026-01-01T01:00:00+00:00"
    assert snapshot["data_asof"] == "2026-01-01T01:00:00+00:00"
    assert snapshot["data_quality_summary"]["stale_data"] is False
    assert "stale_local_data" not in snapshot["reason_codes"]


def _state_snapshot_for_regime(regime: str = "trend_up", *, confidence: float = 0.9):
    return {
        "factory": "bot_factory",
        "schema_version": "market_state_snapshot_v1",
        "run_id": "state_snapshot",
        "generated_at": "2026-01-01T08:00:00+00:00",
        "data_asof": "2026-01-01T08:00:00+00:00",
        "latest_local_candle_at": "2026-01-01T07:55:00+00:00",
        "latest_local_candle_close_at": "2026-01-01T08:00:00+00:00",
        "pair": "BTC/USDT:USDT",
        "pair_group": "btc_major",
        "base_timeframe": "5m",
        "aggregate_label": regime,
        "state_confidence": confidence,
        "uncertainty": round(1.0 - confidence, 6),
        "out_of_distribution_score": round(1.0 - confidence, 6),
        "no_trade_default": False,
        "horizon_conflict": {"conflict_detected": False, "reason_codes": []},
        "feature_quality_summary": {"feature_quality_pass": True, "flags": []},
        "data_quality_summary": {"stale_data": False, "flags": []},
        "state_encoder_version": "deterministic_market_state_encoder_v1",
        "horizon_profile_id": (
            "deterministic_market_state_encoder_v1:"
            f"micro={regime}:intraday={regime}:swing=missing"
        ),
        "cost_model_id": "cost_model_v1",
        "horizons": [
            {
                "schema_version": "market_state_window_v1",
                "horizon": "5m",
                "horizon_group": "micro",
                "label": regime,
                "state_id": (
                    "deterministic_market_state_encoder_v1:"
                    f"5m:{regime}:{'high' if confidence >= 0.7 else 'low'}:ohlcv_state_features_v1"
                ),
                "state_window_id": f"state_window:{regime}:20260101T075500Z:20260101T080000Z",
                "confidence": confidence,
                "uncertainty": round(1.0 - confidence, 6),
                "out_of_distribution_score": round(1.0 - confidence, 6),
                "feature_cutoff_timestamp": "2026-01-01T08:00:00+00:00",
                "label_cutoff_timestamp": "2026-01-01T08:00:00+00:00",
                "decision_window_start": "2026-01-01T07:55:00+00:00",
                "decision_window_end": "2026-01-01T08:00:00+00:00",
                "future_data_used": False,
                "state_encoder_version": "deterministic_market_state_encoder_v1",
                "reason_codes": [f"label_{regime}"],
            }
        ],
        "reason_codes": [f"{regime}_state"],
        "safety_scope": {
            "freqtrade_trade_started": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading_started": False,
            "exchange_order_placement": False,
            "process_control": False,
        },
    }


def _source_observation_state_scope(regime: str = "trend_up", *, confidence: float = 0.9):
    bucket = "high" if confidence >= 0.7 else "low"
    return {
        "state_id": (
            "deterministic_market_state_encoder_v1:"
            f"5m:{regime}:{bucket}:ohlcv_state_features_v1"
        ),
        "horizon_profile_id": (
            "deterministic_market_state_encoder_v1:"
            f"micro={regime}:intraday={regime}:swing=missing"
        ),
        "state_encoder_version": "deterministic_market_state_encoder_v1",
        "state_window_id": f"state_window:{regime}:20260101T075500Z:20260101T080000Z",
        "feature_cutoff_timestamp": "2026-01-01T08:00:00+00:00",
        "label_cutoff_timestamp": "2026-01-01T08:00:00+00:00",
        "decision_window_start": "2026-01-01T07:55:00+00:00",
        "decision_window_end": "2026-01-01T08:00:00+00:00",
        "future_data_used": False,
    }


def _strict_regime_scorecard_for_state_tests(
    *,
    regime: str = "trend_up",
    candidate_id: str = "candidate",
    strategy_id: str = "strategy",
    strategy_class_name: str = "FixtureStrategy",
    normal_returns: tuple[float, float] = (3.0, 2.0),
    stress_returns: tuple[float, float] = (2.0, 1.2),
    lower_confidence_bounds: tuple[float, float] = (0.5, 0.4),
):
    from freqtrade_ext.bot_factory.regime_promotion import (
        RegimePromotionThresholds,
        build_regime_fitness_scorecard,
    )

    observations = [
        _regime_observation(
            "state-btc",
            candidate_id=candidate_id,
            strategy_id=strategy_id,
            strategy_class_name=strategy_class_name,
            source_type="walk_forward",
            regime=regime,
            pair="BTC/USDT:USDT",
            window_start="2026-01-01T00:00:00+00:00",
            window_end="2026-02-01T00:00:00+00:00",
            net_return_normal_cost=normal_returns[0],
            net_return_stress_cost=stress_returns[0],
            lower_confidence_bound=lower_confidence_bounds[0],
        ),
        _regime_observation(
            "state-eth",
            candidate_id=candidate_id,
            strategy_id=strategy_id,
            strategy_class_name=strategy_class_name,
            source_type="walk_forward",
            regime=regime,
            pair="ETH/USDT:USDT",
            window_start="2026-02-01T00:00:00+00:00",
            window_end="2026-03-01T00:00:00+00:00",
            net_return_normal_cost=normal_returns[1],
            net_return_stress_cost=stress_returns[1],
            lower_confidence_bound=lower_confidence_bounds[1],
        ),
    ]
    for observation in observations:
        observation.update(_source_observation_state_scope(regime))
    scorecard = build_regime_fitness_scorecard(
        observations,
        contract=_regime_contract(
            intended_regimes=[regime],
            excluded_regimes=["unknown"],
            maximum_drawdown_by_regime={regime: 8.0},
        ),
        baseline_observations=[],
        thresholds=RegimePromotionThresholds(
            min_sample_days=0.0,
            min_window_count=1,
            min_trade_count=0,
            min_global_regime_count=1,
            max_calendar_concentration=0.6,
        ),
        candidate_identity=observations[0]["candidate_identity"],
    )
    assert scorecard["decision"] == "REGIME_SCOPED_SELECTOR_ELIGIBLE"
    scorecard["baseline_comparison"] = {
        "by_regime": [
            {
                "market_regime": regime,
                "candidate_return": sum(normal_returns),
                "hold_return": 2.0,
                "no_trade_return": 0.0,
                "hold_delta": sum(normal_returns) - 2.0,
                "no_trade_delta": sum(normal_returns),
            }
        ]
    }
    return scorecard


def test_state_conditioned_scorecard_preserves_state_scope_and_baselines():
    from freqtrade_ext.bot_factory.state_conditioning import (
        build_state_conditioned_scorecard,
        validate_state_conditioned_scorecard_for_selector,
    )

    scorecard = build_state_conditioned_scorecard(
        regime_scorecard=_strict_regime_scorecard_for_state_tests(),
        market_state_snapshot=_state_snapshot_for_regime("trend_up"),
        run_id="state_scorecard_test",
        require_walk_forward_evidence=True,
    )
    validation = validate_state_conditioned_scorecard_for_selector(scorecard)

    assert scorecard["schema_version"] == "state_conditioned_scorecard_v1"
    assert scorecard["evidence_eligibility"] == "selector_eligible_candidate"
    assert scorecard["selector_candidate_creation_allowed"] is True
    assert scorecard["walk_forward_gate_passed"] is True
    assert scorecard["rows"][0]["decision"] == "STATE_SELECTOR_ELIGIBLE"
    assert scorecard["rows"][0]["state_id"].endswith("5m:trend_up:high:ohlcv_state_features_v1")
    assert {row["baseline_id"] for row in scorecard["baseline_comparisons"]} == {
        "hold",
        "no_trade",
    }
    assert validation["ok"] is True
    assert validation["safety_scope"]["paper_trading_started"] is False


def test_state_conditioned_scorecard_preserves_snapshot_pair_timeframe_scope():
    from freqtrade_ext.bot_factory.selector_matching import build_selector_matching_decision
    from freqtrade_ext.bot_factory.state_conditioning import build_state_conditioned_scorecard
    from freqtrade_ext.bot_factory.strategy_suitability import build_strategy_suitability_matrix

    regime_scorecard = _strict_regime_scorecard_for_state_tests()
    regime_scorecard["candidate_identity"]["allowed_pairs"] = [
        "ETH/USDT:USDT",
        "BTC/USDT:USDT",
    ]
    regime_scorecard["candidate_identity"]["allowed_timeframes"] = ["15m", "5m"]
    snapshot = _state_snapshot_for_regime("trend_up")

    scorecard = build_state_conditioned_scorecard(
        regime_scorecard=regime_scorecard,
        market_state_snapshot=snapshot,
        run_id="state_scorecard_snapshot_scope",
        require_walk_forward_evidence=True,
    )
    matrix = build_strategy_suitability_matrix(
        state_scorecards=[scorecard],
        market_state_snapshot=snapshot,
        run_id="matrix_snapshot_scope",
    )
    decision = build_selector_matching_decision(
        current_market_state=snapshot,
        strategy_suitability_matrix=matrix,
        decision_id="selector_snapshot_scope",
    )
    selector_row = next(
        row for row in matrix["rows"] if row.get("decision") == "SELECTOR_ELIGIBLE"
    )

    assert scorecard["rows"][0]["pair"] == "BTC/USDT:USDT"
    assert scorecard["rows"][0]["timeframe"] == "5m"
    assert selector_row["pair"] == "BTC/USDT:USDT"
    assert selector_row["timeframe"] == "5m"
    assert decision["selected_action"] == "select_strategy"
    assert decision["selected_candidate_id"] == "candidate"


def test_state_conditioned_scorecard_requires_source_observation_state_scope():
    from freqtrade_ext.bot_factory.state_conditioning import (
        build_state_conditioned_scorecard,
        validate_state_conditioned_scorecard_for_selector,
    )

    regime_scorecard = _strict_regime_scorecard_for_state_tests()
    for row in regime_scorecard["scorecard_by_regime"]:
        for field in (
            "state_id",
            "horizon_profile_id",
            "state_encoder_version",
            "state_window_id",
            "feature_cutoff_timestamp",
            "label_cutoff_timestamp",
            "decision_window_start",
            "decision_window_end",
            "future_data_used",
        ):
            row.pop(field, None)
        row["source_state_observation_scope_complete"] = False
        row["source_state_observation_scope_reason_codes"] = [
            "source_state_observation_scope_incomplete"
        ]

    scorecard = build_state_conditioned_scorecard(
        regime_scorecard=regime_scorecard,
        market_state_snapshot=_state_snapshot_for_regime("trend_up"),
        run_id="state_scorecard_missing_source_state",
        require_walk_forward_evidence=True,
    )
    validation = validate_state_conditioned_scorecard_for_selector(scorecard)

    assert scorecard["diagnostic_only"] is True
    assert scorecard["selector_candidate_creation_allowed"] is False
    assert scorecard["paper_readiness_input_allowed"] is False
    assert scorecard["rows"][0]["decision"] == "STATE_DIAGNOSTIC_ONLY"
    assert scorecard["rows"][0]["state_id"] is None
    assert scorecard["rows"][0]["diagnostic_snapshot_state_id"].endswith(
        "5m:trend_up:high:ohlcv_state_features_v1"
    )
    assert "state_fields_missing_from_source_observation" in scorecard["rows"][0]["blockers"]
    assert "state_fields_missing_from_source_observation" in scorecard["blockers"]
    assert validation["ok"] is False


def test_state_conditioned_scorecard_missing_walk_forward_is_diagnostic_only():
    from freqtrade_ext.bot_factory.state_conditioning import (
        build_state_conditioned_scorecard,
        validate_state_conditioned_scorecard_for_selector,
    )

    regime_scorecard = _strict_regime_scorecard_for_state_tests()
    for row in regime_scorecard["scorecard_by_regime"]:
        row["walk_forward_pass_rate"] = 0.0

    scorecard = build_state_conditioned_scorecard(
        regime_scorecard=regime_scorecard,
        market_state_snapshot=_state_snapshot_for_regime("trend_up"),
        run_id="state_scorecard_diagnostic",
        require_walk_forward_evidence=True,
    )
    validation = validate_state_conditioned_scorecard_for_selector(scorecard)

    assert scorecard["diagnostic_only"] is True
    assert scorecard["evidence_eligibility"] == "diagnostic_only"
    assert scorecard["selector_candidate_creation_allowed"] is False
    assert scorecard["paper_readiness_input_allowed"] is False
    assert "missing_walk_forward_evidence" in scorecard["blockers"]
    assert validation["ok"] is False
    assert "state_conditioned_scorecard_selector_creation_allowed" in validation["reason_codes"]


def test_state_conditioned_scorecard_allow_missing_walk_forward_remains_diagnostic_only():
    from freqtrade_ext.bot_factory.state_conditioning import (
        build_state_conditioned_scorecard,
        validate_state_conditioned_scorecard_for_selector,
    )

    regime_scorecard = _strict_regime_scorecard_for_state_tests()
    for row in regime_scorecard["scorecard_by_regime"]:
        row.pop("walk_forward_pass_rate", None)

    scorecard = build_state_conditioned_scorecard(
        regime_scorecard=regime_scorecard,
        market_state_snapshot=_state_snapshot_for_regime("trend_up"),
        run_id="state_scorecard_allow_missing_wf",
        require_walk_forward_evidence=False,
    )
    validation = validate_state_conditioned_scorecard_for_selector(scorecard)

    assert scorecard["walk_forward_gate_required"] is False
    assert scorecard["walk_forward_gate_passed"] is False
    assert scorecard["diagnostic_only"] is True
    assert scorecard["selector_candidate_creation_allowed"] is False
    assert scorecard["paper_readiness_input_allowed"] is False
    assert "missing_walk_forward_evidence" in scorecard["blockers"]
    assert validation["ok"] is False


def test_state_conditioned_scorecard_requires_source_historical_gate_pass():
    from freqtrade_ext.bot_factory.state_conditioning import (
        build_state_conditioned_scorecard,
        validate_state_conditioned_scorecard_for_selector,
    )

    regime_scorecard = _strict_regime_scorecard_for_state_tests()
    assert any(
        row["decision"] == "REGIME_SCOPED_SELECTOR_ELIGIBLE"
        for row in regime_scorecard["scorecard_by_regime"]
    )
    regime_scorecard["decision"] = "REJECT"

    scorecard = build_state_conditioned_scorecard(
        regime_scorecard=regime_scorecard,
        market_state_snapshot=_state_snapshot_for_regime("trend_up"),
        run_id="state_scorecard_rejected_source",
        require_walk_forward_evidence=True,
    )
    validation = validate_state_conditioned_scorecard_for_selector(scorecard)

    assert scorecard["historical_gate_passed"] is False
    assert scorecard["selector_candidate_creation_allowed"] is False
    assert scorecard["paper_readiness_input_allowed"] is False
    assert "historical_gate_not_selector_eligible" in scorecard["blockers"]
    assert validation["ok"] is False


def test_diagnostic_scorecard_cannot_become_selector_candidate():
    import pytest

    from freqtrade_ext.bot_factory.regime_promotion import (
        candidate_identity_from_logic_spec,
        selection_candidate_from_scorecard,
        strong_uptrend_momentum_logic_spec,
    )

    logic = strong_uptrend_momentum_logic_spec()
    scorecard = {
        "factory": "regime_fitness_scorecard",
        "schema_version": "regime_fitness_scorecard_v1",
        "manual_review_only": False,
        "decision": "REGIME_SCOPED_SELECTOR_ELIGIBLE",
        "diagnostic_only": True,
        "evidence_eligibility": "diagnostic_only",
        "candidate_identity": candidate_identity_from_logic_spec(
            logic,
            candidate_id="candidate",
        ),
    }

    with pytest.raises(ValueError, match="Diagnostic-only"):
        selection_candidate_from_scorecard(
            logic=logic,
            scorecard=scorecard,
            candidate_id="candidate",
        )


def _state_conditioned_scorecard_for_selector_test(
    *,
    regime: str = "trend_up",
    candidate_id: str = "candidate",
    strategy_id: str = "strategy",
    strategy_class_name: str = "FixtureStrategy",
    normal_returns: tuple[float, float] = (3.0, 2.0),
    stress_returns: tuple[float, float] = (2.0, 1.2),
    lower_confidence_bounds: tuple[float, float] = (0.5, 0.4),
):
    from freqtrade_ext.bot_factory.state_conditioning import build_state_conditioned_scorecard

    return build_state_conditioned_scorecard(
        regime_scorecard=_strict_regime_scorecard_for_state_tests(
            regime=regime,
            candidate_id=candidate_id,
            strategy_id=strategy_id,
            strategy_class_name=strategy_class_name,
            normal_returns=normal_returns,
            stress_returns=stress_returns,
            lower_confidence_bounds=lower_confidence_bounds,
        ),
        market_state_snapshot=_state_snapshot_for_regime(regime),
        run_id=f"{candidate_id}_{regime}_state_scorecard",
        require_walk_forward_evidence=True,
    )


def test_regime_observation_state_fields_require_complete_no_future_scope():
    from freqtrade_ext.bot_factory.regime_promotion import validate_observation_record

    observation = _regime_observation("state-observation")
    observation.update(_source_observation_state_scope("trend_up"))
    valid = validate_observation_record(observation)

    partial = dict(observation)
    partial["horizon_profile_id"] = ""
    partial_validation = validate_observation_record(partial)

    future = dict(observation)
    future["future_data_used"] = True
    future_validation = validate_observation_record(future)

    assert valid["ok"] is True
    assert partial_validation["ok"] is False
    assert "state_observation_fields_complete" in {
        check["name"] for check in partial_validation["checks"] if not check["passed"]
    }
    assert future_validation["ok"] is False
    assert "state_observation_no_future_data" in {
        check["name"] for check in future_validation["checks"] if not check["passed"]
    }


def test_strategy_suitability_matrix_is_state_scoped_and_blocks_identity_inheritance():
    from freqtrade_ext.bot_factory.strategy_suitability import (
        build_strategy_suitability_matrix,
        validate_strategy_suitability_matrix_for_selector,
    )

    trend_scorecard = _state_conditioned_scorecard_for_selector_test(
        regime="trend_up",
        candidate_id="trend-candidate",
        strategy_id="trend-strategy",
    )
    range_scorecard = _state_conditioned_scorecard_for_selector_test(
        regime="range",
        candidate_id="range-candidate",
        strategy_id="range-strategy",
    )
    snapshot_with_missing_state = _state_snapshot_for_regime("trend_up")
    snapshot_with_missing_state["horizons"].append(
        {
            "schema_version": "market_state_window_v1",
            "horizon": "15m",
            "horizon_group": "micro",
            "label": "high_volatility",
            "state_id": (
                "deterministic_market_state_encoder_v1:"
                "15m:high_volatility:high:ohlcv_state_features_v1"
            ),
            "confidence": 0.85,
            "uncertainty": 0.15,
            "out_of_distribution_score": 0.15,
            "reason_codes": ["label_high_volatility"],
        }
    )
    matrix = build_strategy_suitability_matrix(
        state_scorecards=[trend_scorecard, range_scorecard],
        market_state_snapshot=snapshot_with_missing_state,
        run_id="suitability_scope_test",
    )
    validation = validate_strategy_suitability_matrix_for_selector(matrix)
    trend_state_id = trend_scorecard["rows"][0]["state_id"]

    range_rows_for_trend_state = [
        row
        for row in matrix["rows"]
        if row.get("candidate_id") == "range-candidate"
        and row.get("state_id") == trend_state_id
        and row.get("decision") == "SELECTOR_ELIGIBLE"
    ]

    tampered = json.loads(json.dumps(trend_scorecard))
    tampered["rows"][0]["candidate_id"] = "other-candidate"
    tampered_matrix = build_strategy_suitability_matrix(
        state_scorecards=[tampered],
        market_state_snapshot=_state_snapshot_for_regime("trend_up"),
        run_id="suitability_identity_test",
    )
    tampered_strategy_rows = [
        row for row in tampered_matrix["rows"] if row.get("row_type") == "strategy"
    ]

    assert validation["ok"] is True
    assert range_rows_for_trend_state == []
    assert any(row["row_type"] == "no_trade" for row in matrix["rows"])
    assert any(row["row_type"] == "missing_state" for row in matrix["rows"])
    assert any(row["decision"] == "IDENTITY_MISMATCH" for row in tampered_strategy_rows)
    assert all(
        row["selector_eligible"] is False
        for row in tampered_strategy_rows
        if row.get("identity_mismatch")
    )


def test_strategy_suitability_matrix_selector_validation_requires_safe_scope():
    from freqtrade_ext.bot_factory.strategy_suitability import (
        build_strategy_suitability_matrix,
        validate_strategy_suitability_matrix_for_selector,
    )

    scorecard = _state_conditioned_scorecard_for_selector_test(
        regime="trend_up",
        candidate_id="trend-candidate",
        strategy_id="trend-strategy",
    )
    matrix = build_strategy_suitability_matrix(
        state_scorecards=[scorecard],
        market_state_snapshot=_state_snapshot_for_regime("trend_up"),
        run_id="suitability_safety_scope_test",
    )
    unsafe_matrix = json.loads(json.dumps(matrix))
    unsafe_matrix["safety_scope"]["live_trading_started"] = True
    unsafe_matrix["safety_scope"]["exchange_order_placement"] = True
    unsafe_matrix["safety_scope"]["process_control"] = True

    validation = validate_strategy_suitability_matrix_for_selector(unsafe_matrix)
    safety_check = _check_by_name(
        validation["checks"],
        "strategy_suitability_matrix_safety_scope",
    )

    assert validation["ok"] is False
    assert safety_check["passed"] is False
    assert "strategy_suitability_matrix_safety_scope" in validation["reason_codes"]
    assert {
        "exchange_order_placement",
        "live_trading_started",
        "process_control",
    }.issubset(set(safety_check["details"]["missing_or_true_required_false_flags"]))


def test_selector_matching_selects_current_state_and_ranks_by_stress_utility():
    from freqtrade_ext.bot_factory.selector_matching import build_selector_matching_decision
    from freqtrade_ext.bot_factory.strategy_suitability import build_strategy_suitability_matrix

    trend_scorecard = _state_conditioned_scorecard_for_selector_test(
        regime="trend_up",
        candidate_id="trend-candidate",
        strategy_id="trend-strategy",
    )
    range_scorecard = _state_conditioned_scorecard_for_selector_test(
        regime="range",
        candidate_id="range-candidate",
        strategy_id="range-strategy",
    )
    matrix = build_strategy_suitability_matrix(
        state_scorecards=[trend_scorecard, range_scorecard],
        market_state_snapshot=_state_snapshot_for_regime("trend_up"),
        run_id="selector_state_test",
    )
    trend_decision = build_selector_matching_decision(
        current_market_state=_state_snapshot_for_regime("trend_up"),
        strategy_suitability_matrix=matrix,
        decision_id="selector_trend",
    )
    range_decision = build_selector_matching_decision(
        current_market_state=_state_snapshot_for_regime("range"),
        strategy_suitability_matrix=matrix,
        decision_id="selector_range",
    )

    raw_high_scorecard = _state_conditioned_scorecard_for_selector_test(
        regime="trend_up",
        candidate_id="raw-high",
        strategy_id="raw-high-strategy",
        normal_returns=(10.0, 8.0),
        stress_returns=(0.3, 0.3),
        lower_confidence_bounds=(0.1, 0.1),
    )
    robust_scorecard = _state_conditioned_scorecard_for_selector_test(
        regime="trend_up",
        candidate_id="robust",
        strategy_id="robust-strategy",
        normal_returns=(4.0, 4.0),
        stress_returns=(2.0, 1.8),
        lower_confidence_bounds=(0.6, 0.5),
    )
    ranking_matrix = build_strategy_suitability_matrix(
        state_scorecards=[raw_high_scorecard, robust_scorecard],
        market_state_snapshot=_state_snapshot_for_regime("trend_up"),
        run_id="selector_ranking_test",
    )
    ranking_decision = build_selector_matching_decision(
        current_market_state=_state_snapshot_for_regime("trend_up"),
        strategy_suitability_matrix=ranking_matrix,
        decision_id="selector_ranking",
    )

    assert trend_decision["selected_action"] == "select_strategy"
    assert trend_decision["selected_candidate_id"] == "trend-candidate"
    assert range_decision["selected_action"] == "select_strategy"
    assert range_decision["selected_candidate_id"] == "range-candidate"
    assert ranking_decision["selected_candidate_id"] == "robust"
    assert "selected_by_stress_cost_utility" in ranking_decision["reason_codes"]


def test_selector_matching_requires_current_market_identity_for_state_match():
    from freqtrade_ext.bot_factory.market_regime import build_current_market_state
    from freqtrade_ext.bot_factory.selector_matching import build_selector_matching_decision
    from freqtrade_ext.bot_factory.strategy_suitability import build_strategy_suitability_matrix

    scorecard = _state_conditioned_scorecard_for_selector_test(
        regime="trend_up",
        candidate_id="trend-candidate",
        strategy_id="trend-strategy",
    )
    matrix = build_strategy_suitability_matrix(
        state_scorecards=[scorecard],
        market_state_snapshot=_state_snapshot_for_regime("trend_up"),
        run_id="selector_market_identity_test",
    )
    valid_decision = build_selector_matching_decision(
        current_market_state=_state_snapshot_for_regime("trend_up"),
        strategy_suitability_matrix=matrix,
        decision_id="selector_market_identity_valid",
    )

    assert valid_decision["selected_action"] == "select_strategy"
    assert valid_decision["selected_candidate_id"] == "trend-candidate"

    current_artifact_decision = build_selector_matching_decision(
        current_market_state=build_current_market_state(_state_snapshot_for_regime("trend_up")),
        strategy_suitability_matrix=matrix,
        decision_id="selector_market_identity_current_artifact",
    )

    assert current_artifact_decision["selected_action"] == "select_strategy"
    assert current_artifact_decision["selected_candidate_id"] == "trend-candidate"

    for field, value in (
        ("pair", "ETH/USDT:USDT"),
        ("base_timeframe", "15m"),
        ("cost_model_id", "cost_model_v2"),
    ):
        current = _state_snapshot_for_regime("trend_up")
        current[field] = value

        decision = build_selector_matching_decision(
            current_market_state=current,
            strategy_suitability_matrix=matrix,
            decision_id=f"selector_market_identity_{field}",
        )

        assert decision["selected_action"] == "no_trade"
        assert decision["selected_candidate_id"] is None
        assert (
            decision["no_trade_reason"] == "no_selector_eligible_strategy_for_current_state"
        )


def test_selector_matching_defaults_no_trade_for_unsafe_current_states_and_cooldown():
    from freqtrade_ext.bot_factory.selector_matching import build_selector_matching_decision
    from freqtrade_ext.bot_factory.strategy_suitability import build_strategy_suitability_matrix

    scorecard = _state_conditioned_scorecard_for_selector_test(
        regime="trend_up",
        candidate_id="trend-candidate",
        strategy_id="trend-strategy",
    )
    matrix = build_strategy_suitability_matrix(
        state_scorecards=[scorecard],
        market_state_snapshot=_state_snapshot_for_regime("trend_up"),
        run_id="selector_no_trade_test",
    )
    mixed = _state_snapshot_for_regime("trend_up")
    mixed["aggregate_label"] = "mixed"
    mixed["no_trade_default"] = True
    mixed["horizon_conflict"] = {"conflict_detected": True, "reason_codes": ["test_conflict"]}
    mixed["reason_codes"] = ["horizon_conflict"]
    stale = _state_snapshot_for_regime("trend_up")
    stale["data_quality_summary"]["stale_data"] = True
    stale["no_trade_default"] = True
    stale["reason_codes"] = ["stale_local_data"]
    ood = _state_snapshot_for_regime("trend_up")
    ood["aggregate_label"] = "out_of_distribution"
    ood["out_of_distribution_score"] = 0.95
    ood["no_trade_default"] = True
    ood["reason_codes"] = ["out_of_distribution_state"]

    for snapshot, reason in (
        (mixed, "horizon_conflict"),
        (stale, "stale_local_data"),
        (ood, "out_of_distribution_state"),
    ):
        decision = build_selector_matching_decision(
            current_market_state=snapshot,
            strategy_suitability_matrix=matrix,
            decision_id=f"selector_{reason}",
        )
        assert decision["selected_action"] == "no_trade"
        assert reason in decision["reason_codes"]

    previous = _state_conditioned_scorecard_for_selector_test(
        regime="trend_up",
        candidate_id="previous",
        strategy_id="previous-strategy",
        stress_returns=(1.0, 1.0),
    )
    challenger = _state_conditioned_scorecard_for_selector_test(
        regime="trend_up",
        candidate_id="challenger",
        strategy_id="challenger-strategy",
        stress_returns=(2.0, 2.0),
    )
    cooldown_matrix = build_strategy_suitability_matrix(
        state_scorecards=[previous, challenger],
        market_state_snapshot=_state_snapshot_for_regime("trend_up"),
        run_id="selector_cooldown_test",
    )
    cooldown_decision = build_selector_matching_decision(
        current_market_state=_state_snapshot_for_regime("trend_up"),
        strategy_suitability_matrix=cooldown_matrix,
        selector_state={
            "last_selected_candidate_id": "previous",
            "observations_since_switch": 1,
        },
        cooldown_observations=3,
        decision_id="selector_cooldown",
    )
    hysteresis_decision = build_selector_matching_decision(
        current_market_state=_state_snapshot_for_regime("trend_up"),
        strategy_suitability_matrix=cooldown_matrix,
        selector_state={
            "last_selected_candidate_id": "previous",
            "observations_since_switch": 4,
        },
        hysteresis_margin=10.0,
        decision_id="selector_hysteresis",
    )

    assert cooldown_decision["selected_action"] == "no_trade"
    assert cooldown_decision["no_trade_reason"] == "selector_cooldown_blocks_switching"
    assert hysteresis_decision["selected_action"] == "select_strategy"
    assert hysteresis_decision["selected_candidate_id"] == "previous"
    assert "selector_hysteresis_kept_previous_candidate" in hysteresis_decision["reason_codes"]


def test_no_trade_scorecard_records_opportunity_cost_without_hindsight_reward():
    from freqtrade_ext.bot_factory.selector_matching import build_no_trade_scorecard
    from freqtrade_ext.bot_factory.strategy_suitability import build_strategy_suitability_matrix

    scorecard = _state_conditioned_scorecard_for_selector_test(
        regime="trend_up",
        candidate_id="trend-candidate",
        strategy_id="trend-strategy",
    )
    matrix = build_strategy_suitability_matrix(
        state_scorecards=[scorecard],
        market_state_snapshot=_state_snapshot_for_regime("trend_up"),
        run_id="no_trade_scorecard_test",
    )
    clear_trend = build_no_trade_scorecard(
        current_market_state=_state_snapshot_for_regime("trend_up"),
        strategy_suitability_matrix=matrix,
        run_id="clear_trend_no_trade",
    )
    unknown = _state_snapshot_for_regime("trend_up")
    unknown["aggregate_label"] = "unknown"
    unknown["no_trade_default"] = True
    unknown["state_confidence"] = 0.2
    unknown["uncertainty"] = 0.8
    unknown_scorecard = build_no_trade_scorecard(
        current_market_state=unknown,
        strategy_suitability_matrix=matrix,
        run_id="unknown_no_trade",
    )
    high_volatility = _state_snapshot_for_regime("high_volatility")
    high_volatility["no_trade_default"] = True
    high_volatility_scorecard = build_no_trade_scorecard(
        current_market_state=high_volatility,
        strategy_suitability_matrix=matrix,
        run_id="high_volatility_no_trade",
    )

    current_clear = next(row for row in clear_trend["rows"] if row["current_state"])
    current_unknown = next(row for row in unknown_scorecard["rows"] if row["current_state"])
    current_high_volatility = next(
        row for row in high_volatility_scorecard["rows"] if row["current_state"]
    )
    assert current_clear["assessment"] == "costly_supported_state"
    assert current_clear["opportunity_cost_vs_best_selector_eligible_strategy"] > 0
    assert "no_hindsight_profit_credit" in current_clear["reason_codes"]
    assert current_unknown["assessment"] == "acceptable_uncertain_or_ood_state"
    assert current_unknown["uncertainty_reduction_value"] > 0
    assert current_high_volatility["assessment"] == "acceptable_uncertain_or_ood_state"
    assert "high_volatility_safety_value" in current_high_volatility["reason_codes"]


def test_paper_readiness_rejects_minimal_market_state_scorecard_json(tmp_path):
    strategy_path = _write_paper_strategy(tmp_path, "PaperStrategy")
    historical_dir, walk_forward_dir, training_dir = _write_paper_evidence(
        tmp_path,
        historical_gate_pass=True,
        walk_forward_recommendation="pass",
        training_recommendation="pass",
    )
    minimal_scorecard = tmp_path / "minimal_state_scorecard.json"
    minimal_scorecard.write_text(
        json.dumps(
            {
                "factory": "state_conditioned_scorecard",
                "schema_version": "state_conditioned_scorecard_v1",
                "diagnostic_only": False,
                "selector_candidate_creation_allowed": True,
                "paper_readiness_input_allowed": True,
                "proxy_evidence": False,
                "relaxed_thresholds_used": False,
                "walk_forward_gate_passed": True,
            }
        ),
        encoding="utf-8",
    )
    inputs = PaperReadinessInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_state_scorecard_check",
        config_path=tmp_path / "config.json",
        strategy_path=strategy_path,
        historical_dir=historical_dir,
        walk_forward_dir=walk_forward_dir,
        training_dir=training_dir,
        market_state_scorecard_path=minimal_scorecard,
        requires_market_state_scorecard=True,
        reviewer_notes=["Reviewed for no-startup paper readiness."],
    )

    readiness, _, _ = evaluate_paper_readiness(
        inputs,
        static_report=scan_paths([strategy_path]),
        config=_paper_config("PaperStrategy"),
        strategy_file=strategy_path,
    )

    assert readiness["readiness"] in {"blocked", "fail"}
    assert "market_state_scorecard_full_schema" in {
        check["name"] for check in readiness["failures"]
    }


def test_paper_readiness_requires_market_state_scorecard_for_target_strategy(tmp_path):
    from freqtrade_ext.bot_factory.candidate_identity import build_strategy_candidate_identity

    strategy_path = tmp_path / "PaperStrategy.py"
    paper_identity = build_strategy_candidate_identity(
        candidate_id="paper-candidate",
        strategy_id="PaperStrategy",
        strategy_class_name="PaperStrategy",
        strategy_source_path=strategy_path,
        strategy_version="strategy_v1",
        signal_version="signal_v1",
        risk_policy_version="risk_v1",
        regime_classifier_version="regime_classifier_v1",
        cost_model_id="cost_model_v1",
        allowed_pairs=["BTC/USDT:USDT"],
        allowed_timeframes=["5m"],
        created_at="2026-05-20T00:00:00+00:00",
        source_artifacts={"strategy_source": strategy_path},
        root_dir=tmp_path,
    )
    strategy_path.write_text(
        "class PaperStrategy:\n"
        f"    bot_factory_candidate_identity: dict[str, object] = {json.dumps(paper_identity, sort_keys=True)}\n"
        "    can_short = False\n"
        "    def populate_entry_trend(self, dataframe, metadata):\n"
        "        dataframe.loc[:, 'enter_long'] = 1\n"
        "        return dataframe\n",
        encoding="utf-8",
    )
    historical_dir, walk_forward_dir, training_dir = _write_paper_evidence(
        tmp_path,
        historical_gate_pass=True,
        walk_forward_recommendation="pass",
        training_recommendation="pass",
    )
    historical_metrics_path = historical_dir / "metrics.json"
    historical_metrics = json.loads(historical_metrics_path.read_text(encoding="utf-8"))
    historical_metrics["candidate_identity"] = paper_identity
    historical_metrics_path.write_text(json.dumps(historical_metrics), encoding="utf-8")
    walk_forward_metrics_path = walk_forward_dir / "walk_forward_metrics.json"
    walk_forward_metrics = json.loads(
        walk_forward_metrics_path.read_text(encoding="utf-8")
    )
    walk_forward_metrics["candidate_identity"] = paper_identity
    walk_forward_metrics_path.write_text(json.dumps(walk_forward_metrics), encoding="utf-8")

    other_scorecard = _state_conditioned_scorecard_for_selector_test(
        regime="trend_up",
        candidate_id="other-candidate",
        strategy_id="OtherStrategy",
        strategy_class_name="OtherStrategy",
    )
    scorecard_path = tmp_path / "other_state_scorecard.json"
    scorecard_path.write_text(json.dumps(other_scorecard), encoding="utf-8")
    inputs = PaperReadinessInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_market_state_strategy_check",
        config_path=tmp_path / "config.json",
        strategy_path=strategy_path,
        historical_dir=historical_dir,
        walk_forward_dir=walk_forward_dir,
        training_dir=training_dir,
        market_state_scorecard_path=scorecard_path,
        requires_market_state_scorecard=True,
        reviewer_notes=["Reviewed for no-startup paper readiness."],
    )

    readiness, _, _ = evaluate_paper_readiness(
        inputs,
        static_report=scan_paths([strategy_path]),
        config=_paper_config("PaperStrategy"),
        strategy_file=strategy_path,
    )

    assert readiness["readiness"] == "fail"
    assert "market_state_scorecard_matches_strategy" in {
        check["name"] for check in readiness["failures"]
    }


def test_paper_readiness_requires_suitability_matrix_for_target_strategy(tmp_path):
    from freqtrade_ext.bot_factory.strategy_suitability import build_strategy_suitability_matrix

    strategy_path = _write_paper_strategy(tmp_path, "PaperStrategy")
    historical_dir, walk_forward_dir, training_dir = _write_paper_evidence(
        tmp_path,
        historical_gate_pass=True,
        walk_forward_recommendation="pass",
        training_recommendation="pass",
    )
    other_scorecard = _state_conditioned_scorecard_for_selector_test(
        regime="trend_up",
        candidate_id="other-candidate",
        strategy_id="OtherStrategy",
        strategy_class_name="OtherStrategy",
    )
    matrix = build_strategy_suitability_matrix(
        state_scorecards=[other_scorecard],
        market_state_snapshot=_state_snapshot_for_regime("trend_up"),
        run_id="other_strategy_matrix",
    )
    matrix_path = tmp_path / "strategy_state_suitability_matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    inputs = PaperReadinessInputs(
        root_dir=tmp_path,
        strategy="PaperStrategy",
        run_id="paper_suitability_strategy_check",
        config_path=tmp_path / "config.json",
        strategy_path=strategy_path,
        historical_dir=historical_dir,
        walk_forward_dir=walk_forward_dir,
        training_dir=training_dir,
        strategy_suitability_matrix_path=matrix_path,
        requires_strategy_suitability_matrix=True,
        reviewer_notes=["Reviewed for no-startup paper readiness."],
    )

    readiness, _, _ = evaluate_paper_readiness(
        inputs,
        static_report=scan_paths([strategy_path]),
        config=_paper_config("PaperStrategy"),
        strategy_file=strategy_path,
    )

    assert readiness["readiness"] == "fail"
    assert "strategy_suitability_matrix_matches_strategy" in {
        check["name"] for check in readiness["failures"]
    }


def test_backtest_evidence_pipeline_writes_observation_scorecard_and_selector_artifacts(tmp_path):
    from freqtrade_ext.bot_factory.backtest_results import BacktestMetrics, write_metrics
    from freqtrade_ext.bot_factory.evidence_pipeline import (
        BacktestEvidencePipelineInputs,
        build_backtest_evidence_pipeline,
        write_backtest_evidence_pipeline_artifacts,
    )

    metrics_path = tmp_path / "data" / "backtests" / "S" / "run" / "metrics.json"
    trades_path = metrics_path.parent / "trades.csv"
    ohlcv_path = tmp_path / "user_data" / "data" / "BTC_USDT-5m.parquet"
    identity = _regime_observation("identity-source")["candidate_identity"]
    metrics = BacktestMetrics(
        strategy_name="S",
        total_return=0.35,
        total_return_pct=35.0,
        cagr=None,
        sharpe=None,
        sortino=1.2,
        calmar=None,
        max_drawdown_pct=4.0,
        profit_factor=1.6,
        win_rate=0.6,
        average_win=None,
        average_loss=None,
        trade_count=2,
        expectancy=0.15,
        fee_paid=0.0,
        backtest_start="2026-01-01",
        backtest_end="2026-01-02",
        generated_at="2026-05-22T00:00:00+00:00",
        candidate_identity=identity,
    )
    write_metrics(metrics, metrics_path)
    trades_path.write_text(
        "open_date,profit_ratio,is_short,leverage\n"
        "2026-01-01T01:00:00+00:00,0.20,False,1.0\n"
        "2026-01-01T02:00:00+00:00,0.12,False,1.0\n",
        encoding="utf-8",
    )
    dates = pd.date_range("2026-01-01", periods=40, freq="5min", tz="UTC")
    close = pd.Series(np.linspace(100.0, 122.0, 40))
    ohlcv_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": dates,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(40, 100.0),
        }
    ).to_parquet(ohlcv_path)

    pipeline = build_backtest_evidence_pipeline(
        BacktestEvidencePipelineInputs(
            root_dir=tmp_path,
            metrics_path=metrics_path,
            trades_path=trades_path,
            ohlcv_path=ohlcv_path,
            strategy="S",
            pair="BTC/USDT:USDT",
            timeframe="5m",
            output_root=tmp_path / "data" / "regime_evidence",
            run_id="run",
            intended_regimes=["trend_up"],
            excluded_regimes=["unknown"],
            reviewer_notes=["pipeline test"],
        )
    )
    paths = write_backtest_evidence_pipeline_artifacts(
        pipeline,
        root_dir=tmp_path,
        output_root=tmp_path / "data" / "regime_evidence",
    )

    assert pipeline["observation_ledger"]["ok"] is True
    assert pipeline["observation_ledger"]["observation_count"] >= 2
    assert pipeline["regime_fitness_scorecard"]["decision"] == "REGIME_SCOPED_SELECTOR_ELIGIBLE"
    assert pipeline["regime_fitness_scorecard"]["promotion_authorized_by_this_command"] is False
    assert pipeline["regime_fitness_scorecard"]["phase3_readiness_required_after_scorecard"] is True
    assert "baseline_comparison" in pipeline["regime_fitness_scorecard"]
    assert pipeline["selector_candidate"]["candidate_identity"]["candidate_id"] == identity["candidate_id"]
    assert {"metrics", "trades", "ohlcv"}.issubset(
        pipeline["selector_candidate"]["candidate_identity"]["source_artifacts"]
    )
    for path in paths.values():
        assert path.exists()


def test_manual_scorecard_cannot_become_selector_candidate():
    from freqtrade_ext.bot_factory.regime_promotion import (
        selection_candidate_from_scorecard,
        strong_uptrend_momentum_logic_spec,
    )

    logic = strong_uptrend_momentum_logic_spec()
    manual_scorecard = {
        "factory": "manual_scorecard",
        "manual_review_only": True,
        "candidate_identity": _regime_observation("manual")["candidate_identity"],
    }

    try:
        selection_candidate_from_scorecard(
            logic=logic,
            scorecard=manual_scorecard,
            candidate_id="candidate",
        )
    except ValueError as exc:
        assert "deterministic" in str(exc)
    else:
        raise AssertionError("manual scorecard should be rejected")


def test_rejected_scorecard_cannot_become_selector_candidate():
    import pytest

    from freqtrade_ext.bot_factory.regime_promotion import (
        candidate_identity_from_logic_spec,
        selection_candidate_from_scorecard,
        strong_uptrend_momentum_logic_spec,
    )

    logic = strong_uptrend_momentum_logic_spec()
    scorecard = {
        "factory": "regime_fitness_scorecard",
        "manual_review_only": False,
        "decision": "REJECT",
        "candidate_identity": candidate_identity_from_logic_spec(
            logic,
            candidate_id="candidate",
        ),
    }

    with pytest.raises(ValueError, match="selector-eligible"):
        selection_candidate_from_scorecard(
            logic=logic,
            scorecard=scorecard,
            candidate_id="candidate",
        )


def test_style_aware_gate_allows_low_trade_trend_candidate_without_scalp_threshold():
    from freqtrade_ext.bot_factory.backtest_results import BacktestMetrics, evaluate_style_aware_gate

    metrics = BacktestMetrics(
        strategy_name="TrendS",
        total_return=0.05,
        total_return_pct=5.0,
        cagr=None,
        sharpe=None,
        sortino=0.9,
        calmar=None,
        max_drawdown_pct=6.0,
        profit_factor=1.3,
        win_rate=0.55,
        average_win=None,
        average_loss=None,
        trade_count=24,
        expectancy=0.01,
        fee_paid=0.0,
        backtest_start="2026-01-01",
        backtest_end="2026-02-01",
        generated_at="2026-05-22T00:00:00+00:00",
    )

    gate = evaluate_style_aware_gate(
        metrics,
        candidate_style="intraday_trend_following",
        hold_baseline_return_pct=3.0,
    )

    assert gate["recommendation"] == "pass"
    assert next(check for check in gate["checks"] if check["name"] == "style_min_trades")[
        "rule"
    ] == ">= 20 for intraday_trend_following"


def test_runtime_selector_fails_closed_on_low_confidence_feature_quality_and_cooldown():
    from freqtrade_ext.bot_factory.feature_quality import build_feature_quality_report
    from freqtrade_ext.bot_factory.regime_promotion import (
        RuntimeRegimeSnapshot,
        RuntimeSelectorState,
        evaluate_runtime_strategy_selection,
        strong_uptrend_momentum_logic_spec,
    )

    logic = strong_uptrend_momentum_logic_spec()
    candidate = _selector_candidate_for_logic(
        logic,
        candidate_id="uptrend-candidate",
        regime="trend_up",
    )
    quality = build_feature_quality_report(
        pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC"),
                "close": [1.0, None, None, 1.1],
                "volume": [100.0, 0.0, 0.0, 100.0],
                "moving_average_slope": [0.1, 0.1, 0.1, 0.1],
                "range_efficiency": [0.8, 0.8, 0.8, 0.8],
                "regime_label": [1.0, 1.0, 1.0, 1.0],
                "cost_model": [1.0, 1.0, 1.0, 1.0],
            }
        ),
        required_features=candidate["required_features"],
        now=datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
        classifier_confidence=0.4,
    )

    selection = evaluate_runtime_strategy_selection(
        runtime=RuntimeRegimeSnapshot(
            current_regime="trend_up",
            pair="BTC/USDT:USDT",
            timeframe="5m",
            regime_classifier_version="regime_classifier_v1",
            data_quality_pass=True,
            available_features=candidate["required_features"],
            regime_confidence=0.4,
            feature_quality_report=quality,
        ),
        candidates=[candidate],
        selector_state=RuntimeSelectorState(
            last_selected_candidate_id="range-candidate",
            last_selected_regime="range",
            observations_since_switch=0,
        ),
        min_confidence_by_regime={"trend_up": 0.6},
        cooldown_observations=2,
    )

    assert selection["action"] == "no_trade"
    assert "runtime_regime_confidence_below_threshold" in selection["reason_codes"]
    assert "runtime_regime_change_cooldown_active" in selection["reason_codes"]


def test_shadow_leaderboard_rejects_future_paper_sources():
    from freqtrade_ext.bot_factory.regime_promotion import build_shadow_observation_leaderboards

    current = _regime_observation("current", source_type="local_shadow_replay")
    future = _regime_observation("future", source_type="future_paper")

    leaderboard = build_shadow_observation_leaderboards([current, future])

    assert leaderboard["accepted_count"] == 1
    assert leaderboard["rejected_count"] == 1
    assert leaderboard["historical_readiness_override_allowed"] is False
    assert leaderboard["parallel_observations_direct_promotion_allowed"] is False


def test_candidate_review_joins_artifacts_and_reason_codes(tmp_path):
    from freqtrade_ext.bot_factory.candidate_review import (
        build_candidate_review,
        write_candidate_review_artifacts,
    )

    identity = _regime_observation("review")["candidate_identity"]
    strategy_path = tmp_path / "Strategy.py"
    strategy_path.write_text("class Strategy:\n    pass\n", encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"strategy_name": "Strategy", "total_return_pct": 2.0, "candidate_identity": identity}),
        encoding="utf-8",
    )
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "scorecard_id": "score",
                "decision": "REGIME_SCOPED_SELECTOR_ELIGIBLE",
                "eligible_regimes": ["trend_up"],
                "candidate_identity": identity,
                "baseline_comparison": {"by_regime": []},
            }
        ),
        encoding="utf-8",
    )
    readiness_path = tmp_path / "paper_readiness.json"
    readiness_path.write_text(
        json.dumps({"readiness": "blocked", "blockers": [{"name": "reviewer_note_present"}]}),
        encoding="utf-8",
    )

    review = build_candidate_review(
        root_dir=tmp_path,
        candidate_id="candidate",
        strategy="Strategy",
        strategy_source_path=strategy_path,
        historical_metrics_path=metrics_path,
        regime_scorecard_path=scorecard_path,
        paper_readiness_path=readiness_path,
        reviewer_notes=["review test"],
    )
    json_path, report_path = write_candidate_review_artifacts(
        review,
        root_dir=tmp_path,
        output_root=tmp_path / "reviews",
    )

    assert "paper_readiness_blocked" in review["reason_codes"]
    assert review["strategy_source"]["sha256"]
    assert "-> regime_fitness_scorecard.json" in review["architecture_diagram"]
    assert json_path.exists()
    assert report_path.exists()


def test_gate_glossary_defines_no_paper_live_approval_by_name():
    from freqtrade_ext.bot_factory.gate_semantics import (
        gate_glossary,
        gate_semantics_payload,
    )

    glossary = gate_glossary()
    payload = gate_semantics_payload("REGIME_SCOPED_SELECTOR_ELIGIBLE", "paper_readiness.pass")

    assert "paper trading" in glossary["REGIME_SCOPED_SELECTOR_ELIGIBLE"]["does_not_permit"]
    assert "starting freqtrade trade" in glossary["paper_readiness.pass"]["does_not_permit"]
    assert payload["promotion_authorized_by_this_command"] is False
    assert payload["paper_live_approval_by_name_allowed"] is False


def test_paper_readiness_requires_regime_scorecard_when_selector_eligibility_claimed(
    tmp_path,
):
    from freqtrade_ext.bot_factory.paper import (
        PaperReadinessInputs,
        _regime_scorecard_evidence_checks,
    )

    inputs = PaperReadinessInputs(
        root_dir=tmp_path,
        strategy="S",
        run_id="paper",
        config_path=tmp_path / "config.json",
        strategy_path=tmp_path / "S.py",
        historical_dir=tmp_path / "historical",
        walk_forward_dir=tmp_path / "walk_forward",
        training_dir=tmp_path / "training",
        requires_regime_scorecard=True,
    )

    missing_checks = _regime_scorecard_evidence_checks(inputs)

    assert missing_checks[0].name == "regime_scorecard_required"
    assert missing_checks[0].status == "blocked"

    scorecard_path = tmp_path / "regime_fitness_scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "decision": "REGIME_SCOPED_SELECTOR_ELIGIBLE",
                "promotion_authorized_by_this_command": False,
                "raw_aggregate_pnl_promotion_allowed": False,
                "phase3_readiness_required_after_scorecard": True,
            }
        ),
        encoding="utf-8",
    )
    present_checks = _regime_scorecard_evidence_checks(
        PaperReadinessInputs(
            root_dir=tmp_path,
            strategy="S",
            run_id="paper",
            config_path=tmp_path / "config.json",
            strategy_path=tmp_path / "S.py",
            historical_dir=tmp_path / "historical",
            walk_forward_dir=tmp_path / "walk_forward",
            training_dir=tmp_path / "training",
            regime_scorecard_path=scorecard_path,
            requires_regime_scorecard=True,
        )
    )

    assert {check.name: check.status for check in present_checks} == {
        "regime_scorecard_required": "pass",
        "regime_scorecard_selector_eligible": "pass",
        "regime_scorecard_does_not_authorize_promotion": "pass",
    }


def test_regime_observation_rejects_mismatched_candidate_identity():
    from freqtrade_ext.bot_factory.candidate_identity import build_strategy_candidate_identity
    from freqtrade_ext.bot_factory.regime_promotion import validate_observation_record

    wrong_identity = build_strategy_candidate_identity(
        candidate_id="candidate",
        strategy_id="strategy-b",
        strategy_class_name="StrategyB",
        strategy_source_path="user_data/strategies/StrategyB.py",
        strategy_version="strategy_v1",
        signal_version="signal_v1",
        risk_policy_version="risk_v1",
        regime_classifier_version="regime_classifier_v1",
        cost_model_id="cost_model_v1",
        allowed_pairs=["BTC/USDT:USDT"],
        allowed_timeframes=["5m"],
        created_at="2026-05-21T00:00:00+00:00",
        source_artifacts={"strategy_source": "user_data/strategies/StrategyB.py"},
    )
    observation = _regime_observation(
        "mismatch",
        strategy_id="strategy-a",
        candidate_identity=wrong_identity,
    )

    result = validate_observation_record(observation)

    assert result["ok"] is False
    assert _check_by_name(
        result["checks"], "candidate_identity_strategy_id_matches_row"
    )["passed"] is False


def test_selector_rejects_scorecard_identity_for_different_strategy():
    import pytest

    from freqtrade_ext.bot_factory.regime_promotion import (
        RegimePromotionThresholds,
        build_regime_fitness_scorecard,
        candidate_identity_from_logic_spec,
        contract_from_logic_spec,
        range_mean_reversion_logic_spec,
        selection_candidate_from_scorecard,
        strong_uptrend_momentum_logic_spec,
    )

    uptrend_logic = strong_uptrend_momentum_logic_spec()
    range_logic = range_mean_reversion_logic_spec()
    candidate_id = "strategy-a-candidate"
    identity = candidate_identity_from_logic_spec(uptrend_logic, candidate_id=candidate_id)
    scorecard = build_regime_fitness_scorecard(
        [
            _regime_observation(
                "strategy-a-btc",
                candidate_id=candidate_id,
                strategy_id=uptrend_logic.strategy_id,
                strategy_class_name=uptrend_logic.strategy_class_name,
                strategy_source_path=uptrend_logic.strategy_source_path,
                strategy_version=uptrend_logic.strategy_version,
                signal_version=uptrend_logic.signal_version,
                risk_policy_version=uptrend_logic.risk_policy_version,
                regime_classifier_version=uptrend_logic.regime_classifier_version,
                cost_model_id=uptrend_logic.cost_model_id,
                allowed_pairs=list(uptrend_logic.allowed_pairs),
                allowed_timeframes=list(uptrend_logic.allowed_timeframes),
                candidate_identity=identity,
                regime="trend_up",
                pair="BTC/USDT:USDT",
                window_start="2026-01-01T00:00:00+00:00",
                window_end="2026-02-01T00:00:00+00:00",
                net_return_normal_cost=7.0,
                net_return_stress_cost=4.0,
                lower_confidence_bound=0.8,
            ),
            _regime_observation(
                "strategy-a-eth",
                candidate_id=candidate_id,
                strategy_id=uptrend_logic.strategy_id,
                strategy_class_name=uptrend_logic.strategy_class_name,
                strategy_source_path=uptrend_logic.strategy_source_path,
                strategy_version=uptrend_logic.strategy_version,
                signal_version=uptrend_logic.signal_version,
                risk_policy_version=uptrend_logic.risk_policy_version,
                regime_classifier_version=uptrend_logic.regime_classifier_version,
                cost_model_id=uptrend_logic.cost_model_id,
                allowed_pairs=list(uptrend_logic.allowed_pairs),
                allowed_timeframes=list(uptrend_logic.allowed_timeframes),
                candidate_identity=identity,
                regime="trend_up",
                pair="ETH/USDT:USDT",
                window_start="2026-02-01T00:00:00+00:00",
                window_end="2026-03-05T00:00:00+00:00",
                net_return_normal_cost=6.0,
                net_return_stress_cost=3.5,
                lower_confidence_bound=0.7,
            ),
        ],
        contract=contract_from_logic_spec(uptrend_logic),
        thresholds=RegimePromotionThresholds(max_calendar_concentration=0.5),
    )

    assert scorecard["decision"] == "REGIME_SCOPED_SELECTOR_ELIGIBLE"
    with pytest.raises(ValueError, match="candidate identity"):
        selection_candidate_from_scorecard(
            logic=range_logic,
            scorecard=scorecard,
            candidate_id=candidate_id,
        )


def test_donchian_strategy_identity_matches_strong_uptrend_logic():
    from freqtrade_ext.bot_factory.candidate_identity import (
        compare_candidate_identities,
        load_candidate_identity_from_strategy_source,
    )
    from freqtrade_ext.bot_factory.regime_promotion import (
        candidate_identity_from_logic_spec,
        strong_uptrend_momentum_logic_spec,
    )

    logic = strong_uptrend_momentum_logic_spec()
    strategy_identity = load_candidate_identity_from_strategy_source(
        Path("user_data/strategies/DonchianTrendBullStrategy.py"),
        strategy_class_name="DonchianTrendBullStrategy",
    )
    logic_identity = candidate_identity_from_logic_spec(
        logic,
        candidate_id="strong-uptrend-historical-ohlcv-candidate",
    )

    result = compare_candidate_identities(logic_identity, strategy_identity)

    assert result["ok"] is True
    assert strategy_identity["strategy_class_name"] == "DonchianTrendBullStrategy"
    assert logic.logic_id == "strong_uptrend_momentum_v1"


def test_strategy_identity_loader_accepts_annotated_assignments(tmp_path):
    from freqtrade_ext.bot_factory.candidate_identity import (
        build_strategy_candidate_identity,
        load_candidate_identity_from_strategy_source,
    )

    cases = {
        "ModuleAnnotatedStrategy": (
            lambda payload: (
                f"BOT_FACTORY_CANDIDATE_IDENTITY: dict[str, object] = {payload}\n"
                "class ModuleAnnotatedStrategy:\n"
                "    pass\n"
            )
        ),
        "ClassAnnotatedStrategy": (
            lambda payload: (
                "class ClassAnnotatedStrategy:\n"
                f"    bot_factory_candidate_identity: dict[str, object] = {payload}\n"
            )
        ),
    }
    for class_name, render_source in cases.items():
        strategy_file = tmp_path / f"{class_name}.py"
        identity = build_strategy_candidate_identity(
            candidate_id=f"{class_name.lower()}-candidate",
            strategy_id=class_name,
            strategy_class_name=class_name,
            strategy_source_path=strategy_file,
            strategy_version=f"{class_name}_v1",
            signal_version="signal_v1",
            risk_policy_version="risk_v1",
            regime_classifier_version="regime_v1",
            cost_model_id="cost_v1",
            allowed_pairs=["BTC/USDT:USDT"],
            allowed_timeframes=["5m"],
            created_at="2026-05-24T00:00:00+00:00",
            source_artifacts={"strategy_source": strategy_file},
            root_dir=tmp_path,
        )
        strategy_file.write_text(
            render_source(json.dumps(identity, sort_keys=True)),
            encoding="utf-8",
        )

        loaded = load_candidate_identity_from_strategy_source(
            strategy_file,
            strategy_class_name=class_name,
            root_dir=tmp_path,
        )

        assert loaded == identity


def test_regime_contract_requires_no_trade_conditions_for_exclusions():
    from freqtrade_ext.bot_factory.regime_promotion import validate_strategy_contract

    contract = _regime_contract(
        intended_regimes=["trend_up"],
        excluded_regimes=["high_volatility"],
        no_trade_conditions=[],
    )

    result = validate_strategy_contract(contract)

    assert result["ok"] is False
    assert _check_by_name(
        result["checks"],
        "no_trade_conditions_present_for_excluded_regimes",
    )["passed"] is False


def test_candidate_iteration_plan_preserves_lineage_and_blocks_execution(tmp_path):
    from freqtrade_ext.bot_factory.candidate_iteration import (
        CandidateIterationInputs,
        build_candidate_iteration_plan,
        write_candidate_iteration_artifacts,
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "candidate_id": "cand-fail",
        "strategy_name": "S",
        "recommendation": "retry",
        "failure_taxonomy_codes": ["FAIL_OVERFIT_WF_GAP"],
        "checks": [
            {
                "name": "walk_forward",
                "status": "fail",
                "path": "wf.json",
                "payload_summary": {
                    "summary": {"pass_rate": 0.0, "max_single_window_profit_dependency": 1.0}
                },
            }
        ],
        "research_brief": {
            "thesis_id": "TH-1",
            "thesis_statement": "Pullback recovery thesis.",
            "research_references": [
                {
                    "reference_id": "paper:research-TH-1",
                    "title": "Research reference",
                    "source": "Local bibliography",
                    "published_at": "2024",
                    "relevance": "Motivates the tested hypothesis.",
                    "motivated_thesis_ids": ["TH-1"],
                }
            ],
        },
        "next_candidate_input": {
            "thesis_id": "TH-1",
            "retry_budget_per_thesis": 3,
            "thesis_retry_count": 1,
            "parameter_only_retry_count": 0,
            "force_distinct_hypothesis_family": False,
        },
    }), encoding="utf-8")
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "S",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "mean_reversion_pullback",
        "thesis_id": "TH-1",
        "thesis_type": "mean_reversion",
        "thesis_statement": "Pullback recovery thesis.",
        "falsification_criteria": "Walk-forward degradation.",
        "evidence_refs": ["local:prior"],
        "retry_budget_per_thesis": 3,
        "thesis_retry_count": 1,
        "parameter_only_retry_limit": 1,
        "parameter_only_retry_count": 0,
    }), encoding="utf-8")

    plan = build_candidate_iteration_plan(CandidateIterationInputs(
        root_dir=tmp_path,
        candidate_manifest_path=manifest,
        proposal_metadata_path=proposal,
        reviewer_findings=["Walk-forward gap indicates regime fragility."],
        changed_assumptions=["Add regime filter before pullback entries."],
        changed_parameters=["Raise volume factor modestly."],
        unchanged_rejection_rules=["Reject if profit depends on one narrow period."],
        prior_timerange="20250101-20250201",
        proposed_timerange="20250101-20250201",
    ))
    plan_path, revision_path, report_path = write_candidate_iteration_artifacts(
        plan,
        root_dir=tmp_path,
        output_root=Path("reviews"),
    )

    assert plan["action"] == "revise"
    assert plan["evaluation_allowed_by_this_plan"] is False
    assert plan["lineage"]["candidate_manifest_path"] == "manifest.json"
    assert plan["proposal_revision_input"]["source_candidate_id"] == "cand-fail"
    assert plan["proposal_revision_input"]["safety_scope"]["live_trading"] is False
    assert plan["failure_evidence_summary"]["failed_checks"][0]["name"] == "walk_forward"
    assert plan["proposal_revision_input"]["research_references"][0]["reference_id"] == (
        "paper:research-TH-1"
    )
    assert plan["proposal_revision_input"]["requires_new_thesis_id"] is False
    assert plan_path.is_file()
    assert revision_path.is_file()
    assert report_path.is_file()


def test_candidate_iteration_blocks_causal_failure_map_blocked_next_action(tmp_path):
    from freqtrade_ext.bot_factory.candidate_iteration import (
        CandidateIterationInputs,
        build_candidate_iteration_plan,
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "candidate_id": "cand-local-reject",
        "strategy_name": "S",
        "recommendation": "retry",
        "failure_taxonomy_codes": ["FAIL_COST_SENSITIVE"],
        "checks": [
            {
                "name": "walk_forward",
                "status": "fail",
                "path": "wf.json",
            }
        ],
        "research_brief": {
            "thesis_id": "TH-LOCAL-REJECTED",
            "research_references": [
                {
                    "reference_id": "paper:research-TH-LOCAL-REJECTED",
                    "title": "Local rejection reference",
                    "source": "Local bibliography",
                    "published_at": "2025",
                    "relevance": "Documents the rejected mechanism.",
                    "motivated_thesis_ids": ["TH-LOCAL-REJECTED"],
                }
            ],
        },
        "next_candidate_input": {
            "thesis_id": "TH-LOCAL-REJECTED",
            "retry_budget_per_thesis": 3,
            "thesis_retry_count": 1,
            "parameter_only_retry_count": 0,
            "force_distinct_hypothesis_family": False,
            "blocked_next_actions": [
                "retry_validated_local_rejection_by_parameter_tuning",
            ],
        },
    }), encoding="utf-8")
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "S",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "mean_reversion_pullback",
        "thesis_id": "TH-LOCAL-REJECTED",
        "thesis_type": "mean_reversion",
        "thesis_statement": "Rejected local thesis.",
        "falsification_criteria": "Positive post-cost edge fails.",
        "evidence_refs": ["local:prior"],
    }), encoding="utf-8")

    plan = build_candidate_iteration_plan(CandidateIterationInputs(
        root_dir=tmp_path,
        candidate_manifest_path=manifest,
        proposal_metadata_path=proposal,
        reviewer_findings=[
            "The causal failure map says the local rejection should not be retried."
        ],
        changed_assumptions=[
            (
                "Retry validated local rejection by parameter tuning while keeping "
                "the same mechanism."
            )
        ],
        unchanged_rejection_rules=[
            "Reject if local falsification still shows non-positive edge."
        ],
        prior_timerange="20250101-20250201",
        proposed_timerange="20250101-20250201",
    ))

    assert plan["action"] == "blocked"
    assert plan["blocked_next_action_matches"] == [
        "retry_validated_local_rejection_by_parameter_tuning"
    ]
    assert "revision_avoids_blocked_next_actions" in {
        check["name"] for check in plan["checks"] if check["status"] == "blocked"
    }
    revision = plan["proposal_revision_input"]
    assert revision["blocked_next_actions"] == [
        "retry_validated_local_rejection_by_parameter_tuning"
    ]
    assert revision["blocked_next_action_matches"] == [
        "retry_validated_local_rejection_by_parameter_tuning"
    ]


def test_candidate_iteration_requires_research_brief_for_theory_trail(tmp_path):
    from freqtrade_ext.bot_factory.candidate_iteration import (
        CandidateIterationInputs,
        build_candidate_iteration_plan,
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "candidate_id": "cand-no-research",
        "strategy_name": "S",
        "recommendation": "retry",
        "checks": [{"name": "walk_forward", "status": "fail", "path": "wf.json"}],
        "next_candidate_input": {"retry_budget_per_thesis": 3, "thesis_retry_count": 0},
    }), encoding="utf-8")
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({"strategy_name": "S"}), encoding="utf-8")

    plan = build_candidate_iteration_plan(CandidateIterationInputs(
        root_dir=tmp_path,
        candidate_manifest_path=manifest,
        proposal_metadata_path=proposal,
        reviewer_findings=["Walk-forward evidence failed."],
        changed_assumptions=["Change the hypothesis family before retry."],
        unchanged_rejection_rules=["Reject if walk-forward pass rate remains low."],
    ))

    assert plan["action"] == "blocked"
    assert "research_brief_available" in {
        check["name"] for check in plan["checks"] if check["status"] == "blocked"
    }


def test_candidate_iteration_force_distinct_requires_new_theory(tmp_path):
    from freqtrade_ext.bot_factory.candidate_iteration import (
        CandidateIterationInputs,
        build_candidate_iteration_plan,
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "candidate_id": "cand-distinct",
        "strategy_name": "S",
        "recommendation": "retry",
        "checks": [{"name": "walk_forward", "status": "fail", "path": "wf.json"}],
        "research_brief": {
            "thesis_id": "TH-OLD",
            "research_references": [
                {
                    "reference_id": "paper:old",
                    "title": "Old reference",
                    "source": "Local bibliography",
                    "published_at": "2024",
                    "relevance": "Old hypothesis support.",
                    "motivated_thesis_ids": ["TH-OLD"],
                }
            ],
        },
        "next_candidate_input": {
            "thesis_id": "TH-OLD",
            "retry_budget_per_thesis": 2,
            "thesis_retry_count": 0,
            "parameter_only_retry_count": 0,
            "force_distinct_hypothesis_family": True,
        },
    }), encoding="utf-8")
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "S",
        "strategy_logic_variant": "trend_continuation",
        "thesis_id": "TH-OLD",
        "thesis_type": "trend_continuation",
    }), encoding="utf-8")

    plan = build_candidate_iteration_plan(CandidateIterationInputs(
        root_dir=tmp_path,
        candidate_manifest_path=manifest,
        proposal_metadata_path=proposal,
        reviewer_findings=["Prior family had too few trades and fragile windows."],
        changed_assumptions=["Move to a volatility expansion hypothesis family."],
        unchanged_rejection_rules=["Reject if pass rate remains below threshold."],
    ))

    revision = plan["proposal_revision_input"]
    assert plan["action"] == "revise"
    assert revision["required_hypothesis_family_change"] is True
    assert revision["previous_thesis_id"] == "TH-OLD"
    assert revision["thesis_id"] is None
    assert revision["requires_new_thesis_id"] is True
    assert revision["requires_new_research_references"] is True
    assert revision["strategy_logic_variant"] == "volatility_breakout"


def test_candidate_iteration_rejects_safety_relaxation(tmp_path):
    from freqtrade_ext.bot_factory.candidate_iteration import (
        CandidateIterationInputs,
        build_candidate_iteration_plan,
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "candidate_id": "cand-fail",
        "strategy_name": "S",
        "recommendation": "retry",
        "checks": [{"name": "walk_forward", "status": "fail", "path": "wf.json"}],
        "next_candidate_input": {"retry_budget_per_thesis": 3, "thesis_retry_count": 1},
    }), encoding="utf-8")
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({"strategy_name": "S"}), encoding="utf-8")

    plan = build_candidate_iteration_plan(CandidateIterationInputs(
        root_dir=tmp_path,
        candidate_manifest_path=manifest,
        proposal_metadata_path=proposal,
        reviewer_findings=["Reviewer found weak walk-forward evidence."],
        changed_assumptions=["Use future close to improve labels."],
        unchanged_rejection_rules=["Reject if walk-forward gap persists."],
    ))

    assert plan["action"] == "reject"
    assert "revision_safety_scope_preserved" in {
        check["name"] for check in plan["checks"] if check["status"] == "blocked"
    }


def test_candidate_iteration_blocks_invalid_timerange_calendar_dates(tmp_path):
    from freqtrade_ext.bot_factory.candidate_iteration import (
        CandidateIterationInputs,
        build_candidate_iteration_plan,
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "candidate_id": "cand-fail",
        "strategy_name": "S",
        "recommendation": "retry",
        "checks": [{"name": "walk_forward", "status": "fail", "path": "wf.json"}],
        "research_references": [
            {
                "title": "Post-cost validation note",
                "url": "local:research",
                "published_at": "2026",
                "relevance": "Documents the failed candidate thesis.",
                "motivated_thesis_ids": ["TH-BAD-DATE"],
            }
        ],
        "next_candidate_input": {
            "thesis_id": "TH-BAD-DATE",
            "retry_budget_per_thesis": 3,
            "thesis_retry_count": 1,
        },
    }), encoding="utf-8")
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "S",
        "thesis_id": "TH-BAD-DATE",
        "thesis_type": "mean_reversion",
    }), encoding="utf-8")

    plan = build_candidate_iteration_plan(CandidateIterationInputs(
        root_dir=tmp_path,
        candidate_manifest_path=manifest,
        proposal_metadata_path=proposal,
        reviewer_findings=["Prior timerange was typed incorrectly."],
        changed_assumptions=["Retest with the same thesis only after timerange correction."],
        unchanged_rejection_rules=["Reject if walk-forward evidence remains fragile."],
        prior_timerange="20250230-20250301",
        proposed_timerange="20250101-20250201",
    ))

    timerange_check = next(
        check for check in plan["checks"] if check["name"] == "timerange_values_valid"
    )
    assert plan["action"] == "blocked"
    assert timerange_check["status"] == "blocked"
    assert timerange_check["details"]["invalid_timeranges"] == [
        {
            "field": "prior_timerange",
            "value": "20250230-20250301",
            "reason": "invalid_calendar_date",
            "message": "day is out of range for month",
        }
    ]


def test_candidate_iteration_blocks_malformed_timerange_strings(tmp_path):
    from freqtrade_ext.bot_factory.candidate_iteration import (
        CandidateIterationInputs,
        build_candidate_iteration_plan,
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "candidate_id": "cand-fail",
        "strategy_name": "S",
        "recommendation": "retry",
        "checks": [{"name": "walk_forward", "status": "fail", "path": "wf.json"}],
        "research_references": [
            {
                "title": "Post-cost validation note",
                "url": "local:research",
                "published_at": "2026",
                "relevance": "Documents the failed candidate thesis.",
                "motivated_thesis_ids": ["TH-BAD-FORMAT"],
            }
        ],
        "next_candidate_input": {
            "thesis_id": "TH-BAD-FORMAT",
            "retry_budget_per_thesis": 3,
            "thesis_retry_count": 1,
        },
    }), encoding="utf-8")
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({
        "strategy_name": "S",
        "thesis_id": "TH-BAD-FORMAT",
        "thesis_type": "mean_reversion",
    }), encoding="utf-8")

    plan = build_candidate_iteration_plan(CandidateIterationInputs(
        root_dir=tmp_path,
        candidate_manifest_path=manifest,
        proposal_metadata_path=proposal,
        reviewer_findings=["Timerange format should be rejected."],
        changed_assumptions=["Retest only after timerange correction."],
        unchanged_rejection_rules=["Reject if walk-forward evidence remains fragile."],
        prior_timerange="2025-01-01-2025-02-01",
        proposed_timerange="20250101-20250201",
    ))

    timerange_check = next(
        check for check in plan["checks"] if check["name"] == "timerange_values_valid"
    )
    assert plan["action"] == "blocked"
    assert timerange_check["status"] == "blocked"
    assert timerange_check["details"]["invalid_timeranges"] == [
        {
            "field": "prior_timerange",
            "value": "2025-01-01-2025-02-01",
            "reason": "invalid_format",
            "message": "Timerange must match YYYYMMDD-YYYYMMDD.",
        }
    ]


def test_strategy_code_generator_volatility_breakout_exit_uses_prior_rolling_low():
    from freqtrade_ext.bot_factory.strategy_code import _exit_logic_for_variant

    exit_logic = _exit_logic_for_variant("volatility_breakout")

    assert 'prior_low = dataframe["rolling_low"].shift(1)' in exit_logic
    assert 'breakout_failure = dataframe["close"] < prior_low' in exit_logic
    assert 'breakout_failure = dataframe["close"] < dataframe["rolling_low"]' not in exit_logic


def test_signal_diagnostics_explains_zero_entry_components(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "ZeroEntryStrategy",
        "candidate_id": "zero-entry-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "mean_reversion_pullback",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 5,
            "buy_rsi_pullback": 32,
            "buy_rsi_recovery": 42,
            "buy_ema_fast": 3,
            "buy_ema_slow": 8,
            "buy_volume_window": 4,
            "buy_volume_factor": 1.0,
            "sell_rsi_exit": 65,
        },
    }), encoding="utf-8")
    ohlcv_path = tmp_path / "ohlcv.csv"
    rows = [
        {
            "date": f"2025-01-01 00:{minute:02d}:00+00:00",
            "open": 100 + minute,
            "high": 101 + minute,
            "low": 99 + minute,
            "close": 100 + minute,
            "volume": 100,
        }
        for minute in range(30)
    ]
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-zero-entry",
        timerange="20250101-20250102",
        reviewer_notes=["diagnostic only"],
    ))

    assert diagnostics["status"] == "completed"
    assert diagnostics["entry_count"] == 0
    assert diagnostics["entry_signal_count"] == 0
    assert diagnostics["first_zero_component"] == "pullback_seen"
    assert "ZERO_ENTRY_SIGNALS" in diagnostics["diagnosis_codes"]
    assert diagnostics["components"]["pullback_seen"]["individual_count"] == 0
    assert diagnostics["components"]["pullback_seen"]["description"]
    assert diagnostics["safety_scope"]["exchange_order_placement"] is False

    json_path, report_path = write_signal_diagnostics_artifacts(
        diagnostics,
        root_dir=tmp_path,
        output_root=Path("diagnostics"),
    )

    assert json_path.is_file()
    assert report_path.is_file()
    assert json.loads(json_path.read_text(encoding="utf-8"))["entry_count"] == 0
    assert "entry_signal_count: 0" in report_path.read_text(encoding="utf-8")


def test_signal_diagnostics_reports_generated_entry_edge_after_cost(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "EdgeDiagnosticStrategy",
        "candidate_id": "edge-diagnostic-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "volatility_breakout",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 3,
            "buy_rsi_pullback": 32,
            "buy_rsi_recovery": 42,
            "buy_ema_fast": 3,
            "buy_ema_slow": 8,
            "buy_volume_window": 4,
            "buy_volume_factor": 0.0,
            "sell_rsi_exit": 65,
            "sell_timeout_candles": 2,
        },
    }), encoding="utf-8")
    ohlcv_path = tmp_path / "ohlcv.csv"
    closes = [100.0, 100.05, 100.10, 100.15, 100.20, 105.0, 94.0, 92.0, 91.0, 90.0]
    rows = []
    for index, close in enumerate(closes):
        wide_breakout_candle = index == 5
        rows.append({
            "date": f"2025-01-01 00:{index:02d}:00+00:00",
            "open": close - 0.02,
            "high": close + (6.0 if wide_breakout_candle else 0.05),
            "low": close - (6.0 if wide_breakout_candle else 0.05),
            "close": close,
            "volume": 100,
        })
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-entry-edge",
        timerange="20250101-20250102",
        entry_edge_hold_candles=2,
        entry_edge_all_in_cost_bps=12.0,
        reviewer_notes=["entry edge diagnostic only"],
    ))

    edge = diagnostics["generated_entry_edge"]
    assert diagnostics["status"] == "completed"
    assert diagnostics["entry_count"] >= 1
    assert edge["sample_count"] >= 1
    assert edge["status"] == "fail"
    assert edge["net_edge_bps"] < 0
    assert "GENERATED_ENTRY_EDGE_NEGATIVE_AFTER_COST" in diagnostics["diagnosis_codes"]

    _, report_path = write_signal_diagnostics_artifacts(
        diagnostics,
        root_dir=tmp_path,
        output_root=Path("diagnostics"),
    )
    report_text = report_path.read_text(encoding="utf-8")
    assert "## Generated Entry Edge" in report_text
    assert "net_edge_bps" in report_text


def test_signal_diagnostics_supports_crowding_unwind_structural_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "CrowdingUnwindDiagnosticStrategy",
        "candidate_id": "crowding-unwind-diagnostic-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "crowding_unwind_reaccumulation",
        "timeframe": "5m",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 24,
            "buy_rsi_pullback": 38,
            "buy_rsi_recovery": 46,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 288,
            "buy_volume_factor": 0.90,
            "sell_rsi_exit": 95,
            "sell_timeout_candles": 72,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=1200, freq="5min", tz="UTC")
    closes = [
        100.0 + index * 0.002 + np.sin(index / 6.0) * 0.20
        for index in range(len(dates))
    ]
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.02 for value in closes],
        "high": [value + 0.10 for value in closes],
        "low": [value - 0.10 for value in closes],
        "close": closes,
        "volume": [100.0 + np.sin(index / 5.0) * 2.0 for index in range(len(dates))],
    }).to_csv(ohlcv_path, index=False)

    structural_root = tmp_path / "data" / "market_structure" / "bybit" / "futures"
    structural_root.mkdir(parents=True)
    context_dates = pd.date_range("2025-01-01", periods=110, freq="1h", tz="UTC")
    pd.DataFrame({
        "date": context_dates,
        "open_interest": [
            1000.0 if index < 72 else 840.0
            for index in range(len(context_dates))
        ],
    }).to_parquet(structural_root / "BTC_USDT_USDT-1h-open_interest.parquet")
    pd.DataFrame({
        "date": context_dates,
        "long_account_ratio": [
            0.62 if index < 72 else 0.42
            for index in range(len(context_dates))
        ],
        "short_account_ratio": [
            0.38 if index < 72 else 0.58
            for index in range(len(context_dates))
        ],
        "long_short_ratio": [
            1.60 if index < 72 else 0.72
            for index in range(len(context_dates))
        ],
    }).to_parquet(structural_root / "BTC_USDT_USDT-1h-long_short_ratio.parquet")

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-crowding-unwind",
        timerange="20250101-20250105",
        entry_edge_hold_candles=2,
        entry_edge_all_in_cost_bps=12.0,
        reviewer_notes=["crowding unwind diagnostic only"],
    ))

    assert diagnostics["status"] == "completed"
    assert diagnostics["structural_data_merge"]["requested"] is True
    assert diagnostics["structural_data_merge"]["open_interest"]["matched_row_count"] > 0
    assert diagnostics["structural_data_merge"]["long_short_ratio"]["matched_row_count"] > 0
    assert diagnostics["entry_signal_count"] > 0
    assert "open_interest_unwinding" in diagnostics["components"]
    assert "short_account_reaccumulation" in diagnostics["components"]
    assert "price_above_sma" in diagnostics["components"]
    assert "volume_participation_floor" in diagnostics["components"]
    assert "negative_funding_pressure" not in diagnostics["components"]
    assert diagnostics["components"]["open_interest_unwinding"]["individual_count"] > 0
    assert diagnostics["components"]["short_account_reaccumulation"]["individual_count"] > 0


def test_signal_diagnostics_merges_freqai_prediction_artifacts(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "HybridSignalStrategy",
        "candidate_id": "hybrid-signal-candidate",
        "generator_mode": "hybrid_ml",
        "strategy_logic_variant": "trend_continuation",
        "target_definition": "future_return",
        "prediction_threshold": 0.001,
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 5,
            "buy_rsi_pullback": 32,
            "buy_rsi_recovery": 52,
            "buy_ema_fast": 3,
            "buy_ema_slow": 8,
            "buy_volume_window": 4,
            "buy_volume_factor": 0.5,
            "sell_rsi_exit": 65,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=80, freq="5min", tz="UTC")
    closes = [100.0] * 25 + [100.0 + index for index in range(55)]
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [value + 1.0 for value in closes],
        "low": [value - 1.0 for value in closes],
        "close": closes,
        "volume": [100.0] * len(closes),
    }).to_csv(ohlcv_path, index=False)
    predictions_dir = tmp_path / "models" / "identifier" / "backtesting_predictions"
    predictions_dir.mkdir(parents=True)
    pd.DataFrame({
        "date": dates,
        "&-future_return": [0.002] * len(dates),
        "&-future_return_mean": [0.0] * len(dates),
        "&-future_return_std": [0.001] * len(dates),
        "do_predict": [1] * len(dates),
    }).to_csv(predictions_dir / "cb_btc_1_prediction.csv", index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        freqai_predictions_dir=predictions_dir,
        diagnostics_id="diag-hybrid-predictions",
        timerange="20250101-20250102",
        reviewer_notes=["diagnostic only"],
    ))

    assert diagnostics["status"] == "completed"
    assert diagnostics["entry_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is True
    assert diagnostics["components"]["ml_filter"]["individual_count"] == len(dates)
    assert diagnostics["prediction_merge"]["prediction_file_count"] == 1
    assert diagnostics["prediction_merge"]["target_column_present_after_merge"] is True
    assert diagnostics["prediction_merge"]["matched_row_count"] == len(dates)
    assert "ML_FILTER_UNAVAILABLE" not in diagnostics["diagnosis_codes"]


def test_signal_diagnostics_supports_downside_liquidity_shock_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "DownsideShockStrategy",
        "candidate_id": "downside-shock-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "downside_liquidity_shock_reversal",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 6,
            "buy_rsi_pullback": 35,
            "buy_rsi_recovery": 41,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 12,
            "buy_volume_factor": 0.95,
            "sell_rsi_exit": 58,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=80, freq="5min", tz="UTC")
    closes = (
        [100.0 + index * 0.05 for index in range(30)]
        + [100.0, 98.8, 97.6, 96.7, 97.2, 97.8, 98.4, 99.0]
        + [99.0 + index * 0.04 for index in range(42)]
    )
    volumes = [100.0] * 30 + [70.0] * 8 + [95.0] * 42
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [value + 0.4 for value in closes],
        "low": [value - 0.4 for value in closes],
        "close": closes,
        "volume": volumes,
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-downside-shock",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert "downside_shock" in diagnostics["components"]
    assert "rsi_washout" in diagnostics["components"]
    assert "quiet_volume" in diagnostics["components"]
    assert "local_low_reclaim" in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_intraday_session_liquidity_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "SessionLiquidityStrategy",
        "candidate_id": "session-liquidity-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "intraday_session_liquidity_reclaim",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 18,
            "buy_rsi_pullback": 42,
            "buy_rsi_recovery": 50,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 12,
            "buy_volume_factor": 1.05,
            "sell_rsi_exit": 62,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01 12:00:00", periods=80, freq="5min", tz="UTC")
    closes = (
        [100.0] * 12
        + [99.0 - index * 0.02 for index in range(16)]
        + [99.4, 99.8, 100.3, 100.6]
        + [100.6 + index * 0.01 for index in range(48)]
    )
    volumes = [100.0] * 28 + [180.0, 210.0, 240.0, 220.0] + [120.0] * 48
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [value + 0.2 for value in closes],
        "low": [value - 0.2 for value in closes],
        "close": closes,
        "volume": volumes,
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-session-liquidity",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert "session_window" in diagnostics["components"]
    assert "weekday_liquidity" in diagnostics["components"]
    assert "prior_vwap_discount" in diagnostics["components"]
    assert "vwap_reclaim" in diagnostics["components"]
    assert "controlled_atr" in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert diagnostics["components"]["vwap_reclaim"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_signed_volume_imbalance_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "SignedVolumeImbalanceStrategy",
        "candidate_id": "signed-volume-imbalance-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "signed_volume_imbalance_accumulation",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 12,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 12,
            "buy_volume_factor": 1.0,
            "sell_rsi_exit": 64,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=80, freq="5min", tz="UTC")
    closes = (
        [100.0 - index * 0.03 for index in range(16)]
        + [99.1, 98.9, 98.7, 98.6, 98.7, 98.8]
        + [99.0, 99.2, 99.3, 99.35, 99.4, 99.45]
        + [99.45 + index * 0.01 for index in range(52)]
    )
    opens = (
        [value + 0.12 for value in closes[:22]]
        + [value - 0.18 for value in closes[22:]]
    )
    volumes = [90.0] * 22 + [180.0] * 10 + [120.0] * 48
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": [max(open_, close) + 0.12 for open_, close in zip(opens, closes)],
        "low": [min(open_, close) - 0.12 for open_, close in zip(opens, closes)],
        "close": closes,
        "volume": volumes,
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-signed-volume-imbalance",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert "positive_signed_imbalance" in diagnostics["components"]
    assert "close_location_accumulation" in diagnostics["components"]
    assert "mid_reclaim" in diagnostics["components"]
    assert "not_breakout_chase" in diagnostics["components"]
    assert "controlled_range" in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert diagnostics["components"]["positive_signed_imbalance"]["individual_count"] > 0
    assert diagnostics["components"]["mid_reclaim"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_liquidity_recovery_horizon_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "LiquidityRecoveryHorizonStrategy",
        "candidate_id": "liquidity-recovery-horizon-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "liquidity_recovery_horizon",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 12,
            "buy_rsi_pullback": 38,
            "buy_rsi_recovery": 46,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 24,
            "buy_volume_factor": 0.95,
            "sell_rsi_exit": 64,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=96, freq="5min", tz="UTC")
    closes = (
        [100.0 + index * 0.01 for index in range(24)]
        + [100.2, 99.5, 98.9, 98.4, 98.0, 97.8, 97.7, 97.8, 97.9, 98.0, 98.1, 98.2]
        + [98.25 + index * 0.04 for index in range(60)]
    )
    opens = [close - 0.03 for close in closes]
    highs = [max(open_, close) + (0.70 if 24 <= index < 32 else 0.08) for index, (open_, close) in enumerate(zip(opens, closes))]
    lows = [min(open_, close) - (0.70 if 24 <= index < 32 else 0.08) for index, (open_, close) in enumerate(zip(opens, closes))]
    volumes = [100.0] * 24 + [260.0] * 8 + [120.0] * 64
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-liquidity-recovery-horizon",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert "recent_liquidity_stress" in diagnostics["components"]
    assert "liquidity_normalizing" in diagnostics["components"]
    assert "participation_recovered" in diagnostics["components"]
    assert "below_recovery_anchor" in diagnostics["components"]
    assert "controlled_cost_proxy" in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert diagnostics["components"]["recent_liquidity_stress"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_bipower_jump_decay_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "BipowerJumpDecayStrategy",
        "candidate_id": "bipower-jump-decay-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "bipower_jump_decay",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 12,
            "buy_rsi_pullback": 42,
            "buy_rsi_recovery": 50,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 24,
            "buy_volume_factor": 0.90,
            "sell_rsi_exit": 66,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=96, freq="5min", tz="UTC")
    closes = [100.0 + index * 0.005 for index in range(30)]
    closes += [102.4, 102.8, 103.0, 103.15, 103.25, 103.30]
    closes += [103.32 + index * 0.01 for index in range(60)]
    opens = [close - 0.02 for close in closes]
    highs = [max(open_, close) + 0.05 for open_, close in zip(opens, closes)]
    lows = [min(open_, close) - 0.05 for open_, close in zip(opens, closes)]
    volumes = [120.0] * len(closes)
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-bipower-jump-decay",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert "positive_jump_detected" in diagnostics["components"]
    assert "jump_dominates_continuous_variance" in diagnostics["components"]
    assert "continuous_variance_decaying" in diagnostics["components"]
    assert "post_jump_drift_positive" in diagnostics["components"]
    assert "not_overextended_after_jump" in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert diagnostics["components"]["positive_jump_detected"]["individual_count"] > 0
    assert diagnostics["components"]["jump_dominates_continuous_variance"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_directional_change_overshoot_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "DirectionalChangeOvershootStrategy",
        "candidate_id": "directional-change-overshoot-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "directional_change_overshoot",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 18,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 50,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 36,
            "buy_volume_factor": 0.95,
            "sell_rsi_exit": 66,
            "sell_timeout_candles": 72,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=120, freq="5min", tz="UTC")
    closes = [100.0] * 18 + [100.0 + index * 0.08 for index in range(102)]
    opens = [close - 0.03 for close in closes]
    highs = [max(open_, close) + 0.04 for open_, close in zip(opens, closes)]
    lows = [min(open_, close) - 0.04 for open_, close in zip(opens, closes)]
    volumes = [150.0] * len(closes)
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-directional-change-overshoot",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert diagnostics["entry_signal_count"] > 0
    assert "directional_change_confirmed" in diagnostics["components"]
    assert "overshoot_persisted" in diagnostics["components"]
    assert "event_time_trend_positive" in diagnostics["components"]
    assert "adverse_reversal_absent" in diagnostics["components"]
    assert "turnover_controlled" in diagnostics["components"]
    assert "positive_jump_detected" not in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert diagnostics["components"]["directional_change_confirmed"]["individual_count"] > 0
    assert diagnostics["components"]["overshoot_persisted"]["individual_count"] > 0
    assert diagnostics["components"]["event_time_trend_positive"]["individual_count"] > 0
    assert diagnostics["components"]["adverse_reversal_absent"]["individual_count"] > 0
    assert diagnostics["components"]["turnover_controlled"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_range_quarticity_vol_of_vol_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "RangeQuarticityVolOfVolStrategy",
        "candidate_id": "range-quarticity-vol-of-vol-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "range_quarticity_vol_of_vol_state",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 24,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 24,
            "buy_volume_factor": 0.90,
            "sell_rsi_exit": 64,
            "sell_timeout_candles": 96,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=144, freq="5min", tz="UTC")
    closes = [100.0 + index * 0.01 for index in range(144)]
    widths = [0.10] * 48 + [2.40] * 24 + [0.55] * 72
    volumes = [100.0] * 48 + [190.0] * 24 + [140.0] * 72
    opens = [close - 0.02 for close in closes]
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": [max(open_, close) + width / 2.0 for open_, close, width in zip(opens, closes, widths)],
        "low": [min(open_, close) - width / 2.0 for open_, close, width in zip(opens, closes, widths)],
        "close": closes,
        "volume": volumes,
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-range-quarticity-vol-of-vol",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert diagnostics["entry_signal_count"] > 0
    assert "range_quarticity_state_decay" in diagnostics["components"]
    assert "post_stress_stabilization" in diagnostics["components"]
    assert "participation_present" in diagnostics["components"]
    assert "range_not_reexpanding" in diagnostics["components"]
    assert "positive_stabilization_drift" in diagnostics["components"]
    assert "turnover_controlled" in diagnostics["components"]
    assert "directional_change_confirmed" not in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert diagnostics["components"]["range_quarticity_state_decay"]["individual_count"] > 0
    assert diagnostics["components"]["post_stress_stabilization"]["individual_count"] > 0
    assert diagnostics["components"]["participation_present"]["individual_count"] > 0
    assert diagnostics["components"]["range_not_reexpanding"]["individual_count"] > 0
    assert diagnostics["components"]["positive_stabilization_drift"]["individual_count"] > 0
    assert diagnostics["components"]["turnover_controlled"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_entropy_regime_transition_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "EntropyRegimeStrategy",
        "candidate_id": "entropy-regime-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "entropy_regime_transition",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 6,
            "buy_rsi_pullback": 38,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 12,
            "buy_volume_factor": 0.95,
            "sell_rsi_exit": 63,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=90, freq="5min", tz="UTC")
    closes = (
        [100.0 + (0.08 if index % 2 else -0.08) for index in range(30)]
        + [99.8 + index * 0.04 for index in range(30)]
        + [101.0 - index * 0.01 for index in range(30)]
    )
    volumes = [120.0] * 30 + [180.0] * 30 + [110.0] * 30
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.03 for value in closes],
        "high": [value + 0.12 for value in closes],
        "low": [value - 0.12 for value in closes],
        "close": closes,
        "volume": volumes,
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-entropy-regime",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert "low_directional_entropy" in diagnostics["components"]
    assert "efficiency_expanding" in diagnostics["components"]
    assert "positive_entropy_drift" in diagnostics["components"]
    assert "midline_hold" in diagnostics["components"]
    assert "range_not_extended" in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert "breakout_filter" not in diagnostics["components"]
    assert diagnostics["components"]["low_directional_entropy"]["individual_count"] > 0
    assert diagnostics["components"]["efficiency_expanding"]["individual_count"] > 0
    assert diagnostics["components"]["positive_entropy_drift"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_fractal_long_memory_regime_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "FractalMemoryStrategy",
        "candidate_id": "fractal-memory-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "fractal_long_memory_regime",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 12,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 18,
            "buy_volume_factor": 0.90,
            "sell_rsi_exit": 62,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=120, freq="5min", tz="UTC")
    increments = (
        [0.03, 0.05, 0.04, 0.06, -0.01, 0.04] * 10
        + [-0.04, -0.05, -0.03, 0.01, -0.04, -0.02] * 4
        + [0.05, 0.04, 0.06, -0.01, 0.05, 0.03] * 6
    )
    closes: list[float] = []
    current = 100.0
    for increment in increments:
        current += increment
        closes.append(current)
    volumes = [120.0 + (index % 7) * 8.0 for index in range(len(closes))]
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.02 for value in closes],
        "high": [value + 0.10 for value in closes],
        "low": [value - 0.10 for value in closes],
        "close": closes,
        "volume": volumes,
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-fractal-memory",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert "persistent_memory_regime" in diagnostics["components"]
    assert "efficient_path" in diagnostics["components"]
    assert "positive_fractal_drift" in diagnostics["components"]
    assert "midline_hold" in diagnostics["components"]
    assert "not_range_extension" in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert "breakout_filter" not in diagnostics["components"]
    assert diagnostics["components"]["persistent_memory_regime"]["individual_count"] > 0
    assert diagnostics["components"]["efficient_path"]["individual_count"] > 0
    assert diagnostics["components"]["positive_fractal_drift"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_semivariance_asymmetry_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "SemivarianceAsymmetryStrategy",
        "candidate_id": "semivariance-asymmetry-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "semivariance_asymmetry_regime",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 12,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 18,
            "buy_volume_factor": 0.85,
            "sell_rsi_exit": 62,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=120, freq="5min", tz="UTC")
    increments = (
        [0.05, 0.04, -0.01, 0.06, 0.03, -0.01] * 10
        + [-0.05, -0.04, 0.02, -0.03, 0.01, -0.02] * 4
        + [0.05, 0.03, -0.01, 0.04, 0.02, -0.01] * 6
    )
    closes: list[float] = []
    current = 100.0
    for increment in increments:
        current += increment
        closes.append(current)
    volumes = [130.0 + (index % 6) * 5.0 for index in range(len(closes))]
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.02 for value in closes],
        "high": [value + 0.08 for value in closes],
        "low": [value - 0.08 for value in closes],
        "close": closes,
        "volume": volumes,
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-semivariance-asymmetry",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert "good_volatility_dominance" in diagnostics["components"]
    assert "bad_volatility_decay" in diagnostics["components"]
    assert "positive_semivariance_drift" in diagnostics["components"]
    assert "controlled_range" in diagnostics["components"]
    assert "not_range_extension" in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert "breakout_filter" not in diagnostics["components"]
    assert diagnostics["components"]["good_volatility_dominance"]["individual_count"] > 0
    assert diagnostics["components"]["bad_volatility_decay"]["individual_count"] > 0
    assert diagnostics["components"]["positive_semivariance_drift"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_funding_pressure_carry_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "FundingPressureCarryStrategy",
        "candidate_id": "funding-pressure-carry-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "funding_pressure_carry",
        "timeframe": "5m",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 12,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 18,
            "buy_volume_factor": 0.90,
            "sell_rsi_exit": 62,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=360, freq="5min", tz="UTC")
    closes = [100.0 + index * 0.015 for index in range(len(dates))]
    volumes = [140.0 + (index % 8) * 6.0 for index in range(len(dates))]
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.02 for value in closes],
        "high": [value + 0.07 for value in closes],
        "low": [value - 0.07 for value in closes],
        "close": closes,
        "volume": volumes,
    }).to_csv(ohlcv_path, index=False)
    funding_dates = pd.date_range("2025-01-01", periods=6, freq="8h", tz="UTC")
    funding_rates = [-0.00035, -0.00025, -0.00015, -0.00008, -0.00003, 0.00001]
    funding_path = tmp_path / "funding.csv"
    pd.DataFrame({
        "date": funding_dates,
        "open": funding_rates,
        "high": [0.0] * len(funding_rates),
        "low": [0.0] * len(funding_rates),
        "close": [0.0] * len(funding_rates),
        "volume": [0.0] * len(funding_rates),
    }).to_csv(funding_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        funding_rate_path=funding_path,
        diagnostics_id="diag-funding-pressure-carry",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert diagnostics["funding_rate_merge"]["matched_row_count"] > 0
    assert "negative_funding_pressure" in diagnostics["components"]
    assert "funding_pressure_releasing" in diagnostics["components"]
    assert "price_resilience" in diagnostics["components"]
    assert "not_positive_crowding" in diagnostics["components"]
    assert "controlled_range" in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert "breakout_filter" not in diagnostics["components"]
    assert diagnostics["components"]["negative_funding_pressure"]["individual_count"] > 0
    assert diagnostics["components"]["funding_pressure_releasing"]["individual_count"] > 0
    assert diagnostics["components"]["price_resilience"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_realized_skewness_tail_shape_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "RealizedSkewnessTailStrategy",
        "candidate_id": "realized-skewness-tail-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "realized_skewness_tail_shape",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 12,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 18,
            "buy_volume_factor": 0.90,
            "sell_rsi_exit": 62,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=144, freq="5min", tz="UTC")
    increments = [0.08, 0.06, 0.05, 0.04, -0.18, 0.07, 0.05, 0.04] * 18
    closes: list[float] = []
    current = 100.0
    for increment in increments:
        current += increment
        closes.append(current)
    volumes = [135.0 + (index % 8) * 7.0 for index in range(len(closes))]
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.02 for value in closes],
        "high": [value + 0.09 for value in closes],
        "low": [value - 0.09 for value in closes],
        "close": closes,
        "volume": volumes,
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-realized-skewness-tail",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert "low_realized_skewness" in diagnostics["components"]
    assert "kurtosis_risk_premium" in diagnostics["components"]
    assert "lottery_tail_cooling" in diagnostics["components"]
    assert "positive_tail_shape_drift" in diagnostics["components"]
    assert "controlled_range" in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert "breakout_filter" not in diagnostics["components"]
    assert diagnostics["components"]["low_realized_skewness"]["individual_count"] > 0
    assert diagnostics["components"]["kurtosis_risk_premium"]["individual_count"] > 0
    assert diagnostics["components"]["positive_tail_shape_drift"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_calendar_turnover_seasonality_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "CalendarTurnoverStrategy",
        "candidate_id": "calendar-turnover-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "calendar_turnover_seasonality",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 24,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 48,
            "buy_volume_factor": 1.00,
            "sell_rsi_exit": 62,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-02", periods=7 * 24 * 12, freq="5min", tz="UTC")
    closes: list[float] = []
    volumes: list[float] = []
    current = 100.0
    for index, date in enumerate(dates):
        current += 0.01 + (0.02 if date.dayofweek in {0, 3} else 0.0)
        closes.append(current)
        if date.dayofweek in {5, 6}:
            volumes.append(85.0 + (index % 12) * 2.0)
        elif date.dayofweek in {0, 3}:
            volumes.append(190.0 + (index % 12) * 8.0)
        else:
            volumes.append(145.0 + (index % 12) * 4.0)
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.02 for value in closes],
        "high": [value + 0.08 for value in closes],
        "low": [value - 0.08 for value in closes],
        "close": closes,
        "volume": volumes,
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-calendar-turnover",
        timerange="20250102-20250109",
    ))

    assert diagnostics["status"] == "completed"
    assert "calendar_risk_window" in diagnostics["components"]
    assert "weekend_discount_context" in diagnostics["components"]
    assert "turnover_recovery" in diagnostics["components"]
    assert "positive_calendar_drift" in diagnostics["components"]
    assert "not_breakout_chase" in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert "breakout_filter" not in diagnostics["components"]
    assert diagnostics["components"]["calendar_risk_window"]["individual_count"] > 0
    assert diagnostics["components"]["weekend_discount_context"]["individual_count"] > 0
    assert diagnostics["components"]["turnover_recovery"]["individual_count"] > 0
    assert diagnostics["components"]["positive_calendar_drift"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_amihud_illiquidity_premium_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "AmihudIlliquidityStrategy",
        "candidate_id": "amihud-illiquidity-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "amihud_illiquidity_premium",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 24,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 36,
            "buy_volume_factor": 0.90,
            "sell_rsi_exit": 62,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=240, freq="5min", tz="UTC")
    closes: list[float] = []
    volumes: list[float] = []
    current = 100.0
    for index, _date in enumerate(dates):
        if index % 16 in {4, 5}:
            current += 0.42 - (index % 2) * 0.10
            volumes.append(70.0 + index % 5)
        else:
            current += 0.03
            volumes.append(145.0 + (index % 10) * 3.0)
        closes.append(current)
    ohlcv_path = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.03 for value in closes],
        "high": [value + 0.10 for value in closes],
        "low": [value - 0.10 for value in closes],
        "close": closes,
        "volume": volumes,
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-amihud-illiquidity",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert "price_impact_premium" in diagnostics["components"]
    assert "illiquidity_releasing" in diagnostics["components"]
    assert "not_extreme_impact" in diagnostics["components"]
    assert "price_resilience" in diagnostics["components"]
    assert "positive_illiquidity_drift" in diagnostics["components"]
    assert "volume_floor" in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert "breakout_filter" not in diagnostics["components"]
    assert diagnostics["components"]["price_impact_premium"]["individual_count"] > 0
    assert diagnostics["components"]["illiquidity_releasing"]["individual_count"] > 0
    assert diagnostics["components"]["not_extreme_impact"]["individual_count"] > 0
    assert diagnostics["components"]["price_resilience"]["individual_count"] > 0
    assert diagnostics["components"]["positive_illiquidity_drift"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_cross_asset_lead_lag_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "CrossAssetLeadLagStrategy",
        "candidate_id": "cross-asset-lead-lag-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "cross_asset_lead_lag",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 24,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 36,
            "buy_volume_factor": 0.90,
            "sell_rsi_exit": 62,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=240, freq="5min", tz="UTC")
    btc_closes: list[float] = []
    eth_closes: list[float] = []
    btc_current = 100.0
    eth_current = 200.0
    for index, _date in enumerate(dates):
        btc_current += 0.03
        eth_current += 0.04
        if index % 24 == 5:
            eth_current += 1.20
        if index % 24 == 7:
            btc_current += 0.45
        btc_closes.append(btc_current)
        eth_closes.append(eth_current)

    ohlcv_path = tmp_path / "btc_ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.02 for value in btc_closes],
        "high": [value + 0.08 for value in btc_closes],
        "low": [value - 0.08 for value in btc_closes],
        "close": btc_closes,
        "volume": [180.0 + (index % 8) * 4.0 for index in range(len(dates))],
    }).to_csv(ohlcv_path, index=False)
    informative_path = tmp_path / "eth_ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.04 for value in eth_closes],
        "high": [value + 0.12 for value in eth_closes],
        "low": [value - 0.12 for value in eth_closes],
        "close": eth_closes,
        "volume": [220.0 + (index % 9) * 5.0 for index in range(len(dates))],
    }).to_csv(informative_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        informative_ohlcv_path=informative_path,
        diagnostics_id="diag-cross-asset-lead-lag",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert diagnostics["informative_ohlcv_merge"]["requested"] is True
    assert diagnostics["informative_ohlcv_merge"]["matched_row_count"] > 0
    assert "eth_positive_lead" in diagnostics["components"]
    assert "btc_lag_discount" in diagnostics["components"]
    assert "spread_not_extreme" in diagnostics["components"]
    assert "btc_resilience" in diagnostics["components"]
    assert "positive_cross_asset_drift" in diagnostics["components"]
    assert "trend_filter" not in diagnostics["components"]
    assert "breakout_filter" not in diagnostics["components"]
    assert diagnostics["components"]["eth_positive_lead"]["individual_count"] > 0
    assert diagnostics["components"]["btc_lag_discount"]["individual_count"] > 0
    assert diagnostics["components"]["spread_not_extreme"]["individual_count"] > 0
    assert diagnostics["components"]["btc_resilience"]["individual_count"] > 0
    assert diagnostics["components"]["positive_cross_asset_drift"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_cross_asset_cointegration_spread_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "CrossAssetCointegrationStrategy",
        "candidate_id": "cross-asset-cointegration-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "cross_asset_cointegration_spread",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 36,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 72,
            "buy_volume_factor": 0.85,
            "sell_rsi_exit": 62,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=300, freq="5min", tz="UTC")
    btc_closes: list[float] = []
    eth_closes: list[float] = []
    btc_current = 100.0
    eth_current = 200.0
    for index, _date in enumerate(dates):
        if 80 <= index < 160:
            btc_current += 0.015
            eth_current += 0.085
        elif 160 <= index < 240:
            btc_current += 0.105
            eth_current += 0.055
        else:
            btc_current += 0.045
            eth_current += 0.055
        btc_closes.append(btc_current)
        eth_closes.append(eth_current)

    ohlcv_path = tmp_path / "btc_ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.03 for value in btc_closes],
        "high": [value + 0.18 for value in btc_closes],
        "low": [value - 0.18 for value in btc_closes],
        "close": btc_closes,
        "volume": [190.0 + (index % 10) * 3.0 for index in range(len(dates))],
    }).to_csv(ohlcv_path, index=False)
    informative_path = tmp_path / "eth_ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.04 for value in eth_closes],
        "high": [value + 0.24 for value in eth_closes],
        "low": [value - 0.24 for value in eth_closes],
        "close": eth_closes,
        "volume": [230.0 + (index % 12) * 4.0 for index in range(len(dates))],
    }).to_csv(informative_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        informative_ohlcv_path=informative_path,
        diagnostics_id="diag-cross-asset-cointegration",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert diagnostics["informative_ohlcv_merge"]["requested"] is True
    assert diagnostics["informative_ohlcv_merge"]["matched_row_count"] > 0
    assert "btc_discount_to_eth" in diagnostics["components"]
    assert "spread_reversion_turn" in diagnostics["components"]
    assert "eth_market_support" in diagnostics["components"]
    assert "btc_resilience" in diagnostics["components"]
    assert "cointegration_spread_not_extreme" in diagnostics["components"]
    assert "eth_positive_lead" not in diagnostics["components"]
    assert "variance_ratio_expansion" not in diagnostics["components"]
    assert diagnostics["components"]["btc_discount_to_eth"]["individual_count"] > 0
    assert diagnostics["components"]["spread_reversion_turn"]["individual_count"] > 0
    assert diagnostics["components"]["eth_market_support"]["individual_count"] > 0
    assert diagnostics["components"]["btc_resilience"]["individual_count"] > 0
    assert diagnostics["components"]["cointegration_spread_not_extreme"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_cross_asset_correlation_recovery_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "CrossAssetCorrelationStrategy",
        "candidate_id": "cross-asset-correlation-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "cross_asset_correlation_recovery",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 36,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 72,
            "buy_volume_factor": 0.85,
            "sell_rsi_exit": 62,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=300, freq="5min", tz="UTC")
    btc_closes: list[float] = []
    eth_closes: list[float] = []
    btc_current = 100.0
    eth_current = 200.0
    for index, _date in enumerate(dates):
        alternating = 1.0 if index % 2 == 0 else -1.0
        if index < 140:
            btc_current += 0.020 * alternating + 0.030
            eth_current += -0.018 * alternating + 0.040
        elif index < 240:
            btc_current += 0.045 * alternating + 0.055
            eth_current += 0.035 * alternating + 0.040
        else:
            btc_current += 0.030 * alternating + 0.050
            eth_current += 0.025 * alternating + 0.045
        btc_closes.append(btc_current)
        eth_closes.append(eth_current)

    ohlcv_path = tmp_path / "btc_ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.03 for value in btc_closes],
        "high": [value + 0.18 for value in btc_closes],
        "low": [value - 0.18 for value in btc_closes],
        "close": btc_closes,
        "volume": [190.0 + (index % 10) * 3.0 for index in range(len(dates))],
    }).to_csv(ohlcv_path, index=False)
    informative_path = tmp_path / "eth_ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.04 for value in eth_closes],
        "high": [value + 0.24 for value in eth_closes],
        "low": [value - 0.24 for value in eth_closes],
        "close": eth_closes,
        "volume": [230.0 + (index % 12) * 4.0 for index in range(len(dates))],
    }).to_csv(informative_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        informative_ohlcv_path=informative_path,
        diagnostics_id="diag-cross-asset-correlation",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert diagnostics["informative_ohlcv_merge"]["requested"] is True
    assert diagnostics["informative_ohlcv_merge"]["matched_row_count"] > 0
    assert "correlation_breakdown" in diagnostics["components"]
    assert "correlation_recovery" in diagnostics["components"]
    assert "btc_relative_recovery" in diagnostics["components"]
    assert "btc_discount_to_eth" not in diagnostics["components"]
    assert "eth_positive_lead" not in diagnostics["components"]
    assert diagnostics["components"]["correlation_breakdown"]["individual_count"] > 0
    assert diagnostics["components"]["correlation_recovery"]["individual_count"] > 0
    assert diagnostics["components"]["btc_relative_recovery"]["individual_count"] > 0
    assert diagnostics["components"]["eth_market_support"]["individual_count"] > 0
    assert diagnostics["components"]["btc_resilience"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_market_beta_drawdown_carry_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "MarketBetaDrawdownCarryStrategy",
        "candidate_id": "market-beta-drawdown-carry-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "market_beta_drawdown_carry",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 24,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 72,
            "buy_volume_factor": 0.90,
            "sell_rsi_exit": 78,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=420, freq="5min", tz="UTC")
    closes: list[float] = []
    current = 100.0
    for index, _date in enumerate(dates):
        if index < 120:
            current += 0.025
        elif index < 220:
            current -= 0.035
        else:
            current += 0.018 if index % 2 == 0 else 0.006
        closes.append(current)

    ohlcv_path = tmp_path / "btc_ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - (0.08 if index >= 220 else 0.02) for index, value in enumerate(closes)],
        "high": [value + 0.18 for value in closes],
        "low": [value - 0.18 for value in closes],
        "close": closes,
        "volume": [220.0 + (index % 12) * 4.0 for index in range(len(dates))],
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-market-beta-drawdown-carry",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert "moderate_drawdown" in diagnostics["components"]
    assert "volatility_budget" in diagnostics["components"]
    assert "positive_candle_reentry" in diagnostics["components"]
    assert "beta_resilience" in diagnostics["components"]
    assert "participation_floor" in diagnostics["components"]
    assert "not_overheated" in diagnostics["components"]
    assert "correlation_breakdown" not in diagnostics["components"]
    assert "negative_funding_pressure" not in diagnostics["components"]
    assert diagnostics["components"]["moderate_drawdown"]["individual_count"] > 0
    assert diagnostics["components"]["volatility_budget"]["individual_count"] > 0
    assert diagnostics["components"]["positive_candle_reentry"]["individual_count"] > 0
    assert diagnostics["components"]["beta_resilience"]["individual_count"] > 0
    assert diagnostics["components"]["participation_floor"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_regime_state_reentry_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "RegimeStateReentryStrategy",
        "candidate_id": "regime-state-reentry-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "regime_state_reentry",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 24,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 72,
            "buy_volume_factor": 0.75,
            "sell_rsi_exit": 82,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=420, freq="5min", tz="UTC")
    closes: list[float] = []
    current = 100.0
    for index, _date in enumerate(dates):
        current += 0.030 if index % 5 else 0.010
        closes.append(current)

    ohlcv_path = tmp_path / "btc_ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.02 for value in closes],
        "high": [value + 0.16 for value in closes],
        "low": [value - 0.16 for value in closes],
        "close": closes,
        "volume": [210.0 + (index % 10) * 4.0 for index in range(len(dates))],
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-regime-state-reentry",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert "positive_regime_drift" in diagnostics["components"]
    assert "state_stability" in diagnostics["components"]
    assert "volatility_state_budget" in diagnostics["components"]
    assert "trendline_support" in diagnostics["components"]
    assert "closed_candle_reentry" in diagnostics["components"]
    assert "drawdown_state_intact" in diagnostics["components"]
    assert "participation_floor" in diagnostics["components"]
    assert "correlation_breakdown" not in diagnostics["components"]
    assert "negative_funding_pressure" not in diagnostics["components"]
    assert diagnostics["components"]["positive_regime_drift"]["individual_count"] > 0
    assert diagnostics["components"]["state_stability"]["individual_count"] > 0
    assert diagnostics["components"]["volatility_state_budget"]["individual_count"] > 0
    assert diagnostics["components"]["trendline_support"]["individual_count"] > 0
    assert diagnostics["components"]["closed_candle_reentry"]["individual_count"] > 0
    assert diagnostics["components"]["drawdown_state_intact"]["individual_count"] > 0
    assert diagnostics["components"]["participation_floor"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_mark_price_dislocation_reclaim_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "MarkPriceDislocationStrategy",
        "candidate_id": "mark-price-dislocation-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "mark_price_dislocation_reclaim",
        "timeframe": "5m",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 12,
            "buy_rsi_pullback": 38,
            "buy_rsi_recovery": 46,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 72,
            "buy_volume_factor": 0.50,
            "sell_rsi_exit": 66,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=420, freq="5min", tz="UTC")
    closes = [98.5 + index * 0.004 for index in range(len(dates))]
    ohlcv_path = tmp_path / "btc_ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.03 for value in closes],
        "high": [value + 0.12 for value in closes],
        "low": [value - 0.12 for value in closes],
        "close": closes,
        "volume": [180.0 + (index % 8) * 3.0 for index in range(len(dates))],
    }).to_csv(ohlcv_path, index=False)

    mark_dates = pd.date_range("2025-01-01", periods=12, freq="4h", tz="UTC")
    mark_path = tmp_path / "btc_mark.csv"
    pd.DataFrame({
        "date": mark_dates,
        "open": [100.0 for _ in mark_dates],
        "high": [100.2 for _ in mark_dates],
        "low": [99.8 for _ in mark_dates],
        "close": [100.0 for _ in mark_dates],
        "volume": [0.0 for _ in mark_dates],
    }).to_csv(mark_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        informative_ohlcv_path=mark_path,
        diagnostics_id="diag-mark-price-dislocation",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert diagnostics["informative_ohlcv_merge"]["mark_close_column_present_after_merge"] is True
    assert "mark_discount_pressure" in diagnostics["components"]
    assert "mark_gap_reclaiming" in diagnostics["components"]
    assert "mark_price_support" in diagnostics["components"]
    assert "discount_not_extreme" in diagnostics["components"]
    assert "price_resilience" in diagnostics["components"]
    assert "participation_floor" in diagnostics["components"]
    assert "negative_funding_pressure" not in diagnostics["components"]
    assert "eth_market_support" not in diagnostics["components"]
    assert diagnostics["components"]["mark_discount_pressure"]["individual_count"] > 0
    assert diagnostics["components"]["mark_gap_reclaiming"]["individual_count"] > 0
    assert diagnostics["components"]["mark_price_support"]["individual_count"] > 0
    assert diagnostics["components"]["discount_not_extreme"]["individual_count"] > 0
    assert diagnostics["components"]["participation_floor"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_mark_discount_reclaim_continuation_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "MarkDiscountReclaimStrategy",
        "candidate_id": "mark-discount-reclaim-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "mark_discount_reclaim_continuation",
        "timeframe": "5m",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 6,
            "buy_rsi_pullback": 38,
            "buy_rsi_recovery": 46,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 72,
            "buy_volume_factor": 0.0,
            "sell_rsi_exit": 66,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=420, freq="5min", tz="UTC")
    closes = [99.80 + index * 0.002 for index in range(len(dates))]
    ohlcv_path = tmp_path / "btc_ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.02 for value in closes],
        "high": [value + 0.08 for value in closes],
        "low": [value - 0.08 for value in closes],
        "close": closes,
        "volume": [150.0 + (index % 6) * 2.0 for index in range(len(dates))],
    }).to_csv(ohlcv_path, index=False)

    mark_dates = pd.date_range("2025-01-01", periods=12, freq="4h", tz="UTC")
    mark_path = tmp_path / "btc_mark.csv"
    pd.DataFrame({
        "date": mark_dates,
        "open": [100.0 for _ in mark_dates],
        "high": [100.2 for _ in mark_dates],
        "low": [99.8 for _ in mark_dates],
        "close": [100.0 for _ in mark_dates],
        "volume": [0.0 for _ in mark_dates],
    }).to_csv(mark_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        informative_ohlcv_path=mark_path,
        diagnostics_id="diag-mark-discount-reclaim",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert diagnostics["informative_ohlcv_merge"]["mark_close_column_present_after_merge"] is True
    assert "mark_discount_pressure" in diagnostics["components"]
    assert "six_candle_discount_reclaim" in diagnostics["components"]
    assert "short_return_nonnegative" in diagnostics["components"]
    assert "negative_funding_pressure" not in diagnostics["components"]
    assert "mark_gap_reclaiming" not in diagnostics["components"]
    assert diagnostics["components"]["mark_discount_pressure"]["individual_count"] > 0
    assert diagnostics["components"]["six_candle_discount_reclaim"]["individual_count"] > 0
    assert diagnostics["components"]["short_return_nonnegative"]["individual_count"] > 0
    assert diagnostics["entry_signal_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_mark_fair_value_momentum_lag_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "MarkFairValueMomentumLagStrategy",
        "candidate_id": "mark-fair-value-momentum-lag-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "mark_fair_value_momentum_lag",
        "timeframe": "5m",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 12,
            "buy_rsi_pullback": 38,
            "buy_rsi_recovery": 46,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 48,
            "buy_volume_factor": 0.0,
            "sell_rsi_exit": 72,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=420, freq="5min", tz="UTC")
    closes = [100.0 for _ in range(len(dates))]
    ohlcv_path = tmp_path / "btc_ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [value + 0.10 for value in closes],
        "low": [value - 0.10 for value in closes],
        "close": closes,
        "volume": [150.0 for _ in closes],
    }).to_csv(ohlcv_path, index=False)

    mark_dates = pd.date_range("2025-01-01", periods=12, freq="4h", tz="UTC")
    mark_closes = [100.0 + index * 0.40 for index in range(len(mark_dates))]
    mark_path = tmp_path / "btc_mark.csv"
    pd.DataFrame({
        "date": mark_dates,
        "open": mark_closes,
        "high": [value + 0.05 for value in mark_closes],
        "low": [value - 0.05 for value in mark_closes],
        "close": mark_closes,
        "volume": [0.0 for _ in mark_dates],
    }).to_csv(mark_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        informative_ohlcv_path=mark_path,
        diagnostics_id="diag-mark-fair-value-momentum-lag",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert diagnostics["informative_ohlcv_merge"]["mark_close_column_present_after_merge"] is True
    assert "mark_fair_value_momentum" in diagnostics["components"]
    assert "traded_price_lag" in diagnostics["components"]
    assert "range_budget" in diagnostics["components"]
    assert "participation_floor" in diagnostics["components"]
    assert "event_cooldown" in diagnostics["components"]
    assert "mark_discount_pressure" not in diagnostics["components"]
    assert "negative_funding_pressure" not in diagnostics["components"]
    assert diagnostics["components"]["mark_fair_value_momentum"]["individual_count"] > 0
    assert diagnostics["components"]["traded_price_lag"]["individual_count"] > 0
    assert diagnostics["components"]["range_budget"]["individual_count"] > 0
    assert diagnostics["components"]["participation_floor"]["individual_count"] > 0
    assert diagnostics["components"]["event_cooldown"]["individual_count"] > 0
    assert diagnostics["entry_signal_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_microstructure_spread_reversion_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "MicrostructureSpreadStrategy",
        "candidate_id": "microstructure-spread-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "microstructure_spread_reversion",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 24,
            "buy_rsi_pullback": 40,
            "buy_rsi_recovery": 48,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 72,
            "buy_volume_factor": 0.50,
            "sell_rsi_exit": 64,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=360, freq="5min", tz="UTC")
    closes = [
        100.0 + (0.05 if index % 2 == 0 else -0.04) + index * 0.002
        for index in range(len(dates))
    ]
    ohlcv_path = tmp_path / "btc_ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.02 for value in closes],
        "high": [value + 0.08 for value in closes],
        "low": [value - 0.08 for value in closes],
        "close": closes,
        "volume": [220.0 + (index % 10) * 5.0 for index in range(len(dates))],
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-microstructure-spread",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert "spread_pressure" in diagnostics["components"]
    assert "spread_compressing" in diagnostics["components"]
    assert "hl_spread_normalizing" in diagnostics["components"]
    assert "price_resilience" in diagnostics["components"]
    assert "positive_recovery" in diagnostics["components"]
    assert "participation_floor" in diagnostics["components"]
    assert "mark_discount_pressure" not in diagnostics["components"]
    assert "negative_funding_pressure" not in diagnostics["components"]
    assert diagnostics["components"]["spread_pressure"]["individual_count"] > 0
    assert diagnostics["components"]["spread_compressing"]["individual_count"] > 0
    assert diagnostics["components"]["hl_spread_normalizing"]["individual_count"] > 0
    assert diagnostics["components"]["positive_recovery"]["individual_count"] > 0
    assert diagnostics["components"]["participation_floor"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_supports_variance_ratio_regime_switch_variant(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "VarianceRatioRegimeStrategy",
        "candidate_id": "variance-ratio-regime-candidate",
        "generator_mode": "rule_based",
        "strategy_logic_variant": "variance_ratio_regime_switch",
        "parameter_defaults": {
            "buy_rsi_window": 14,
            "buy_pullback_lookback": 24,
            "buy_rsi_pullback": 42,
            "buy_rsi_recovery": 50,
            "buy_ema_fast": 12,
            "buy_ema_slow": 48,
            "buy_volume_window": 72,
            "buy_volume_factor": 0.90,
            "sell_rsi_exit": 62,
        },
    }), encoding="utf-8")
    dates = pd.date_range("2025-01-01", periods=260, freq="5min", tz="UTC")
    closes: list[float] = []
    current = 100.0
    increment = 0.02
    for index, _date in enumerate(dates):
        regime_boost = 0.035 if 80 <= index < 180 else 0.008
        increment = (increment * 0.82) + regime_boost
        current += increment
        closes.append(current)

    ohlcv_path = tmp_path / "btc_ohlcv.csv"
    pd.DataFrame({
        "date": dates,
        "open": [value - 0.04 for value in closes],
        "high": [value + 0.42 for value in closes],
        "low": [value - 0.42 for value in closes],
        "close": closes,
        "volume": [200.0 + (index % 12) * 3.0 for index in range(len(dates))],
    }).to_csv(ohlcv_path, index=False)

    diagnostics = diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        ohlcv_path=ohlcv_path,
        diagnostics_id="diag-variance-ratio-regime",
        timerange="20250101-20250102",
    ))

    assert diagnostics["status"] == "completed"
    assert "variance_ratio_expansion" in diagnostics["components"]
    assert "positive_autocorr_regime" in diagnostics["components"]
    assert "positive_regime_drift" in diagnostics["components"]
    assert "controlled_regime_return" in diagnostics["components"]
    assert "midline_resilience" in diagnostics["components"]
    assert "persistent_memory_regime" not in diagnostics["components"]
    assert "eth_positive_lead" not in diagnostics["components"]
    assert diagnostics["components"]["variance_ratio_expansion"]["individual_count"] > 0
    assert diagnostics["components"]["positive_autocorr_regime"]["individual_count"] > 0
    assert diagnostics["components"]["positive_regime_drift"]["individual_count"] > 0
    assert diagnostics["components"]["controlled_regime_return"]["individual_count"] > 0
    assert diagnostics["components"]["midline_resilience"]["individual_count"] > 0
    assert diagnostics["ml_filter"]["target_column_present"] is None


def test_signal_diagnostics_blocks_paths_outside_workspace(tmp_path):
    import pytest

    outside = tmp_path.parent / f"{tmp_path.name}_outside_metadata.json"
    outside.write_text(json.dumps({"strategy_name": "Outside"}), encoding="utf-8")

    with pytest.raises(ValueError, match="inside the workspace"):
        diagnose_candidate_signals(CandidateSignalDiagnosticsInputs(
            root_dir=tmp_path,
            generated_metadata_path=outside,
            ohlcv_path=tmp_path / "ohlcv.csv",
        ))


def test_candidate_failure_synthesis_builds_theory_first_next_brief(tmp_path):
    from freqtrade_ext.bot_factory.candidate_failure_synthesis import (
        CandidateFailureSynthesisInputs,
        synthesize_candidate_failures,
        write_candidate_failure_synthesis_artifacts,
    )

    manifest_a = tmp_path / "manifest_a.json"
    manifest_a.write_text(json.dumps({
        "checks": [
            {
                "name": "historical_backtest",
                "status": "fail",
                "payload_summary": {"trade_count": 0, "total_return_pct": 0.0},
            },
            {"name": "walk_forward", "status": "fail"},
        ]
    }), encoding="utf-8")
    manifest_b = tmp_path / "manifest_b.json"
    manifest_b.write_text(json.dumps({
        "checks": [
            {
                "name": "historical_backtest",
                "status": "fail",
                "payload_summary": {"trade_count": 12, "total_return_pct": -0.5},
            },
            {"name": "walk_forward", "status": "fail"},
        ]
    }), encoding="utf-8")
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(json.dumps({
        "ranking_id": "rank-test",
        "paper_ready_candidate_ids": [],
        "ranked_candidates": [
            {
                "candidate_id": "cand-a",
                "strategy_name": "StrategyA",
                "manifest_path": "manifest_a.json",
                "recommendation": "retry",
                "paper_ready_eligible": False,
                "paper_ready_blockers": ["historical_backtest", "walk_forward"],
                "failure_taxonomy_codes": ["FAIL_REGIME_FRAGILE"],
                "hypothesis_family": "liquidity_mean_reversion",
                "thesis": {"thesis_id": "TH-A"},
                "metrics": {
                    "historical_trade_count": 0,
                    "historical_total_return_pct": 0.0,
                    "walk_forward_pass_rate": 0.0,
                },
                "rank": 1,
            },
            {
                "candidate_id": "cand-b",
                "strategy_name": "StrategyB",
                "manifest_path": "manifest_b.json",
                "recommendation": "retry",
                "paper_ready_eligible": False,
                "paper_ready_blockers": ["historical_backtest", "walk_forward"],
                "failure_taxonomy_codes": ["FAIL_COST_SENSITIVE"],
                "hypothesis_family": "volatility_breakout",
                "thesis": {"thesis_id": "TH-B"},
                "metrics": {
                    "historical_trade_count": 12,
                    "historical_total_return_pct": -0.5,
                    "walk_forward_pass_rate": 0.0,
                },
                "rank": 2,
            },
        ],
    }), encoding="utf-8")
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text(json.dumps({
        "candidate_id": "cand-a",
        "status": "completed",
        "entry_count": 0,
        "diagnosis_codes": ["ZERO_ENTRY_SIGNALS", "ML_FILTER_UNAVAILABLE"],
        "first_zero_component": "ml_filter",
        "bottleneck_components": [{"name": "ml_filter", "all_except_count": 4}],
    }), encoding="utf-8")
    freqai_predictions_path = tmp_path / "freqai_predictions.json"
    freqai_predictions_path.write_text(json.dumps({
        "candidate_id": "cand-a",
        "status": "completed",
        "expected_target_column": "&-future_return",
        "expected_target_column_present": False,
        "target_columns": ["&-long_return"],
        "model_label_columns": ["&-long_return"],
        "prediction_file_count": 1,
        "row_count": 2,
        "diagnosis_codes": [
            "PREDICTION_FILES_PRESENT",
            "EXPECTED_TARGET_PREDICTION_MISSING",
            "PREDICTION_TARGET_MISMATCH",
            "MODEL_LABEL_MISMATCH",
        ],
    }), encoding="utf-8")
    local_falsification_path = tmp_path / "local_falsification.json"
    local_falsification_path.write_text(json.dumps({
        "factory": "research_local_falsification",
        "status": "failed",
        "thesis_id": "TH-LOCAL-REJECTED",
        "mechanism_class": "low_range_volume_absorption",
        "expected_edge_bps": -4.0,
        "all_in_cost_bps": 12.0,
        "net_edge_bps": -16.0,
        "sample_count": 78,
        "data_span_days": 397.0,
        "profitable_windows_ratio": 0.0,
        "calendar_window_frequency": "quarter",
        "calendar_window_count": 2,
        "profitable_calendar_windows_ratio": 0.0,
        "calendar_window_summaries": [
            {
                "calendar_window": "2026Q1",
                "sample_count": 40,
                "expected_edge_bps": -2.0,
                "net_edge_bps": -14.0,
                "win_rate": 0.45,
                "profitable": False,
            },
            {
                "calendar_window": "2026Q2",
                "sample_count": 38,
                "expected_edge_bps": -6.0,
                "net_edge_bps": -18.0,
                "win_rate": 0.42,
                "profitable": False,
            },
        ],
        "event_source": {
            "used": True,
            "factory": "research_local_event_builder",
            "factory_valid": True,
            "status": "completed",
            "status_completed": True,
            "thesis_id": "TH-LOCAL-REJECTED",
            "thesis_matches": True,
            "events_csv_path": "registry/strategies/research_decisions/events.csv",
            "event_path_matches": True,
            "source_ohlcv_path": "user_data/data/bybit/futures/BTC_USDT-5m.parquet",
            "ohlcv_path_matches": True,
            "event_count": 78,
            "safety_scope_valid": True,
            "context_features_used": False,
            "required_contexts": [],
            "failure_synthesis_used": True,
            "failure_synthesis_parseable": True,
            "failure_synthesis_path": (
                "registry/strategies/synthesis/candidate_failure_synthesis.json"
            ),
            "failure_synthesis_thesis_repeats": False,
            "failure_synthesis_mechanism_repeats": False,
            "failure_synthesis_allow_failed_thesis_or_family": False,
            "failure_synthesis_guard_valid": True,
        },
        "safety_scope": {
            "historical_only": True,
            "backtest_started": False,
            "strategy_code_generated": False,
            "paper_trading_started": False,
            "dry_run_trading_started": False,
            "live_trading": False,
            "exchange_order_placement": False,
            "shorting": False,
            "leverage": 1.0,
            "process_control": False,
        },
        "blockers": [{"name": "net_edge_exceeds_cost"}],
    }), encoding="utf-8")

    synthesis = synthesize_candidate_failures(CandidateFailureSynthesisInputs(
        root_dir=tmp_path,
        ranking_path=ranking_path,
        signal_diagnostics_paths=[diagnostics_path],
        freqai_prediction_diagnostics_paths=[freqai_predictions_path],
        local_falsification_paths=[local_falsification_path],
        synthesis_id="synth-test",
        reviewer_notes=["synthesis test only"],
    ))

    brief = synthesis["next_research_brief"]
    aggregate = synthesis["aggregate_failure_summary"]
    assert synthesis["status"] == "completed"
    assert aggregate["paper_ready_count"] == 0
    assert aggregate["zero_trade_candidate_ids"] == ["cand-a"]
    assert aggregate["negative_return_candidate_ids"] == ["cand-b"]
    assert aggregate["signal_bottlenecks"][0]["first_zero_component"] == "ml_filter"
    assert aggregate["freqai_target_mismatch_candidate_ids"] == ["cand-a"]
    assert aggregate["local_falsification_rejection_artifact_count"] == 1
    assert aggregate["local_falsification_rejection_count"] == 1
    assert aggregate["local_falsification_invalid_rejection_count"] == 0
    rejection = aggregate["local_falsification_rejection_artifacts"][0]
    assert rejection["rejection_valid"] is True
    assert rejection["safety_scope_valid"] is True
    assert rejection["event_source_valid"] is True
    assert rejection["event_source_failure_synthesis_guard_valid"] is True
    assert rejection["calendar_window_frequency"] == "quarter"
    assert rejection["calendar_window_count"] == 2.0
    assert rejection["profitable_calendar_windows_ratio"] == 0.0
    assert rejection["calendar_window_summaries"][0] == {
        "calendar_window": "2026Q1",
        "sample_count": 40.0,
        "expected_edge_bps": -2.0,
        "net_edge_bps": -14.0,
        "win_rate": 0.45,
        "profitable": False,
    }
    assert aggregate["local_falsification_failed_thesis_ids"] == ["TH-LOCAL-REJECTED"]
    assert aggregate["local_falsification_failed_mechanism_classes"] == [
        "low_range_volume_absorption"
    ]
    assert "TH-LOCAL-REJECTED" in aggregate["thesis_ids_tried"]
    assert "low_range_volume_absorption" in aggregate["hypothesis_families_tried"]
    assert brief["requires_new_thesis_id"] is True
    assert brief["requires_new_research_references"] is True
    assert brief["parameter_only_retry_allowed"] is False
    assert any(
        "model-prediction diagnostics" in question
        for question in brief["recommended_research_questions"]
    )
    assert any(
        "expect &-future_return" in question or "&-future_return" in question
        for question in brief["recommended_research_questions"]
    )
    assert any(
        "TH-LOCAL-REJECTED" in question
        for question in brief["recommended_research_questions"]
    )
    assert "parameter_only_threshold_loosen" in brief["blocked_next_actions"]

    json_path, report_path = write_candidate_failure_synthesis_artifacts(
        synthesis,
        root_dir=tmp_path,
        output_root=Path("synthesis"),
    )

    assert json_path.is_file()
    assert report_path.is_file()
    assert "parameter_only_retry_allowed: False" in report_path.read_text(
        encoding="utf-8"
    )
    assert "Local Falsification Rejections" in report_path.read_text(
        encoding="utf-8"
    )


def test_candidate_failure_synthesis_uses_ranking_research_context_without_manifest(
    tmp_path,
):
    from freqtrade_ext.bot_factory.candidate_failure_synthesis import (
        CandidateFailureSynthesisInputs,
        synthesize_candidate_failures,
    )

    question_handoff = {
        "required": True,
        "passed": False,
        "computed_missing_research_question_response_indexes": [1],
    }
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(json.dumps({
        "ranking_id": "rank-handoff-context",
        "paper_ready_candidate_ids": [],
        "ranked_candidates": [
            {
                "candidate_id": "cand-handoff",
                "strategy_name": "StrategyHandoff",
                "recommendation": "retry",
                "paper_ready_eligible": False,
                "paper_ready_blockers": ["historical_backtest"],
                "blocked_next_actions": [
                    "retry_validated_local_rejection_by_parameter_tuning",
                ],
                "research_brief": {
                    "thesis_id": "TH-HANDOFF",
                    "blocked_next_actions": ["parameter_only_threshold_loosen"],
                    "research_decision_question_handoff": question_handoff,
                },
                "research_handoff_summary": {
                    "research_decision_question_handoff": question_handoff,
                },
                "hypothesis_family": "local_absorption",
                "thesis": {"thesis_id": "TH-HANDOFF"},
                "metrics": {
                    "historical_trade_count": 3,
                    "historical_total_return_pct": -0.8,
                    "walk_forward_pass_rate": 0.0,
                },
                "rank": 1,
            },
        ],
    }), encoding="utf-8")

    synthesis = synthesize_candidate_failures(CandidateFailureSynthesisInputs(
        root_dir=tmp_path,
        ranking_path=ranking_path,
        synthesis_id="synth-handoff-context",
    ))

    candidate = synthesis["candidates"][0]
    aggregate = synthesis["aggregate_failure_summary"]
    brief = synthesis["next_research_brief"]
    assert candidate["blocked_next_actions"] == [
        "retry_validated_local_rejection_by_parameter_tuning",
        "parameter_only_threshold_loosen",
    ]
    assert candidate["research_handoff_summary"] == {
        "research_decision_question_handoff": question_handoff,
    }
    assert aggregate["blocked_next_actions"] == candidate["blocked_next_actions"]
    assert "retry_validated_local_rejection_by_parameter_tuning" in brief[
        "blocked_next_actions"
    ]
    assert brief["research_handoff_summaries"] == [
        {
            "candidate_id": "cand-handoff",
            "research_handoff_summary": {
                "research_decision_question_handoff": question_handoff,
            },
        }
    ]


def test_candidate_failure_synthesis_ignores_unsafe_local_falsification_rejection(
    tmp_path,
):
    from freqtrade_ext.bot_factory.candidate_failure_synthesis import (
        CandidateFailureSynthesisInputs,
        synthesize_candidate_failures,
    )

    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(json.dumps({
        "ranking_id": "rank-test",
        "paper_ready_candidate_ids": [],
        "ranked_candidates": [
            {
                "candidate_id": "cand-a",
                "strategy_name": "StrategyA",
                "recommendation": "retry",
                "paper_ready_eligible": False,
                "paper_ready_blockers": ["historical_backtest"],
                "failure_taxonomy_codes": ["FAIL_COST_SENSITIVE"],
                "hypothesis_family": "low_range_volume_absorption",
                "thesis": {"thesis_id": "TH-A"},
                "metrics": {
                    "historical_trade_count": 4,
                    "historical_total_return_pct": -1.2,
                    "walk_forward_pass_rate": 0.0,
                },
                "rank": 1,
            },
        ],
    }), encoding="utf-8")
    local_falsification_path = tmp_path / "crafted_failed_local_falsification.json"
    local_falsification_path.write_text(json.dumps({
        "factory": "research_local_falsification",
        "status": "failed",
        "thesis_id": "TH-CRAFTED-LOCAL-REJECTED",
        "mechanism_class": "crafted_cost_edge",
        "expected_edge_bps": -2.0,
        "all_in_cost_bps": 12.0,
        "net_edge_bps": -14.0,
        "sample_count": 80,
        "data_span_days": 365.0,
        "blockers": [{"name": "expected_edge_exceeds_all_in_cost"}],
    }), encoding="utf-8")

    synthesis = synthesize_candidate_failures(CandidateFailureSynthesisInputs(
        root_dir=tmp_path,
        ranking_path=ranking_path,
        local_falsification_paths=[local_falsification_path],
        synthesis_id="synth-test",
    ))

    aggregate = synthesis["aggregate_failure_summary"]
    artifact = aggregate["local_falsification_rejection_artifacts"][0]
    assert aggregate["local_falsification_rejection_artifact_count"] == 1
    assert aggregate["local_falsification_rejection_count"] == 0
    assert aggregate["local_falsification_invalid_rejection_count"] == 1
    assert aggregate["local_falsification_failed_thesis_ids"] == []
    assert artifact["factory_valid"] is True
    assert artifact["status_rejected"] is True
    assert artifact["rejection_valid"] is False
    assert artifact["safety_scope_valid"] is False
    assert artifact["event_source_valid"] is False
    assert "safety_scope_invalid" in artifact["failure_reasons"]
    assert "event_source_invalid" in artifact["failure_reasons"]


def test_candidate_failure_synthesis_blocks_paths_outside_workspace(tmp_path):
    import pytest

    from freqtrade_ext.bot_factory.candidate_failure_synthesis import (
        CandidateFailureSynthesisInputs,
        synthesize_candidate_failures,
    )

    outside = tmp_path.parent / f"{tmp_path.name}_outside_ranking.json"
    outside.write_text(json.dumps({"ranked_candidates": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="inside the workspace"):
        synthesize_candidate_failures(CandidateFailureSynthesisInputs(
            root_dir=tmp_path,
            ranking_path=outside,
        ))


def test_candidate_failure_map_builds_causal_categories(tmp_path):
    from freqtrade_ext.bot_factory.candidate_failure_map import (
        CandidateFailureMapInputs,
        build_candidate_failure_map,
        write_candidate_failure_map_artifacts,
    )

    synthesis_path = tmp_path / "registry" / "strategies" / "synthesis" / "synth.json"
    synthesis_path.parent.mkdir(parents=True)
    synthesis_path.write_text(
        json.dumps(
            {
                "factory": "candidate_failure_synthesis",
                "synthesis_id": "synth-map-test",
                "aggregate_failure_summary": {
                    "paper_ready_candidate_ids": [],
                    "hypothesis_families_tried": [
                        "hybrid_ml_return_filter",
                        "volatility_breakout",
                    ],
                    "thesis_ids_tried": ["TH-A", "TH-B"],
                    "local_falsification_rejections": [
                        {
                            "path": (
                                "registry/strategies/research_decisions/"
                                "local_rejection.json"
                            ),
                            "thesis_id": "TH-LOCAL-REJECTED",
                            "mechanism_class": "mark_fair_value_momentum_lag",
                            "net_edge_bps": -4.2,
                            "profitable_windows_ratio": 0.0,
                            "calendar_window_frequency": "quarter",
                            "calendar_window_count": 3,
                            "profitable_calendar_windows_ratio": 0.0,
                            "calendar_window_summaries": [
                                {
                                    "calendar_window": "2026Q1",
                                    "sample_count": 20,
                                    "net_edge_bps": -5.0,
                                    "profitable": False,
                                }
                            ],
                        }
                    ],
                },
                "next_research_brief": {
                    "requires_new_thesis_id": True,
                    "requires_new_research_references": True,
                    "minimum_research_reference_count": 2,
                    "blocked_next_actions": [
                        "ranking_context_retry_block",
                    ],
                    "research_handoff_summaries": [
                        {
                            "candidate_id": "cand-a",
                            "research_handoff_summary": {
                                "research_decision_question_handoff": {
                                    "required": True,
                                    "passed": False,
                                }
                            },
                        }
                    ],
                    "recommended_research_questions": [
                        (
                            "Why did the generated entry set for cand-b have "
                            "non-positive edge after costs?"
                        )
                    ],
                },
                "candidates": [
                    {
                        "candidate_id": "cand-a",
                        "strategy_name": "HybridA",
                        "hypothesis_family": "hybrid_ml_return_filter",
                        "thesis_id": "TH-A",
                        "failure_taxonomy_codes": ["FAIL_REGIME_FRAGILE"],
                        "failed_checks": ["historical_backtest", "walk_forward"],
                        "skipped_checks": ["training_factory"],
                        "metrics": {
                            "historical_trade_count": 0,
                            "historical_total_return_pct": 0.0,
                            "walk_forward_pass_rate": 0.0,
                            "walk_forward_profitable_windows_ratio": 0.0,
                        },
                        "signal_diagnostics": {
                            "available": True,
                            "entry_count": 0,
                            "diagnosis_codes": ["ZERO_ENTRY_SIGNALS"],
                            "first_zero_component": "ml_filter",
                            "rarest_component": "momentum_confirmed",
                            "bottleneck_components": [
                                {"name": "ml_filter", "all_except_count": 10}
                            ],
                        },
                        "freqai_prediction_diagnostics": {
                            "available": True,
                            "expected_target_column": "&-future_return",
                            "target_columns": ["&-long_return"],
                            "diagnosis_codes": ["PREDICTION_TARGET_MISMATCH"],
                        },
                    },
                    {
                        "candidate_id": "cand-b",
                        "strategy_name": "RuleB",
                        "hypothesis_family": "volatility_breakout",
                        "thesis_id": "TH-B",
                        "failure_taxonomy_codes": [
                            "FAIL_COST_SENSITIVE",
                            "FAIL_OVERFIT_WF_GAP",
                        ],
                        "failed_checks": ["historical_backtest", "walk_forward"],
                        "skipped_checks": [],
                        "metrics": {
                            "historical_trade_count": 24,
                            "historical_total_return_pct": -0.8,
                            "walk_forward_pass_rate": 0.0,
                            "walk_forward_profitable_windows_ratio": 0.0,
                        },
                        "signal_diagnostics": {
                            "available": True,
                            "entry_count": 42,
                            "diagnosis_codes": [
                                "GENERATED_ENTRY_EDGE_NEGATIVE_AFTER_COST"
                            ],
                            "first_zero_component": None,
                            "rarest_component": "volume_filter",
                            "bottleneck_components": [
                                {"name": "volume_filter", "all_except_count": 6}
                            ],
                            "generated_entry_edge": {
                                "status": "fail",
                                "sample_count": 42,
                                "net_edge_bps": -8.5,
                                "profitable_windows_ratio": 0.0,
                            },
                        },
                        "freqai_prediction_diagnostics": {"available": False},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    failure_map = build_candidate_failure_map(
        CandidateFailureMapInputs(
            root_dir=tmp_path,
            synthesis_path=synthesis_path,
            map_id="map-test",
            reviewer_notes=["causal map test only"],
        )
    )

    categories = failure_map["causal_failure_categories"]["categories"]
    assert failure_map["status"] == "completed"
    assert categories["zero_trade_or_signal_sparsity"]["candidate_count"] == 1
    assert categories["ml_rule_alignment_failure"]["candidate_ids"] == ["cand-a"]
    assert categories["entry_exists_negative_edge"]["candidate_ids"] == ["cand-b"]
    assert categories["generated_entry_negative_edge"]["candidate_ids"] == ["cand-b"]
    assert categories["no_profitable_walk_forward_windows"]["candidate_count"] == 2
    guidance = failure_map["research_selection_guidance"]
    assert guidance["requires_research_decision_before_proposal"] is True
    assert guidance["minimum_research_selection_score"] == 80
    assert guidance["research_selection_rubric"][0]["component"] == (
        "novelty_against_failure_set"
    )
    risk_by_category = {
        item["category"]: item for item in guidance["causal_risk_weights"]
    }
    assert risk_by_category["walk_forward_fragility"]["risk_score"] == 100.0
    assert risk_by_category["walk_forward_fragility"][
        "required_for_next_research"
    ] is True
    assert risk_by_category["entry_exists_negative_edge"]["risk_score"] == 62.5
    assert "positive expectancy after fees when entries exist" in risk_by_category[
        "entry_exists_negative_edge"
    ]["response_focus"]
    assert "generated entry set expectancy after costs" in risk_by_category[
        "generated_entry_negative_edge"
    ]["response_focus"]
    assert any(
        "generated entry set for cand-b" in question
        for question in guidance["required_research_questions"]
    )
    assert guidance["validated_local_falsification_rejections"] == [
        {
            "path": "registry/strategies/research_decisions/local_rejection.json",
            "thesis_id": "TH-LOCAL-REJECTED",
            "mechanism_class": "mark_fair_value_momentum_lag",
            "net_edge_bps": -4.2,
            "profitable_windows_ratio": 0.0,
            "calendar_window_frequency": "quarter",
            "calendar_window_count": 3,
            "profitable_calendar_windows_ratio": 0.0,
            "calendar_window_summaries": [
                {
                    "calendar_window": "2026Q1",
                    "sample_count": 20,
                    "net_edge_bps": -5.0,
                    "profitable": False,
                }
            ],
        }
    ]
    assert any(
        "validated local falsification rejection for TH-LOCAL-REJECTED" in question
        for question in guidance["required_research_questions"]
    )
    assert "retry_validated_local_rejection_by_parameter_tuning" in guidance[
        "blocked_next_actions"
    ]
    assert "proposal_generation_without_approved_research_decision" in guidance[
        "blocked_next_actions"
    ]
    assert "ranking_context_retry_block" in guidance["blocked_next_actions"]
    assert guidance["research_handoff_summaries"] == [
        {
            "candidate_id": "cand-a",
            "research_handoff_summary": {
                "research_decision_question_handoff": {
                    "required": True,
                    "passed": False,
                }
            },
        }
    ]

    json_path, report_path = write_candidate_failure_map_artifacts(
        failure_map,
        root_dir=tmp_path,
        output_root=Path("failure_maps"),
    )

    assert json_path.is_file()
    assert report_path.is_file()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Dominant Failure Categories" in report_text
    assert "Causal Risk Weights" in report_text
    assert "Validated Local Falsification Rejections" in report_text
    assert "TH-LOCAL-REJECTED / mark_fair_value_momentum_lag" in report_text
    assert "profitable_calendar_windows_ratio=0.0" in report_text
    assert "Research Selection Rubric" in report_text


def test_candidate_failure_map_blocks_paths_outside_workspace(tmp_path):
    import pytest

    from freqtrade_ext.bot_factory.candidate_failure_map import (
        CandidateFailureMapInputs,
        build_candidate_failure_map,
    )

    outside = tmp_path.parent / f"{tmp_path.name}_outside_synthesis.json"
    outside.write_text(json.dumps({"factory": "candidate_failure_synthesis"}), encoding="utf-8")

    with pytest.raises(ValueError, match="inside the workspace"):
        build_candidate_failure_map(
            CandidateFailureMapInputs(root_dir=tmp_path, synthesis_path=outside)
        )


def test_local_event_builder_writes_closed_candle_events_for_falsification(tmp_path):
    ohlcv_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.csv"
    spec_path = tmp_path / "registry" / "strategies" / "research_decisions" / "event_spec.json"
    synthesis_path = (
        tmp_path
        / "registry"
        / "strategies"
        / "synthesis"
        / "candidate_failure_synthesis.json"
    )
    ohlcv_path.parent.mkdir(parents=True)
    spec_path.parent.mkdir(parents=True)
    synthesis_path.parent.mkdir(parents=True)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    drop_indexes = {10, 20, 30, 40, 50}
    rows = []
    price = 100.0
    for index in range(70):
        if index in drop_indexes:
            price *= 0.99
        elif index - 1 in drop_indexes:
            price *= 1.015
        else:
            price *= 1.0002
        rows.append(
            {
                "date": start + pd.Timedelta(minutes=5 * index),
                "open": price,
                "high": price * 1.001,
                "low": price * 0.999,
                "close": price,
                "volume": 1000.0 + index,
            }
        )
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_spec",
                "event_id": "local-event-test",
                "thesis_id": "TH-LOCAL-EVENT-001",
                "mechanism_class": "closed_candle_drop_recovery_event",
                "cooldown_candles": 3,
                "conditions": [
                    {
                        "feature": "return_bps",
                        "lookback_candles": 1,
                        "operator": "<=",
                        "value": -50.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    synthesis_path.write_text(
        json.dumps(
            {
                "aggregate_failure_summary": {
                    "failed_thesis_ids": ["TH-OTHER-LOCAL-EVENT-001"],
                    "failed_hypothesis_families": ["other_closed_candle_event"],
                }
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_events(
        LocalEventBuildInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            event_spec_path=spec_path,
            failure_synthesis_path=synthesis_path,
            event_id="local-event-test",
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "completed"
    assert artifact["event_count"] == 5
    assert artifact["safety_scope"]["future_data"] is False
    assert artifact["safety_scope"]["strategy_code_generated"] is False
    assert artifact["context_merge"]["semantics"] == (
        "closed_context_candle_availability_v1"
    )
    assert artifact["context_merge"]["context_features_used"] is False
    assert artifact["context_merge"]["closed_context_candle_alignment"] is True
    assert artifact["failure_synthesis_summary"]["used"] is True
    assert artifact["failure_synthesis_summary"][
        "thesis_repeats_failed_synthesis"
    ] is False
    assert artifact["failure_synthesis_summary"][
        "mechanism_repeats_failed_synthesis"
    ] is False

    json_path, report_path, events_path = write_local_event_artifacts(
        artifact,
        root_dir=tmp_path,
        output_root=Path("registry/strategies/research_decisions"),
    )

    assert json_path.is_file()
    assert report_path.is_file()
    assert events_path.is_file()
    assert len(pd.read_csv(events_path)) == 5

    falsification = build_local_falsification(
        LocalFalsificationInputs(
            root_dir=tmp_path,
            thesis_id="TH-LOCAL-EVENT-001",
            mechanism_class="closed_candle_drop_recovery_event",
            ohlcv_path=ohlcv_path,
            event_path=events_path,
            event_source_path=json_path,
            hold_candles=1,
            all_in_cost_bps=5.0,
            min_sample_count=5,
            min_profitable_windows_ratio=0.75,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert falsification["status"] == "passed"
    assert falsification["sample_count"] == 5
    assert falsification["net_edge_bps"] > 0.0
    assert falsification["event_source"]["factory_valid"] is True
    assert falsification["event_source"]["event_path_matches"] is True
    assert falsification["event_source"]["ohlcv_path_matches"] is True
    assert falsification["event_source"]["context_features_used"] is False
    assert (
        falsification["event_source"]["closed_context_candle_alignment_valid"] is True
    )
    assert falsification["event_source"]["failure_synthesis_used"] is True
    assert falsification["event_source"]["failure_synthesis_parseable"] is True
    assert falsification["event_source"]["failure_synthesis_guard_valid"] is True


def test_local_event_builder_supports_utc_session_features(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    spec_path = tmp_path / "event_spec.json"
    start = pd.Timestamp("2026-01-05T00:00:00Z")
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 1000.0 + index,
            }
            for index in range(48)
        ]
    ).to_csv(ohlcv_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_spec",
                "event_id": "utc-session-event-test",
                "thesis_id": "TH-UTC-SESSION-001",
                "mechanism_class": "utc_session_calendar_event",
                "conditions": [
                    {
                        "feature": "hour_utc",
                        "lookback_candles": 1,
                        "operator": "==",
                        "value": 13.0,
                    },
                    {
                        "feature": "weekday",
                        "lookback_candles": 1,
                        "operator": "<=",
                        "value": 4.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_events(
        LocalEventBuildInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            event_spec_path=spec_path,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "completed"
    assert artifact["event_count"] == 2
    assert artifact["feature_columns"] == ["hour_utc_1", "weekday_1"]
    assert [event["date"] for event in artifact["events"]] == [
        "2026-01-05T13:00:00+00:00",
        "2026-01-06T13:00:00+00:00",
    ]
    assert [event["hour_utc_1"] for event in artifact["events"]] == [13.0, 13.0]
    assert [event["weekday_1"] for event in artifact["events"]] == [0.0, 1.0]
    assert artifact["safety_scope"]["closed_candle_ohlcv_only"] is True
    assert artifact["safety_scope"]["strategy_code_generated"] is False


def test_local_event_builder_supports_futures_context_features(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    funding_path = tmp_path / "funding.csv"
    mark_path = tmp_path / "mark.csv"
    spec_path = tmp_path / "event_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    closes = [
        99.0,
        99.5,
        100.0,
        100.5,
        100.0,
        100.2,
        100.4,
        100.6,
        104.0,
        104.5,
        105.0,
        105.5,
    ]
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0 + index,
            }
            for index, close in enumerate(closes)
        ]
    ).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [start, start + pd.Timedelta(hours=4), start + pd.Timedelta(hours=8)],
            "open": [-0.00040, -0.00025, -0.00010],
            "high": [0.0, 0.0, 0.0],
            "low": [0.0, 0.0, 0.0],
            "close": [0.0, 0.0, 0.0],
            "volume": [0.0, 0.0, 0.0],
        }
    ).to_csv(funding_path, index=False)
    pd.DataFrame(
        {
            "date": [start, start + pd.Timedelta(hours=4), start + pd.Timedelta(hours=8)],
            "open": [100.0, 105.0, 110.0],
            "high": [100.0, 105.0, 110.0],
            "low": [100.0, 105.0, 110.0],
            "close": [100.0, 105.0, 110.0],
            "volume": [None, None, None],
        }
    ).to_csv(mark_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_spec",
                "event_id": "futures-context-event-test",
                "thesis_id": "TH-FUTURES-CONTEXT-001",
                "mechanism_class": "funding_mark_dislocation_reclaim",
                "cooldown_candles": 4,
                "conditions": [
                    {
                        "feature": "funding_rate_bps",
                        "lookback_candles": 1,
                        "operator": "<=",
                        "value": -1.0,
                    },
                    {
                        "feature": "funding_rate_delta_bps",
                        "lookback_candles": 1,
                        "operator": ">=",
                        "value": 1.0,
                    },
                    {
                        "feature": "mark_price_gap_bps",
                        "lookback_candles": 1,
                        "operator": "<=",
                        "value": -50.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_events(
        LocalEventBuildInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            funding_rate_path=funding_path,
            mark_price_path=mark_path,
            event_spec_path=spec_path,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "completed"
    assert artifact["event_count"] == 2
    assert artifact["feature_columns"] == [
        "funding_rate_bps_1",
        "funding_rate_delta_bps_1",
        "mark_price_gap_bps_1",
    ]
    diagnostics = {
        item["feature_column"]: item for item in artifact["condition_diagnostics"]
    }
    assert diagnostics["funding_rate_bps_1"]["match_count"] > 0
    assert diagnostics["mark_price_gap_bps_1"]["non_null_count"] == len(closes) - 3
    assert artifact["combined_match_count_before_cooldown"] >= artifact["event_count"]
    assert artifact["cumulative_condition_match_counts"][-1]["match_count"] == artifact[
        "combined_match_count_before_cooldown"
    ]
    assert [event["date"] for event in artifact["events"]] == [
        "2026-01-01T07:00:00+00:00",
        "2026-01-01T11:00:00+00:00",
    ]
    assert artifact["auxiliary_sources"]["funding_rate"]["row_count"] == 3
    assert artifact["auxiliary_sources"]["mark_price"]["row_count"] == 3
    assert artifact["context_merge"]["semantics"] == (
        "closed_context_candle_availability_v1"
    )
    assert artifact["context_merge"]["context_features_used"] is True
    assert artifact["context_merge"]["required_contexts"] == [
        "funding_rate",
        "mark_price",
    ]
    assert artifact["context_merge"]["closed_context_candle_alignment"] is True
    assert artifact["context_merge"]["contexts"]["mark_price"][
        "closed_context_shift_seconds"
    ] == 10800.0
    assert artifact["safety_scope"]["closed_candle_ohlcv_only"] is False
    assert artifact["safety_scope"]["closed_candle_local_market_data_only"] is True
    assert artifact["safety_scope"]["closed_context_candle_alignment"] is True
    assert artifact["safety_scope"]["future_data"] is False

    json_path, _, events_path = write_local_event_artifacts(
        artifact,
        root_dir=tmp_path,
        output_root=Path("registry/strategies/research_decisions"),
    )
    falsification = build_local_falsification(
        LocalFalsificationInputs(
            root_dir=tmp_path,
            thesis_id="TH-FUTURES-CONTEXT-001",
            mechanism_class="funding_mark_dislocation_reclaim",
            ohlcv_path=ohlcv_path,
            event_path=events_path,
            event_source_path=json_path,
            hold_candles=1,
            all_in_cost_bps=5.0,
            min_sample_count=1,
            min_profitable_windows_ratio=0.0,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert falsification["event_source"]["safety_scope_valid"] is True
    assert falsification["event_source"]["context_features_used"] is True
    assert falsification["event_source"]["required_contexts"] == [
        "funding_rate",
        "mark_price",
    ]
    assert (
        falsification["event_source"]["context_merge_semantics"]
        == "closed_context_candle_availability_v1"
    )
    assert (
        falsification["event_source"]["closed_context_candle_alignment_valid"] is True
    )


def test_local_event_builder_supports_open_interest_context_features(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    open_interest_path = tmp_path / "open_interest.csv"
    quality_path = tmp_path / "open_interest_quality.json"
    spec_path = tmp_path / "event_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    closes = [100.0 + index for index in range(12)]
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0,
            }
            for index, close in enumerate(closes)
        ]
    ).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [start, start + pd.Timedelta(hours=4), start + pd.Timedelta(hours=8)],
            "open_interest": [1000.0, 1100.0, 1300.0],
        }
    ).to_csv(open_interest_path, index=False)
    _write_passing_structural_quality_report(quality_path, rows=3)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_spec",
                "event_id": "open-interest-context-event-test",
                "thesis_id": "TH-OPEN-INTEREST-CONTEXT-001",
                "mechanism_class": "open_interest_impulse_reversion",
                "cooldown_candles": 4,
                "conditions": [
                    {
                        "feature": "open_interest_delta_pct",
                        "lookback_candles": 1,
                        "operator": ">=",
                        "value": 5.0,
                    },
                    {
                        "feature": "open_interest_zscore",
                        "lookback_candles": 4,
                        "operator": ">=",
                        "value": 1.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_events(
        LocalEventBuildInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            open_interest_path=open_interest_path,
            open_interest_quality_report_paths=[quality_path],
            event_spec_path=spec_path,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "completed"
    assert artifact["event_count"] == 2
    assert artifact["feature_columns"] == [
        "open_interest_delta_pct_1",
        "open_interest_zscore_4",
    ]
    assert [event["date"] for event in artifact["events"]] == [
        "2026-01-01T07:00:00+00:00",
        "2026-01-01T11:00:00+00:00",
    ]
    assert artifact["auxiliary_sources"]["open_interest"]["row_count"] == 3
    assert artifact["open_interest_quality_reports"][0]["ok"] is True
    assert artifact["context_merge"]["required_contexts"] == ["open_interest"]
    assert artifact["context_merge"]["contexts"]["open_interest"][
        "closed_context_shift_seconds"
    ] == 10800.0
    assert artifact["safety_scope"]["closed_candle_ohlcv_only"] is False
    assert artifact["safety_scope"]["closed_context_candle_alignment"] is True


def test_local_event_builder_supports_long_short_ratio_context_features(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    ratio_path = tmp_path / "long_short_ratio.csv"
    quality_path = tmp_path / "long_short_ratio_quality.json"
    spec_path = tmp_path / "event_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    closes = [100.0 + index for index in range(12)]
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0,
            }
            for index, close in enumerate(closes)
        ]
    ).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [start, start + pd.Timedelta(hours=4), start + pd.Timedelta(hours=8)],
            "buyRatio": [0.50, 0.60, 0.70],
            "sellRatio": [0.50, 0.40, 0.30],
        }
    ).to_csv(ratio_path, index=False)
    _write_passing_structural_quality_report(quality_path, rows=3)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_spec",
                "event_id": "long-short-context-event-test",
                "thesis_id": "TH-LONG-SHORT-CONTEXT-001",
                "mechanism_class": "long_short_crowding_reversion",
                "cooldown_candles": 4,
                "conditions": [
                    {
                        "feature": "long_short_ratio",
                        "lookback_candles": 1,
                        "operator": ">=",
                        "value": 1.4,
                    },
                    {
                        "feature": "long_account_ratio_delta_bps",
                        "lookback_candles": 1,
                        "operator": ">=",
                        "value": 500.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_events(
        LocalEventBuildInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            long_short_ratio_path=ratio_path,
            long_short_ratio_quality_report_paths=[quality_path],
            event_spec_path=spec_path,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "completed"
    assert artifact["event_count"] == 2
    assert artifact["feature_columns"] == [
        "long_short_ratio_1",
        "long_account_ratio_delta_bps_1",
    ]
    assert [event["date"] for event in artifact["events"]] == [
        "2026-01-01T07:00:00+00:00",
        "2026-01-01T11:00:00+00:00",
    ]
    assert artifact["events"][0]["long_short_ratio_1"] == 1.5
    assert artifact["events"][0]["long_account_ratio_delta_bps_1"] == 1000.0
    assert artifact["auxiliary_sources"]["long_short_ratio"]["row_count"] == 3
    assert artifact["long_short_ratio_quality_reports"][0]["ok"] is True
    assert artifact["context_merge"]["required_contexts"] == ["long_short_ratio"]
    assert artifact["context_merge"]["contexts"]["long_short_ratio"][
        "closed_context_shift_seconds"
    ] == 10800.0
    assert artifact["safety_scope"]["closed_candle_ohlcv_only"] is False
    assert artifact["safety_scope"]["closed_context_candle_alignment"] is True


def test_local_event_builder_blocks_open_interest_context_without_quality_report(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    open_interest_path = tmp_path / "open_interest.csv"
    spec_path = tmp_path / "event_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.0 + index,
                "volume": 1000.0,
            }
            for index in range(6)
        ]
    ).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [start, start + pd.Timedelta(hours=2)],
            "open_interest": [1000.0, 1100.0],
        }
    ).to_csv(open_interest_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_spec",
                "event_id": "open-interest-missing-quality-event-test",
                "thesis_id": "TH-OPEN-INTEREST-MISSING-QUALITY-001",
                "mechanism_class": "open_interest_impulse_reversion",
                "conditions": [
                    {
                        "feature": "open_interest",
                        "lookback_candles": 1,
                        "operator": ">=",
                        "value": 1000.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_events(
        LocalEventBuildInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            open_interest_path=open_interest_path,
            event_spec_path=spec_path,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "blocked"
    assert artifact["event_count"] == 0
    assert "open_interest_quality_report_passed_when_required" in {
        check["name"] for check in artifact["blockers"]
    }


def test_local_event_builder_blocks_long_short_ratio_context_without_quality_report(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    ratio_path = tmp_path / "long_short_ratio.csv"
    spec_path = tmp_path / "event_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.0 + index,
                "volume": 1000.0,
            }
            for index in range(6)
        ]
    ).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [start, start + pd.Timedelta(hours=2)],
            "buyRatio": [0.55, 0.65],
            "sellRatio": [0.45, 0.35],
        }
    ).to_csv(ratio_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_spec",
                "event_id": "long-short-missing-quality-event-test",
                "thesis_id": "TH-LONG-SHORT-MISSING-QUALITY-001",
                "mechanism_class": "long_short_crowding_reversion",
                "conditions": [
                    {
                        "feature": "long_short_ratio",
                        "lookback_candles": 1,
                        "operator": ">=",
                        "value": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_events(
        LocalEventBuildInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            long_short_ratio_path=ratio_path,
            event_spec_path=spec_path,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "blocked"
    assert artifact["event_count"] == 0
    assert "long_short_ratio_quality_report_passed_when_required" in {
        check["name"] for check in artifact["blockers"]
    }


def test_local_event_builder_supports_informative_ohlcv_context_features(tmp_path):
    ohlcv_path = tmp_path / "btc_ohlcv.csv"
    informative_path = tmp_path / "eth_ohlcv.csv"
    spec_path = tmp_path / "event_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    btc_closes = [100.0, 101.0, 102.0, 103.0, 110.0, 111.0, 120.0, 121.0]
    eth_closes = [100.0, 100.2, 100.4, 100.6, 101.0, 101.2, 102.0, 102.2]
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0 + index,
            }
            for index, close in enumerate(btc_closes)
        ]
    ).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": close,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": 2000.0 + index,
            }
            for index, close in enumerate(eth_closes)
        ]
    ).to_csv(informative_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_spec",
                "event_id": "informative-ohlcv-context-event-test",
                "thesis_id": "TH-INFORMATIVE-OHLCV-CONTEXT-001",
                "mechanism_class": "cross_asset_relative_impulse",
                "cooldown_candles": 2,
                "conditions": [
                    {
                        "feature": "informative_return_bps",
                        "lookback_candles": 1,
                        "operator": ">=",
                        "value": 10.0,
                    },
                    {
                        "feature": "relative_return_bps",
                        "lookback_candles": 1,
                        "operator": ">=",
                        "value": 400.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_events(
        LocalEventBuildInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            informative_ohlcv_path=informative_path,
            event_spec_path=spec_path,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "completed"
    assert artifact["event_count"] == 2
    assert artifact["feature_columns"] == [
        "informative_return_bps_1",
        "relative_return_bps_1",
    ]
    assert [event["date"] for event in artifact["events"]] == [
        "2026-01-01T04:00:00+00:00",
        "2026-01-01T06:00:00+00:00",
    ]
    assert artifact["events"][0]["informative_return_bps_1"] >= 10.0
    assert artifact["events"][0]["relative_return_bps_1"] >= 400.0
    assert artifact["source_informative_ohlcv_path"] == "eth_ohlcv.csv"
    assert artifact["auxiliary_sources"]["informative_ohlcv"]["row_count"] == 8
    assert artifact["context_merge"]["required_contexts"] == ["informative_ohlcv"]
    assert artifact["context_merge"]["contexts"]["informative_ohlcv"][
        "closed_context_shift_seconds"
    ] == 0.0
    assert artifact["safety_scope"]["closed_candle_ohlcv_only"] is False
    assert artifact["safety_scope"]["closed_context_candle_alignment"] is True


def test_local_event_builder_supports_liquidation_context_features(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    liquidation_path = tmp_path / "liquidation.parquet"
    quality_path = tmp_path / "liquidation_quality.json"
    spec_path = tmp_path / "event_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    closes = [100.0, 100.0, 101.0, 102.0, 103.0]
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0,
            }
            for index, close in enumerate(closes)
        ]
    ).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "T": [
                int((start + pd.Timedelta(hours=1, minutes=15)).timestamp() * 1000),
                int((start + pd.Timedelta(hours=1, minutes=30)).timestamp() * 1000),
                int((start + pd.Timedelta(hours=3, minutes=10)).timestamp() * 1000),
            ],
            "S": ["Sell", "Sell", "Buy"],
            "v": [2.0, 1.0, 3.0],
            "p": [100.0, 100.0, 100.0],
        }
    ).to_parquet(liquidation_path)
    _write_passing_structural_quality_report(quality_path, rows=3)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_spec",
                "event_id": "liquidation-context-event-test",
                "thesis_id": "TH-LIQUIDATION-CONTEXT-001",
                "mechanism_class": "liquidation_absorption_probe",
                "conditions": [
                    {
                        "feature": "liquidation_sell_notional",
                        "lookback_candles": 1,
                        "operator": ">=",
                        "value": 250.0,
                    },
                    {
                        "feature": "liquidation_imbalance",
                        "lookback_candles": 1,
                        "operator": "<=",
                        "value": -0.9,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_events(
        LocalEventBuildInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            liquidation_path=liquidation_path,
            liquidation_quality_report_paths=[quality_path],
            event_spec_path=spec_path,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "completed"
    assert artifact["event_count"] == 1
    assert artifact["feature_columns"] == [
        "liquidation_sell_notional_1",
        "liquidation_imbalance_1",
    ]
    assert artifact["events"][0]["date"] == "2026-01-01T01:00:00+00:00"
    assert artifact["events"][0]["liquidation_sell_notional_1"] == 300.0
    assert artifact["events"][0]["liquidation_imbalance_1"] == -1.0
    assert artifact["source_liquidation_path"] == "liquidation.parquet"
    assert artifact["liquidation_quality_reports"][0]["ok"] is True
    assert artifact["auxiliary_sources"]["liquidation"]["row_count"] == 3
    assert artifact["context_merge"]["required_contexts"] == ["liquidation"]
    assert artifact["context_merge"]["contexts"]["liquidation"][
        "closed_context_shift_seconds"
    ] == 0.0
    assert artifact["safety_scope"]["closed_candle_ohlcv_only"] is False
    assert artifact["safety_scope"]["closed_context_candle_alignment"] is True


def test_local_event_builder_supports_order_book_context_features(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    order_book_path = tmp_path / "order_book.parquet"
    quality_path = tmp_path / "order_book_quality.json"
    spec_path = tmp_path / "event_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    closes = [100.0, 100.0, 101.0, 102.0, 103.0]
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0,
            }
            for index, close in enumerate(closes)
        ]
    ).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [
                start + pd.Timedelta(hours=1, minutes=10),
                start + pd.Timedelta(hours=1, minutes=40),
                start + pd.Timedelta(hours=3, minutes=10),
            ],
            "best_bid": [99.9, 99.8, 102.9],
            "best_ask": [100.1, 100.3, 103.2],
            "bid_size": [9.0, 8.0, 2.0],
            "ask_size": [1.0, 2.0, 8.0],
        }
    ).to_parquet(order_book_path)
    _write_passing_structural_quality_report(quality_path, rows=3)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_spec",
                "event_id": "order-book-context-event-test",
                "thesis_id": "TH-ORDER-BOOK-CONTEXT-001",
                "mechanism_class": "top_of_book_imbalance_probe",
                "conditions": [
                    {
                        "feature": "order_book_depth_imbalance",
                        "lookback_candles": 1,
                        "operator": ">=",
                        "value": 0.5,
                    },
                    {
                        "feature": "order_book_spread_bps",
                        "lookback_candles": 1,
                        "operator": ">=",
                        "value": 30.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_events(
        LocalEventBuildInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            order_book_path=order_book_path,
            order_book_quality_report_paths=[quality_path],
            event_spec_path=spec_path,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "completed"
    assert artifact["event_count"] == 1
    assert artifact["feature_columns"] == [
        "order_book_depth_imbalance_1",
        "order_book_spread_bps_1",
    ]
    assert artifact["events"][0]["date"] == "2026-01-01T01:00:00+00:00"
    assert artifact["events"][0]["order_book_depth_imbalance_1"] == 0.6
    assert artifact["events"][0]["order_book_spread_bps_1"] >= 30.0
    assert artifact["source_order_book_path"] == "order_book.parquet"
    assert artifact["order_book_quality_reports"][0]["ok"] is True
    assert artifact["auxiliary_sources"]["order_book"]["row_count"] == 3
    assert artifact["context_merge"]["required_contexts"] == ["order_book"]
    assert artifact["context_merge"]["contexts"]["order_book"][
        "closed_context_shift_seconds"
    ] == 0.0
    assert artifact["safety_scope"]["closed_candle_ohlcv_only"] is False
    assert artifact["safety_scope"]["closed_context_candle_alignment"] is True


def test_local_event_builder_blocks_order_book_context_without_quality_report(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    order_book_path = tmp_path / "order_book.parquet"
    spec_path = tmp_path / "event_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.0 + index,
                "volume": 1000.0,
            }
            for index in range(3)
        ]
    ).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [start + pd.Timedelta(hours=1, minutes=10)],
            "best_bid": [99.9],
            "best_ask": [100.1],
            "bid_size": [9.0],
            "ask_size": [1.0],
        }
    ).to_parquet(order_book_path)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_spec",
                "event_id": "order-book-missing-quality-test",
                "thesis_id": "TH-ORDER-BOOK-MISSING-QUALITY-001",
                "mechanism_class": "top_of_book_imbalance_probe",
                "conditions": [
                    {
                        "feature": "order_book_depth_imbalance",
                        "lookback_candles": 1,
                        "operator": ">=",
                        "value": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_events(
        LocalEventBuildInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            order_book_path=order_book_path,
            event_spec_path=spec_path,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "blocked"
    assert artifact["event_count"] == 0
    assert "order_book_quality_report_passed_when_required" in {
        check["name"] for check in artifact["blockers"]
    }


def test_local_event_cli_maps_informative_ohlcv_path(tmp_path):
    from scripts.bot_factory_build_local_events import build_inputs_from_args

    args = SimpleNamespace(
        ohlcv_path="btc.csv",
        event_spec_json="event_spec.json",
        funding_rate_path=None,
        mark_price_path=None,
        informative_ohlcv_path="eth.csv",
        open_interest_path="open_interest.csv",
        open_interest_quality_report_json=["open_interest_quality.json"],
        long_short_ratio_path="long_short.csv",
        long_short_ratio_quality_report_json=["long_short_quality.json"],
        liquidation_path="liquidation.csv",
        liquidation_quality_report_json=["liquidation_quality.json"],
        order_book_path="order_book.csv",
        order_book_quality_report_json=["order_book_quality.json"],
        failure_synthesis_json=None,
        allow_failed_thesis_or_family=False,
        event_id="event-id",
        output_root="registry/strategies/research_decisions",
        reviewer_note=[],
        created_at="2026-05-07T00:00:00+00:00",
    )

    inputs = build_inputs_from_args(args, root_dir=tmp_path)

    assert inputs.root_dir == tmp_path
    assert inputs.informative_ohlcv_path == Path("eth.csv")
    assert inputs.open_interest_path == Path("open_interest.csv")
    assert inputs.open_interest_quality_report_paths == [Path("open_interest_quality.json")]
    assert inputs.long_short_ratio_path == Path("long_short.csv")
    assert inputs.long_short_ratio_quality_report_paths == [Path("long_short_quality.json")]
    assert inputs.liquidation_path == Path("liquidation.csv")
    assert inputs.liquidation_quality_report_paths == [Path("liquidation_quality.json")]
    assert inputs.order_book_path == Path("order_book.csv")
    assert inputs.order_book_quality_report_paths == [Path("order_book_quality.json")]
    assert inputs.ohlcv_path == Path("btc.csv")


def test_local_event_builder_blocks_missing_required_futures_context(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    spec_path = tmp_path / "event_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(minutes=5 * index),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 + index,
                "volume": 1000.0,
            }
            for index in range(10)
        ]
    ).to_csv(ohlcv_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_spec",
                "event_id": "missing-context-event-test",
                "thesis_id": "TH-MISSING-CONTEXT-001",
                "mechanism_class": "funding_context_required",
                "conditions": [
                    {
                        "feature": "funding_rate_bps",
                        "lookback_candles": 1,
                        "operator": "<=",
                        "value": -1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_events(
        LocalEventBuildInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            event_spec_path=spec_path,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "blocked"
    assert artifact["event_count"] == 0
    assert {"funding_rate_file_present", "funding_rate_parseable"} <= {
        check["name"] for check in artifact["blockers"]
    }


def test_local_event_builder_blocks_failed_family_when_synthesis_supplied(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    spec_path = tmp_path / "event_spec.json"
    synthesis_path = tmp_path / "candidate_failure_synthesis.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(minutes=5 * index),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 - index,
                "volume": 1000.0,
            }
            for index in range(20)
        ]
    ).to_csv(ohlcv_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_spec",
                "event_id": "failed-family-event-test",
                "thesis_id": "TH-NEW-MARK-RETRY",
                "mechanism_class": "mark_price_dislocation_reclaim",
                "conditions": [
                    {
                        "feature": "return_bps",
                        "lookback_candles": 1,
                        "operator": "<=",
                        "value": -1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    synthesis_path.write_text(
        json.dumps(
            {
                "factory": "candidate_failure_synthesis",
                "aggregate_failure_summary": {
                    "all_candidates_failed_gates": True,
                    "hypothesis_families_tried": ["mark_price_dislocation_reclaim"],
                    "thesis_ids_tried": ["THESIS-MARK-PRICE-DISLOCATION-20260506T161500Z"],
                },
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_events(
        LocalEventBuildInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            event_spec_path=spec_path,
            failure_synthesis_path=synthesis_path,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "blocked"
    assert artifact["event_count"] == 0
    assert artifact["failure_synthesis_summary"]["mechanism_repeats_failed_synthesis"] is True
    assert "event_spec_mechanism_not_in_failure_synthesis" in {
        check["name"] for check in artifact["blockers"]
    }


def test_local_event_builder_blocks_unsupported_feature(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    spec_path = tmp_path / "event_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(minutes=5 * index),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0 + index,
                "volume": 1000.0,
            }
            for index in range(10)
        ]
    ).to_csv(ohlcv_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_spec",
                "event_id": "bad-event-spec",
                "thesis_id": "TH-BAD-EVENT-001",
                "mechanism_class": "unsupported_future_event",
                "conditions": [
                    {
                        "feature": "future_return_bps",
                        "lookback_candles": 1,
                        "operator": ">=",
                        "value": 10.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_events(
        LocalEventBuildInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            event_spec_path=spec_path,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "blocked"
    assert artifact["event_count"] == 0
    assert "condition_1_valid" in {check["name"] for check in artifact["blockers"]}


def test_edge_discovery_builds_multi_horizon_cost_edge_artifact(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    spec_path = tmp_path / "edge_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    price = 100.0
    for index in range(80):
        price *= 1.002
        rows.append(
            {
                "date": start + pd.Timedelta(minutes=5 * index),
                "open": price * 0.999,
                "high": price * 1.001,
                "low": price * 0.998,
                "close": price,
                "volume": 1000.0 + index,
            }
        )
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_edge_discovery_spec",
                "edge_discovery_id": "ED-UNIT-001",
                "thesis_id": "TH-EDGE-DISCOVERY-001",
                "mechanism_class": "closed_candle_momentum_probe",
                "hypothesis_scope": "cross_asset",
                "instrument_universe": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                "market_structure_domains": ["ohlcv", "cross_asset"],
                "hypothesis": "Positive closed-candle impulse persists after costs.",
                "conditions": [
                    {
                        "feature": "return_bps",
                        "operator": ">",
                        "value": 0.0,
                        "lookback_candles": 1,
                    }
                ],
                "horizons": [1, 3],
                "all_in_cost_bps": 1.0,
                "cooldown_candles": 1,
            }
        ),
        encoding="utf-8",
    )

    artifact = build_edge_discovery(
        EdgeDiscoveryInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            edge_spec_path=spec_path,
            min_sample_count=20,
            min_profitable_windows_ratio=0.5,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["factory"] == "research_edge_discovery"
    assert artifact["status"] == "passed"
    assert artifact["proposal_generation_allowed"] is False
    assert artifact["candidate_generation_allowed"] is False
    assert artifact["candidate_generation_result"] == "no candidate generated"
    assert artifact["strategy_codegen_allowed"] is False
    assert artifact["hypothesis_scope"] == "cross_asset"
    assert artifact["instrument_universe"] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert artifact["market_structure_domains"] == ["ohlcv", "cross_asset"]
    assert artifact["passing_horizon_count"] == 2
    assert artifact["best_horizon_by_net_edge"]["net_edge_bps"] > 0.0
    concentration = artifact["concentration_diagnostics"]
    assert concentration["event_count"] == artifact["event_count"]
    assert concentration["date_parseable_event_count"] == artifact["event_count"]
    assert concentration["active_day_count"] == 1
    assert concentration["max_day_event_share"] == 1.0
    assert artifact["safety_scope"]["backtest_started"] is False
    assert artifact["safety_scope"]["strategy_code_generated"] is False
    assert "proposal_generation_without_passing_research_gate" in artifact[
        "blocked_next_actions"
    ]
    assert "not_single_pair_dependent" in {
        check["name"] for check in artifact["research_gate"]["blockers"]
    }

    json_path, report_path = write_edge_discovery_artifacts(
        artifact,
        root_dir=tmp_path,
        output_root=tmp_path / "registry" / "strategies" / "research_decisions",
    )
    assert json_path.is_file()
    assert report_path.is_file()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Concentration Diagnostics" in report_text
    assert "max_day_event_share" in report_text
    assert "hypothesis_scope: cross_asset" in report_text


def test_cost_model_supports_default_scenarios_and_context_override():
    defaults = default_cost_scenarios()
    assert defaults["normal"].total_cost_bps == 12.0
    assert defaults["stress"].total_cost_bps >= defaults["normal"].total_cost_bps * 1.5

    scenarios = cost_scenarios_from_spec(
        {
            "pair": "BTC/USDT:USDT",
            "timeframe": "5m",
            "order_type": "maker",
            "cost_model": {
                "overrides": [
                    {
                        "pair": "BTC/USDT:USDT",
                        "timeframe": "5m",
                        "order_type": "maker",
                        "scenarios": [
                            {
                                "scenario_name": "normal",
                                "fee_bps_entry": 1.0,
                                "fee_bps_exit": 1.0,
                                "spread_bps": 0.5,
                                "slippage_bps_entry": 0.25,
                                "slippage_bps_exit": 0.25,
                                "adverse_selection_bps": 2.0,
                                "no_fill_rate": 0.2,
                                "partial_fill_rate": 0.3,
                                "exit_taker_rate": 0.4,
                                "stress_multiplier": 1.0,
                            },
                            {
                                "scenario_name": "stress",
                                "fee_bps_entry": 1.0,
                                "fee_bps_exit": 1.0,
                                "spread_bps": 0.5,
                                "slippage_bps_entry": 0.25,
                                "slippage_bps_exit": 0.25,
                                "adverse_selection_bps": 2.0,
                                "no_fill_rate": 0.3,
                                "partial_fill_rate": 0.4,
                                "exit_taker_rate": 0.8,
                                "stress_multiplier": 2.0,
                            },
                        ],
                    }
                ]
            },
        },
        context=CostModelContext(
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="maker",
        ),
    )

    assert set(scenarios) == {"best", "normal", "stress"}
    assert scenarios["normal"]["total_cost_bps"] == 5.0
    assert scenarios["stress"]["total_cost_bps"] == 10.0
    assert scenarios["normal"]["no_fill_rate"] == 0.2
    assert scenarios["normal"]["partial_fill_rate"] == 0.3
    assert scenarios["normal"]["adverse_selection_bps"] == 2.0


def test_cost_model_preserves_zero_top_level_all_in_cost_bps():
    scenarios = cost_scenarios_from_spec({"all_in_cost_bps": 0})

    assert scenarios["normal"]["total_cost_bps"] == 0.0


def test_cost_model_preserves_inherited_total_cost_for_non_price_scenario_override():
    scenarios = cost_scenarios_from_spec(
        {
            "all_in_cost_bps": 0,
            "cost_model": {
                "scenarios": [
                    {
                        "scenario_name": "normal",
                        "no_fill_rate": 0.25,
                        "partial_fill_rate": 0.4,
                    }
                ]
            },
        }
    )

    assert scenarios["normal"]["total_cost_bps"] == 0.0
    assert scenarios["normal"]["no_fill_rate"] == 0.25
    assert scenarios["normal"]["partial_fill_rate"] == 0.4


def test_cost_model_selects_most_specific_matching_override():
    scenarios = cost_scenarios_from_spec(
        {
            "pair": "BTC/USDT:USDT",
            "timeframe": "5m",
            "order_type": "maker",
            "cost_model": {
                "overrides": [
                    {
                        "scenarios": [
                            {"scenario_name": "normal", "total_cost_bps": 20.0}
                        ]
                    },
                    {
                        "pair": "BTC/USDT:USDT",
                        "timeframe": "5m",
                        "order_type": "maker",
                        "scenarios": [
                            {"scenario_name": "normal", "total_cost_bps": 4.0}
                        ],
                    },
                ]
            },
        }
    )

    assert scenarios["normal"]["total_cost_bps"] == 4.0


def test_cost_model_merges_selected_override_with_base_scenarios():
    scenarios = cost_scenarios_from_spec(
        {
            "pair": "ETH/USDT:USDT",
            "timeframe": "5m",
            "cost_model": {
                "scenarios": [
                    {"scenario_name": "normal", "total_cost_bps": 8.0},
                    {"scenario_name": "stress", "total_cost_bps": 30.0},
                ],
                "overrides": [
                    {
                        "pair": "ETH/USDT:USDT",
                        "timeframe": "5m",
                        "scenarios": [
                            {"scenario_name": "normal", "total_cost_bps": 4.0}
                        ],
                    }
                ],
            },
        }
    )

    assert scenarios["normal"]["total_cost_bps"] == 4.0
    assert scenarios["stress"]["total_cost_bps"] == 30.0


def _write_cost_calibration_ohlcv(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=24, freq="5min"),
            "open": np.linspace(100.0, 103.0, 24),
            "high": np.linspace(100.4, 103.8, 24),
            "low": np.linspace(99.8, 102.6, 24),
            "close": np.linspace(100.1, 103.4, 24),
            "volume": np.linspace(1000.0, 1600.0, 24),
        }
    )
    frame.to_csv(path, index=False)


def _write_cost_calibration_order_book(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=24, freq="5min"),
            "best_bid": np.linspace(99.95, 103.2, 24),
            "best_ask": np.linspace(100.05, 103.35, 24),
            "bid_size": np.linspace(10.0, 14.0, 24),
            "ask_size": np.linspace(9.0, 12.0, 24),
        }
    )
    frame.to_csv(path, index=False)


def _write_cost_calibration_spread(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=24, freq="5min"),
            "spread_bps": np.linspace(1.0, 3.5, 24),
        }
    )
    frame.to_csv(path, index=False)


def test_cost_calibration_builds_scenarios_and_artifacts_from_local_data(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    order_book_path = tmp_path / "order_book.csv"
    fills_path = tmp_path / "fills.csv"
    _write_cost_calibration_ohlcv(ohlcv_path)
    _write_cost_calibration_order_book(order_book_path)
    pd.DataFrame(
        [
            {
                "scenario_name": "best",
                "no_fill_rate": 0.04,
                "partial_fill_rate": 0.08,
                "adverse_selection_bps": 0.25,
                "exit_taker_rate": 0.2,
            },
            {
                "scenario_name": "normal",
                "no_fill_rate": 0.10,
                "partial_fill_rate": 0.18,
                "adverse_selection_bps": 0.75,
                "exit_taker_rate": 0.5,
            },
            {
                "scenario_name": "stress",
                "no_fill_rate": 0.22,
                "partial_fill_rate": 0.35,
                "adverse_selection_bps": 1.75,
                "exit_taker_rate": 0.85,
            },
        ]
    ).to_csv(fills_path, index=False)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            order_book_path=order_book_path,
            fills_path=fills_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="maker",
            liquidity_tier="liquid",
            volatility_regime="normal",
            cost_calibration_id="unit_cost_calibration",
            output_root=tmp_path / "artifacts",
            created_at="2026-05-10T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "completed"
    assert artifact["candidate_generation_allowed"] is False
    assert artifact["proposal_generation_allowed"] is False
    assert artifact["strategy_codegen_allowed"] is False
    assert artifact["candidate_generation_result"] == "no candidate generated"
    assert set(artifact["cost_scenarios"]) == {"best", "normal", "stress"}
    normal = artifact["cost_scenarios"]["normal"]
    stress = artifact["cost_scenarios"]["stress"]
    assert normal["total_cost_bps"] is not None
    assert stress["total_cost_bps"] >= normal["total_cost_bps"]
    for scenario in artifact["cost_scenarios"].values():
        assert scenario["no_fill_rate"] is not None
        assert scenario["partial_fill_rate"] is not None
        assert scenario["adverse_selection_bps"] is not None
        assert scenario["exit_taker_rate"] is not None
        assert scenario["pair"] == "BTC/USDT:USDT"
        assert scenario["timeframe"] == "5m"
        assert scenario["order_type"] == "maker"

    json_path, report_path, table_path = write_cost_calibration_artifacts(
        artifact,
        root_dir=tmp_path,
        output_root=tmp_path / "artifacts",
    )
    assert json_path.is_file()
    assert report_path.is_file()
    assert table_path.is_file()
    report_text = report_path.read_text(encoding="utf-8")
    table_text = table_path.read_text(encoding="utf-8")
    assert "candidate_generation_result: no candidate generated" in report_text
    assert "normal" in table_text
    assert "stress" in table_text


def test_cost_calibration_sanitizes_id_before_writing_artifacts(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    _write_cost_calibration_ohlcv(ohlcv_path)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            order_type="taker",
            cost_calibration_id="../other-dir",
        )
    )

    assert artifact["cost_calibration_id"] == "other-dir"
    json_path, report_path, table_path = write_cost_calibration_artifacts(
        artifact,
        root_dir=tmp_path,
        output_root=tmp_path / "artifacts",
    )

    output_root = (tmp_path / "artifacts").resolve()
    json_path.resolve().relative_to(output_root)
    assert json_path.parent == output_root / "other-dir"
    assert report_path.parent == json_path.parent
    assert table_path.parent == json_path.parent
    assert not (tmp_path / "other-dir" / "cost_calibration.json").exists()
    assert artifact["candidate_generation_result"] == "no candidate generated"


def test_cost_calibration_accepts_depth_only_order_book_with_spread_artifact(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    order_book_path = tmp_path / "depth_only_order_book.csv"
    spread_path = tmp_path / "spread.csv"
    _write_cost_calibration_ohlcv(ohlcv_path)
    _write_cost_calibration_spread(spread_path)
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=24, freq="5min"),
            "bid_size": np.linspace(10.0, 14.0, 24),
            "ask_size": np.linspace(8.0, 13.0, 24),
        }
    ).to_csv(order_book_path, index=False)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            order_book_path=order_book_path,
            spread_path=spread_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="maker",
            cost_calibration_id="depth-only-order-book",
        )
    )

    assert artifact["status"] == "completed"
    assert artifact["sources"]["order_book"]["status"] == "loaded"
    assert artifact["sources"]["spread"]["status"] == "loaded"
    assert artifact["cost_scenarios"]["normal"]["no_fill_rate"] is not None
    assert artifact["cost_scenarios"]["normal"]["provenance"]["spread_source"] == "spread_artifact"
    assert "order_book_usable_columns_missing" not in {
        blocker["name"] for blocker in artifact["blockers"]
    }


def test_cost_calibration_blocks_unusable_spread_artifact(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    order_book_path = tmp_path / "order_book.csv"
    spread_path = tmp_path / "spread.csv"
    _write_cost_calibration_ohlcv(ohlcv_path)
    _write_cost_calibration_order_book(order_book_path)
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3, freq="5min"),
            "spread_bps": ["bad", None, "nan"],
        }
    ).to_csv(spread_path, index=False)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            order_book_path=order_book_path,
            spread_path=spread_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="taker",
            cost_calibration_id="bad-spread",
        )
    )

    blockers = {blocker["name"] for blocker in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert artifact["sources"]["spread"]["status"] == "blocked"
    assert artifact["sources"]["spread"]["blocker_name"] == "spread_numeric_rows_missing"
    assert "spread_numeric_rows_missing" in blockers
    assert artifact["candidate_generation_result"] == "no candidate generated"


def test_cost_calibration_blocks_negative_spread_artifact_rows(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    order_book_path = tmp_path / "order_book.csv"
    spread_path = tmp_path / "negative_spread.csv"
    _write_cost_calibration_ohlcv(ohlcv_path)
    _write_cost_calibration_order_book(order_book_path)
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3, freq="5min"),
            "spread_bps": [-1.0, -2.0, -3.0],
        }
    ).to_csv(spread_path, index=False)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            order_book_path=order_book_path,
            spread_path=spread_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="taker",
            cost_calibration_id="negative-spread",
        )
    )

    blockers = {blocker["name"] for blocker in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert artifact["sources"]["spread"]["status"] == "blocked"
    assert artifact["sources"]["spread"]["blocker_name"] == "spread_negative_rows_present"
    assert "spread_negative_rows_present" in blockers
    assert artifact["candidate_generation_result"] == "no candidate generated"


def test_cost_calibration_blocks_order_book_nonfinite_spread_values(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    order_book_path = tmp_path / "order_book.csv"
    _write_cost_calibration_ohlcv(ohlcv_path)
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3, freq="5min"),
            "best_bid": [-1.0, -2.0, -3.0],
            "best_ask": [1.0, 2.0, 3.0],
        }
    ).to_csv(order_book_path, index=False)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            order_book_path=order_book_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="taker",
            cost_calibration_id="nonfinite-order-book-spread",
        )
    )

    blockers = {blocker["name"] for blocker in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert artifact["sources"]["order_book"]["status"] == "blocked"
    assert (
        artifact["sources"]["order_book"]["blocker_name"]
        == "order_book_spread_numeric_rows_missing"
    )
    assert "order_book_spread_numeric_rows_missing" in blockers
    assert artifact["candidate_generation_result"] == "no candidate generated"


def test_cost_calibration_blocks_maker_order_book_without_numeric_depth(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    order_book_path = tmp_path / "order_book.csv"
    spread_path = tmp_path / "spread.csv"
    _write_cost_calibration_ohlcv(ohlcv_path)
    _write_cost_calibration_spread(spread_path)
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3, freq="5min"),
            "bid_size": ["bad", None, "nan"],
            "ask_size": ["bad", None, "nan"],
        }
    ).to_csv(order_book_path, index=False)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            order_book_path=order_book_path,
            spread_path=spread_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="maker",
            cost_calibration_id="bad-maker-depth",
        )
    )

    blockers = {blocker["name"] for blocker in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert artifact["sources"]["order_book"]["status"] == "blocked"
    assert (
        artifact["sources"]["order_book"]["blocker_name"]
        == "order_book_depth_numeric_rows_missing"
    )
    assert "order_book_depth_numeric_rows_missing" in blockers
    assert artifact["candidate_generation_result"] == "no candidate generated"


def test_cost_calibration_blocks_negative_maker_order_book_depth(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    order_book_path = tmp_path / "order_book.csv"
    spread_path = tmp_path / "spread.csv"
    _write_cost_calibration_ohlcv(ohlcv_path)
    _write_cost_calibration_spread(spread_path)
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3, freq="5min"),
            "bid_size": [-10.0, -11.0, -12.0],
            "ask_size": [15.0, 16.0, 17.0],
        }
    ).to_csv(order_book_path, index=False)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            order_book_path=order_book_path,
            spread_path=spread_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="maker",
            cost_calibration_id="negative-maker-depth",
        )
    )

    blockers = {blocker["name"] for blocker in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert artifact["sources"]["order_book"]["status"] == "blocked"
    assert (
        artifact["sources"]["order_book"]["blocker_name"]
        == "order_book_depth_negative_rows_present"
    )
    assert "order_book_depth_negative_rows_present" in blockers
    assert artifact["candidate_generation_result"] == "no candidate generated"


def test_cost_calibration_blocks_missing_normal_cost_with_structured_blocker(tmp_path):
    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=None,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="taker",
            cost_calibration_id="missing-normal",
        )
    )

    blockers = {blocker["name"] for blocker in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert "ohlcv_path_missing" in blockers
    assert "normal_cost_missing" in blockers
    assert artifact["candidate_generation_result"] == "no candidate generated"


def test_cost_calibration_blocks_stress_cost_below_normal(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    fills_path = tmp_path / "fills.json"
    _write_cost_calibration_ohlcv(ohlcv_path)
    fills_path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_name": "normal",
                        "total_cost_bps": 20.0,
                    },
                    {
                        "scenario_name": "stress",
                        "total_cost_bps": 10.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            fills_path=fills_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="taker",
            cost_calibration_id="stress-lt-normal",
        )
    )

    blockers = {blocker["name"] for blocker in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert "stress_cost_below_normal" in blockers


def test_cost_calibration_blocks_maker_missing_fill_risk_fields(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    _write_cost_calibration_ohlcv(ohlcv_path)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="maker",
            cost_calibration_id="maker-missing-risk",
        )
    )

    blockers = {blocker["name"] for blocker in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert "maker_no_fill_rate_missing" in blockers
    assert "maker_partial_fill_rate_missing" in blockers
    assert "maker_adverse_selection_bps_missing" in blockers
    assert "maker_exit_taker_rate_missing" in blockers
    assert "strategy_generation_from_cost_calibration" in artifact["blocked_next_actions"]


def test_cost_calibration_keeps_fills_when_context_selectors_are_unset(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    fills_path = tmp_path / "fills.csv"
    _write_cost_calibration_ohlcv(ohlcv_path)
    pd.DataFrame(
        [
            {
                "scenario_name": "normal",
                "pair": "BTC/USDT:USDT",
                "timeframe": "5m",
                "order_type": "maker",
                "no_fill_rate": 0.11,
                "partial_fill_rate": 0.22,
                "adverse_selection_bps": 1.5,
                "exit_taker_rate": 0.66,
                "total_cost_bps": 18.0,
            },
            {
                "scenario_name": "stress",
                "pair": "BTC/USDT:USDT",
                "timeframe": "5m",
                "order_type": "maker",
                "no_fill_rate": 0.25,
                "partial_fill_rate": 0.4,
                "adverse_selection_bps": 3.0,
                "exit_taker_rate": 0.9,
                "total_cost_bps": 30.0,
            },
        ]
    ).to_csv(fills_path, index=False)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            fills_path=fills_path,
            cost_calibration_id="unset-context-fills",
        )
    )

    scenarios = artifact["cost_scenarios"]
    assert scenarios["normal"]["total_cost_bps"] == 18.0
    assert scenarios["normal"]["no_fill_rate"] == 0.11
    assert scenarios["normal"]["partial_fill_rate"] == 0.22
    assert scenarios["normal"]["adverse_selection_bps"] == 1.5
    assert scenarios["normal"]["exit_taker_rate"] == 0.66
    assert scenarios["stress"]["total_cost_bps"] == 30.0


def test_cost_calibration_prefers_specific_fills_over_later_generic_duplicate(
    tmp_path,
):
    ohlcv_path = tmp_path / "ohlcv.csv"
    fills_path = tmp_path / "fills.csv"
    _write_cost_calibration_ohlcv(ohlcv_path)
    pd.DataFrame(
        [
            {
                "scenario_name": "normal",
                "pair": "BTC/USDT:USDT",
                "timeframe": "5m",
                "order_type": "taker",
                "total_cost_bps": 18.0,
            },
            {
                "scenario_name": "normal",
                "total_cost_bps": 99.0,
            },
            {
                "scenario_name": "stress",
                "total_cost_bps": 32.0,
            },
        ]
    ).to_csv(fills_path, index=False)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            fills_path=fills_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="taker",
            cost_calibration_id="specific-over-generic-fills",
        )
    )

    assert artifact["status"] == "completed"
    assert artifact["sources"]["fills"]["status"] == "loaded"
    assert artifact["sources"]["fills"]["row_count"] == 3
    assert artifact["cost_scenarios"]["normal"]["total_cost_bps"] == 18.0
    assert artifact["cost_scenarios"]["stress"]["total_cost_bps"] == 32.0
    assert artifact["candidate_generation_result"] == "no candidate generated"


def test_cost_calibration_blocks_fills_artifact_with_zero_matching_scenarios(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    fills_path = tmp_path / "fills.csv"
    _write_cost_calibration_ohlcv(ohlcv_path)
    pd.DataFrame(
        [
            {
                "scenario_name": "normal",
                "pair": "ETH/USDT:USDT",
                "timeframe": "5m",
                "order_type": "maker",
                "no_fill_rate": 0.10,
                "partial_fill_rate": 0.20,
                "adverse_selection_bps": 1.0,
                "exit_taker_rate": 0.5,
                "total_cost_bps": 18.0,
            }
        ]
    ).to_csv(fills_path, index=False)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            fills_path=fills_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="maker",
            cost_calibration_id="zero-matching-fills",
        )
    )

    blockers = {blocker["name"] for blocker in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert artifact["sources"]["fills"]["status"] == "blocked"
    assert artifact["sources"]["fills"]["blocker_name"] == "fills_scenarios_missing"
    assert "fills_scenarios_missing" in blockers
    assert artifact["candidate_generation_result"] == "no candidate generated"


def test_cost_calibration_blocks_json_fills_with_zero_matching_scenarios(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    fills_path = tmp_path / "fills.json"
    _write_cost_calibration_ohlcv(ohlcv_path)
    fills_path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_name": "normal",
                        "pair": "ETH/USDT:USDT",
                        "timeframe": "5m",
                        "total_cost_bps": 18.0,
                    },
                    {
                        "scenario_name": "not_a_scenario",
                        "pair": "BTC/USDT:USDT",
                        "timeframe": "5m",
                        "total_cost_bps": 20.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            fills_path=fills_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="taker",
            cost_calibration_id="zero-matching-json-fills",
        )
    )

    blockers = {blocker["name"] for blocker in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert artifact["sources"]["fills"]["row_count"] == 2
    assert artifact["sources"]["fills"]["blocker_name"] == "fills_scenarios_missing"
    assert "fills_scenarios_missing" in blockers


def test_cost_calibration_loads_top_level_json_fills_array(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    fills_path = tmp_path / "fills.json"
    _write_cost_calibration_ohlcv(ohlcv_path)
    fills_path.write_text(
        json.dumps(
            [
                {
                    "scenario_name": "normal",
                    "pair": "BTC/USDT:USDT",
                    "timeframe": "5m",
                    "total_cost_bps": 18.0,
                },
                {
                    "scenario_name": "stress",
                    "pair": "BTC/USDT:USDT",
                    "timeframe": "5m",
                    "total_cost_bps": 32.0,
                },
            ]
        ),
        encoding="utf-8",
    )

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            fills_path=fills_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="taker",
            cost_calibration_id="top-level-json-fills",
        )
    )

    assert artifact["status"] == "completed"
    assert artifact["sources"]["fills"]["status"] == "loaded"
    assert artifact["sources"]["fills"]["row_count"] == 2
    assert artifact["cost_scenarios"]["normal"]["total_cost_bps"] == 18.0
    assert artifact["cost_scenarios"]["stress"]["total_cost_bps"] == 32.0
    assert artifact["candidate_generation_result"] == "no candidate generated"


def test_cost_calibration_default_id_preserves_subsecond_resolution(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    _write_cost_calibration_ohlcv(ohlcv_path)

    first = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            order_type="taker",
            created_at="2026-05-10T09:57:04.111111+00:00",
        )
    )
    second = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            order_type="taker",
            created_at="2026-05-10T09:57:04.222222+00:00",
        )
    )

    assert first["cost_calibration_id"] != second["cost_calibration_id"]
    assert "111111" in first["cost_calibration_id"]
    assert "222222" in second["cost_calibration_id"]
    assert first["candidate_generation_result"] == "no candidate generated"
    assert second["candidate_generation_result"] == "no candidate generated"


def test_cost_calibration_returns_structured_blocker_for_fills_parse_error(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    fills_path = tmp_path / "fills.json"
    _write_cost_calibration_ohlcv(ohlcv_path)
    fills_path.write_text("{not-json", encoding="utf-8")

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            fills_path=fills_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="taker",
            cost_calibration_id="bad-fills",
        )
    )

    blockers = {blocker["name"] for blocker in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert "fills_artifact_parse_error" in blockers
    assert artifact["candidate_generation_allowed"] is False
    assert artifact["candidate_generation_result"] == "no candidate generated"


def test_cost_calibration_returns_structured_blocker_for_missing_ohlcv_columns(tmp_path):
    ohlcv_path = tmp_path / "missing_close_ohlcv.csv"
    pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
        }
    ).to_csv(ohlcv_path, index=False)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="taker",
            cost_calibration_id="missing-ohlcv-column",
        )
    )

    blockers = {blocker["name"] for blocker in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert artifact["sources"]["ohlcv"]["status"] == "blocked"
    assert artifact["sources"]["ohlcv"]["blocker_name"] == "ohlcv_required_columns_missing"
    assert "ohlcv_required_columns_missing" in blockers
    assert "normal_cost_missing" in blockers
    assert artifact["candidate_generation_result"] == "no candidate generated"


def test_cost_calibration_returns_structured_blocker_for_unusable_ohlcv_rows(tmp_path):
    ohlcv_path = tmp_path / "bad_ohlcv.csv"
    pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "open": ["bad"],
            "high": ["bad"],
            "low": ["bad"],
            "close": ["bad"],
        }
    ).to_csv(ohlcv_path, index=False)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="taker",
            cost_calibration_id="bad-ohlcv",
        )
    )

    blockers = {blocker["name"] for blocker in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert "ohlcv_numeric_rows_missing" in blockers
    assert artifact["candidate_generation_result"] == "no candidate generated"


def test_cost_calibration_blocks_nonfinite_ohlcv_rows(tmp_path):
    ohlcv_path = tmp_path / "nonfinite_ohlcv.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3, freq="5min"),
            "open": [100.0, 101.0, 102.0],
            "high": [float("inf"), float("inf"), float("inf")],
            "low": [float("-inf"), float("-inf"), float("-inf")],
            "close": [float("inf"), float("-inf"), float("inf")],
        }
    ).to_csv(ohlcv_path, index=False)

    artifact = build_cost_calibration(
        CostCalibrationInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            pair="BTC/USDT:USDT",
            timeframe="5m",
            order_type="taker",
            cost_calibration_id="nonfinite-ohlcv",
        )
    )

    blockers = {blocker["name"] for blocker in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert artifact["sources"]["ohlcv"]["status"] == "blocked"
    assert artifact["sources"]["ohlcv"]["blocker_name"] == "ohlcv_numeric_rows_missing"
    assert "ohlcv_numeric_rows_missing" in blockers
    assert artifact["candidate_generation_result"] == "no candidate generated"


def test_event_level_report_ignores_gate_pass_on_structurally_failed_horizon():
    report = _event_level_post_cost_report(
        {
            "thesis_id": "TH-STRUCTURAL-HORIZON-GATE-001",
            "mechanism_class": "structural_gate_selection_probe",
        },
        best_horizon={
            "hold_candles": 1,
            "net_edge_bps_normal": 50.0,
        },
        horizon_results=[
            {
                "hold_candles": 1,
                "status": "failed",
                "sample_count": 2,
                "net_edge_bps_normal": 50.0,
                "passes_research_gate": True,
                "research_gate": {
                    "passes_research_gate": True,
                    "rejection_reason": None,
                    "candidate_generation_result": "candidate generation allowed",
                },
            },
            {
                "hold_candles": 2,
                "status": "passed",
                "sample_count": 64,
                "net_edge_bps_normal": 5.0,
                "passes_research_gate": False,
                "research_gate": {
                    "passes_research_gate": False,
                    "rejection_reason": "random_entry_control_beaten",
                    "candidate_generation_result": "no candidate generated",
                },
            },
        ],
        concentration={"max_quarter_event_share": 0.25},
    )

    assert report["holding_period"] == 2
    assert report["passes_research_gate"] is False
    assert report["candidate_generation_result"] == "no candidate generated"


def test_edge_discovery_price_frame_prefers_populated_symbol_column():
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "pair": "unknown",
                "symbol": symbol,
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000.0,
            }
            for index, (symbol, close) in enumerate(
                [
                    ("BTC/USDT:USDT", 500.0),
                    ("ETH/USDT:USDT", 100.0),
                    ("ETH/USDT:USDT", 105.0),
                ]
            )
        ]
    )

    subset = _price_frame_for_label(frame, "ETH/USDT:USDT")

    assert list(subset["symbol"]) == ["ETH/USDT:USDT", "ETH/USDT:USDT"]
    assert list(subset["close"]) == [100.0, 105.0]


def test_local_events_grouped_features_prefer_populated_symbol_column():
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    frame = pd.DataFrame(
        [
            {
                "date": start,
                "pair": "unknown",
                "symbol": "BTC/USDT:USDT",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
            },
            {
                "date": start,
                "pair": "unknown",
                "symbol": "ETH/USDT:USDT",
                "open": 200.0,
                "high": 201.0,
                "low": 199.0,
                "close": 200.0,
                "volume": 2000.0,
            },
            {
                "date": start + pd.Timedelta(hours=1),
                "pair": "unknown",
                "symbol": "BTC/USDT:USDT",
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.0,
                "volume": 1100.0,
            },
            {
                "date": start + pd.Timedelta(hours=1),
                "pair": "unknown",
                "symbol": "ETH/USDT:USDT",
                "open": 202.0,
                "high": 203.0,
                "low": 201.0,
                "close": 202.0,
                "volume": 2200.0,
            },
        ]
    )

    returns = _feature_series(
        frame,
        {"feature": "return_bps", "lookback_candles": 1},
    )
    sma_distance = _feature_series(
        frame,
        {"feature": "sma_distance_bps", "lookback_candles": 2},
    )

    assert pd.isna(returns.iloc[0])
    assert pd.isna(returns.iloc[1])
    assert round(float(returns.iloc[2]), 6) == 100.0
    assert round(float(returns.iloc[3]), 6) == 100.0
    assert pd.isna(sma_distance.iloc[0])
    assert pd.isna(sma_distance.iloc[1])
    assert round(float(sma_distance.iloc[2]), 6) == round(
        (101.0 / 100.5 - 1.0) * 10000.0,
        6,
    )
    assert round(float(sma_distance.iloc[3]), 6) == round(
        (202.0 / 201.0 - 1.0) * 10000.0,
        6,
    )


def test_edge_discovery_uses_next_candle_open_entry_semantics(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    spec_path = tmp_path / "edge_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = [
        {
            "date": start,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000.0,
        },
        {
            "date": start + pd.Timedelta(minutes=5),
            "open": 100.0,
            "high": 111.0,
            "low": 100.0,
            "close": 110.0,
            "volume": 1000.0,
        },
        {
            "date": start + pd.Timedelta(minutes=10),
            "open": 200.0,
            "high": 201.0,
            "low": 199.0,
            "close": 200.5,
            "volume": 1000.0,
        },
        {
            "date": start + pd.Timedelta(minutes=15),
            "open": 201.0,
            "high": 221.0,
            "low": 200.0,
            "close": 220.0,
            "volume": 1000.0,
        },
    ]
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_edge_discovery_spec",
                "thesis_id": "TH-NEXT-OPEN-001",
                "mechanism_class": "next_open_semantics_probe",
                "hypothesis_scope": "cross_asset",
                "instrument_universe": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                "conditions": [
                    {
                        "feature": "return_bps",
                        "operator": ">",
                        "value": 500.0,
                        "lookback_candles": 1,
                    }
                ],
                "horizons": [1],
                "all_in_cost_bps": 1.0,
            }
        ),
        encoding="utf-8",
    )

    artifact = build_edge_discovery(
        EdgeDiscoveryInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            edge_spec_path=spec_path,
            min_sample_count=1,
            min_profitable_windows_ratio=0.0,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    sample = artifact["horizon_results"][0]["sample_preview"][0]
    assert sample["event_time"] == "2026-01-01T00:05:00+00:00"
    assert sample["entry_time"] == "2026-01-01T00:10:00+00:00"
    assert sample["entry_semantics"] == "next_candle_open"
    assert sample["entry_price_type"] == "open"
    assert sample["entry_price"] == 200.0
    assert sample["exit_time"] == "2026-01-01T00:10:00+00:00"
    assert sample["exit_price_type"] == "close"
    assert sample["exit_price"] == 200.5
    assert sample["exit_close"] == 200.5


def test_edge_discovery_matches_event_pair_to_ohlcv_price_series(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    spec_path = tmp_path / "edge_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    prices = {
        "BTC/USDT:USDT": [(50.0, 50.0), (50.0, 50.0), (10.0, 10.0)],
        "ETH/USDT:USDT": [(100.0, 100.0), (100.0, 102.0), (100.0, 105.0)],
    }
    for index in range(3):
        for pair, candles in prices.items():
            open_, close = candles[index]
            rows.append(
                {
                    "date": start + pd.Timedelta(days=index),
                    "pair": pair,
                    "open": open_,
                    "high": max(open_, close) + 0.5,
                    "low": min(open_, close) - 0.5,
                    "close": close,
                    "volume": 1000.0,
                }
            )
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_edge_discovery_spec",
                "thesis_id": "TH-PAIR-PRICE-MATCH-001",
                "mechanism_class": "pair_price_alignment_probe",
                "hypothesis_scope": "cross_asset",
                "instrument_universe": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                "conditions": [
                    {
                        "feature": "return_bps",
                        "operator": ">",
                        "value": 100.0,
                        "lookback_candles": 1,
                    }
                ],
                "horizons": [1],
                "all_in_cost_bps": 1.0,
            }
        ),
        encoding="utf-8",
    )

    artifact = build_edge_discovery(
        EdgeDiscoveryInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            edge_spec_path=spec_path,
            min_sample_count=1,
            min_profitable_windows_ratio=0.0,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    sample = artifact["horizon_results"][0]["sample_preview"][0]
    assert sample["pair"] == "ETH/USDT:USDT"
    assert sample["event_time"] == "2026-01-02T00:00:00+00:00"
    assert sample["entry_time"] == "2026-01-03T00:00:00+00:00"
    assert sample["entry_price"] == 100.0
    assert sample["exit_price"] == 105.0
    assert sample["price_series_instrument"] == "ETH/USDT:USDT"


def test_edge_discovery_research_gate_passes_synthetic_positive_case(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    spec_path = tmp_path / "edge_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    pair_event_indices = {
        "BTC/USDT:USDT": {10, 90, 170},
        "ETH/USDT:USDT": {50, 130, 210},
    }
    rows = []
    for index in range(240):
        for pair, event_indices in pair_event_indices.items():
            close = 100.0
            open_ = 100.0
            if index in event_indices:
                close = 102.0
            if index - 1 in event_indices:
                close = 103.0
            rows.append(
                {
                    "date": start + pd.Timedelta(days=index),
                    "pair": pair,
                    "open": open_,
                    "high": max(open_, close) + 0.5,
                    "low": min(open_, close) - 0.5,
                    "close": close,
                    "volume": 1000.0,
                }
            )
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_edge_discovery_spec",
                "thesis_id": "TH-RESEARCH-GATE-PASS-001",
                "mechanism_class": "rare_forced_flow_reversal",
                "hypothesis_scope": "cross_asset",
                "instrument_universe": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                "market_structure_domains": ["ohlcv"],
                "conditions": [
                    {
                        "feature": "return_bps",
                        "operator": ">",
                        "value": 100.0,
                        "lookback_candles": 1,
                    }
                ],
                "horizons": [1],
                "cost_model": {
                    "scenarios": [
                        {"scenario_name": "best", "total_cost_bps": 1.0},
                        {"scenario_name": "normal", "total_cost_bps": 2.0},
                        {"scenario_name": "stress", "total_cost_bps": 3.0},
                    ]
                },
                "cooldown_candles": 5,
            }
        ),
        encoding="utf-8",
    )

    artifact = build_edge_discovery(
        EdgeDiscoveryInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            edge_spec_path=spec_path,
            min_sample_count=5,
            min_profitable_windows_ratio=0.7,
            min_calendar_window_count=3,
            min_profitable_calendar_windows_ratio=0.6,
            min_negative_control_delta_bps=1.0,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    report = artifact["event_level_post_cost_edge_report"]
    assert artifact["status"] == "passed"
    assert artifact["candidate_generation_allowed"] is True
    assert report["passes_research_gate"] is True
    assert report["pair_concentration"] == 0.5
    assert report["pair_evidence_unique_count"] == 2
    assert report["pair_price_series"]["multi_instrument_price_series_aligned"] is True
    assert report["net_edge_bps_normal"] >= 6.0
    assert report["net_edge_bps_stress"] > 0.0
    assert report["lower_confidence_bound_bps"] > 0.0
    assert report["negative_control_random_entry_delta_bps"] >= 1.0
    assert report["negative_control_shuffled_signal_delta_bps"] >= 1.0
    assert report["negative_control_shifted_signal_delta_bps"] >= 1.0
    horizon = artifact["horizon_results"][0]
    expected_distribution = {"BTC/USDT:USDT": 3, "ETH/USDT:USDT": 3}
    assert horizon["negative_controls"]["random_entry"][
        "pair_evidence_distribution"
    ] == expected_distribution
    assert horizon["negative_controls"]["shuffled_signal"][
        "pair_evidence_distribution"
    ] == expected_distribution
    assert horizon["negative_controls"]["shifted_signal"][
        "pair_evidence_distribution"
    ] == expected_distribution


def test_edge_discovery_research_gate_rejects_pair_labels_without_price_series(
    tmp_path,
):
    ohlcv_path = tmp_path / "ohlcv.csv"
    spec_path = tmp_path / "edge_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    event_indices = {10, 31, 50, 71, 90}
    rows = []
    for index in range(120):
        pair = "BTC/USDT:USDT" if index % 2 == 0 else "ETH/USDT:USDT"
        close = 100.0
        if index in event_indices:
            close = 102.0
        if index - 2 in event_indices:
            close = 103.0
        rows.append(
            {
                "date": start + pd.Timedelta(days=index),
                "pair": pair,
                "open": 100.0,
                "high": close + 0.5,
                "low": 99.5,
                "close": close,
                "volume": 1000.0,
            }
        )
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_edge_discovery_spec",
                "thesis_id": "TH-DECLARED-MULTI-PAIR-ONLY-001",
                "mechanism_class": "declared_pair_list_is_not_evidence",
                "hypothesis_scope": "cross_asset",
                "instrument_universe": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                "conditions": [
                    {
                        "feature": "return_bps",
                        "operator": ">",
                        "value": 100.0,
                        "lookback_candles": 1,
                    }
                ],
                "horizons": [1],
                "cost_model": {
                    "scenarios": [
                        {"scenario_name": "best", "total_cost_bps": 1.0},
                        {"scenario_name": "normal", "total_cost_bps": 2.0},
                        {"scenario_name": "stress", "total_cost_bps": 3.0},
                    ]
                },
                "cooldown_candles": 5,
            }
        ),
        encoding="utf-8",
    )

    artifact = build_edge_discovery(
        EdgeDiscoveryInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            edge_spec_path=spec_path,
            min_sample_count=5,
            min_profitable_windows_ratio=0.7,
            min_negative_control_delta_bps=0.0,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    gate_blockers = {
        check["name"] for check in artifact["research_gate"]["blockers"]
    }
    report = artifact["event_level_post_cost_edge_report"]
    assert artifact["status"] == "passed"
    assert artifact["candidate_generation_allowed"] is False
    assert artifact["proposal_generation_allowed"] is False
    assert report["pair_concentration"] == 1.0
    assert report["pair_evidence_count"] == 5
    assert report["pair_evidence_unique_count"] == 2
    assert report["pair_price_series"]["shared_timestamp_count"] == 0
    assert "not_single_pair_dependent" in gate_blockers


def test_negative_controls_include_last_valid_next_open_start():
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    ohlcv = pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(days=index),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 1000.0,
            }
            for index in range(4)
        ]
    )
    events = pd.DataFrame({"date": ohlcv["date"].iloc[:3].tolist()})

    controls = _negative_control_summary(
        ohlcv,
        events,
        hold_candles=1,
        funding_rate=None,
        normal_cost_bps=0.0,
        real_net_edge_bps=None,
    )

    assert controls["random_entry"]["sample_count"] == 3
    assert controls["shuffled_signal"]["sample_count"] == 3
    assert controls["random_entry"]["sample_preview"][-1]["event_time"] == (
        "2026-01-03T00:00:00+00:00"
    )


def test_shifted_negative_controls_preserve_sample_count_near_boundaries():
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    ohlcv = pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(days=index),
                "pair": "BTC/USDT:USDT",
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 1000.0,
            }
            for index in range(8)
        ]
    )
    events = pd.DataFrame(
        {
            "date": [start + pd.Timedelta(days=index) for index in (3, 4, 5)],
            "pair": ["BTC/USDT:USDT"] * 3,
        }
    )

    controls = _control_events(
        ohlcv,
        events,
        hold_candles=2,
        event_count=len(events),
        mode="shifted_future",
    )

    assert len(controls) == 3
    assert len({control["date"] for control in controls}) == 3
    assert {control["pair"] for control in controls} == {"BTC/USDT:USDT"}


def test_edge_discovery_research_gate_blocks_negative_controls_and_reports_no_candidate(
    tmp_path,
):
    ohlcv_path = tmp_path / "ohlcv.csv"
    spec_path = tmp_path / "edge_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(days=index),
                "open": 99.0 + index,
                "high": 101.0 + index,
                "low": 98.0 + index,
                "close": 100.0 + index,
                "volume": 1000.0,
            }
            for index in range(90)
        ]
    ).to_csv(ohlcv_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_edge_discovery_spec",
                "thesis_id": "TH-NEGATIVE-CONTROL-BLOCK-001",
                "mechanism_class": "market_beta_disguised_as_signal",
                "hypothesis_scope": "cross_asset",
                "instrument_universe": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                "conditions": [
                    {
                        "feature": "return_bps",
                        "operator": ">",
                        "value": 0.0,
                        "lookback_candles": 1,
                    }
                ],
                "horizons": [1],
                "cost_model": {
                    "scenarios": [
                        {"scenario_name": "best", "total_cost_bps": 1.0},
                        {"scenario_name": "normal", "total_cost_bps": 1.0},
                        {"scenario_name": "stress", "total_cost_bps": 1.5},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    artifact = build_edge_discovery(
        EdgeDiscoveryInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            edge_spec_path=spec_path,
            min_sample_count=20,
            min_profitable_windows_ratio=0.7,
            min_negative_control_delta_bps=1.0,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    report = artifact["event_level_post_cost_edge_report"]
    gate_blockers = {
        check["name"] for check in artifact["research_gate"]["blockers"]
    }
    assert artifact["status"] == "passed"
    assert artifact["candidate_generation_allowed"] is False
    assert artifact["proposal_generation_allowed"] is False
    assert artifact["promotion_gate"]["proposal_generation_allowed"] is False
    assert artifact["candidate_generation_result"] == "no candidate generated"
    assert report["passes_research_gate"] is False
    assert "random_entry_control_beaten" in gate_blockers
    assert "shuffled_signal_control_beaten" in gate_blockers
    assert "no candidate generated" == report["candidate_generation_result"]

    _, report_path = write_edge_discovery_artifacts(
        artifact,
        root_dir=tmp_path,
        output_root=tmp_path / "registry" / "strategies" / "research_decisions",
    )
    report_text = report_path.read_text(encoding="utf-8")
    assert "candidate_generation_result: no candidate generated" in report_text
    assert "negative_control_random_entry_delta_bps" in report_text


def test_edge_discovery_supports_liquidation_context_features(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    liquidation_path = tmp_path / "liquidation.parquet"
    quality_path = tmp_path / "liquidation_quality.json"
    spec_path = tmp_path / "edge_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    closes = [100.0, 100.0, 110.0, 111.0, 112.0, 113.0]
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": close - 1.0,
                "high": close * 1.001,
                "low": (close - 1.0) * 0.999,
                "close": close,
                "volume": 1000.0,
            }
            for index, close in enumerate(closes)
        ]
    ).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [
                start + pd.Timedelta(hours=1, minutes=15),
                start + pd.Timedelta(hours=1, minutes=30),
            ],
            "side": ["Sell", "Sell"],
            "quantity": [2.0, 1.0],
            "price": [100.0, 100.0],
        }
    ).to_parquet(liquidation_path)
    _write_passing_structural_quality_report(quality_path, rows=2)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_edge_discovery_spec",
                "edge_discovery_id": "ED-LIQUIDATION-UNIT-001",
                "thesis_id": "TH-EDGE-LIQUIDATION-CONTEXT-001",
                "mechanism_class": "liquidation_absorption_probe",
                "hypothesis_scope": "microstructure",
                "market_structure_domains": ["liquidation"],
                "conditions": [
                    {
                        "feature": "liquidation_sell_notional",
                        "operator": ">=",
                        "value": 250.0,
                        "lookback_candles": 1,
                    }
                ],
                "horizons": [1],
                "all_in_cost_bps": 1.0,
            }
        ),
        encoding="utf-8",
    )

    artifact = build_edge_discovery(
        EdgeDiscoveryInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            edge_spec_path=spec_path,
            liquidation_path=liquidation_path,
            liquidation_quality_report_paths=[quality_path],
            min_sample_count=1,
            min_profitable_windows_ratio=0.0,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "passed"
    assert artifact["source_liquidation_path"] == "liquidation.parquet"
    assert artifact["liquidation_quality_reports"][0]["ok"] is True
    assert artifact["auxiliary_sources"]["liquidation"]["row_count"] == 2
    assert artifact["context_merge"]["required_contexts"] == ["liquidation"]
    assert artifact["feature_columns"] == ["liquidation_sell_notional_1"]
    assert artifact["event_count"] == 1
    assert artifact["best_horizon_by_net_edge"]["net_edge_bps"] > 0.0


def test_edge_discovery_supports_order_book_context_features(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    order_book_path = tmp_path / "order_book.parquet"
    quality_path = tmp_path / "order_book_quality.json"
    spec_path = tmp_path / "edge_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    closes = [100.0, 100.0, 110.0, 111.0, 112.0, 113.0]
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": close - 1.0,
                "high": close * 1.001,
                "low": (close - 1.0) * 0.999,
                "close": close,
                "volume": 1000.0,
            }
            for index, close in enumerate(closes)
        ]
    ).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [
                start + pd.Timedelta(hours=1, minutes=10),
                start + pd.Timedelta(hours=1, minutes=40),
            ],
            "best_bid": [99.9, 99.8],
            "best_ask": [100.1, 100.3],
            "bid_size": [9.0, 8.0],
            "ask_size": [1.0, 2.0],
        }
    ).to_parquet(order_book_path)
    _write_passing_structural_quality_report(quality_path, rows=2)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_edge_discovery_spec",
                "edge_discovery_id": "ED-ORDER-BOOK-UNIT-001",
                "thesis_id": "TH-EDGE-ORDER-BOOK-CONTEXT-001",
                "mechanism_class": "top_of_book_imbalance_probe",
                "hypothesis_scope": "microstructure",
                "market_structure_domains": ["order_book"],
                "conditions": [
                    {
                        "feature": "order_book_depth_imbalance",
                        "operator": ">=",
                        "value": 0.5,
                        "lookback_candles": 1,
                    }
                ],
                "horizons": [1],
                "all_in_cost_bps": 1.0,
            }
        ),
        encoding="utf-8",
    )

    artifact = build_edge_discovery(
        EdgeDiscoveryInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            edge_spec_path=spec_path,
            order_book_path=order_book_path,
            order_book_quality_report_paths=[quality_path],
            min_sample_count=1,
            min_profitable_windows_ratio=0.0,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "passed"
    assert artifact["source_order_book_path"] == "order_book.parquet"
    assert artifact["order_book_quality_reports"][0]["ok"] is True
    assert artifact["auxiliary_sources"]["order_book"]["row_count"] == 2
    assert artifact["context_merge"]["required_contexts"] == ["order_book"]
    assert artifact["feature_columns"] == ["order_book_depth_imbalance_1"]
    assert artifact["event_count"] == 1
    assert artifact["best_horizon_by_net_edge"]["net_edge_bps"] > 0.0
    assert artifact["strategy_codegen_allowed"] is False


def test_edge_discovery_blocks_order_book_context_without_quality_report(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    order_book_path = tmp_path / "order_book.parquet"
    spec_path = tmp_path / "edge_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    closes = [100.0, 100.0, 110.0, 111.0]
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0,
            }
            for index, close in enumerate(closes)
        ]
    ).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [start + pd.Timedelta(hours=1, minutes=10)],
            "best_bid": [99.9],
            "best_ask": [100.1],
            "bid_size": [9.0],
            "ask_size": [1.0],
        }
    ).to_parquet(order_book_path)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_edge_discovery_spec",
                "edge_discovery_id": "ED-ORDER-BOOK-MISSING-QUALITY-001",
                "thesis_id": "TH-EDGE-ORDER-BOOK-MISSING-QUALITY-001",
                "mechanism_class": "top_of_book_imbalance_probe",
                "hypothesis_scope": "microstructure",
                "market_structure_domains": ["order_book"],
                "conditions": [
                    {
                        "feature": "order_book_depth_imbalance",
                        "operator": ">=",
                        "value": 0.5,
                        "lookback_candles": 1,
                    }
                ],
                "horizons": [1],
                "all_in_cost_bps": 1.0,
            }
        ),
        encoding="utf-8",
    )

    artifact = build_edge_discovery(
        EdgeDiscoveryInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            edge_spec_path=spec_path,
            order_book_path=order_book_path,
            min_sample_count=1,
            min_profitable_windows_ratio=0.0,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "blocked"
    assert artifact["event_count"] == 0
    assert artifact["proposal_generation_allowed"] is False
    assert "order_book_quality_report_passed_when_required" in {
        check["name"] for check in artifact["blockers"]
    }


def test_edge_discovery_blocks_positioning_context_without_quality_reports(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    open_interest_path = tmp_path / "open_interest.csv"
    ratio_path = tmp_path / "long_short_ratio.csv"
    spec_path = tmp_path / "edge_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(hours=index),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.0 + index,
                "volume": 1000.0,
            }
            for index in range(6)
        ]
    ).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [start, start + pd.Timedelta(hours=2)],
            "open_interest": [1000.0, 1100.0],
        }
    ).to_csv(open_interest_path, index=False)
    pd.DataFrame(
        {
            "date": [start, start + pd.Timedelta(hours=2)],
            "buyRatio": [0.55, 0.65],
            "sellRatio": [0.45, 0.35],
        }
    ).to_csv(ratio_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_edge_discovery_spec",
                "edge_discovery_id": "ED-POSITIONING-MISSING-QUALITY-001",
                "thesis_id": "TH-EDGE-POSITIONING-MISSING-QUALITY-001",
                "mechanism_class": "positioning_quality_probe",
                "hypothesis_scope": "single_instrument",
                "market_structure_domains": ["open_interest", "long_short_ratio"],
                "conditions": [
                    {
                        "feature": "open_interest",
                        "operator": ">=",
                        "value": 1000.0,
                        "lookback_candles": 1,
                    },
                    {
                        "feature": "long_short_ratio",
                        "operator": ">=",
                        "value": 1.0,
                        "lookback_candles": 1,
                    },
                ],
                "horizons": [1],
                "all_in_cost_bps": 1.0,
            }
        ),
        encoding="utf-8",
    )

    artifact = build_edge_discovery(
        EdgeDiscoveryInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            edge_spec_path=spec_path,
            open_interest_path=open_interest_path,
            long_short_ratio_path=ratio_path,
            min_sample_count=1,
            min_profitable_windows_ratio=0.0,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    blocker_names = {check["name"] for check in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert artifact["event_count"] == 0
    assert artifact["proposal_generation_allowed"] is False
    assert "open_interest_quality_report_passed_when_required" in blocker_names
    assert "long_short_ratio_quality_report_passed_when_required" in blocker_names


def test_edge_discovery_blocks_parameter_search_grid(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    spec_path = tmp_path / "edge_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(minutes=5 * index),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.0 + index,
                "volume": 1000.0,
            }
            for index in range(30)
        ]
    ).to_csv(ohlcv_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_edge_discovery_spec",
                "thesis_id": "TH-EDGE-DISCOVERY-GRID",
                "mechanism_class": "parameter_search_probe",
                "conditions": [
                    {
                        "feature": "return_bps",
                        "operator": ">",
                        "value": 0.0,
                        "lookback_candles": 1,
                    }
                ],
                "horizons": [1, 2],
                "all_in_cost_bps": 1.0,
                "threshold_grid": {"return_bps": [0.0, 5.0, 10.0]},
            }
        ),
        encoding="utf-8",
    )

    artifact = build_edge_discovery(
        EdgeDiscoveryInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            edge_spec_path=spec_path,
            min_sample_count=5,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "blocked"
    assert artifact["proposal_generation_allowed"] is False
    blocker_names = {check["name"] for check in artifact["blockers"]}
    assert "edge_spec_no_parameter_search_grid" in blocker_names
    assert (
        "parameter_only_threshold_loosen_after_failed_edge_discovery"
        in artifact["blocked_next_actions"]
    )


def test_edge_discovery_blocks_cross_asset_scope_without_universe(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    spec_path = tmp_path / "edge_spec.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    pd.DataFrame(
        [
            {
                "date": start + pd.Timedelta(minutes=5 * index),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.0 + index,
                "volume": 1000.0,
            }
            for index in range(30)
        ]
    ).to_csv(ohlcv_path, index=False)
    spec_path.write_text(
        json.dumps(
            {
                "factory": "research_edge_discovery_spec",
                "thesis_id": "TH-EDGE-DISCOVERY-CROSS-ASSET-SCOPE",
                "mechanism_class": "cross_asset_probe_without_universe",
                "hypothesis_scope": "cross_asset",
                "conditions": [
                    {
                        "feature": "return_bps",
                        "operator": ">",
                        "value": 0.0,
                        "lookback_candles": 1,
                    }
                ],
                "horizons": [1, 2],
                "all_in_cost_bps": 1.0,
            }
        ),
        encoding="utf-8",
    )

    artifact = build_edge_discovery(
        EdgeDiscoveryInputs(
            root_dir=tmp_path,
            ohlcv_path=ohlcv_path,
            edge_spec_path=spec_path,
            min_sample_count=5,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    blocker_names = {check["name"] for check in artifact["blockers"]}
    assert artifact["status"] == "blocked"
    assert artifact["hypothesis_scope"] == "cross_asset"
    assert "edge_spec_instrument_universe_sufficient_for_scope" in blocker_names


def test_edge_discovery_cli_maps_context_and_quality_report_paths(tmp_path):
    from scripts.bot_factory_build_edge_discovery import build_inputs_from_args

    args = SimpleNamespace(
        ohlcv_path="btc.csv",
        edge_spec_json="edge_spec.json",
        funding_rate_path="funding.csv",
        mark_price_path=None,
        informative_ohlcv_path="eth.csv",
        open_interest_path="open_interest.csv",
        open_interest_quality_report_json=["open_interest_quality.json"],
        long_short_ratio_path="long_short.csv",
        long_short_ratio_quality_report_json=["long_short_quality.json"],
        liquidation_path="liquidation.csv",
        liquidation_quality_report_json=["liquidation_quality.json"],
        order_book_path="order_book.csv",
        order_book_quality_report_json=["order_book_quality.json"],
        failure_synthesis_json="synthesis.json",
        allow_failed_thesis_or_family=False,
        min_sample_count=30,
        min_profitable_windows_ratio=0.75,
        min_calendar_window_count=2,
        min_profitable_calendar_windows_ratio=0.5,
        min_data_span_days=45.0,
        min_passing_horizon_count=1,
        max_horizon_count=4,
        edge_discovery_id="ED-CLI",
        output_root="registry/strategies/research_decisions",
        reviewer_note=["note"],
        created_at="2026-05-07T00:00:00+00:00",
    )

    inputs = build_inputs_from_args(args, root_dir=tmp_path)

    assert inputs.ohlcv_path == Path("btc.csv")
    assert inputs.edge_spec_path == Path("edge_spec.json")
    assert inputs.funding_rate_path == Path("funding.csv")
    assert inputs.informative_ohlcv_path == Path("eth.csv")
    assert inputs.open_interest_path == Path("open_interest.csv")
    assert inputs.open_interest_quality_report_paths == [Path("open_interest_quality.json")]
    assert inputs.long_short_ratio_path == Path("long_short.csv")
    assert inputs.long_short_ratio_quality_report_paths == [Path("long_short_quality.json")]
    assert inputs.liquidation_path == Path("liquidation.csv")
    assert inputs.liquidation_quality_report_paths == [Path("liquidation_quality.json")]
    assert inputs.order_book_path == Path("order_book.csv")
    assert inputs.order_book_quality_report_paths == [Path("order_book_quality.json")]
    assert inputs.failure_synthesis_path == Path("synthesis.json")
    assert inputs.min_sample_count == 30
    assert inputs.max_horizon_count == 4


def test_local_falsification_builds_cost_edge_artifact(tmp_path):
    ohlcv_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.csv"
    event_path = tmp_path / "registry" / "strategies" / "research_decisions" / "events.csv"
    ohlcv_path.parent.mkdir(parents=True)
    event_path.parent.mkdir(parents=True)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    price = 100.0
    for index in range(90):
        price *= 1.001
        rows.append(
            {
                "date": start + pd.Timedelta(minutes=5 * index),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1000.0,
            }
        )
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    pd.DataFrame({"date": [rows[index]["date"] for index in range(0, 60, 2)]}).to_csv(
        event_path,
        index=False,
    )

    artifact = build_local_falsification(
        LocalFalsificationInputs(
            root_dir=tmp_path,
            thesis_id="TH-COST-FLOOR-001",
            mechanism_class="closed_candle_event_study",
            ohlcv_path=ohlcv_path,
            event_path=event_path,
            hold_candles=1,
            all_in_cost_bps=5.0,
            min_sample_count=20,
            min_profitable_windows_ratio=0.75,
            falsification_id="local-falsification-test",
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "passed"
    assert artifact["sample_count"] == 30
    assert artifact["expected_edge_bps"] > artifact["all_in_cost_bps"]
    assert artifact["net_edge_bps"] > 0.0
    assert artifact["profitable_windows_ratio"] == 1.0
    assert artifact["calendar_window_frequency"] == "quarter"
    assert artifact["calendar_window_count"] == 1
    assert artifact["profitable_calendar_windows_ratio"] == 1.0
    assert artifact["calendar_window_summaries"][0]["calendar_window"] == "2026Q1"
    assert artifact["safety_scope"]["strategy_code_generated"] is False
    assert artifact["safety_scope"]["paper_trading_started"] is False

    json_path, report_path = write_local_falsification_artifacts(
        artifact,
        root_dir=tmp_path,
        output_root=Path("registry/strategies/research_decisions"),
    )

    assert json_path.is_file()
    assert report_path.is_file()
    assert json.loads(json_path.read_text(encoding="utf-8"))["factory"] == (
        "research_local_falsification"
    )
    report_text = report_path.read_text(encoding="utf-8")
    assert "expected_edge_bps" in report_text
    assert "profitable_calendar_windows_ratio" in report_text


def test_local_falsification_labeled_events_fall_back_to_single_price_series(
    tmp_path,
):
    ohlcv_path = tmp_path / "ohlcv.csv"
    event_path = tmp_path / "events.csv"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = [
        {
            "date": start + pd.Timedelta(hours=index),
            "open": 100.0 + index,
            "high": 101.0 + index,
            "low": 99.0 + index,
            "close": 100.0 + index,
            "volume": 1000.0,
        }
        for index in range(6)
    ]
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [rows[index]["date"] for index in range(3)],
            "pair": ["BTC/USDT:USDT"] * 3,
        }
    ).to_csv(event_path, index=False)

    artifact = build_local_falsification(
        LocalFalsificationInputs(
            root_dir=tmp_path,
            thesis_id="TH-LABELED-SINGLE-SERIES-001",
            mechanism_class="single_series_labeled_event_study",
            ohlcv_path=ohlcv_path,
            event_path=event_path,
            hold_candles=1,
            all_in_cost_bps=0.0,
            min_sample_count=3,
            min_profitable_windows_ratio=1.0,
            created_at="2026-05-09T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "passed"
    assert artifact["sample_count"] == 3
    sample = artifact["sample_preview"][0]
    assert sample["pair"] == "BTC/USDT:USDT"
    assert sample["price_series_instrument_unverified"] is True
    assert "price_series_instrument_column" not in sample


def test_local_falsification_filters_labeled_events_to_matching_price_series(
    tmp_path,
):
    ohlcv_path = tmp_path / "ohlcv.csv"
    event_path = tmp_path / "events.csv"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    btc_closes = [500.0, 1000.0, 1.0, 1.0]
    eth_closes = [100.0, 100.0, 105.0, 105.0]
    for index in range(4):
        for pair, closes in (
            ("BTC/USDT:USDT", btc_closes),
            ("ETH/USDT:USDT", eth_closes),
        ):
            close = closes[index]
            rows.append(
                {
                    "date": start + pd.Timedelta(hours=index),
                    "pair": pair,
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0,
                }
            )
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [start + pd.Timedelta(hours=1)],
            "pair": ["ETH/USDT:USDT"],
        }
    ).to_csv(event_path, index=False)

    artifact = build_local_falsification(
        LocalFalsificationInputs(
            root_dir=tmp_path,
            thesis_id="TH-LOCAL-PAIR-PRICE-MATCH-001",
            mechanism_class="local_pair_price_alignment_probe",
            ohlcv_path=ohlcv_path,
            event_path=event_path,
            hold_candles=1,
            all_in_cost_bps=0.0,
            min_sample_count=1,
            min_profitable_windows_ratio=1.0,
            created_at="2026-05-09T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "passed"
    assert artifact["sample_count"] == 1
    sample = artifact["sample_preview"][0]
    assert sample["pair"] == "ETH/USDT:USDT"
    assert sample["entry_price"] == 100.0
    assert sample["exit_price"] == 105.0
    assert sample["price_series_instrument"] == "ETH/USDT:USDT"
    assert sample["price_series_instrument_column"] == "pair"


def test_local_falsification_prefers_populated_symbol_column_for_labeled_events(
    tmp_path,
):
    ohlcv_path = tmp_path / "ohlcv.csv"
    event_path = tmp_path / "events.csv"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    btc_closes = [500.0, 1000.0, 1.0]
    eth_closes = [100.0, 100.0, 105.0]
    for index in range(3):
        for symbol, closes in (
            ("BTC/USDT:USDT", btc_closes),
            ("ETH/USDT:USDT", eth_closes),
        ):
            close = closes[index]
            rows.append(
                {
                    "date": start + pd.Timedelta(hours=index),
                    "pair": "unknown",
                    "symbol": symbol,
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0,
                }
            )
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [start + pd.Timedelta(hours=1)],
            "symbol": ["ETH/USDT:USDT"],
        }
    ).to_csv(event_path, index=False)

    artifact = build_local_falsification(
        LocalFalsificationInputs(
            root_dir=tmp_path,
            thesis_id="TH-LOCAL-SYMBOL-PRICE-MATCH-001",
            mechanism_class="local_symbol_price_alignment_probe",
            ohlcv_path=ohlcv_path,
            event_path=event_path,
            hold_candles=1,
            all_in_cost_bps=0.0,
            min_sample_count=1,
            min_profitable_windows_ratio=1.0,
            created_at="2026-05-09T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "passed"
    assert artifact["sample_count"] == 1
    sample = artifact["sample_preview"][0]
    assert sample["symbol"] == "ETH/USDT:USDT"
    assert sample["entry_price"] == 100.0
    assert sample["exit_price"] == 105.0
    assert sample["price_series_instrument"] == "ETH/USDT:USDT"
    assert sample["price_series_instrument_column"] == "symbol"


def test_local_falsification_prefers_event_label_matching_price_column(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    event_path = tmp_path / "events.csv"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    btc_closes = [500.0, 1000.0, 1.0]
    eth_closes = [100.0, 100.0, 105.0]
    for index in range(3):
        for symbol, closes in (
            ("BTC/USDT:USDT", btc_closes),
            ("ETH/USDT:USDT", eth_closes),
        ):
            close = closes[index]
            rows.append(
                {
                    "date": start + pd.Timedelta(hours=index),
                    "pair": "perpetual_futures",
                    "symbol": symbol,
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0,
                }
            )
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    pd.DataFrame(
        {
            "date": [start + pd.Timedelta(hours=1)],
            "pair": ["perpetual_futures"],
            "symbol": ["ETH/USDT:USDT"],
        }
    ).to_csv(event_path, index=False)

    artifact = build_local_falsification(
        LocalFalsificationInputs(
            root_dir=tmp_path,
            thesis_id="TH-LOCAL-MATCHING-EVENT-LABEL-001",
            mechanism_class="local_matching_event_label_probe",
            ohlcv_path=ohlcv_path,
            event_path=event_path,
            hold_candles=1,
            all_in_cost_bps=0.0,
            min_sample_count=1,
            min_profitable_windows_ratio=1.0,
            created_at="2026-05-09T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "passed"
    assert artifact["sample_count"] == 1
    sample = artifact["sample_preview"][0]
    assert sample["symbol"] == "ETH/USDT:USDT"
    assert sample["entry_price"] == 100.0
    assert sample["exit_price"] == 105.0
    assert sample["price_series_instrument"] == "ETH/USDT:USDT"
    assert sample["price_series_instrument_column"] == "symbol"


def test_local_falsification_can_include_realized_long_funding_adjustment(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    event_path = tmp_path / "events.csv"
    funding_path = tmp_path / "funding.csv"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for index in range(36):
        rows.append(
            {
                "date": start + pd.Timedelta(hours=index),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000.0,
            }
        )
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    pd.DataFrame({"date": [rows[index]["date"] for index in range(0, 6)]}).to_csv(
        event_path,
        index=False,
    )
    pd.DataFrame(
        {
            "date": [
                start + pd.Timedelta(hours=8),
                start + pd.Timedelta(hours=16),
                start + pd.Timedelta(hours=24),
            ],
            "open": [-0.002, -0.002, -0.002],
        }
    ).to_csv(funding_path, index=False)

    artifact = build_local_falsification(
        LocalFalsificationInputs(
            root_dir=tmp_path,
            thesis_id="TH-FUNDING-ADJUSTED-001",
            mechanism_class="funding_adjusted_event_study",
            ohlcv_path=ohlcv_path,
            event_path=event_path,
            funding_rate_path=funding_path,
            hold_candles=10,
            all_in_cost_bps=5.0,
            min_sample_count=4,
            min_profitable_windows_ratio=0.75,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "passed"
    assert artifact["expected_price_edge_bps"] == 0.0
    assert artifact["expected_funding_adjustment_bps"] == 20.0
    assert artifact["expected_edge_bps"] == 20.0
    assert artifact["net_edge_bps"] == 15.0
    assert artifact["funding_rate_adjustment"]["used"] is True
    assert artifact["safety_scope"]["closed_candle_ohlcv_only"] is False
    assert artifact["safety_scope"]["closed_candle_local_market_data_only"] is True
    assert "funding_rate_parseable" in {
        check["name"] for check in artifact["checks"] if check["status"] == "pass"
    }


def test_local_falsification_requires_funding_path_for_funding_event_source(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    event_path = tmp_path / "events.csv"
    event_source_path = tmp_path / "local_events.json"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for index in range(12):
        rows.append(
            {
                "date": start + pd.Timedelta(hours=index),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000.0,
            }
        )
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    pd.DataFrame({"date": [rows[0]["date"], rows[1]["date"]]}).to_csv(
        event_path,
        index=False,
    )
    event_source_path.write_text(
        json.dumps(
            {
                "factory": "research_local_event_builder",
                "status": "completed",
                "thesis_id": "TH-FUNDING-CONTEXT-001",
                "events_csv_path": str(event_path.relative_to(tmp_path)),
                "source_ohlcv_path": str(ohlcv_path.relative_to(tmp_path)),
                "event_count": 2,
                "context_merge": {
                    "semantics": "closed_context_candle_availability_v1",
                    "required_contexts": ["funding_rate"],
                    "closed_context_candle_alignment": True,
                },
                "safety_scope": {
                    "historical_only": True,
                    "closed_candle_local_market_data_only": True,
                    "leverage": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    artifact = build_local_falsification(
        LocalFalsificationInputs(
            root_dir=tmp_path,
            thesis_id="TH-FUNDING-CONTEXT-001",
            mechanism_class="funding_context_event_study",
            ohlcv_path=ohlcv_path,
            event_path=event_path,
            event_source_path=event_source_path,
            hold_candles=1,
            all_in_cost_bps=0.0,
            min_sample_count=1,
            min_profitable_windows_ratio=0.0,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "failed"
    assert artifact["funding_rate_adjustment"]["required_by_event_source"] is True
    assert "funding_rate_path_present_for_funding_event_source" in {
        check["name"] for check in artifact["blockers"]
    }


def test_local_falsification_blocks_cost_edge_below_cost(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    event_path = tmp_path / "events.csv"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    price = 100.0
    for index in range(40):
        price *= 0.999
        rows.append(
            {
                "date": start + pd.Timedelta(minutes=5 * index),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1000.0,
            }
        )
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    pd.DataFrame({"date": [rows[index]["date"] for index in range(0, 25)]}).to_csv(
        event_path,
        index=False,
    )

    artifact = build_local_falsification(
        LocalFalsificationInputs(
            root_dir=tmp_path,
            thesis_id="TH-COST-FLOOR-NEGATIVE-001",
            mechanism_class="closed_candle_event_study",
            ohlcv_path=ohlcv_path,
            event_path=event_path,
            hold_candles=1,
            all_in_cost_bps=1.0,
            min_sample_count=20,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "failed"
    assert artifact["sample_count"] >= 20
    assert artifact["net_edge_bps"] < 0.0
    assert "expected_edge_exceeds_all_in_cost" in {
        check["name"] for check in artifact["blockers"]
    }


def test_local_falsification_blocks_too_short_data_span(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    event_path = tmp_path / "events.csv"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    price = 100.0
    for index in range(48):
        price *= 1.001
        rows.append(
            {
                "date": start + pd.Timedelta(hours=index),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1000.0,
            }
        )
    pd.DataFrame(rows).to_csv(ohlcv_path, index=False)
    pd.DataFrame({"date": [rows[index]["date"] for index in range(0, 20, 2)]}).to_csv(
        event_path,
        index=False,
    )

    artifact = build_local_falsification(
        LocalFalsificationInputs(
            root_dir=tmp_path,
            thesis_id="TH-SHORT-SPAN-001",
            mechanism_class="short_span_positive_event_study",
            ohlcv_path=ohlcv_path,
            event_path=event_path,
            hold_candles=1,
            all_in_cost_bps=0.0,
            min_sample_count=5,
            min_profitable_windows_ratio=0.5,
            min_data_span_days=10.0,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "failed"
    assert artifact["data_span_days"] < 10.0
    assert artifact["sample_count"] >= 5
    assert artifact["net_edge_bps"] > 0.0
    assert "ohlcv_data_span_sufficient" in {check["name"] for check in artifact["blockers"]}


def test_local_falsification_blocks_calendar_unstable_positive_edge(tmp_path):
    ohlcv_path = tmp_path / "ohlcv.csv"
    event_path = tmp_path / "events.csv"
    rows = []
    event_times = []

    for quarter_start, event_count, exit_multiplier in [
        ("2026-01-01T00:00:00Z", 20, 1.01),
        ("2026-04-01T00:00:00Z", 20, 0.998),
    ]:
        start = pd.Timestamp(quarter_start)
        for index in range(event_count):
            event_time = start + pd.Timedelta(days=2 * index)
            event_times.append(event_time)
            rows.extend(
                [
                    {
                        "date": event_time,
                        "open": 100.0,
                        "high": 100.0,
                        "low": 100.0,
                        "close": 100.0,
                        "volume": 1000.0,
                    },
                    {
                        "date": event_time + pd.Timedelta(days=1),
                        "open": 100.0 * exit_multiplier,
                        "high": 100.0 * exit_multiplier,
                        "low": 100.0 * exit_multiplier,
                        "close": 100.0 * exit_multiplier,
                        "volume": 1000.0,
                    },
                ]
            )

    pd.DataFrame(rows).sort_values("date").to_csv(ohlcv_path, index=False)
    pd.DataFrame({"date": event_times}).to_csv(event_path, index=False)

    artifact = build_local_falsification(
        LocalFalsificationInputs(
            root_dir=tmp_path,
            thesis_id="TH-CALENDAR-UNSTABLE-POSITIVE-EDGE-001",
            mechanism_class="calendar_concentrated_event_study",
            ohlcv_path=ohlcv_path,
            event_path=event_path,
            hold_candles=1,
            all_in_cost_bps=5.0,
            min_sample_count=20,
            min_profitable_windows_ratio=0.0,
            min_calendar_window_count=3,
            min_profitable_calendar_windows_ratio=0.75,
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "failed"
    assert artifact["sample_count"] == 40
    assert artifact["net_edge_bps"] > 0.0
    assert artifact["calendar_window_count"] == 2
    assert artifact["min_calendar_window_count"] == 3
    assert artifact["profitable_calendar_windows_ratio"] == 0.5
    assert artifact["min_profitable_calendar_windows_ratio"] == 0.75
    blocker_names = {check["name"] for check in artifact["blockers"]}
    assert "calendar_window_count_sufficient" in blocker_names
    assert "profitable_calendar_windows_ratio_sufficient" in blocker_names


def test_research_selection_template_exports_causal_map_questions(tmp_path):
    map_path = tmp_path / "causal_failure_map.json"
    map_path.write_text(
        json.dumps(
            {
                "factory": "causal_failure_map",
                "status": "completed",
                "map_id": "map-template-test",
                "source_synthesis_id": "synth-template-test",
                "source_synthesis_path": (
                    "registry/strategies/synthesis/synth/candidate_failure_synthesis.json"
                ),
                "research_selection_guidance": {
                    "requires_research_decision_before_proposal": True,
                    "requires_research_question_responses": True,
                    "minimum_research_selection_score": 80,
                    "dominant_failure_categories": [
                        {"category": "regime_fragile_mechanism", "candidate_count": 32},
                        {"category": "cost_sensitive_mechanism", "candidate_count": 31},
                    ],
                    "required_research_questions": [
                        "What mechanism survives after failed families are excluded?",
                        "Why should expected edge exceed fee and turnover costs?",
                    ],
                    "validated_local_falsification_rejections": [
                        {
                            "path": "registry/strategies/research_decisions/reject/local_falsification.json",
                            "thesis_id": "TH-REJECTED",
                            "mechanism_class": "rejected_mechanism",
                            "net_edge_bps": -5.0,
                            "profitable_windows_ratio": 0.0,
                            "profitable_calendar_windows_ratio": 0.25,
                        }
                    ],
                    "blocked_next_actions": [
                        "parameter_only_threshold_loosen",
                        "proposal_generation_without_approved_research_decision",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    artifact = build_research_selection_template(
        ResearchSelectionTemplateInputs(
            root_dir=tmp_path,
            causal_failure_map_path=map_path,
            template_id="template-test",
            created_at="2026-05-07T00:00:00+00:00",
        )
    )

    assert artifact["status"] == "completed"
    assert artifact["required_causal_failure_response_count"] == 2
    assert artifact["required_research_question_response_count"] == 2
    assert artifact["validated_local_falsification_rejection_count"] == 1
    assert artifact["required_causal_failure_responses"][0]["cli_argument"] == (
        '--causal-failure-response "regime_fragile_mechanism=<substantive response>"'
    )
    assert artifact["required_research_question_responses"][1]["cli_argument"] == (
        '--research-question-response "2=<substantive response>"'
    )
    input_template = artifact["research_selection_input_template"]
    assert input_template["failure_synthesis_json"] == (
        "registry/strategies/synthesis/synth/candidate_failure_synthesis.json"
    )
    assert input_template["causal_failure_map_json"] == "causal_failure_map.json"
    assert input_template["causal_failure_responses"] == {
        "regime_fragile_mechanism": "",
        "cost_sensitive_mechanism": "",
    }
    assert input_template["research_question_responses"] == {"1": "", "2": ""}
    assert (
        "--research-selection-input-json <filled-research-selection-input.json>"
        in artifact["select_research_thesis_input_json_command_template"]
    )
    command_template = artifact["select_research_thesis_command_template"]
    assert r"scripts\bot_factory_select_research_thesis.py" in command_template
    assert (
        "--failure-synthesis-json "
        "registry/strategies/synthesis/synth/candidate_failure_synthesis.json"
    ) in command_template
    assert "--causal-failure-map-json causal_failure_map.json" in command_template
    assert (
        '--causal-failure-response "regime_fragile_mechanism=<substantive response>"'
    ) in command_template
    assert '--research-question-response "2=<substantive response>"' in command_template
    assert artifact["safety_scope"]["strategy_code_generated"] is False
    assert artifact["safety_scope"]["paper_trading_started"] is False

    json_path, report_path = write_research_selection_template_artifacts(
        artifact,
        root_dir=tmp_path,
        output_root=Path("registry/strategies/research_decisions"),
    )

    assert json_path.is_file()
    assert report_path.is_file()
    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["factory"] == "research_selection_response_template"
    report_text = report_path.read_text(encoding="utf-8")
    assert "regime_fragile_mechanism" in report_text
    assert "Why should expected edge exceed fee" in report_text
    assert "Select Research Thesis Input JSON Command Template" in report_text
    assert "Select Research Thesis Command Template" in report_text
    assert "Research Selection Input JSON Template" in report_text


def test_research_selection_cli_loads_filled_input_json(tmp_path):
    from scripts.bot_factory_select_research_thesis import build_inputs_from_args

    input_path = tmp_path / "filled_research_selection_input.json"
    input_path.write_text(
        json.dumps(
            {
                "research_selection_input_template": {
                    "failure_synthesis_json": (
                        "registry/strategies/synthesis/synth/"
                        "candidate_failure_synthesis.json"
                    ),
                    "causal_failure_map_json": (
                        "registry/strategies/failure_maps/map/causal_failure_map.json"
                    ),
                    "thesis_id": "TH-CLOSED-CANDLE-RESILIENCE-JSON",
                    "thesis_family": "closed_candle_liquidity_resilience",
                    "mechanism_class": "closed_candle_resilience_reclaim",
                    "thesis_statement": (
                        "Closed-candle BTC futures resilience after local stress "
                        "can be tested before proposal generation."
                    ),
                    "mechanism_summary": (
                        "Use local historical OHLCV, cost evidence, and "
                        "walk-forward rejection rules to test the mechanism."
                    ),
                    "novelty_rationale": (
                        "This JSON-filled thesis is outside failed families and "
                        "does not retune thresholds from rejected mechanisms."
                    ),
                    "required_data": ["BTC futures 5m closed-candle OHLCV"],
                    "local_data_paths": [
                        "user_data/data/bybit/futures/BTC_USDT_USDT-5m-futures.parquet"
                    ],
                    "edge_rationale": (
                        "Expected edge must beat fee, spread, slippage, and "
                        "turnover costs in local historical evidence."
                    ),
                    "transaction_cost_exposure": (
                        "Assume at least 12 bps all-in fee, spread, slippage, "
                        "and turnover drag."
                    ),
                    "falsification_plan": (
                        "Reject before proposal if local closed-candle evidence "
                        "fails post-cost edge or walk-forward stability."
                    ),
                    "stop_conditions": [
                        "Reject if post-cost edge or walk-forward evidence fails."
                    ],
                    "research_references": [
                        {
                            "reference_id": "paper:resilience-json",
                            "title": "Research reference for resilience",
                            "source": "Local bibliography",
                            "published_at": "2024",
                            "relevance": (
                                "Supports a falsifiable market resilience "
                                "mechanism using local historical evidence."
                            ),
                            "motivated_thesis_ids": [
                                "TH-CLOSED-CANDLE-RESILIENCE-JSON"
                            ],
                        }
                    ],
                    "causal_failure_responses": {
                        "regime_fragile_mechanism": (
                            "Segment regimes with local historical evidence and "
                            "reject when the mechanism fails across states."
                        )
                    },
                    "research_question_responses": {
                        "1": (
                            "Require local falsification and reject before "
                            "proposal when evidence is absent."
                        )
                    },
                    "decision_id": "json-input-decision",
                    "created_at": "2026-05-07T12:00:00+00:00",
                    "reviewer_notes": ["Loaded from filled JSON input."],
                }
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        research_selection_input_json=str(input_path),
        failure_synthesis_json=None,
        thesis_id=None,
        thesis_family=None,
        mechanism_class=None,
        thesis_statement=None,
        mechanism_summary=None,
        novelty_rationale=None,
        required_data=[],
        local_data_path=[],
        local_data_quality_report_json=[],
        structural_data_capability_report_json=[],
        local_falsification_json=[],
        prior_local_falsification_json=[],
        causal_failure_map_json=None,
        causal_failure_response=[],
        research_question_response=[],
        edge_rationale=None,
        transaction_cost_exposure=None,
        falsification_plan=None,
        stop_condition=[],
        research_reference=[],
        decision_id=None,
        output_root=None,
        reviewer_note=[],
        created_at=None,
    )

    inputs = build_inputs_from_args(args, root_dir=tmp_path)

    assert inputs.failure_synthesis_path == Path(
        "registry/strategies/synthesis/synth/candidate_failure_synthesis.json"
    )
    assert inputs.causal_failure_map_path == Path(
        "registry/strategies/failure_maps/map/causal_failure_map.json"
    )
    assert inputs.thesis_id == "TH-CLOSED-CANDLE-RESILIENCE-JSON"
    assert inputs.required_data == ["BTC futures 5m closed-candle OHLCV"]
    assert inputs.local_data_paths == [
        Path("user_data/data/bybit/futures/BTC_USDT_USDT-5m-futures.parquet")
    ]
    assert inputs.causal_failure_responses == [
        (
            "regime_fragile_mechanism=Segment regimes with local historical "
            "evidence and reject when the mechanism fails across states."
        )
    ]
    assert inputs.research_question_responses == [
        (
            "1=Require local falsification and reject before proposal when "
            "evidence is absent."
        )
    ]
    assert inputs.research_references[0].reference_id == "paper:resilience-json"
    assert inputs.decision_id == "json-input-decision"


def test_research_selection_gate_approves_distinct_local_falsifiable_thesis(tmp_path):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
        )
    )

    assert decision["status"] == "approved_for_proposal_generation"
    assert decision["proposal_generation_allowed"] is True
    assert decision["code_generation_allowed"] is False
    assert decision["safety_scope"]["historical_only"] is True
    assert decision["safety_scope"]["paper_trading_started"] is False
    assert decision["novelty_assessment"]["repeated_failed_family_matches"] == []
    assert decision["research_references"][0]["motivated_thesis_ids"] == [
        "TH-ORDERBOOK-RESILIENCE-001"
    ]

    json_path, report_path = write_research_selection_artifacts(
        decision,
        root_dir=tmp_path,
        output_root=Path("research_decisions"),
    )

    assert json_path.is_file()
    assert report_path.is_file()
    assert "code_generation_allowed: False" in report_path.read_text(encoding="utf-8")


def test_research_selection_gate_accepts_causal_failure_map_responses(tmp_path):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    map_path = _write_research_selection_causal_failure_map(tmp_path)

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                (
                    "regime_fragile_mechanism=Use closed-candle regime segmentation "
                    "and reject the thesis when segments do not hold."
                ),
                (
                    "walk_forward_fragility=Predefine which historical "
                    "walk-forward windows should pass and reject absent support."
                ),
                (
                    "cost_sensitive_mechanism=Reject frequent entries when "
                    "fee and turnover drag dominate expected edge."
                ),
            ],
        )
    )

    assert decision["status"] == "approved_for_proposal_generation"
    assert decision["causal_failure_map"]["map_id"] == "map-test"
    assert decision["causal_failure_map"]["required_categories_to_address"] == [
        "regime_fragile_mechanism",
        "walk_forward_fragility",
        "cost_sensitive_mechanism",
    ]
    assert decision["causal_failure_map"]["missing_response_categories"] == []
    assert decision["causal_failure_map"]["minimum_research_selection_score"] == 80.0
    assert decision["research_selection_score"]["score"] == 100.0
    assert decision["research_selection_score"]["passes_minimum"] is True
    assert decision["research_selection_score"]["failed_components"] == []
    assert "causal_failure_responses_cover_required_categories" in {
        check["name"] for check in decision["checks"] if check["status"] == "pass"
    }


def test_research_selection_gate_blocks_missing_causal_failure_responses(tmp_path):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    map_path = _write_research_selection_causal_failure_map(tmp_path)

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
        )
    )

    assert decision["status"] == "blocked"
    assert decision["causal_failure_map"]["missing_response_categories"] == [
        "regime_fragile_mechanism",
        "walk_forward_fragility",
        "cost_sensitive_mechanism",
    ]
    assert "causal_failure_responses_cover_required_categories" in {
        check["name"] for check in decision["blockers"]
    }


def test_research_selection_gate_blocks_missing_required_research_question_responses(
    tmp_path,
):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    map_path = _write_research_selection_causal_failure_map(tmp_path)

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                (
                    "regime_fragile_mechanism=Use closed-candle regime segmentation "
                    "and reject the thesis when segments do not hold."
                ),
                (
                    "walk_forward_fragility=Predefine which historical "
                    "walk-forward windows should pass and reject absent support."
                ),
                (
                    "cost_sensitive_mechanism=Reject frequent entries when "
                    "fee and turnover drag dominate expected edge."
                ),
            ],
            research_question_responses=[],
        )
    )

    assert decision["status"] == "blocked"
    assert decision["causal_failure_map"][
        "requires_research_question_responses"
    ] is True
    assert decision["causal_failure_map"][
        "missing_research_question_response_indexes"
    ] == [1, 2, 3]
    assert "research_question_responses_cover_required_questions" in {
        check["name"] for check in decision["blockers"]
    }


def test_research_selection_gate_requires_local_rejection_question_response(
    tmp_path,
):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    local_rejection_question = (
        "What materially different market mechanism avoids the validated local "
        "falsification rejection for TH-LOCAL-REJECTED / "
        "mark_fair_value_momentum_lag while preserving closed-candle evidence "
        "and positive post-cost edge?"
    )
    map_path = _write_research_selection_causal_failure_map(
        tmp_path,
        required_research_questions=[
            "What mechanism survives after failed families are excluded?",
            "Why should expected edge exceed fee and turnover costs?",
            "Which walk-forward regimes should pass or fail?",
            local_rejection_question,
        ],
        validated_local_falsification_rejections=[
            {
                "path": "registry/strategies/research_decisions/local_rejection.json",
                "thesis_id": "TH-LOCAL-REJECTED",
                "mechanism_class": "mark_fair_value_momentum_lag",
                "net_edge_bps": -4.2,
                "profitable_windows_ratio": 0.0,
            }
        ],
    )
    causal_responses = [
        (
            "regime_fragile_mechanism=Use closed-candle regime segmentation "
            "and reject the thesis when segments do not hold."
        ),
        (
            "walk_forward_fragility=Predefine which historical walk-forward "
            "windows should pass and reject absent support."
        ),
        (
            "cost_sensitive_mechanism=Reject frequent entries when fee and "
            "turnover drag dominate expected edge."
        ),
    ]

    missing_local_answer = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=causal_responses,
        )
    )

    assert missing_local_answer["status"] == "blocked"
    assert missing_local_answer["causal_failure_map"][
        "validated_local_falsification_rejections"
    ][0]["thesis_id"] == "TH-LOCAL-REJECTED"
    assert local_rejection_question in missing_local_answer["causal_failure_map"][
        "required_research_questions"
    ]
    assert missing_local_answer["causal_failure_map"][
        "missing_research_question_response_indexes"
    ] == [4]
    assert "research_question_responses_cover_required_questions" in {
        check["name"] for check in missing_local_answer["blockers"]
    }

    json_path, report_path = write_research_selection_artifacts(
        missing_local_answer,
        root_dir=tmp_path,
        output_root=Path("research_decisions"),
    )
    assert json_path.is_file()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Validated Local Falsification Rejections" in report_text
    assert "TH-LOCAL-REJECTED / mark_fair_value_momentum_lag" in report_text

    answered = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=causal_responses,
            research_question_responses=[
                (
                    "1=Use a closed-candle liquidity resilience mechanism that "
                    "is outside the failed families and reject it on local evidence."
                ),
                (
                    "2=Require expected edge to exceed fee, spread, slippage, "
                    "and turnover costs before proposal generation."
                ),
                (
                    "3=Define walk-forward regimes upfront and reject when "
                    "profitable windows are absent or unstable."
                ),
                (
                    "4=Avoid the rejected mark fair-value momentum lag by using "
                    "a materially different closed-candle liquidity recovery "
                    "mechanism with positive post-cost edge evidence."
                ),
            ],
        )
    )

    assert answered["status"] == "approved_for_proposal_generation"
    assert answered["causal_failure_map"][
        "missing_research_question_response_indexes"
    ] == []
    assert answered["causal_failure_map"][
        "weak_research_question_response_indexes"
    ] == []


def test_research_selection_gate_requires_calendar_window_rejection_response(
    tmp_path,
):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    calendar_question = (
        "What materially different market mechanism avoids the validated local "
        "falsification rejection for TH-CALENDAR-REJECTED / "
        "calendar_fragile_reversion while preserving closed-candle evidence, "
        "positive post-cost edge and quarterly calendar-window stability "
        "(profitable_calendar_windows_ratio=0.0)?"
    )
    map_path = _write_research_selection_causal_failure_map(
        tmp_path,
        required_research_questions=[
            "What mechanism survives after failed families are excluded?",
            "Why should expected edge exceed fee and turnover costs?",
            "Which walk-forward regimes should pass or fail?",
            calendar_question,
        ],
        validated_local_falsification_rejections=[
            {
                "path": "registry/strategies/research_decisions/calendar_rejection.json",
                "thesis_id": "TH-CALENDAR-REJECTED",
                "mechanism_class": "calendar_fragile_reversion",
                "net_edge_bps": -9.0,
                "profitable_windows_ratio": 0.0,
                "profitable_calendar_windows_ratio": 0.0,
            }
        ],
    )
    causal_responses = [
        (
            "regime_fragile_mechanism=Use closed-candle regime segmentation "
            "and reject the thesis when segments do not hold."
        ),
        (
            "walk_forward_fragility=Predefine historical walk-forward windows "
            "that should pass and reject absent support."
        ),
        (
            "cost_sensitive_mechanism=Reject frequent entries when fee and "
            "turnover drag dominate expected edge."
        ),
    ]
    base_question_responses = [
        (
            "1=Use a closed-candle liquidity mechanism outside failed families "
            "and reject it with local evidence before proposal generation."
        ),
        (
            "2=Require expected edge to exceed fee, spread, slippage, and "
            "turnover costs before proposal generation."
        ),
        (
            "3=Define walk-forward regimes upfront and reject when profitable "
            "windows are absent or unstable."
        ),
    ]

    missing_calendar_evidence = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=causal_responses,
            research_question_responses=[
                *base_question_responses,
                (
                    "4=Use a materially different closed-candle liquidity "
                    "recovery mechanism with positive post-cost edge evidence."
                ),
            ],
        )
    )

    assert missing_calendar_evidence["status"] == "blocked"
    assert missing_calendar_evidence["causal_failure_map"][
        "weak_research_question_response_indexes"
    ] == [4]
    question_quality = missing_calendar_evidence["causal_failure_map"][
        "research_question_response_quality_by_index"
    ]["4"]
    assert question_quality["missing_requirement_groups"] == [
        "calendar_window_evidence"
    ]
    assert "research_question_responses_are_substantive" in {
        check["name"] for check in missing_calendar_evidence["blockers"]
    }

    answered = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=causal_responses,
            research_question_responses=[
                *base_question_responses,
                (
                    "4=Use a materially different closed-candle mechanism and "
                    "reject it unless quarterly calendar-window evidence shows "
                    "positive post-cost edge across calendar regimes."
                ),
            ],
        )
    )

    assert answered["status"] == "approved_for_proposal_generation"
    assert answered["causal_failure_map"][
        "weak_research_question_response_indexes"
    ] == []


def test_research_selection_gate_requires_material_causal_failure_categories_beyond_top_three(
    tmp_path,
):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    map_path = _write_research_selection_causal_failure_map(
        tmp_path,
        candidate_count=24,
        dominant_categories=[
            {"category": "regime_fragile_mechanism", "candidate_count": 24},
            {"category": "walk_forward_fragility", "candidate_count": 24},
            {"category": "cost_sensitive_mechanism", "candidate_count": 23},
            {"category": "no_profitable_walk_forward_windows", "candidate_count": 18},
            {"category": "entry_exists_negative_edge", "candidate_count": 17},
            {"category": "overfit_or_window_dependency", "candidate_count": 14},
        ],
    )
    base_responses = [
        (
            "regime_fragile_mechanism=Use closed-candle regime segmentation "
            "and reject the thesis when segments do not hold."
        ),
        (
            "walk_forward_fragility=Predefine which historical "
            "walk-forward windows should pass and reject absent support."
        ),
        (
            "cost_sensitive_mechanism=Reject frequent entries when "
            "fee and turnover drag dominate expected edge."
        ),
    ]

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=base_responses,
        )
    )

    assert decision["status"] == "blocked"
    assert decision["causal_failure_map"]["material_category_min_share"] == 0.70
    assert decision["causal_failure_map"]["required_categories_to_address"] == [
        "regime_fragile_mechanism",
        "walk_forward_fragility",
        "cost_sensitive_mechanism",
        "no_profitable_walk_forward_windows",
        "entry_exists_negative_edge",
    ]
    assert decision["causal_failure_map"]["missing_response_categories"] == [
        "no_profitable_walk_forward_windows",
        "entry_exists_negative_edge",
    ]

    approved = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                *base_responses,
                (
                    "no_profitable_walk_forward_windows=Reject unless predefined "
                    "walk-forward windows show profitable returns and negative "
                    "expectancy is absent across historical splits."
                ),
                (
                    "entry_exists_negative_edge=Reject the thesis when entry signals "
                    "keep negative expectancy or loss after fees in historical evidence."
                ),
            ],
        )
    )

    assert approved["status"] == "approved_for_proposal_generation"
    assert approved["causal_failure_map"]["missing_response_categories"] == []


def test_research_selection_score_weights_unanswered_causal_risk(tmp_path):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    map_path = _write_research_selection_causal_failure_map(
        tmp_path,
        candidate_count=10,
        dominant_categories=[
            {"category": "walk_forward_fragility", "candidate_count": 10},
            {"category": "cost_sensitive_mechanism", "candidate_count": 6},
            {"category": "entry_exists_negative_edge", "candidate_count": 5},
        ],
        causal_risk_weights=[
            {
                "category": "walk_forward_fragility",
                "risk_score": 100.0,
                "required_for_next_research": True,
            },
            {
                "category": "cost_sensitive_mechanism",
                "risk_score": 20.0,
                "required_for_next_research": True,
            },
            {
                "category": "entry_exists_negative_edge",
                "risk_score": 20.0,
                "required_for_next_research": True,
            },
        ],
    )

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                (
                    "cost_sensitive_mechanism=Reject frequent entries when "
                    "fee, spread, slippage, and turnover drag dominate edge "
                    "or expectancy in historical evidence."
                ),
                (
                    "entry_exists_negative_edge=Reject the thesis when entry "
                    "signals keep negative expectancy or loss after fees in "
                    "historical evidence."
                ),
            ],
        )
    )

    causal_component = next(
        component
        for component in decision["research_selection_score"]["components"]
        if component["name"] == "causal_failure_response_quality"
    )
    assert decision["status"] == "blocked"
    assert decision["causal_failure_map"]["missing_response_categories"] == [
        "walk_forward_fragility"
    ]
    assert causal_component["details"]["weighted_response_score"] == 8.57
    assert causal_component["details"]["unanswered_required_risk_weight"] == 100.0
    walk_forward_score = next(
        item
        for item in causal_component["details"]["category_scores"]
        if item["category"] == "walk_forward_fragility"
    )
    assert walk_forward_score["quality_ratio"] == 0.0
    assert walk_forward_score["missing_reasons"] == ["missing_response"]
    assert "research_selection_score_meets_minimum" in {
        check["name"] for check in decision["blockers"]
    }


def test_research_selection_gate_requires_quantified_cost_response_for_high_risk_map(
    tmp_path,
):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    map_path = _write_research_selection_causal_failure_map(
        tmp_path,
        candidate_count=10,
        dominant_categories=[
            {"category": "cost_sensitive_mechanism", "candidate_count": 10},
            {"category": "regime_fragile_mechanism", "candidate_count": 10},
            {"category": "walk_forward_fragility", "candidate_count": 10},
        ],
        causal_risk_weights=[
            {"category": "cost_sensitive_mechanism", "risk_score": 100.0},
            {"category": "regime_fragile_mechanism", "risk_score": 100.0},
            {"category": "walk_forward_fragility", "risk_score": 100.0},
        ],
    )
    base_responses = [
        (
            "regime_fragile_mechanism=Use closed-candle regime segmentation "
            "and reject the thesis when historical evidence fails."
        ),
        (
            "walk_forward_fragility=Predefine historical walk-forward windows "
            "that should pass and reject absent split support."
        ),
    ]

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                *base_responses,
                (
                    "cost_sensitive_mechanism=Reject frequent entries when "
                    "fee, spread, slippage, and turnover drag dominate edge "
                    "or expectancy in historical evidence."
                ),
            ],
        )
    )

    assert decision["status"] == "blocked"
    cost_quality = decision["causal_failure_map"]["response_quality_by_category"][
        "cost_sensitive_mechanism"
    ]
    assert cost_quality["risk_score"] == 100.0
    assert "quantified_cost_terms" in cost_quality["missing_requirement_groups"]
    assert decision["causal_failure_map"]["category_evidence_gaps"] == [
        {
            "category": "cost_sensitive_mechanism",
            "missing_requirement_groups": ["quantified_cost_terms"],
        }
    ]

    quantified_without_artifact = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                *base_responses,
                (
                    "cost_sensitive_mechanism=Reject unless expected edge "
                    "exceeds 12 bps of all-in fee, spread, slippage, and "
                    "turnover drag in historical evidence."
                ),
            ],
        )
    )

    assert quantified_without_artifact["status"] == "blocked"
    assert quantified_without_artifact["causal_failure_map"]["category_evidence_gaps"] == []
    assert quantified_without_artifact["local_falsification_evidence"][
        "high_risk_cost_evidence_required"
    ] is True
    assert "local_falsification_cost_evidence_present" in {
        check["name"] for check in quantified_without_artifact["blockers"]
    }

    crafted_path = (
        tmp_path
        / "registry"
        / "strategies"
        / "research_decisions"
        / "crafted_cost_falsification.json"
    )
    crafted_path.parent.mkdir(parents=True)
    crafted_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "thesis_id": "TH-ORDERBOOK-RESILIENCE-001",
                "expected_edge_bps": 18.0,
                "all_in_cost_bps": 12.0,
                "sample_count": 64,
            }
        ),
        encoding="utf-8",
    )

    crafted_artifact_decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            local_falsification_paths=[crafted_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                *base_responses,
                (
                    "cost_sensitive_mechanism=Reject unless expected edge "
                    "exceeds 12 bps of all-in fee, spread, slippage, and "
                    "turnover drag in historical evidence."
                ),
            ],
        )
    )

    assert crafted_artifact_decision["status"] == "blocked"
    crafted_artifact = crafted_artifact_decision["local_falsification_evidence"][
        "artifacts"
    ][0]
    assert crafted_artifact["factory_valid"] is False
    assert crafted_artifact["safety_scope_valid"] is False
    assert crafted_artifact["event_source_valid"] is False
    assert crafted_artifact["event_source_failure_synthesis_guard_valid"] is False
    assert crafted_artifact["cost_edge_passes"] is False
    assert "local_falsification_cost_evidence_factory_valid" in {
        check["name"] for check in crafted_artifact_decision["blockers"]
    }
    assert "local_falsification_cost_evidence_safety_scope_valid" in {
        check["name"] for check in crafted_artifact_decision["blockers"]
    }
    assert "local_falsification_cost_evidence_event_source_valid" in {
        check["name"] for check in crafted_artifact_decision["blockers"]
    }

    def local_falsification_payload(
        data_span_days,
        *,
        guard=True,
        context_features_used=False,
        include_context_alignment=True,
    ):
        event_source = {
            "used": True,
            "factory": "research_local_event_builder",
            "factory_valid": True,
            "status": "completed",
            "status_completed": True,
            "thesis_id": "TH-ORDERBOOK-RESILIENCE-001",
            "thesis_matches": True,
            "events_csv_path": "registry/strategies/research_decisions/events.csv",
            "event_path_matches": True,
            "source_ohlcv_path": "user_data/data/bybit/futures/BTC_USDT-5m.parquet",
            "ohlcv_path_matches": True,
            "event_count": 64,
            "safety_scope_valid": True,
            "context_features_used": context_features_used,
            "required_contexts": ["mark_price"] if context_features_used else [],
        }
        if include_context_alignment:
            event_source.update(
                {
                    "context_merge_semantics": (
                        "closed_context_candle_availability_v1"
                    ),
                    "closed_context_candle_alignment_valid": True,
                }
            )
        if guard:
            event_source.update(
                {
                    "failure_synthesis_used": True,
                    "failure_synthesis_parseable": True,
                    "failure_synthesis_path": (
                        "registry/strategies/synthesis/candidate_failure_synthesis.json"
                    ),
                    "failure_synthesis_failed_thesis_id_count": 26,
                    "failure_synthesis_failed_family_count": 26,
                    "failure_synthesis_thesis_repeats": False,
                    "failure_synthesis_mechanism_repeats": False,
                    "failure_synthesis_allow_failed_thesis_or_family": False,
                    "failure_synthesis_guard_valid": True,
                }
            )
        return {
            "factory": "research_local_falsification",
            "status": "passed",
            "thesis_id": "TH-ORDERBOOK-RESILIENCE-001",
            "expected_edge_bps": 18.0,
            "all_in_cost_bps": 12.0,
            "sample_count": 64,
            "data_span_days": data_span_days,
            "event_source": event_source,
            "safety_scope": {
                "historical_only": True,
                "backtest_started": False,
                "strategy_code_generated": False,
                "paper_trading_started": False,
                "dry_run_trading_started": False,
                "live_trading": False,
                "exchange_order_placement": False,
                "shorting": False,
                "leverage": 1.0,
                "process_control": False,
            },
        }

    short_span_falsification_path = (
        tmp_path
        / "registry"
        / "strategies"
        / "research_decisions"
        / "short_span_cost_falsification.json"
    )
    short_span_falsification_path.write_text(
        json.dumps(local_falsification_payload(30.0)),
        encoding="utf-8",
    )

    short_span_decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            local_falsification_paths=[short_span_falsification_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                *base_responses,
                (
                    "cost_sensitive_mechanism=Reject unless expected edge "
                    "exceeds 12 bps of all-in fee, spread, slippage, and "
                    "turnover drag in historical evidence."
                ),
            ],
        )
    )

    assert short_span_decision["status"] == "blocked"
    short_span_artifact = short_span_decision["local_falsification_evidence"][
        "artifacts"
    ][0]
    assert short_span_artifact["sample_sufficient"] is True
    assert short_span_artifact["data_span_sufficient"] is False
    assert "insufficient_data_span" in short_span_artifact["failure_reasons"]

    unguarded_falsification_path = (
        tmp_path
        / "registry"
        / "strategies"
        / "research_decisions"
        / "unguarded_cost_falsification.json"
    )
    unguarded_falsification_path.write_text(
        json.dumps(local_falsification_payload(365.0, guard=False)),
        encoding="utf-8",
    )

    unguarded_decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            local_falsification_paths=[unguarded_falsification_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                *base_responses,
                (
                    "cost_sensitive_mechanism=Reject unless expected edge "
                    "exceeds 12 bps of all-in fee, spread, slippage, and "
                    "turnover drag in historical evidence."
                ),
            ],
        )
    )

    assert unguarded_decision["status"] == "blocked"
    unguarded_artifact = unguarded_decision["local_falsification_evidence"][
        "artifacts"
    ][0]
    assert unguarded_artifact["event_source_valid"] is True
    assert unguarded_artifact["event_source_failure_synthesis_guard_valid"] is False
    assert unguarded_artifact["cost_edge_passes"] is False
    assert "event_source_failure_synthesis_guard_missing_or_failed" in (
        unguarded_artifact["failure_reasons"]
    )
    assert (
        "local_falsification_cost_evidence_event_source_failure_synthesis_guarded"
        in {check["name"] for check in unguarded_decision["blockers"]}
    )

    stale_context_falsification_path = (
        tmp_path
        / "registry"
        / "strategies"
        / "research_decisions"
        / "stale_context_cost_falsification.json"
    )
    stale_context_falsification_path.write_text(
        json.dumps(
            local_falsification_payload(
                365.0,
                context_features_used=True,
                include_context_alignment=False,
            )
        ),
        encoding="utf-8",
    )

    stale_context_decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            local_falsification_paths=[stale_context_falsification_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                *base_responses,
                (
                    "cost_sensitive_mechanism=Reject unless expected edge "
                    "exceeds 12 bps of all-in fee, spread, slippage, and "
                    "turnover drag in historical evidence."
                ),
            ],
        )
    )

    assert stale_context_decision["status"] == "blocked"
    stale_context_artifact = stale_context_decision["local_falsification_evidence"][
        "artifacts"
    ][0]
    assert stale_context_artifact["event_source_valid"] is False
    assert stale_context_artifact["event_source_context_alignment_valid"] is False
    assert "event_source_context_alignment_missing_or_invalid" in (
        stale_context_artifact["failure_reasons"]
    )
    assert (
        "local_falsification_cost_evidence_event_source_context_alignment_valid"
        in {check["name"] for check in stale_context_decision["blockers"]}
    )

    falsification_path = (
        tmp_path
        / "registry"
        / "strategies"
        / "research_decisions"
        / "cost_falsification.json"
    )
    falsification_path.parent.mkdir(parents=True, exist_ok=True)
    falsification_path.write_text(
        json.dumps(local_falsification_payload(365.0)),
        encoding="utf-8",
    )

    approved = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            local_falsification_paths=[falsification_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                *base_responses,
                (
                    "cost_sensitive_mechanism=Reject unless expected edge "
                    "exceeds 12 bps of all-in fee, spread, slippage, and "
                    "turnover drag in historical evidence."
                ),
            ],
        )
    )

    assert approved["status"] == "approved_for_proposal_generation"
    assert approved["causal_failure_map"]["category_evidence_gaps"] == []
    assert approved["local_falsification_evidence"][
        "passing_cost_edge_artifact_count"
    ] == 1
    assert approved["local_falsification_evidence"]["artifacts"][0][
        "event_source_valid"
    ] is True
    assert approved["local_falsification_evidence"]["artifacts"][0][
        "event_source_failure_synthesis_guard_valid"
    ] is True
    assert "local_falsification_cost_edge_exceeds_costs" in {
        check["name"] for check in approved["checks"] if check["status"] == "pass"
    }


def test_research_selection_gate_blocks_thin_causal_failure_responses(tmp_path):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    map_path = _write_research_selection_causal_failure_map(tmp_path)

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                "regime_fragile_mechanism=Handle it better.",
                "walk_forward_fragility=Make windows pass.",
                "cost_sensitive_mechanism=Reduce costs.",
            ],
        )
    )

    assert decision["status"] == "blocked"
    assert decision["causal_failure_map"]["weak_response_categories"] == [
        "regime_fragile_mechanism",
        "walk_forward_fragility",
        "cost_sensitive_mechanism",
    ]
    assert "causal_failure_responses_are_substantive" in {
        check["name"] for check in decision["blockers"]
    }
    assert "causal_failure_responses_address_category_evidence" in {
        check["name"] for check in decision["blockers"]
    }


def test_research_selection_gate_blocks_parameter_only_causal_failure_responses(tmp_path):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    map_path = _write_research_selection_causal_failure_map(tmp_path)
    parameter_only_response = (
        "Tune thresholds, retune lookback lengths, loosen filters, run grid "
        "search optimization, and adjust roi only."
    )

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                f"regime_fragile_mechanism={parameter_only_response}",
                f"walk_forward_fragility={parameter_only_response}",
                f"cost_sensitive_mechanism={parameter_only_response}",
            ],
        )
    )

    assert decision["status"] == "blocked"
    assert decision["causal_failure_map"]["parameter_only_response_categories"] == [
        "regime_fragile_mechanism",
        "walk_forward_fragility",
        "cost_sensitive_mechanism",
    ]
    assert "causal_failure_responses_not_parameter_only" in {
        check["name"] for check in decision["blockers"]
    }


def test_research_selection_gate_blocks_below_minimum_selection_score(tmp_path):
    map_path = _write_research_selection_causal_failure_map(
        tmp_path,
        minimum_research_selection_score=95,
    )

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                (
                    "regime_fragile_mechanism=Use closed-candle regime segmentation "
                    "and reject the thesis when segments do not hold."
                ),
                (
                    "walk_forward_fragility=Predefine which historical "
                    "walk-forward windows should pass and reject absent support."
                ),
                (
                    "cost_sensitive_mechanism=Reject frequent entries when "
                    "fee and turnover drag dominate expected edge."
                ),
            ],
        )
    )

    assert decision["status"] == "blocked"
    assert decision["research_selection_score"]["score"] == 91.0
    assert decision["research_selection_score"]["minimum_score_required"] == 95.0
    assert "local_historical_falsification" in decision["research_selection_score"][
        "failed_components"
    ]
    assert "research_selection_score_meets_minimum" in {
        check["name"] for check in decision["blockers"]
    }


def test_research_selection_gate_requires_quality_report_for_structural_data(tmp_path):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT_USDT-1h-open_interest.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder open interest data", encoding="utf-8")

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            thesis_id="TH-OPEN-INTEREST-QUALITY-001",
            thesis_family="open_interest_positioning",
            mechanism_class="open_interest_positioning_reversal",
            thesis_statement=(
                "Open interest positioning contractions can reveal structural "
                "deleveraging pressure before a closed-candle rebound."
            ),
            mechanism_summary=(
                "Use historical open interest and OHLCV, never live-only data, "
                "to test whether positioning has a durable edge after costs."
            ),
            required_data=[
                "BTCUSDT 1h open interest with a passing local quality report",
                "BTC/USDT:USDT 5m closed-candle OHLCV",
            ],
            local_data_paths=[data_path],
        )
    )

    assert decision["status"] == "blocked"
    assert "structural_data_quality_report_present" in {
        check["name"] for check in decision["blockers"]
    }


def test_research_selection_gate_accepts_passing_structural_quality_report(tmp_path):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT_USDT-1h-open_interest.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder open interest data", encoding="utf-8")
    report_path = tmp_path / "registry" / "strategies" / "checks" / "open_interest_quality.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "reports": [
                    {
                        "path": str(data_path),
                        "ok": True,
                        "rows": 100,
                        "findings": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    capability_path = _write_structural_capability_report(
        tmp_path,
        local_research_usable=["open_interest"],
        blocked_without_new_data=["liquidation", "order_book"],
        must_not_codegen=["open_interest", "liquidation", "order_book"],
    )

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            thesis_id="TH-OPEN-INTEREST-QUALITY-002",
            thesis_family="open_interest_positioning",
            mechanism_class="open_interest_positioning_reversal",
            thesis_statement=(
                "Open interest positioning contractions can reveal structural "
                "deleveraging pressure before a closed-candle rebound."
            ),
            mechanism_summary=(
                "Use historical open interest and OHLCV, never live-only data, "
                "to test whether positioning has a durable edge after costs."
            ),
            required_data=[
                "BTCUSDT 1h open interest with a passing local quality report",
                "BTC/USDT:USDT 5m closed-candle OHLCV",
            ],
            local_data_paths=[data_path],
            local_data_quality_report_paths=[report_path],
            structural_data_capability_report_paths=[capability_path],
        )
    )

    assert "structural_data_quality_report_present" not in {
        check["name"] for check in decision["blockers"]
    }
    assert "structural_data_capability_supports_required_classes" not in {
        check["name"] for check in decision["blockers"]
    }
    assert decision["thesis"]["local_data_quality_report_paths"] == [
        "registry\\strategies\\checks\\open_interest_quality.json"
    ]
    assert decision["thesis"]["structural_data_capability_report_paths"] == [
        "registry\\strategies\\checks\\structural_capability.json"
    ]


def test_research_selection_gate_blocks_structural_class_without_capability_support(
    tmp_path,
):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT_USDT-1h-liquidation.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder liquidation data", encoding="utf-8")
    report_path = tmp_path / "registry" / "strategies" / "checks" / "liquidation_quality.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps({"ok": True, "reports": [{"ok": True, "rows": 100}]}),
        encoding="utf-8",
    )
    capability_path = _write_structural_capability_report(
        tmp_path,
        local_research_usable=["open_interest"],
        blocked_without_new_data=["liquidation", "order_book"],
        must_not_codegen=["open_interest", "liquidation", "order_book"],
    )

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            thesis_id="TH-LIQUIDATION-CASCADE-QUALITY-001",
            thesis_family="liquidation_cascade_reversal",
            mechanism_class="liquidation_absorption_rebound",
            thesis_statement=(
                "Historical liquidation clusters may identify forced-flow exhaustion "
                "before a closed-candle rebound."
            ),
            mechanism_summary=(
                "Use local historical liquidation events, not open-interest retunes, "
                "to test whether forced-flow exhaustion rebounds after costs."
            ),
            required_data=[
                "Local BTC/USDT futures closed-candle OHLCV",
                "Local historical liquidation events",
            ],
            local_data_paths=[data_path],
            local_data_quality_report_paths=[report_path],
            structural_data_capability_report_paths=[capability_path],
        )
    )

    blocker = next(
        check
        for check in decision["blockers"]
        if check["name"] == "structural_data_capability_supports_required_classes"
    )
    assert "liquidation" in blocker["details"]["required_classes"]
    assert blocker["details"]["unsupported_required_classes"] == ["liquidation"]


def test_research_selection_gate_blocks_parameter_only_core_thesis_fields(tmp_path):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    parameter_only_claim = (
        "Tune thresholds, retune lookback lengths, loosen filters, run grid "
        "search optimization, adjust roi, and change stoploss only."
    )

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            thesis_statement=parameter_only_claim,
            mechanism_summary=parameter_only_claim,
            novelty_rationale=parameter_only_claim,
            edge_rationale=parameter_only_claim,
            falsification_plan=parameter_only_claim,
            stop_conditions=[parameter_only_claim],
        )
    )

    assert decision["status"] == "blocked"
    assert decision["research_quality"]["parameter_only_field_names"] == [
        "thesis_statement",
        "mechanism_summary",
        "novelty_rationale",
        "edge_rationale",
        "falsification_plan",
        "stop_conditions[1]",
    ]
    assert "research_thesis_not_parameter_only" in {
        check["name"] for check in decision["blockers"]
    }


def test_research_selection_gate_preserves_causal_map_handoff_summaries(tmp_path):
    question_handoff = {
        "required": True,
        "passed": False,
        "computed_missing_research_question_response_indexes": [2],
    }
    handoff_summaries = [
        {
            "candidate_id": "cand-handoff",
            "research_handoff_summary": {
                "research_decision_question_handoff": question_handoff,
            },
        }
    ]
    map_path = _write_research_selection_causal_failure_map(
        tmp_path,
        research_handoff_summaries=handoff_summaries,
        blocked_next_actions=[
            "retry_validated_local_rejection_by_parameter_tuning",
        ],
    )

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            causal_failure_map_path=map_path,
        )
    )

    assert decision["causal_failure_map"]["research_handoff_summaries"] == (
        handoff_summaries
    )
    assert decision["causal_failure_map"]["blocked_next_actions"] == [
        "retry_validated_local_rejection_by_parameter_tuning"
    ]


def test_research_selection_gate_blocks_causal_failure_map_synthesis_mismatch(tmp_path):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    map_path = _write_research_selection_causal_failure_map(
        tmp_path,
        source_synthesis_id="older-synth",
    )

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            causal_failure_map_path=map_path,
            causal_failure_responses=[
                "regime_fragile_mechanism=Reject when regimes are not stable.",
                "walk_forward_fragility=Reject without historical split support.",
                "cost_sensitive_mechanism=Reject when fee drag dominates.",
            ],
        )
    )

    assert decision["status"] == "blocked"
    assert "causal_failure_map_matches_failure_synthesis" in {
        check["name"] for check in decision["blockers"]
    }


def test_research_selection_gate_blocks_stale_failure_synthesis(tmp_path):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    synthesis_root = tmp_path / "registry" / "strategies" / "synthesis"
    old_path = synthesis_root / "old_synthesis" / "candidate_failure_synthesis.json"
    new_path = synthesis_root / "new_synthesis" / "candidate_failure_synthesis.json"
    old_path.parent.mkdir(parents=True)
    new_path.parent.mkdir(parents=True)

    def payload(synthesis_id: str, generated_at: str) -> dict[str, Any]:
        return {
            "factory": "candidate_failure_synthesis",
            "status": "completed",
            "generated_at": generated_at,
            "synthesis_id": synthesis_id,
            "candidate_count": 2,
            "aggregate_failure_summary": {
                "paper_ready_candidate_ids": [],
                "paper_ready_count": 0,
                "all_candidates_failed_gates": True,
                "hypothesis_families_tried": ["failed_family"],
                "thesis_ids_tried": ["TH-FAILED"],
            },
            "next_research_brief": {
                "requires_new_thesis_id": True,
                "requires_new_research_references": True,
                "minimum_research_reference_count": 2,
                "parameter_only_retry_allowed": False,
                "paper_or_live_promotion_allowed": False,
                "prior_hypothesis_families_to_avoid_as_default": [
                    "failed_family"
                ],
                "failed_thesis_ids": ["TH-FAILED"],
            },
        }

    old_path.write_text(
        json.dumps(payload("old-synthesis", "2026-05-07T00:00:00+00:00")),
        encoding="utf-8",
    )
    new_path.write_text(
        json.dumps(payload("new-synthesis", "2026-05-07T01:00:00+00:00")),
        encoding="utf-8",
    )

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            failure_synthesis_path=old_path,
            local_data_paths=[data_path],
        )
    )

    assert decision["status"] == "blocked"
    novelty = decision["novelty_assessment"]
    assert novelty["failure_synthesis_latest_checked"] is True
    assert novelty["failure_synthesis_is_latest"] is False
    assert novelty["latest_failure_synthesis_id"] == "new-synthesis"
    latest_path = novelty["latest_failure_synthesis_path"].replace("\\", "/")
    assert latest_path.endswith(
        "new_synthesis/candidate_failure_synthesis.json"
    )
    assert "failure_synthesis_is_latest" in {
        check["name"] for check in decision["blockers"]
    }


def test_research_selection_gate_blocks_repeated_failed_family(tmp_path):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            thesis_id="TH-MICROSTRUCTURE-REPEAT-001",
            thesis_family="microstructure_spread_reversion",
            mechanism_class="roll_spread_reversion",
            local_data_paths=[data_path],
            research_references=[
                _research_selection_reference("TH-MICROSTRUCTURE-REPEAT-001", "repeat-a"),
                _research_selection_reference("TH-MICROSTRUCTURE-REPEAT-001", "repeat-b"),
            ],
        )
    )

    assert decision["status"] == "blocked"
    assert decision["proposal_generation_allowed"] is False
    assert "microstructure_spread_reversion" in decision["novelty_assessment"][
        "repeated_failed_family_matches"
    ]
    assert "thesis_family_outside_failed_families" in {
        check["name"] for check in decision["blockers"]
    }


def test_research_selection_gate_blocks_validated_local_rejection_from_synthesis(
    tmp_path,
):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    thesis_id = "TH-LOCAL-REJECTION-REPEAT-001"
    mechanism_class = "validated_local_rejection_mechanism"
    synthesis_path, synthesis = _write_research_selection_synthesis_with_local_rejection(
        tmp_path,
        thesis_id="TH-VALIDATED-LOCAL-REJECTED",
        mechanism_class=mechanism_class,
        valid=True,
    )

    aggregate = synthesis["aggregate_failure_summary"]
    assert aggregate["local_falsification_rejection_count"] == 1
    assert aggregate["local_falsification_invalid_rejection_count"] == 0
    assert aggregate["local_falsification_failed_mechanism_classes"] == [
        mechanism_class
    ]

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            failure_synthesis_path=synthesis_path,
            thesis_id=thesis_id,
            thesis_family=mechanism_class,
            mechanism_class=mechanism_class,
            local_data_paths=[data_path],
            research_references=[
                _research_selection_reference(thesis_id, "local-rejection-a"),
                _research_selection_reference(thesis_id, "local-rejection-b"),
            ],
        )
    )

    novelty_component = next(
        component
        for component in decision["research_selection_score"]["components"]
        if component["name"] == "novelty_against_failure_set"
    )
    assert decision["status"] == "blocked"
    assert mechanism_class in decision["novelty_assessment"][
        "repeated_failed_family_matches"
    ]
    assert decision["novelty_assessment"][
        "local_falsification_failed_mechanism_class_matches"
    ] == [mechanism_class]
    assert decision["novelty_assessment"][
        "local_falsification_failed_thesis_ids"
    ] == ["TH-VALIDATED-LOCAL-REJECTED"]
    assert novelty_component["details"][
        "local_falsification_failed_mechanism_class_matches"
    ] == [mechanism_class]
    assert novelty_component["passed"] is False
    assert "thesis_family_outside_failed_families" in {
        check["name"] for check in decision["blockers"]
    }
    assert "research_thesis_outside_failure_synthesis_local_rejections" in {
        check["name"] for check in decision["blockers"]
    }


def test_research_selection_gate_ignores_invalid_local_rejection_from_synthesis(
    tmp_path,
):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    thesis_id = "TH-INVALID-LOCAL-REJECTION-001"
    mechanism_class = "invalid_local_rejection_mechanism"
    synthesis_path, synthesis = _write_research_selection_synthesis_with_local_rejection(
        tmp_path,
        thesis_id="TH-INVALID-LOCAL-REJECTED",
        mechanism_class=mechanism_class,
        valid=False,
    )

    aggregate = synthesis["aggregate_failure_summary"]
    assert aggregate["local_falsification_rejection_count"] == 0
    assert aggregate["local_falsification_invalid_rejection_count"] == 1
    assert aggregate["local_falsification_failed_mechanism_classes"] == []

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            failure_synthesis_path=synthesis_path,
            thesis_id=thesis_id,
            thesis_family=mechanism_class,
            mechanism_class=mechanism_class,
            local_data_paths=[data_path],
            research_references=[
                _research_selection_reference(thesis_id, "invalid-rejection-a"),
                _research_selection_reference(thesis_id, "invalid-rejection-b"),
            ],
        )
    )

    assert decision["status"] == "approved_for_proposal_generation"
    assert decision["novelty_assessment"]["repeated_failed_family_matches"] == []
    assert decision["novelty_assessment"][
        "local_falsification_failed_mechanism_class_matches"
    ] == []
    assert decision["novelty_assessment"][
        "local_falsification_failed_thesis_ids"
    ] == []
    assert decision["research_selection_score"]["failed_components"] == []


def test_failure_synthesis_and_causal_map_ingest_edge_discovery_rejection(
    tmp_path,
):
    from freqtrade_ext.bot_factory.candidate_failure_map import (
        CandidateFailureMapInputs,
        build_candidate_failure_map,
    )

    mechanism_class = "failed_edge_discovery_mechanism"
    synthesis_path, synthesis = _write_research_selection_synthesis_with_edge_rejection(
        tmp_path,
        thesis_id="TH-EDGE-DISCOVERY-REJECTED",
        mechanism_class=mechanism_class,
        valid=True,
    )

    aggregate = synthesis["aggregate_failure_summary"]
    assert aggregate["edge_discovery_rejection_count"] == 1
    assert aggregate["edge_discovery_invalid_rejection_count"] == 0
    assert aggregate["edge_discovery_failed_mechanism_classes"] == [
        mechanism_class
    ]
    assert aggregate["edge_discovery_rejections"][0]["net_edge_bps"] == -14.0
    assert any(
        "edge discovery" in question.lower()
        for question in synthesis["next_research_brief"][
            "recommended_research_questions"
        ]
    )

    failure_map = build_candidate_failure_map(
        CandidateFailureMapInputs(
            root_dir=tmp_path,
            synthesis_path=synthesis_path,
            map_id="edge-rejection-map-test",
        )
    )
    guidance = failure_map["research_selection_guidance"]
    assert guidance["validated_edge_discovery_rejections"][0][
        "mechanism_class"
    ] == mechanism_class
    assert any(
        "edge discovery rejection" in question.lower()
        for question in guidance["required_research_questions"]
    )


def test_research_selection_gate_blocks_validated_edge_discovery_rejection(
    tmp_path,
):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    thesis_id = "TH-EDGE-REJECTION-REPEAT-001"
    mechanism_class = "validated_edge_discovery_mechanism"
    synthesis_path, synthesis = _write_research_selection_synthesis_with_edge_rejection(
        tmp_path,
        thesis_id="TH-VALIDATED-EDGE-REJECTED",
        mechanism_class=mechanism_class,
        valid=True,
    )

    aggregate = synthesis["aggregate_failure_summary"]
    assert aggregate["edge_discovery_rejection_count"] == 1
    assert aggregate["edge_discovery_failed_mechanism_classes"] == [
        mechanism_class
    ]

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            failure_synthesis_path=synthesis_path,
            thesis_id=thesis_id,
            thesis_family=mechanism_class,
            mechanism_class=mechanism_class,
            local_data_paths=[data_path],
            research_references=[
                _research_selection_reference(thesis_id, "edge-rejection-a"),
                _research_selection_reference(thesis_id, "edge-rejection-b"),
            ],
        )
    )

    novelty_component = next(
        component
        for component in decision["research_selection_score"]["components"]
        if component["name"] == "novelty_against_failure_set"
    )
    assert decision["status"] == "blocked"
    assert decision["novelty_assessment"][
        "edge_discovery_failed_mechanism_class_matches"
    ] == [mechanism_class]
    assert decision["novelty_assessment"][
        "edge_discovery_failed_thesis_ids"
    ] == ["TH-VALIDATED-EDGE-REJECTED"]
    assert novelty_component["details"][
        "edge_discovery_failed_mechanism_class_matches"
    ] == [mechanism_class]
    assert "research_thesis_outside_failure_synthesis_edge_rejections" in {
        check["name"] for check in decision["blockers"]
    }


def test_research_selection_gate_ignores_invalid_edge_discovery_rejection(
    tmp_path,
):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    thesis_id = "TH-INVALID-EDGE-REJECTION-001"
    mechanism_class = "invalid_edge_discovery_mechanism"
    synthesis_path, synthesis = _write_research_selection_synthesis_with_edge_rejection(
        tmp_path,
        thesis_id="TH-INVALID-EDGE-REJECTED",
        mechanism_class=mechanism_class,
        valid=False,
    )

    aggregate = synthesis["aggregate_failure_summary"]
    assert aggregate["edge_discovery_rejection_count"] == 0
    assert aggregate["edge_discovery_invalid_rejection_count"] == 1
    assert aggregate["edge_discovery_failed_mechanism_classes"] == []

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            failure_synthesis_path=synthesis_path,
            thesis_id=thesis_id,
            thesis_family=mechanism_class,
            mechanism_class=mechanism_class,
            local_data_paths=[data_path],
            research_references=[
                _research_selection_reference(thesis_id, "invalid-edge-a"),
                _research_selection_reference(thesis_id, "invalid-edge-b"),
            ],
        )
    )

    assert decision["status"] == "approved_for_proposal_generation"
    assert decision["novelty_assessment"][
        "edge_discovery_failed_mechanism_class_matches"
    ] == []
    assert decision["novelty_assessment"]["edge_discovery_failed_thesis_ids"] == []
    assert decision["research_selection_score"]["failed_components"] == []


def test_research_selection_gate_blocks_prior_local_falsification_rejection(tmp_path):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")
    prior_path = (
        tmp_path
        / "registry"
        / "strategies"
        / "research_decisions"
        / "prior_failed_local_falsification.json"
    )
    prior_path.parent.mkdir(parents=True)
    prior_path.write_text(
        json.dumps(
            {
                "factory": "research_local_falsification",
                "status": "failed",
                "thesis_id": "TH-OLDER-LOCAL-REJECTION",
                "mechanism_class": "closed_candle_resilience_reclaim",
                "expected_edge_bps": 0.25,
                "all_in_cost_bps": 12.0,
                "net_edge_bps": -11.75,
                "sample_count": 64,
                "blockers": [
                    {
                        "name": "expected_edge_exceeds_all_in_cost",
                        "status": "fail",
                    }
                ],
                "safety_scope": {
                    "historical_only": True,
                    "backtest_started": False,
                    "strategy_code_generated": False,
                    "paper_trading_started": False,
                    "dry_run_trading_started": False,
                    "live_trading": False,
                    "exchange_order_placement": False,
                    "shorting": False,
                    "leverage": 1.0,
                    "process_control": False,
                },
            }
        ),
        encoding="utf-8",
    )

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            prior_local_falsification_paths=[prior_path],
        )
    )

    assert decision["status"] == "blocked"
    assert decision["prior_local_falsification_rejections"][
        "matching_rejection_count"
    ] == 1
    prior = decision["prior_local_falsification_rejections"]["artifacts"][0]
    assert prior["rejection_valid"] is True
    assert prior["mechanism_matches"] is True
    assert "research_thesis_not_previously_rejected_by_local_falsification" in {
        check["name"] for check in decision["blockers"]
    }


def test_research_selection_gate_blocks_stale_research_reference_mapping(tmp_path):
    data_path = tmp_path / "user_data" / "data" / "bybit" / "futures" / "BTC_USDT-5m.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("placeholder local closed-candle data", encoding="utf-8")

    decision = select_research_thesis(
        _research_selection_inputs(
            tmp_path,
            local_data_paths=[data_path],
            research_references=[
                _research_selection_reference("TH-OTHER", "stale-a"),
                _research_selection_reference("TH-OTHER", "stale-b"),
            ],
        )
    )

    assert decision["status"] == "blocked"
    assert decision["proposal_generation_allowed"] is False
    assert "research_references_motivate_current_thesis" in {
        check["name"] for check in decision["blockers"]
    }


def test_freqai_prediction_diagnostics_detects_target_mismatch(tmp_path):
    from freqtrade_ext.bot_factory.freqai_prediction_diagnostics import (
        FreqAIPredictionDiagnosticsInputs,
        diagnose_freqai_predictions,
        write_freqai_prediction_diagnostics_artifacts,
    )

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "strategy_name": "HybridStrategy",
        "candidate_id": "hybrid-candidate",
        "generator_mode": "hybrid_ml",
        "target_definition": "future_return",
        "freqai_identifier": "bf_hybrid_candidate",
        "prediction_threshold": 0.001,
    }), encoding="utf-8")
    predictions_dir = tmp_path / "models" / "identifier" / "backtesting_predictions"
    predictions_dir.mkdir(parents=True)
    pd.DataFrame([
        {
            "date": "2025-01-01T00:00:00Z",
            "&-long_return": 0.002,
            "&-long_return_mean": 0.0,
            "&-long_return_std": 0.001,
            "do_predict": 1,
        },
        {
            "date": "2025-01-01T00:05:00Z",
            "&-long_return": -0.001,
            "&-long_return_mean": 0.0,
            "&-long_return_std": 0.001,
            "do_predict": 1,
        },
    ]).to_csv(predictions_dir / "predictions.csv", index=False)
    metadata_dir = tmp_path / "models" / "identifier" / "sub-train-BTC_1"
    metadata_dir.mkdir()
    (metadata_dir / "cb_btc_1_metadata.json").write_text(json.dumps({
        "label_list": ["&-long_return"],
        "labels_mean": {"&-long_return": 0.0},
    }), encoding="utf-8")
    signal_path = tmp_path / "signal.json"
    signal_path.write_text(json.dumps({
        "candidate_id": "hybrid-candidate",
        "diagnostics_id": "signal",
        "entry_count": 0,
        "diagnosis_codes": ["ML_FILTER_UNAVAILABLE"],
        "first_zero_component": "ml_filter",
    }), encoding="utf-8")
    freqai_metadata_path = tmp_path / "freqai_metadata.json"
    freqai_metadata_path.write_text(json.dumps({
        "freqai_identifier": "old_shared_identifier",
    }), encoding="utf-8")

    diagnostics = diagnose_freqai_predictions(FreqAIPredictionDiagnosticsInputs(
        root_dir=tmp_path,
        generated_metadata_path=metadata_path,
        predictions_dir=predictions_dir,
        signal_diagnostics_path=signal_path,
        freqai_metadata_path=freqai_metadata_path,
        diagnostics_id="freqai-pred-test",
        reviewer_notes=["prediction diagnostic only"],
    ))

    assert diagnostics["status"] == "completed"
    assert diagnostics["expected_target_column"] == "&-future_return"
    assert diagnostics["expected_target_column_present"] is False
    assert diagnostics["target_columns"] == ["&-long_return"]
    assert diagnostics["model_label_columns"] == ["&-long_return"]
    assert "PREDICTION_TARGET_MISMATCH" in diagnostics["diagnosis_codes"]
    assert "MODEL_LABEL_MISMATCH" in diagnostics["diagnosis_codes"]
    assert "FREQAI_IDENTIFIER_MISMATCH" in diagnostics["diagnosis_codes"]
    assert diagnostics["expected_freqai_identifier"] == "bf_hybrid_candidate"
    assert diagnostics["observed_freqai_identifiers"] == ["old_shared_identifier"]
    assert diagnostics["freqai_identifier_match"] is False
    assert diagnostics["alternate_target_summaries"]["&-long_return"][
        "above_threshold_count"
    ] == 1
    assert diagnostics["do_predict_summary"]["positive_count"] == 2
    assert diagnostics["safety_scope"]["backtest_started"] is False

    json_path, report_path = write_freqai_prediction_diagnostics_artifacts(
        diagnostics,
        root_dir=tmp_path,
        output_root=Path("diagnostics"),
    )

    assert json_path.is_file()
    assert report_path.is_file()
    assert "expected_target_column_present: False" in report_path.read_text(
        encoding="utf-8"
    )


def test_freqai_prediction_diagnostics_blocks_paths_outside_workspace(tmp_path):
    import pytest

    from freqtrade_ext.bot_factory.freqai_prediction_diagnostics import (
        FreqAIPredictionDiagnosticsInputs,
        diagnose_freqai_predictions,
    )

    outside = tmp_path.parent / f"{tmp_path.name}_outside_metadata.json"
    outside.write_text(json.dumps({"strategy_name": "Outside"}), encoding="utf-8")

    with pytest.raises(ValueError, match="inside the workspace"):
        diagnose_freqai_predictions(FreqAIPredictionDiagnosticsInputs(
            root_dir=tmp_path,
            generated_metadata_path=outside,
            predictions_dir=tmp_path / "predictions",
        ))
