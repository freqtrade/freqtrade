import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from freqtrade_ext.bot_factory.backtest_results import (
    GateThresholds,
    evaluate_initial_gate,
    load_backtest_result,
    load_gate_thresholds,
    summarize,
)
from freqtrade_ext.bot_factory.data_quality import check_ohlcv_parquet, pair_to_ohlcv_filename
from freqtrade_ext.bot_factory.freqai_backtest import (
    build_freqai_metadata,
    freqai_input_pairs,
    freqai_input_timeframes,
    freqai_model_name,
    resolve_ohlcv_input_paths,
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
from freqtrade_ext.bot_factory.strategy_proposals import (
    REQUIRED_PROPOSAL_SECTIONS,
    StrategyProposalEvidenceInput,
    StrategyProposalInputs,
    build_strategy_proposal,
    write_strategy_proposal_artifacts,
)
from freqtrade_ext.bot_factory.strategy_code import (
    StrategyCodeInputs,
    build_strategy_code,
    write_strategy_code_artifacts,
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
    assert {finding.rule for finding in report.findings} == {
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


def test_training_child_run_id_sanitizes_timerange():
    assert training_child_run_id("train", "20250105-20250107") == (
        "train_20250105_20250107"
    )


def test_training_backtest_command_uses_checked_wrapper_only(tmp_path):
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


def test_training_walk_forward_command_accepts_windows_and_rules(tmp_path):
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
        min_pass_rate=0.5,
    )

    assert cmd[:2] == [
        ".venv/Scripts/python.exe",
        "scripts/bot_factory_run_walk_forward.py",
    ]
    assert cmd.count("--window") == 2
    assert "20250105-20250107" in cmd
    assert cmd[cmd.index("--min-pass-rate") + 1] == "0.5"


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


def test_strategy_code_generator_writes_long_only_strategy_and_metadata(tmp_path):
    proposal_artifacts = build_strategy_proposal(_strategy_proposal_inputs(tmp_path))
    write_strategy_proposal_artifacts(proposal_artifacts)
    _write_hypothesis_metadata(proposal_artifacts.metadata_path)
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
    assert "enter_short" not in generated_code
    assert "exit_short" not in generated_code
    assert "def leverage" not in generated_code
    assert "shift(-1" not in generated_code
    assert ".iloc[-1" not in generated_code
    assert artifacts.metadata["source_proposal_content_hash"] == (
        proposal_artifacts.metadata["proposal_content_hash"]
    )
    assert artifacts.metadata["parameter_defaults"]["buy_rsi_window"] == 14
    assert artifacts.metadata["static_check"]["ran"] is True
    assert artifacts.metadata["static_check"]["ok"] is True
    assert scan_paths([artifacts.strategy_path]).ok


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
    assert 'shift(-12)' in code
    assert artifacts.metadata["label_horizon"] == 12


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
        "evidence_paths": [],
        "output_root": tmp_path / "registry" / "strategies" / "proposals",
        "created_by_agent": "codex-test",
        "created_at": "2026-05-04T00:00:00+00:00",
        "command": ["python", "scripts/bot_factory_generate_strategy_proposal.py"],
    }
    data.update(overrides)
    return StrategyProposalInputs(**data)




def _apply_hypothesis_metadata(metadata: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    defaults = {
        "thesis_id": "THESIS-MR-001",
        "thesis_type": "mean_reversion",
        "thesis_statement": "Pullback recoveries in liquid BTC futures revert on closed-candle confirmation.",
        "falsification_criteria": "Reject if walk-forward return remains negative with acceptable trade count.",
        "novelty_vs_previous": "Adds explicit liquidity filter and timeout risk guard versus prior baseline.",
        "evidence_refs": ["paper:10.2139/ssrn.1968356"],
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
    }), encoding="utf-8")
    generated = tmp_path / "generated.json"
    generated.write_text(json.dumps({"strategy_name": "LongOnlyRsiPullbackCandidate", "candidate_evaluation_eligible": True}), encoding="utf-8")
    static = tmp_path / "static.json"
    static.write_text(json.dumps({"ok": True}), encoding="utf-8")
    ohlcv = tmp_path / "ohlcv.json"
    ohlcv.write_text(json.dumps({"ok": True}), encoding="utf-8")
    backtest = tmp_path / "backtest.json"
    backtest.write_text(json.dumps({"recommendation": "pass"}), encoding="utf-8")
    walk = tmp_path / "walk.json"
    walk.write_text(json.dumps({"recommendation": "pass"}), encoding="utf-8")
    training = tmp_path / "train.json"
    training.write_text(json.dumps({"recommendation": "pass"}), encoding="utf-8")

    manifest = evaluate_candidate(CandidateEvaluationInputs(
        root_dir=tmp_path,
        proposal_metadata_path=proposal,
        generated_metadata_path=generated,
        candidate_id="cand-1",
        static_check_path=static,
        ohlcv_quality_path=ohlcv,
        backtest_metrics_path=backtest,
        walk_forward_metrics_path=walk,
        training_manifest_path=training,
    ))
    assert manifest["recommendation"] == "pass"
    manifest_path, index_path = write_candidate_artifacts(
        manifest, root_dir=tmp_path, output_root=Path("out"), index_path=Path("idx.jsonl")
    )
    assert manifest_path.is_file()
    lines = index_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_candidate_evaluation_rejects_ineligible_candidate(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import CandidateEvaluationInputs, evaluate_candidate

    p = tmp_path / "p.json"
    p.write_text(json.dumps({"strategy_name": "S", "code_generation_eligible": False}), encoding="utf-8")
    g = tmp_path / "g.json"
    g.write_text(json.dumps({"strategy_name": "S", "candidate_evaluation_eligible": False}), encoding="utf-8")
    manifest = evaluate_candidate(CandidateEvaluationInputs(root_dir=tmp_path, proposal_metadata_path=p, generated_metadata_path=g, candidate_id="c"))
    assert manifest["recommendation"] == "reject"


def test_candidate_evaluation_rule_based_does_not_require_training_manifest(tmp_path):
    from freqtrade_ext.bot_factory.candidate_evaluation import CandidateEvaluationInputs, evaluate_candidate

    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps({"strategy_name": "RuleS", "code_generation_eligible": True}), encoding="utf-8")
    generated = tmp_path / "generated.json"
    generated.write_text(json.dumps({"strategy_name": "RuleS", "candidate_evaluation_eligible": True, "generator_mode": "rule_based"}), encoding="utf-8")
    static = tmp_path / "static.json"
    static.write_text(json.dumps({"ok": True}), encoding="utf-8")
    ohlcv = tmp_path / "ohlcv.json"
    ohlcv.write_text(json.dumps({"ok": True}), encoding="utf-8")
    backtest = tmp_path / "backtest.json"
    backtest.write_text(json.dumps({"recommendation": "pass"}), encoding="utf-8")
    walk = tmp_path / "walk.json"
    walk.write_text(json.dumps({"recommendation": "pass"}), encoding="utf-8")

    manifest = evaluate_candidate(CandidateEvaluationInputs(
        root_dir=tmp_path,
        proposal_metadata_path=proposal,
        generated_metadata_path=generated,
        candidate_id="cand-rule",
        static_check_path=static,
        ohlcv_quality_path=ohlcv,
        backtest_metrics_path=backtest,
        walk_forward_metrics_path=walk,
        training_manifest_path=None,
    ))
    training_check = next(c for c in manifest["checks"] if c["name"] == "training_factory")
    assert training_check["status"] == "skipped"
    assert manifest["recommendation"] == "pass"


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
