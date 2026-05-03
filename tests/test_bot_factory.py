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
