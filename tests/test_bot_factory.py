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
