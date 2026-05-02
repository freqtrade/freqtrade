import json
import zipfile
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
from freqtrade_ext.bot_factory.freqai_checks import DependencySpec, check_freqai_dependencies
from freqtrade_ext.bot_factory.safety import scan_paths


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
