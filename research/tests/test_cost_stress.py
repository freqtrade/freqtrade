from pathlib import Path

import numpy as np
import pytest

from freqtrade.data import history
from freqtrade.data.history import get_timerange
from freqtrade.enums import RunMode
from freqtrade.optimize.backtesting import Backtesting
from research.cost_stress import fee_sensitivity
from research.statistics import deflated_sharpe_ratio
from research.walkforward import WalkForwardRunner, generate_windows
from tests.conftest import get_default_conf, patch_exchange


TESTDATADIR = Path(__file__).resolve().parents[2] / "tests" / "testdata"
EXMS = "freqtrade.exchange.exchange.Exchange"


def _conf():
    conf = get_default_conf(TESTDATADIR)
    conf["runmode"] = RunMode.BACKTEST
    conf["max_open_trades"] = 10
    conf["use_exit_signal"] = False
    return conf


def _patch(mocker):
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_pair_base_currency", lambda _, x: x.split("/")[0])


def _window_results(conf):
    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    total_days = max(8, (max_date - min_date).days)
    train_days = max(1, total_days // 8)
    test_days = max(1, total_days // 16)
    windows = generate_windows(min_date, max_date, train_days, test_days)
    runner = WalkForwardRunner(conf, pairs=["UNITTEST/BTC"], timeframe="5m", datadir=TESTDATADIR)
    return runner, runner.run(windows, [{"buy_rsi": 25}, {"buy_rsi": 35}])


def test_fee_sensitivity_reports_one_entry_per_multiplier(mocker):
    conf = _conf()
    _patch(mocker)
    _runner, results = _window_results(conf)

    report = fee_sensitivity(
        conf,
        pairs=["UNITTEST/BTC"],
        timeframe="5m",
        datadir=TESTDATADIR,
        window_results=results,
        multipliers=(1.0, 1.5),
    )

    assert set(report) == {1.0, 1.5}
    for stats in report.values():
        assert isinstance(stats["mean_test_sharpe"], float)
        assert 0.0 <= stats["deflated_sharpe"] <= 1.0
        assert stats["n_windows"] == len(results)


def test_fee_sensitivity_baseline_matches_direct_recomputation(mocker):
    conf = _conf()
    _patch(mocker)
    runner, results = _window_results(conf)

    report = fee_sensitivity(
        conf,
        pairs=["UNITTEST/BTC"],
        timeframe="5m",
        datadir=TESTDATADIR,
        window_results=results,
        multipliers=(1.0,),
    )

    base_fee = Backtesting(conf).fee
    direct = [
        runner.evaluate_fixed_params(wr.window, wr.best_params, fee_override=base_fee)
        for wr in results
    ]
    expected_mean_sharpe = float(np.mean([r.test_sharpe for r in direct]))
    expected_n_obs = sum(len(r.test_returns) for r in direct)
    expected_deflated = deflated_sharpe_ratio(
        expected_mean_sharpe, n_obs=expected_n_obs, n_trials=1, periods_per_year=365
    )

    assert report[1.0]["mean_test_sharpe"] == pytest.approx(expected_mean_sharpe)
    assert report[1.0]["deflated_sharpe"] == pytest.approx(expected_deflated)


def test_fee_sensitivity_raises_on_empty_or_non_positive_multipliers(mocker):
    conf = _conf()
    _patch(mocker)
    _runner, results = _window_results(conf)

    with pytest.raises(ValueError, match="multipliers"):
        fee_sensitivity(
            conf,
            pairs=["UNITTEST/BTC"],
            timeframe="5m",
            datadir=TESTDATADIR,
            window_results=results,
            multipliers=(),
        )
    with pytest.raises(ValueError, match="multipliers"):
        fee_sensitivity(
            conf,
            pairs=["UNITTEST/BTC"],
            timeframe="5m",
            datadir=TESTDATADIR,
            window_results=results,
            multipliers=(0.0,),
        )
