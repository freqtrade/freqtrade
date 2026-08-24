# research/tests/test_walkforward.py
from datetime import timedelta
from pathlib import Path

from freqtrade.data import history
from freqtrade.data.history import get_timerange
from freqtrade.enums import RunMode
from research.walkforward import WalkForwardRunner, Window, variant_key
from tests.conftest import get_default_conf, patch_exchange


TESTDATADIR = Path(__file__).resolve().parents[2] / "tests" / "testdata"
EXMS = "freqtrade.exchange.exchange.Exchange"


def _conf():
    conf = get_default_conf(TESTDATADIR)
    conf["runmode"] = RunMode.BACKTEST
    conf["max_open_trades"] = 10
    conf["use_exit_signal"] = False
    return conf


def test_run_window_selects_best_train_params_and_reports_oos_result(mocker):
    conf = _conf()
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_pair_base_currency", lambda _, x: x.split("/")[0])

    full_data = history.load_data(datadir=TESTDATADIR, timeframe="5m", pairs=["UNITTEST/BTC"])
    min_date, max_date = get_timerange(full_data)
    train_days = max(1, int((max_date - min_date).days * 0.7))
    train_end = min_date + timedelta(days=train_days)
    window = Window(
        train_start=min_date, train_end=train_end, test_start=train_end, test_end=max_date
    )

    runner = WalkForwardRunner(conf, pairs=["UNITTEST/BTC"], timeframe="5m", datadir=TESTDATADIR)
    param_grid = [{"buy_rsi": 25}, {"buy_rsi": 35}]
    result = runner.run_window(window, param_grid)

    assert result.best_params in param_grid
    assert set(result.variant_returns) == {variant_key(p) for p in param_grid}
    assert isinstance(result.train_sharpe, float)
    assert isinstance(result.test_sharpe, float)
    assert isinstance(result.test_returns, list)
    assert result.test_n_trades == len(result.test_returns)


def test_generate_windows_are_contiguous_and_non_overlapping():
    from datetime import UTC, datetime

    from research.walkforward import generate_windows

    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2020, 3, 1, tzinfo=UTC)
    windows = generate_windows(start, end, train_days=20, test_days=10)

    assert len(windows) > 0
    for w in windows:
        assert w.train_end == w.test_start
        assert w.test_end <= end
    from itertools import pairwise

    for a, b in pairwise(windows):
        assert b.test_start == a.test_end
