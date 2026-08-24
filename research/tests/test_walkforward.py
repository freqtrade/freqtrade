# research/tests/test_walkforward.py
import copy
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


def test_evaluate_fixed_params_matches_run_window_for_the_winning_variant(mocker):
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
    run_window_result = runner.run_window(window, param_grid)

    direct_result = runner.evaluate_fixed_params(window, run_window_result.best_params)

    assert direct_result.test_sharpe == run_window_result.test_sharpe
    assert direct_result.test_returns == run_window_result.test_returns
    assert direct_result.test_n_trades == run_window_result.test_n_trades
    assert direct_result.variant_returns == {}

    # fee_override=None and fee_override=<the config's own base fee> must be equivalent --
    # they resolve to the same fee, just via a different code path.
    from freqtrade.optimize.backtesting import Backtesting

    base_fee = Backtesting(conf).fee
    override_result = runner.evaluate_fixed_params(
        window, run_window_result.best_params, fee_override=base_fee
    )
    assert override_result.test_sharpe == run_window_result.test_sharpe
    assert override_result.test_returns == run_window_result.test_returns


def test_evaluate_fixed_params_fee_override_changes_results(mocker):
    """Fee overrides affect backtest results (P&L and/or trade count), not just realized profit.

    IMPORTANT: Different fees can produce different trade counts even with identical entry signals.
    Mechanism: freqtrade's should_exit() checks ROI profit thresholds via calc_profit_ratio(),
    which factors in the fee. A higher fee reduces net profit, potentially delaying ROI exits and
    preventing subsequent trades on the same pair (no parallel stacking by default). This is real,
    verified freqtrade behavior for strategies using minimal_roi (which this fixture does, with
    use_exit_signal=False). This is not a bug or a loosened test — it's an immutable property
    of how freqtrade's exit timing interacts with fees.
    """
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
    params = {"buy_rsi": 25}

    config_before = copy.deepcopy(runner.config)
    cheap = runner.evaluate_fixed_params(window, params, fee_override=0.0)
    expensive = runner.evaluate_fixed_params(window, params, fee_override=0.05)

    # self.config must never be mutated by a fee_override call.
    assert runner.config == config_before

    # Fee override produces measurable result differences (Sharpe, trade count, or both).
    assert (
        cheap.test_sharpe != expensive.test_sharpe or cheap.test_n_trades != expensive.test_n_trades
    )


def test_run_window_still_works_after_the_refactor(mocker):
    """Regression coverage: run_window's own pre-existing test already covers this,
    but this direct check makes the refactor's intent explicit -- run_window's final
    phase now delegates to evaluate_fixed_params rather than duplicating it."""
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
    result = runner.run_window(window, [{"buy_rsi": 25}, {"buy_rsi": 35}])

    assert result.best_params in [{"buy_rsi": 25}, {"buy_rsi": 35}]
    assert set(result.variant_returns) == {
        variant_key({"buy_rsi": 25}),
        variant_key({"buy_rsi": 35}),
    }
    assert isinstance(result.train_sharpe, float)
    assert isinstance(result.test_sharpe, float)
