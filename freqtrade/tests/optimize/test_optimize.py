# pragma pylint: disable=missing-docstring, protected-access, C0103
from freqtrade import optimize
from freqtrade.arguments import TimeRange
from freqtrade.data import history
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy.default_strategy import DefaultStrategy
from freqtrade.tests.conftest import log_has, patch_exchange


def test_get_timeframe(default_conf, mocker) -> None:
    patch_exchange(mocker)
    strategy = DefaultStrategy(default_conf)

    data = strategy.tickerdata_to_dataframe(
        history.load_data(
            datadir=None,
            ticker_interval='1m',
            pairs=['UNITTEST/BTC']
        )
    )
    min_date, max_date = optimize.get_timeframe(data)
    assert min_date.isoformat() == '2017-11-04T23:02:00+00:00'
    assert max_date.isoformat() == '2017-11-14T22:58:00+00:00'


def test_validate_backtest_data_warn(default_conf, mocker, caplog) -> None:
    patch_exchange(mocker)
    strategy = DefaultStrategy(default_conf)

    data = strategy.tickerdata_to_dataframe(
        history.load_data(
            datadir=None,
            ticker_interval='1m',
            pairs=['UNITTEST/BTC'],
            fill_up_missing=False
        )
    )
    min_date, max_date = optimize.get_timeframe(data)
    caplog.clear()
    assert optimize.validate_backtest_data(data, min_date, max_date,
                                           timeframe_to_minutes('1m'))
    assert len(caplog.record_tuples) == 1
    assert log_has(
        "UNITTEST/BTC has missing frames: expected 14396, got 13680, that's 716 missing values",
        caplog.record_tuples)


def test_validate_backtest_data(default_conf, mocker, caplog) -> None:
    patch_exchange(mocker)
    strategy = DefaultStrategy(default_conf)

    timerange = TimeRange('index', 'index', 200, 250)
    data = strategy.tickerdata_to_dataframe(
        history.load_data(
            datadir=None,
            ticker_interval='5m',
            pairs=['UNITTEST/BTC'],
            timerange=timerange
        )
    )

    min_date, max_date = optimize.get_timeframe(data)
    caplog.clear()
    assert not optimize.validate_backtest_data(data, min_date, max_date,
                                               timeframe_to_minutes('5m'))
    assert len(caplog.record_tuples) == 0
