from unittest.mock import MagicMock

from pandas import DataFrame

from freqtrade.data.dataprovider import DataProvider
from freqtrade.state import RunMode
from freqtrade.tests.conftest import get_patched_exchange


def test_ohlcv(mocker, default_conf, ticker_history):
    default_conf['runmode'] = RunMode.DRY_RUN
    tick_interval = default_conf['ticker_interval']
    exchange = get_patched_exchange(mocker, default_conf)
    exchange._klines[('XRP/BTC', tick_interval)] = ticker_history
    exchange._klines[('UNITTEST/BTC', tick_interval)] = ticker_history
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.DRY_RUN
    assert ticker_history.equals(dp.ohlcv('UNITTEST/BTC', tick_interval))
    assert isinstance(dp.ohlcv('UNITTEST/BTC', tick_interval), DataFrame)
    assert dp.ohlcv('UNITTEST/BTC', tick_interval) is not ticker_history
    assert dp.ohlcv('UNITTEST/BTC', tick_interval, copy=False) is ticker_history
    assert not dp.ohlcv('UNITTEST/BTC', tick_interval).empty
    assert dp.ohlcv('NONESENSE/AAA', tick_interval).empty

    # Test with and without parameter
    assert dp.ohlcv('UNITTEST/BTC', tick_interval).equals(dp.ohlcv('UNITTEST/BTC'))

    default_conf['runmode'] = RunMode.LIVE
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.LIVE
    assert isinstance(dp.ohlcv('UNITTEST/BTC', tick_interval), DataFrame)

    default_conf['runmode'] = RunMode.BACKTEST
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.BACKTEST
    assert dp.ohlcv('UNITTEST/BTC', tick_interval).empty


def test_historic_ohlcv(mocker, default_conf, ticker_history):

    historymock = MagicMock(return_value=ticker_history)
    mocker.patch('freqtrade.data.dataprovider.load_pair_history', historymock)

    # exchange = get_patched_exchange(mocker, default_conf)
    dp = DataProvider(default_conf, None)
    data = dp.historic_ohlcv('UNITTEST/BTC', "5m")
    assert isinstance(data, DataFrame)
    assert historymock.call_count == 1
    assert historymock.call_args_list[0][1]['datadir'] is None
    assert historymock.call_args_list[0][1]['refresh_pairs'] is False
    assert historymock.call_args_list[0][1]['ticker_interval'] == '5m'


def test_available_pairs(mocker, default_conf, ticker_history):
    exchange = get_patched_exchange(mocker, default_conf)

    tick_interval = default_conf['ticker_interval']
    exchange._klines[('XRP/BTC', tick_interval)] = ticker_history
    exchange._klines[('UNITTEST/BTC', tick_interval)] = ticker_history
    dp = DataProvider(default_conf, exchange)

    assert len(dp.available_pairs) == 2
    assert dp.available_pairs == ['XRP/BTC', 'UNITTEST/BTC']
