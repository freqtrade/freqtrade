from unittest.mock import MagicMock

from pandas import DataFrame

from freqtrade.data.dataprovider import DataProvider
from freqtrade.tests.conftest import get_patched_exchange


def test_ohlcv(mocker, default_conf, ticker_history):

    exchange = get_patched_exchange(mocker, default_conf)
    exchange._klines['XRP/BTC'] = ticker_history
    exchange._klines['UNITEST/BTC'] = ticker_history
    dp = DataProvider(default_conf, exchange)
    assert ticker_history.equals(dp.ohlcv('UNITEST/BTC'))
    assert isinstance(dp.ohlcv('UNITEST/BTC'), DataFrame)
    assert dp.ohlcv('UNITEST/BTC') is not ticker_history
    assert dp.ohlcv('UNITEST/BTC', copy=False) is ticker_history
    assert dp.ohlcv('NONESENSE/AAA') is None


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
