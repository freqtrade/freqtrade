from unittest.mock import MagicMock

from pandas import DataFrame

from freqtrade.data.dataprovider import DataProvider
from freqtrade.state import RunMode
from tests.conftest import get_patched_exchange


def test_ohlcv(mocker, default_conf, ticker_history):
    default_conf["runmode"] = RunMode.DRY_RUN
    ticker_interval = default_conf["ticker_interval"]
    exchange = get_patched_exchange(mocker, default_conf)
    exchange._klines[("XRP/BTC", ticker_interval)] = ticker_history
    exchange._klines[("UNITTEST/BTC", ticker_interval)] = ticker_history

    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.DRY_RUN
    assert ticker_history.equals(dp.ohlcv("UNITTEST/BTC", ticker_interval))
    assert isinstance(dp.ohlcv("UNITTEST/BTC", ticker_interval), DataFrame)
    assert dp.ohlcv("UNITTEST/BTC", ticker_interval) is not ticker_history
    assert dp.ohlcv("UNITTEST/BTC", ticker_interval, copy=False) is ticker_history
    assert not dp.ohlcv("UNITTEST/BTC", ticker_interval).empty
    assert dp.ohlcv("NONESENSE/AAA", ticker_interval).empty

    # Test with and without parameter
    assert dp.ohlcv("UNITTEST/BTC", ticker_interval).equals(dp.ohlcv("UNITTEST/BTC"))

    default_conf["runmode"] = RunMode.LIVE
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.LIVE
    assert isinstance(dp.ohlcv("UNITTEST/BTC", ticker_interval), DataFrame)

    default_conf["runmode"] = RunMode.BACKTEST
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.BACKTEST
    assert dp.ohlcv("UNITTEST/BTC", ticker_interval).empty


def test_historic_ohlcv(mocker, default_conf, ticker_history):
    historymock = MagicMock(return_value=ticker_history)
    mocker.patch("freqtrade.data.dataprovider.load_pair_history", historymock)

    dp = DataProvider(default_conf, None)
    data = dp.historic_ohlcv("UNITTEST/BTC", "5m")
    assert isinstance(data, DataFrame)
    assert historymock.call_count == 1
    assert historymock.call_args_list[0][1]["ticker_interval"] == "5m"


def test_get_pair_dataframe(mocker, default_conf, ticker_history):
    default_conf["runmode"] = RunMode.DRY_RUN
    ticker_interval = default_conf["ticker_interval"]
    exchange = get_patched_exchange(mocker, default_conf)
    exchange._klines[("XRP/BTC", ticker_interval)] = ticker_history
    exchange._klines[("UNITTEST/BTC", ticker_interval)] = ticker_history

    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.DRY_RUN
    assert ticker_history.equals(dp.get_pair_dataframe("UNITTEST/BTC", ticker_interval))
    assert isinstance(dp.get_pair_dataframe("UNITTEST/BTC", ticker_interval), DataFrame)
    assert dp.get_pair_dataframe("UNITTEST/BTC", ticker_interval) is not ticker_history
    assert not dp.get_pair_dataframe("UNITTEST/BTC", ticker_interval).empty
    assert dp.get_pair_dataframe("NONESENSE/AAA", ticker_interval).empty

    # Test with and without parameter
    assert dp.get_pair_dataframe("UNITTEST/BTC",
                                 ticker_interval).equals(dp.get_pair_dataframe("UNITTEST/BTC"))

    default_conf["runmode"] = RunMode.LIVE
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.LIVE
    assert isinstance(dp.get_pair_dataframe("UNITTEST/BTC", ticker_interval), DataFrame)
    assert dp.get_pair_dataframe("NONESENSE/AAA", ticker_interval).empty

    historymock = MagicMock(return_value=ticker_history)
    mocker.patch("freqtrade.data.dataprovider.load_pair_history", historymock)
    default_conf["runmode"] = RunMode.BACKTEST
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.BACKTEST
    assert isinstance(dp.get_pair_dataframe("UNITTEST/BTC", ticker_interval), DataFrame)
    # assert dp.get_pair_dataframe("NONESENSE/AAA", ticker_interval).empty


def test_available_pairs(mocker, default_conf, ticker_history):
    exchange = get_patched_exchange(mocker, default_conf)
    ticker_interval = default_conf["ticker_interval"]
    exchange._klines[("XRP/BTC", ticker_interval)] = ticker_history
    exchange._klines[("UNITTEST/BTC", ticker_interval)] = ticker_history

    dp = DataProvider(default_conf, exchange)
    assert len(dp.available_pairs) == 2
    assert dp.available_pairs == [
        ("XRP/BTC", ticker_interval),
        ("UNITTEST/BTC", ticker_interval),
    ]


def test_refresh(mocker, default_conf, ticker_history):
    refresh_mock = MagicMock()
    mocker.patch("freqtrade.exchange.Exchange.refresh_latest_ohlcv", refresh_mock)

    exchange = get_patched_exchange(mocker, default_conf, id="binance")
    ticker_interval = default_conf["ticker_interval"]
    pairs = [("XRP/BTC", ticker_interval), ("UNITTEST/BTC", ticker_interval)]

    pairs_non_trad = [("ETH/USDT", ticker_interval), ("BTC/TUSD", "1h")]

    dp = DataProvider(default_conf, exchange)
    dp.refresh(pairs)

    assert refresh_mock.call_count == 1
    assert len(refresh_mock.call_args[0]) == 1
    assert len(refresh_mock.call_args[0][0]) == len(pairs)
    assert refresh_mock.call_args[0][0] == pairs

    refresh_mock.reset_mock()
    dp.refresh(pairs, pairs_non_trad)
    assert refresh_mock.call_count == 1
    assert len(refresh_mock.call_args[0]) == 1
    assert len(refresh_mock.call_args[0][0]) == len(pairs) + len(pairs_non_trad)
    assert refresh_mock.call_args[0][0] == pairs + pairs_non_trad


def test_orderbook(mocker, default_conf, order_book_l2):
    api_mock = MagicMock()
    api_mock.fetch_l2_order_book = order_book_l2
    exchange = get_patched_exchange(mocker, default_conf, api_mock=api_mock)

    dp = DataProvider(default_conf, exchange)
    res = dp.orderbook('ETH/BTC', 5)
    assert order_book_l2.call_count == 1
    assert order_book_l2.call_args_list[0][0][0] == 'ETH/BTC'
    assert order_book_l2.call_args_list[0][0][1] == 5

    assert type(res) is dict
    assert 'bids' in res
    assert 'asks' in res


def test_market(mocker, default_conf, markets):
    api_mock = MagicMock()
    api_mock.markets = markets
    exchange = get_patched_exchange(mocker, default_conf, api_mock=api_mock)

    dp = DataProvider(default_conf, exchange)
    res = dp.market('ETH/BTC')

    assert type(res) is dict
    assert 'symbol' in res
    assert res['symbol'] == 'ETH/BTC'

    res = dp.market('UNITTEST/BTC')
    assert res is None
