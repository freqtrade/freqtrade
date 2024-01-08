from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from pandas import DataFrame, Timestamp

from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import CandleType, RunMode
from freqtrade.exceptions import ExchangeError, OperationalException
from freqtrade.plugins.pairlistmanager import PairListManager
from tests.conftest import EXMS, generate_test_data, get_patched_exchange


@pytest.mark.parametrize('candle_type', [
    'mark',
    '',
])
def test_dp_ohlcv(mocker, default_conf, ohlcv_history, candle_type):
    default_conf["runmode"] = RunMode.DRY_RUN
    timeframe = default_conf["timeframe"]
    exchange = get_patched_exchange(mocker, default_conf)
    candletype = CandleType.from_string(candle_type)
    exchange._klines[("XRP/BTC", timeframe, candletype)] = ohlcv_history
    exchange._klines[("UNITTEST/BTC", timeframe, candletype)] = ohlcv_history

    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.DRY_RUN
    assert ohlcv_history.equals(dp.ohlcv("UNITTEST/BTC", timeframe, candle_type=candletype))
    assert isinstance(dp.ohlcv("UNITTEST/BTC", timeframe, candle_type=candletype), DataFrame)
    assert dp.ohlcv("UNITTEST/BTC", timeframe, candle_type=candletype) is not ohlcv_history
    assert dp.ohlcv("UNITTEST/BTC", timeframe, copy=False, candle_type=candletype) is ohlcv_history
    assert not dp.ohlcv("UNITTEST/BTC", timeframe, candle_type=candletype).empty
    assert dp.ohlcv("NONESENSE/AAA", timeframe, candle_type=candletype).empty

    # Test with and without parameter
    assert dp.ohlcv(
        "UNITTEST/BTC",
        timeframe,
        candle_type=candletype
    ).equals(dp.ohlcv("UNITTEST/BTC", candle_type=candle_type))

    default_conf["runmode"] = RunMode.LIVE
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.LIVE
    assert isinstance(dp.ohlcv("UNITTEST/BTC", timeframe, candle_type=candle_type), DataFrame)

    default_conf["runmode"] = RunMode.BACKTEST
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.BACKTEST
    assert dp.ohlcv("UNITTEST/BTC", timeframe, candle_type=candle_type).empty


def test_historic_ohlcv(mocker, default_conf, ohlcv_history):
    historymock = MagicMock(return_value=ohlcv_history)
    mocker.patch("freqtrade.data.dataprovider.load_pair_history", historymock)

    dp = DataProvider(default_conf, None)
    data = dp.historic_ohlcv("UNITTEST/BTC", "5m")
    assert isinstance(data, DataFrame)
    assert historymock.call_count == 1
    assert historymock.call_args_list[0][1]["timeframe"] == "5m"


def test_historic_ohlcv_dataformat(mocker, default_conf, ohlcv_history):
    hdf5loadmock = MagicMock(return_value=ohlcv_history)
    featherloadmock = MagicMock(return_value=ohlcv_history)
    mocker.patch("freqtrade.data.history.hdf5datahandler.HDF5DataHandler._ohlcv_load", hdf5loadmock)
    mocker.patch("freqtrade.data.history.featherdatahandler.FeatherDataHandler._ohlcv_load",
                 featherloadmock)

    default_conf["runmode"] = RunMode.BACKTEST
    exchange = get_patched_exchange(mocker, default_conf)
    dp = DataProvider(default_conf, exchange)
    data = dp.historic_ohlcv("UNITTEST/BTC", "5m")
    assert isinstance(data, DataFrame)
    hdf5loadmock.assert_not_called()
    featherloadmock.assert_called_once()

    # Switching to dataformat hdf5
    hdf5loadmock.reset_mock()
    featherloadmock.reset_mock()
    default_conf["dataformat_ohlcv"] = "hdf5"
    dp = DataProvider(default_conf, exchange)
    data = dp.historic_ohlcv("UNITTEST/BTC", "5m")
    assert isinstance(data, DataFrame)
    hdf5loadmock.assert_called_once()
    featherloadmock.assert_not_called()


@pytest.mark.parametrize('candle_type', [
    'mark',
    'futures',
    '',
])
def test_get_pair_dataframe(mocker, default_conf, ohlcv_history, candle_type):
    default_conf["runmode"] = RunMode.DRY_RUN
    timeframe = default_conf["timeframe"]
    exchange = get_patched_exchange(mocker, default_conf)
    candletype = CandleType.from_string(candle_type)
    exchange._klines[("XRP/BTC", timeframe, candletype)] = ohlcv_history
    exchange._klines[("UNITTEST/BTC", timeframe, candletype)] = ohlcv_history

    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.DRY_RUN
    assert ohlcv_history.equals(dp.get_pair_dataframe(
        "UNITTEST/BTC", timeframe, candle_type=candle_type))
    assert ohlcv_history.equals(dp.get_pair_dataframe(
        "UNITTEST/BTC", timeframe, candle_type=candletype))
    assert isinstance(dp.get_pair_dataframe(
        "UNITTEST/BTC", timeframe, candle_type=candle_type), DataFrame)
    assert dp.get_pair_dataframe("UNITTEST/BTC", timeframe,
                                 candle_type=candle_type) is not ohlcv_history
    assert not dp.get_pair_dataframe("UNITTEST/BTC", timeframe, candle_type=candle_type).empty
    assert dp.get_pair_dataframe("NONESENSE/AAA", timeframe, candle_type=candle_type).empty

    # Test with and without parameter
    assert dp.get_pair_dataframe("UNITTEST/BTC", timeframe, candle_type=candle_type)\
        .equals(dp.get_pair_dataframe("UNITTEST/BTC", candle_type=candle_type))

    default_conf["runmode"] = RunMode.LIVE
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.LIVE
    assert isinstance(dp.get_pair_dataframe(
        "UNITTEST/BTC", timeframe, candle_type=candle_type), DataFrame)
    assert dp.get_pair_dataframe("NONESENSE/AAA", timeframe, candle_type=candle_type).empty

    historymock = MagicMock(return_value=ohlcv_history)
    mocker.patch("freqtrade.data.dataprovider.load_pair_history", historymock)
    default_conf["runmode"] = RunMode.BACKTEST
    dp = DataProvider(default_conf, exchange)
    assert dp.runmode == RunMode.BACKTEST
    df = dp.get_pair_dataframe("UNITTEST/BTC", timeframe, candle_type=candle_type)
    assert isinstance(df, DataFrame)
    assert len(df) == 3  # ohlcv_history mock has just 3 rows

    dp._set_dataframe_max_date(ohlcv_history.iloc[-1]['date'])
    df = dp.get_pair_dataframe("UNITTEST/BTC", timeframe, candle_type=candle_type)
    assert isinstance(df, DataFrame)
    assert len(df) == 2  # ohlcv_history is limited to 2 rows now


def test_available_pairs(mocker, default_conf, ohlcv_history):
    exchange = get_patched_exchange(mocker, default_conf)
    timeframe = default_conf["timeframe"]
    exchange._klines[("XRP/BTC", timeframe)] = ohlcv_history
    exchange._klines[("UNITTEST/BTC", timeframe)] = ohlcv_history

    dp = DataProvider(default_conf, exchange)
    assert len(dp.available_pairs) == 2
    assert dp.available_pairs == [("XRP/BTC", timeframe), ("UNITTEST/BTC", timeframe), ]


def test_producer_pairs(default_conf):
    dataprovider = DataProvider(default_conf, None)

    producer = "default"
    whitelist = ["XRP/BTC", "ETH/BTC"]
    assert len(dataprovider.get_producer_pairs(producer)) == 0

    dataprovider._set_producer_pairs(whitelist, producer)
    assert len(dataprovider.get_producer_pairs(producer)) == 2

    new_whitelist = ["BTC/USDT"]
    dataprovider._set_producer_pairs(new_whitelist, producer)
    assert dataprovider.get_producer_pairs(producer) == new_whitelist

    assert dataprovider.get_producer_pairs("bad") == []


def test_get_producer_df(default_conf):
    dataprovider = DataProvider(default_conf, None)
    ohlcv_history = generate_test_data('5m', 150)
    pair = 'BTC/USDT'
    timeframe = default_conf['timeframe']
    candle_type = CandleType.SPOT

    empty_la = datetime.fromtimestamp(0, tz=timezone.utc)
    now = datetime.now(timezone.utc)

    # no data has been added, any request should return an empty dataframe
    dataframe, la = dataprovider.get_producer_df(pair, timeframe, candle_type)
    assert dataframe.empty
    assert la == empty_la

    # the data is added, should return that added dataframe
    dataprovider._add_external_df(pair, ohlcv_history, now, timeframe, candle_type)
    dataframe, la = dataprovider.get_producer_df(pair, timeframe, candle_type)
    assert len(dataframe) > 0
    assert la > empty_la

    # no data on this producer, should return empty dataframe
    dataframe, la = dataprovider.get_producer_df(pair, producer_name='bad')
    assert dataframe.empty
    assert la == empty_la

    # non existent timeframe, empty dataframe
    datframe, la = dataprovider.get_producer_df(pair, timeframe='1h')
    assert dataframe.empty
    assert la == empty_la


def test_emit_df(mocker, default_conf, ohlcv_history):
    mocker.patch('freqtrade.rpc.rpc_manager.RPCManager.__init__', MagicMock())
    rpc_mock = mocker.patch('freqtrade.rpc.rpc_manager.RPCManager', MagicMock())
    send_mock = mocker.patch('freqtrade.rpc.rpc_manager.RPCManager.send_msg', MagicMock())

    dataprovider = DataProvider(default_conf, exchange=None, rpc=rpc_mock)
    dataprovider_no_rpc = DataProvider(default_conf, exchange=None)

    pair = "BTC/USDT"

    # No emit yet
    assert send_mock.call_count == 0

    # Rpc is added, we call emit, should call send_msg
    dataprovider._emit_df(pair, ohlcv_history, False)
    assert send_mock.call_count == 1

    send_mock.reset_mock()
    dataprovider._emit_df(pair, ohlcv_history, True)
    assert send_mock.call_count == 2

    send_mock.reset_mock()

    # No rpc added, emit called, should not call send_msg
    dataprovider_no_rpc._emit_df(pair, ohlcv_history, False)
    assert send_mock.call_count == 0


def test_refresh(mocker, default_conf):
    refresh_mock = MagicMock()
    mocker.patch(f"{EXMS}.refresh_latest_ohlcv", refresh_mock)

    exchange = get_patched_exchange(mocker, default_conf, id="binance")
    timeframe = default_conf["timeframe"]
    pairs = [("XRP/BTC", timeframe), ("UNITTEST/BTC", timeframe)]

    pairs_non_trad = [("ETH/USDT", timeframe), ("BTC/TUSD", "1h")]

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
    assert order_book_l2.call_args_list[0][0][1] >= 5

    assert isinstance(res, dict)
    assert 'bids' in res
    assert 'asks' in res


def test_market(mocker, default_conf, markets):
    api_mock = MagicMock()
    api_mock.markets = markets
    exchange = get_patched_exchange(mocker, default_conf, api_mock=api_mock)

    dp = DataProvider(default_conf, exchange)
    res = dp.market('ETH/BTC')

    assert isinstance(res, dict)
    assert 'symbol' in res
    assert res['symbol'] == 'ETH/BTC'

    res = dp.market('UNITTEST/BTC')
    assert res is None


def test_ticker(mocker, default_conf, tickers):
    ticker_mock = MagicMock(return_value=tickers()['ETH/BTC'])
    mocker.patch(f"{EXMS}.fetch_ticker", ticker_mock)
    exchange = get_patched_exchange(mocker, default_conf)
    dp = DataProvider(default_conf, exchange)
    res = dp.ticker('ETH/BTC')
    assert isinstance(res, dict)
    assert 'symbol' in res
    assert res['symbol'] == 'ETH/BTC'

    ticker_mock = MagicMock(side_effect=ExchangeError('Pair not found'))
    mocker.patch(f"{EXMS}.fetch_ticker", ticker_mock)
    exchange = get_patched_exchange(mocker, default_conf)
    dp = DataProvider(default_conf, exchange)
    res = dp.ticker('UNITTEST/BTC')
    assert res == {}


def test_current_whitelist(mocker, default_conf, tickers):
    # patch default conf to volumepairlist
    default_conf['pairlists'][0] = {'method': 'VolumePairList', "number_assets": 5}

    mocker.patch.multiple(EXMS,
                          exchange_has=MagicMock(return_value=True),
                          get_tickers=tickers)
    exchange = get_patched_exchange(mocker, default_conf)

    pairlist = PairListManager(exchange, default_conf)
    dp = DataProvider(default_conf, exchange, pairlist)

    # Simulate volumepairs from exchange.
    pairlist.refresh_pairlist()

    assert dp.current_whitelist() == pairlist._whitelist
    # The identity of the 2 lists should not be identical, but a copy
    assert dp.current_whitelist() is not pairlist._whitelist

    with pytest.raises(OperationalException):
        dp = DataProvider(default_conf, exchange)
        dp.current_whitelist()


def test_get_analyzed_dataframe(mocker, default_conf, ohlcv_history):

    default_conf["runmode"] = RunMode.DRY_RUN

    timeframe = default_conf["timeframe"]
    exchange = get_patched_exchange(mocker, default_conf)

    dp = DataProvider(default_conf, exchange)
    dp._set_cached_df("XRP/BTC", timeframe, ohlcv_history, CandleType.SPOT)
    dp._set_cached_df("UNITTEST/BTC", timeframe, ohlcv_history, CandleType.SPOT)

    assert dp.runmode == RunMode.DRY_RUN
    dataframe, time = dp.get_analyzed_dataframe("UNITTEST/BTC", timeframe)
    assert ohlcv_history.equals(dataframe)
    assert isinstance(time, datetime)

    dataframe, time = dp.get_analyzed_dataframe("XRP/BTC", timeframe)
    assert ohlcv_history.equals(dataframe)
    assert isinstance(time, datetime)

    dataframe, time = dp.get_analyzed_dataframe("NOTHING/BTC", timeframe)
    assert dataframe.empty
    assert isinstance(time, datetime)
    assert time == datetime(1970, 1, 1, tzinfo=timezone.utc)

    # Test backtest mode
    default_conf["runmode"] = RunMode.BACKTEST
    dp._set_dataframe_max_index(1)
    dataframe, time = dp.get_analyzed_dataframe("XRP/BTC", timeframe)

    assert len(dataframe) == 1

    dp._set_dataframe_max_index(2)
    dataframe, time = dp.get_analyzed_dataframe("XRP/BTC", timeframe)
    assert len(dataframe) == 2

    dp._set_dataframe_max_index(3)
    dataframe, time = dp.get_analyzed_dataframe("XRP/BTC", timeframe)
    assert len(dataframe) == 3

    dp._set_dataframe_max_index(500)
    dataframe, time = dp.get_analyzed_dataframe("XRP/BTC", timeframe)
    assert len(dataframe) == len(ohlcv_history)


def test_no_exchange_mode(default_conf):
    dp = DataProvider(default_conf, None)

    message = "Exchange is not available to DataProvider."

    with pytest.raises(OperationalException, match=message):
        dp.refresh([()])

    with pytest.raises(OperationalException, match=message):
        dp.ohlcv('XRP/USDT', '5m', '')

    with pytest.raises(OperationalException, match=message):
        dp.market('XRP/USDT')

    with pytest.raises(OperationalException, match=message):
        dp.ticker('XRP/USDT')

    with pytest.raises(OperationalException, match=message):
        dp.orderbook('XRP/USDT', 20)

    with pytest.raises(OperationalException, match=message):
        dp.available_pairs()


def test_dp_send_msg(default_conf):

    default_conf["runmode"] = RunMode.DRY_RUN

    default_conf["timeframe"] = '1h'
    dp = DataProvider(default_conf, None)
    msg = 'Test message'
    dp.send_msg(msg)

    assert msg in dp._msg_queue
    dp._msg_queue.pop()
    assert msg not in dp._msg_queue
    # Message is not resent due to caching
    dp.send_msg(msg)
    assert msg not in dp._msg_queue
    dp.send_msg(msg, always_send=True)
    assert msg in dp._msg_queue

    default_conf["runmode"] = RunMode.BACKTEST
    dp = DataProvider(default_conf, None)
    dp.send_msg(msg, always_send=True)
    assert msg not in dp._msg_queue


def test_dp__add_external_df(default_conf_usdt):
    timeframe = '1h'
    default_conf_usdt["timeframe"] = timeframe
    dp = DataProvider(default_conf_usdt, None)
    df = generate_test_data(timeframe, 24, '2022-01-01 00:00:00+00:00')
    last_analyzed = datetime.now(timezone.utc)

    res = dp._add_external_df('ETH/USDT', df, last_analyzed, timeframe, CandleType.SPOT)
    assert res[0] is False
    # Why 1000 ??
    assert res[1] == 1000

    # Hard add dataframe
    dp._replace_external_df('ETH/USDT', df, last_analyzed, timeframe, CandleType.SPOT)
    # BTC is not stored yet
    res = dp._add_external_df('BTC/USDT', df, last_analyzed, timeframe, CandleType.SPOT)
    assert res[0] is False
    df_res, _ = dp.get_producer_df('ETH/USDT', timeframe, CandleType.SPOT)
    assert len(df_res) == 24

    # Add the same dataframe again - dataframe size shall not change.
    res = dp._add_external_df('ETH/USDT', df, last_analyzed, timeframe, CandleType.SPOT)
    assert res[0] is True
    assert isinstance(res[1], int)
    assert res[1] == 0
    df, _ = dp.get_producer_df('ETH/USDT', timeframe, CandleType.SPOT)
    assert len(df) == 24

    # Add a new day.
    df2 = generate_test_data(timeframe, 24, '2022-01-02 00:00:00+00:00')

    res = dp._add_external_df('ETH/USDT', df2, last_analyzed, timeframe, CandleType.SPOT)
    assert res[0] is True
    assert isinstance(res[1], int)
    assert res[1] == 0
    df, _ = dp.get_producer_df('ETH/USDT', timeframe, CandleType.SPOT)
    assert len(df) == 48

    # Add a dataframe with a 12 hour offset - so 12 candles are overlapping, and 12 valid.
    df3 = generate_test_data(timeframe, 24, '2022-01-02 12:00:00+00:00')

    res = dp._add_external_df('ETH/USDT', df3, last_analyzed, timeframe, CandleType.SPOT)
    assert res[0] is True
    assert isinstance(res[1], int)
    assert res[1] == 0
    df, _ = dp.get_producer_df('ETH/USDT', timeframe, CandleType.SPOT)
    # New length = 48 + 12 (since we have a 12 hour offset).
    assert len(df) == 60
    assert df.iloc[-1]['date'] == df3.iloc[-1]['date']
    assert df.iloc[-1]['date'] == Timestamp('2022-01-03 11:00:00+00:00')

    # Generate 1 new candle
    df4 = generate_test_data(timeframe, 1, '2022-01-03 12:00:00+00:00')
    res = dp._add_external_df('ETH/USDT', df4, last_analyzed, timeframe, CandleType.SPOT)
    # assert res[0] is True
    # assert res[1] == 0
    df, _ = dp.get_producer_df('ETH/USDT', timeframe, CandleType.SPOT)
    # New length = 61 + 1
    assert len(df) == 61
    assert df.iloc[-2]['date'] == Timestamp('2022-01-03 11:00:00+00:00')
    assert df.iloc[-1]['date'] == Timestamp('2022-01-03 12:00:00+00:00')

    # Gap in the data ...
    df4 = generate_test_data(timeframe, 1, '2022-01-05 00:00:00+00:00')
    res = dp._add_external_df('ETH/USDT', df4, last_analyzed, timeframe, CandleType.SPOT)
    assert res[0] is False
    # 36 hours - from 2022-01-03 12:00:00+00:00 to 2022-01-05 00:00:00+00:00
    assert isinstance(res[1], int)
    assert res[1] == 36
    df, _ = dp.get_producer_df('ETH/USDT', timeframe, CandleType.SPOT)
    # New length = 61 + 1
    assert len(df) == 61

    # Empty dataframe
    df4 = generate_test_data(timeframe, 0, '2022-01-05 00:00:00+00:00')
    res = dp._add_external_df('ETH/USDT', df4, last_analyzed, timeframe, CandleType.SPOT)
    assert res[0] is False
    # 36 hours - from 2022-01-03 12:00:00+00:00 to 2022-01-05 00:00:00+00:00
    assert isinstance(res[1], int)
    assert res[1] == 0


def test_dp_get_required_startup(default_conf_usdt):
    timeframe = '1h'
    default_conf_usdt["timeframe"] = timeframe
    dp = DataProvider(default_conf_usdt, None)

    # No FreqAI config
    assert dp.get_required_startup('5m') == 0
    assert dp.get_required_startup('1h') == 0
    assert dp.get_required_startup('1d') == 0

    dp._config['startup_candle_count'] = 20
    assert dp.get_required_startup('5m') == 20
    assert dp.get_required_startup('1h') == 20
    assert dp.get_required_startup('1h') == 20

    # With freqAI config

    dp._config['freqai'] = {
        'enabled': True,
        'train_period_days': 20,
        'feature_parameters': {
            'indicator_periods_candles': [
                5,
                20,
            ]
        }
    }
    assert dp.get_required_startup('5m') == 5780
    assert dp.get_required_startup('1h') == 500
    assert dp.get_required_startup('1d') == 40

    # FreqAI kindof ignores startup_candle_count if it's below indicator_periods_candles
    dp._config['startup_candle_count'] = 0
    assert dp.get_required_startup('5m') == 5780
    assert dp.get_required_startup('1h') == 500
    assert dp.get_required_startup('1d') == 40

    dp._config['freqai']['feature_parameters']['indicator_periods_candles'][1] = 50
    assert dp.get_required_startup('5m') == 5810
    assert dp.get_required_startup('1h') == 530
    assert dp.get_required_startup('1d') == 70

    # scenario from issue https://github.com/freqtrade/freqtrade/issues/9432
    dp._config['freqai'] = {
        'enabled': True,
        'train_period_days': 180,
        'feature_parameters': {
            'indicator_periods_candles': [
                10,
                20,
            ]
        }
    }
    dp._config['startup_candle_count'] = 40
    assert dp.get_required_startup('5m') == 51880
    assert dp.get_required_startup('1h') == 4360
    assert dp.get_required_startup('1d') == 220
