# pragma pylint: disable=missing-docstring, too-many-arguments, too-many-ancestors, C0103
from datetime import datetime
from copy import deepcopy
from unittest.mock import MagicMock
from sqlalchemy import create_engine

from freqtrade.rpc import init, cleanup, send_msg
from freqtrade.persistence import Trade
import freqtrade.main as main
import freqtrade.misc as misc
import freqtrade.rpc as rpc


def prec_satoshi(a, b):
    """
    :return: True if A and B differs less than one satoshi.
    """
    return abs(a - b) < 0.00000001


def test_init_telegram_enabled(default_conf, mocker):
    module_list = []
    mocker.patch('freqtrade.rpc.REGISTERED_MODULES', module_list)
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.init', MagicMock())

    init(default_conf)

    assert telegram_mock.call_count == 1
    assert 'telegram' in module_list


def test_init_telegram_disabled(default_conf, mocker):
    module_list = []
    mocker.patch('freqtrade.rpc.REGISTERED_MODULES', module_list)
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.init', MagicMock())

    conf = deepcopy(default_conf)
    conf['telegram']['enabled'] = False
    init(conf)

    assert telegram_mock.call_count == 0
    assert 'telegram' not in module_list


def test_cleanup_telegram_enabled(mocker):
    mocker.patch('freqtrade.rpc.REGISTERED_MODULES', ['telegram'])
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.cleanup', MagicMock())
    cleanup()
    assert telegram_mock.call_count == 1


def test_cleanup_telegram_disabled(mocker):
    mocker.patch('freqtrade.rpc.REGISTERED_MODULES', [])
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.cleanup', MagicMock())
    cleanup()
    assert telegram_mock.call_count == 0


def test_send_msg_telegram_enabled(mocker):
    mocker.patch('freqtrade.rpc.REGISTERED_MODULES', ['telegram'])
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.send_msg', MagicMock())
    send_msg('test')
    assert telegram_mock.call_count == 1


def test_send_msg_telegram_disabled(mocker):
    mocker.patch('freqtrade.rpc.REGISTERED_MODULES', [])
    telegram_mock = mocker.patch('freqtrade.rpc.telegram.send_msg', MagicMock())
    send_msg('test')
    assert telegram_mock.call_count == 0


def test_rpc_forcesell(default_conf, update, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock())
    cancel_order_mock = MagicMock()
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker,
                          cancel_order=cancel_order_mock,
                          get_order=MagicMock(return_value={
                              'closed': True,
                              'type': 'LIMIT_BUY',
                          }))
    main.init(default_conf, create_engine('sqlite://'))

    misc.update_state(misc.State.STOPPED)
    (error, res) = rpc.rpc_forcesell(None)
    assert error
    assert res == '`trader is not running`'
    misc.update_state(misc.State.RUNNING)
    (error, res) = rpc.rpc_forcesell(None)
    assert error
    assert res == 'Invalid argument.'

    (error, res) = rpc.rpc_forcesell('all')
    assert not error
    assert res == ''

    main.create_trade(0.001, 5)
    (error, res) = rpc.rpc_forcesell('all')
    assert not error
    assert res == ''

    (error, res) = rpc.rpc_forcesell('1')
    assert not error
    assert res == ''

    misc.update_state(misc.State.STOPPED)

    (error, res) = rpc.rpc_forcesell(None)
    assert error
    assert res == '`trader is not running`'

    (error, res) = rpc.rpc_forcesell('all')
    assert error
    assert res == '`trader is not running`'

    misc.update_state(misc.State.RUNNING)

    assert cancel_order_mock.call_count == 0
    # make an limit-buy open trade
    mocker.patch.multiple('freqtrade.exchange',
                          get_order=MagicMock(return_value={
                              'closed': None,
                              'type': 'LIMIT_BUY'
                          }))
    # check that the trade is called, which is done
    # by ensuring exchange.cancel_order is called
    (error, res) = rpc.rpc_forcesell('1')
    assert not error
    assert res == ''
    assert cancel_order_mock.call_count == 1

    main.create_trade(0.001, 5)
    # make an limit-sell open trade
    mocker.patch.multiple('freqtrade.exchange',
                          get_order=MagicMock(return_value={
                              'closed': None,
                              'type': 'LIMIT_SELL'
                          }))
    (error, res) = rpc.rpc_forcesell('2')
    assert not error
    assert res == ''
    # status quo, no exchange calls
    assert cancel_order_mock.call_count == 1


def test_rpc_trade_status(default_conf, update, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    main.init(default_conf, create_engine('sqlite://'))

    misc.update_state(misc.State.STOPPED)
    (error, result) = rpc.rpc_trade_status()
    assert error
    assert result.find('trader is not running') >= 0

    misc.update_state(misc.State.RUNNING)
    (error, result) = rpc.rpc_trade_status()
    assert error
    assert result.find('no active trade') >= 0

    main.create_trade(0.001, 5)
    (error, result) = rpc.rpc_trade_status()
    assert not error
    trade = result[0]
    assert trade.find('[BTC_ETH]') >= 0


def test_rpc_daily_profit(default_conf, update, ticker, limit_buy_order, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    mocker.patch.multiple('freqtrade.fiat_convert.Pymarketcap',
                          ticker=MagicMock(return_value={'price_usd': 15000.0}),
                          _cache_symbols=MagicMock(return_value={'BTC': 1}))
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    main.init(default_conf, create_engine('sqlite://'))
    stake_currency = default_conf['stake_currency']
    fiat_display_currency = default_conf['fiat_display_currency']

    # Create some test data
    main.create_trade(0.001, 5)
    trade = Trade.query.first()
    assert trade

    # Simulate buy & sell
    trade.update(limit_buy_order)
    trade.update(limit_sell_order)
    trade.close_date = datetime.utcnow()
    trade.is_open = False

    # Try valid data
    update.message.text = '/daily 2'
    (error, days) = rpc.rpc_daily_profit(7, stake_currency,
                                         fiat_display_currency)
    assert not error
    assert len(days) == 7
    for day in days:
        # [datetime.date(2018, 1, 11), '0.00000000 BTC', '0.000 USD']
        assert (day[1] == '0.00000000 BTC' or
                day[1] == '0.00006217 BTC')

        assert (day[2] == '0.000 USD' or
                day[2] == '0.933 USD')
    # ensure first day is current date
    assert str(days[0][0]) == str(datetime.utcnow().date())

    # Try invalid data
    (error, days) = rpc.rpc_daily_profit(0, stake_currency,
                                         fiat_display_currency)
    assert error
    assert days.find('must be an integer greater than 0') >= 0


def test_rpc_trade_statistics(
        default_conf, update, ticker, ticker_sell_up, limit_buy_order, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    mocker.patch.multiple('freqtrade.fiat_convert.Pymarketcap',
                          ticker=MagicMock(return_value={'price_usd': 15000.0}),
                          _cache_symbols=MagicMock(return_value={'BTC': 1}))
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    main.init(default_conf, create_engine('sqlite://'))
    stake_currency = default_conf['stake_currency']
    fiat_display_currency = default_conf['fiat_display_currency']

    (error, stats) = rpc.rpc_trade_statistics(stake_currency,
                                              fiat_display_currency)
    assert error
    assert stats.find('no closed trade') >= 0

    # Create some test data
    main.create_trade(0.001, 5)
    trade = Trade.query.first()
    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)
    # Update the ticker with a market going up
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker_sell_up)
    trade.update(limit_sell_order)
    trade.close_date = datetime.utcnow()
    trade.is_open = False

    (error, stats) = rpc.rpc_trade_statistics(stake_currency,
                                              fiat_display_currency)
    assert not error
    assert prec_satoshi(stats['profit_closed_coin'], 6.217e-05)
    assert prec_satoshi(stats['profit_closed_percent'], 6.2)
    assert prec_satoshi(stats['profit_closed_fiat'], 0.93255)
    assert prec_satoshi(stats['profit_all_coin'], 6.217e-05)
    assert prec_satoshi(stats['profit_all_percent'], 6.2)
    assert prec_satoshi(stats['profit_all_fiat'], 0.93255)
    assert stats['trade_count'] == 1
    assert stats['first_trade_date'] == 'just now'
    assert stats['latest_trade_date'] == 'just now'
    assert stats['avg_duration'] == '0:00:00'
    assert stats['best_pair'] == 'BTC_ETH'
    assert prec_satoshi(stats['best_rate'], 6.2)


# Test that rpc_trade_statistics can handle trades that lacks
# trade.open_rate (it is set to None)
def test_rpc_trade_statistics_closed(
        default_conf, update, ticker, ticker_sell_up, limit_buy_order, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock())
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    mocker.patch.multiple('freqtrade.fiat_convert.Pymarketcap',
                          ticker=MagicMock(return_value={'price_usd': 15000.0}),
                          _cache_symbols=MagicMock(return_value={'BTC': 1}))
    mocker.patch('freqtrade.fiat_convert.CryptoToFiatConverter._find_price', return_value=15000.0)
    main.init(default_conf, create_engine('sqlite://'))
    stake_currency = default_conf['stake_currency']
    fiat_display_currency = default_conf['fiat_display_currency']

    # Create some test data
    main.create_trade(0.001, 5)
    trade = Trade.query.first()
    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)
    # Update the ticker with a market going up
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker_sell_up)
    trade.update(limit_sell_order)
    trade.close_date = datetime.utcnow()
    trade.is_open = False

    for trade in Trade.query.order_by(Trade.id).all():
        trade.open_rate = None

    (error, stats) = rpc.rpc_trade_statistics(stake_currency,
                                              fiat_display_currency)
    assert not error
    assert prec_satoshi(stats['profit_closed_coin'], 0)
    assert prec_satoshi(stats['profit_closed_percent'], 0)
    assert prec_satoshi(stats['profit_closed_fiat'], 0)
    assert prec_satoshi(stats['profit_all_coin'], 0)
    assert prec_satoshi(stats['profit_all_percent'], 0)
    assert prec_satoshi(stats['profit_all_fiat'], 0)
    assert stats['trade_count'] == 1
    assert stats['first_trade_date'] == 'just now'
    assert stats['latest_trade_date'] == 'just now'
    assert stats['avg_duration'] == '0:00:00'
    assert stats['best_pair'] == 'BTC_ETH'
    assert prec_satoshi(stats['best_rate'], 6.2)


def test_rpc_balance_handle(default_conf, update, mocker):
    mock_balance = [{
        'Currency': 'BTC',
        'Balance': 10.0,
        'Available': 12.0,
        'Pending': 0.0,
        'CryptoAddress': 'XXXX',
    }, {
        'Currency': 'ETH',
        'Balance': 0.0,
        'Available': 0.0,
        'Pending': 0.0,
        'CryptoAddress': 'XXXX',
    }]
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch.multiple('freqtrade.main.exchange',
                          get_balances=MagicMock(return_value=mock_balance))
    mocker.patch.multiple('freqtrade.fiat_convert.Pymarketcap',
                          ticker=MagicMock(return_value={'price_usd': 15000.0}),
                          _cache_symbols=MagicMock(return_value={'BTC': 1}))

    (error, res) = rpc.rpc_balance(default_conf['fiat_display_currency'])
    assert not error
    (trade, x, y, z) = res
    assert prec_satoshi(x, 10)
    assert prec_satoshi(z, 150000)
    assert y == 'USD'
    assert len(trade) == 1
    assert trade[0]['currency'] == 'BTC'
    assert prec_satoshi(trade[0]['available'], 12)
    assert prec_satoshi(trade[0]['balance'], 10)
    assert prec_satoshi(trade[0]['pending'], 0)
    assert prec_satoshi(trade[0]['est_btc'], 10)


def test_performance_handle(
        default_conf, update, ticker, limit_buy_order, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: (True, False))
    msg_mock = MagicMock()
    mocker.patch('freqtrade.main.rpc.send_msg', MagicMock())
    mocker.patch.multiple('freqtrade.rpc.telegram',
                          _CONF=default_conf,
                          init=MagicMock(),
                          send_msg=msg_mock)
    mocker.patch.multiple('freqtrade.main.exchange',
                          validate_pairs=MagicMock(),
                          get_ticker=ticker)
    main.init(default_conf, create_engine('sqlite://'))

    # Create some test data
    main.create_trade(0.001, int(default_conf['ticker_interval']))
    trade = Trade.query.first()
    assert trade

    # Simulate fulfilled LIMIT_BUY order for trade
    trade.update(limit_buy_order)

    # Simulate fulfilled LIMIT_SELL order for trade
    trade.update(limit_sell_order)

    trade.close_date = datetime.utcnow()
    trade.is_open = False
    (error, res) = rpc.rpc_performance()
    assert not error
    assert len(res) == 1
    assert res[0]['pair'] == 'BTC_ETH'
    assert res[0]['count'] == 1
    assert prec_satoshi(res[0]['profit'], 6.2)
