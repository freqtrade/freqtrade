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


def test_rpc_trade_status(default_conf, update, ticker, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
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

    main.create_trade(0.001)
    (error, result) = rpc.rpc_trade_status()
    assert not error
    trade = result[0]
    assert trade.find('[BTC_ETH]') >= 0


def test_rpc_daily_profit(default_conf, update, ticker, limit_buy_order, limit_sell_order, mocker):
    mocker.patch.dict('freqtrade.main._CONF', default_conf)
    mocker.patch('freqtrade.main.get_signal', side_effect=lambda s, t: True)
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
    main.create_trade(0.001)
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
