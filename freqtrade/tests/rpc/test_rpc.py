# pragma pylint: disable=missing-docstring, too-many-arguments, too-many-ancestors, C0103
from copy import deepcopy
from unittest.mock import MagicMock

from freqtrade.rpc import init, cleanup, send_msg
from sqlalchemy import create_engine
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
