from pathlib import Path
from unittest.mock import MagicMock

import pytest
import rapidjson

from freqtrade.commands.build_config_commands import (ask_user_config, ask_user_overwrite,
                                                      start_new_config, validate_is_float,
                                                      validate_is_int)
from freqtrade.exceptions import OperationalException
from tests.conftest import get_args, log_has_re


def test_validate_is_float():
    assert validate_is_float('2.0')
    assert validate_is_float('2.1')
    assert validate_is_float('0.1')
    assert validate_is_float('-0.5')
    assert not validate_is_float('-0.5e')


def test_validate_is_int():
    assert validate_is_int('2')
    assert validate_is_int('6')
    assert validate_is_int('-1')
    assert validate_is_int('500')
    assert not validate_is_int('2.0')
    assert not validate_is_int('2.1')
    assert not validate_is_int('-2.1')
    assert not validate_is_int('-ee')


@pytest.mark.parametrize('exchange', ['bittrex', 'binance', 'kraken', 'ftx'])
def test_start_new_config(mocker, caplog, exchange):
    wt_mock = mocker.patch.object(Path, "write_text", MagicMock())
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    unlink_mock = mocker.patch.object(Path, "unlink", MagicMock())
    mocker.patch('freqtrade.commands.build_config_commands.ask_user_overwrite', return_value=True)

    sample_selections = {
        'max_open_trades': 3,
        'stake_currency': 'USDT',
        'stake_amount': 100,
        'fiat_display_currency': 'EUR',
        'timeframe': '15m',
        'dry_run': True,
        'exchange_name': exchange,
        'exchange_key': 'sampleKey',
        'exchange_secret': 'Samplesecret',
        'telegram': False,
        'telegram_token': 'asdf1244',
        'telegram_chat_id': '1144444',
        'api_server': False,
        'api_server_listen_addr': '127.0.0.1',
        'api_server_username': 'freqtrader',
        'api_server_password': 'MoneyMachine',
    }
    mocker.patch('freqtrade.commands.build_config_commands.ask_user_config',
                 return_value=sample_selections)
    args = [
        "new-config",
        "--config",
        "coolconfig.json"
    ]
    start_new_config(get_args(args))

    assert log_has_re("Writing config to .*", caplog)
    assert wt_mock.call_count == 1
    assert unlink_mock.call_count == 1
    result = rapidjson.loads(wt_mock.call_args_list[0][0][0],
                             parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS)
    assert result['exchange']['name'] == exchange
    assert result['timeframe'] == '15m'


def test_start_new_config_exists(mocker, caplog):
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    mocker.patch('freqtrade.commands.build_config_commands.ask_user_overwrite', return_value=False)
    args = [
        "new-config",
        "--config",
        "coolconfig.json"
    ]
    with pytest.raises(OperationalException, match=r"Configuration .* already exists\."):
        start_new_config(get_args(args))


def test_ask_user_overwrite(mocker):
    """
    Once https://github.com/tmbo/questionary/issues/35 is implemented, improve this test.
    """
    prompt_mock = mocker.patch('freqtrade.commands.build_config_commands.prompt',
                               return_value={'overwrite': False})
    assert not ask_user_overwrite(Path('test.json'))
    assert prompt_mock.call_count == 1

    prompt_mock.reset_mock()
    prompt_mock = mocker.patch('freqtrade.commands.build_config_commands.prompt',
                               return_value={'overwrite': True})
    assert ask_user_overwrite(Path('test.json'))
    assert prompt_mock.call_count == 1


def test_ask_user_config(mocker):
    """
    Once https://github.com/tmbo/questionary/issues/35 is implemented, improve this test.
    """
    prompt_mock = mocker.patch('freqtrade.commands.build_config_commands.prompt',
                               return_value={'overwrite': False})
    answers = ask_user_config()
    assert isinstance(answers, dict)
    assert prompt_mock.call_count == 1

    prompt_mock = mocker.patch('freqtrade.commands.build_config_commands.prompt',
                               return_value={})

    with pytest.raises(OperationalException, match=r"User interrupted interactive questions\."):
        ask_user_config()
