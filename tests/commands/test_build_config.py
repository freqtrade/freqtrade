import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from freqtrade.commands.build_config_commands import (ask_user_config,
                                                      start_new_config)
from freqtrade.exceptions import OperationalException
from tests.conftest import get_args, log_has_re


@pytest.mark.parametrize('exchange', ['bittrex', 'binance', 'kraken', 'ftx'])
def test_start_new_config(mocker, caplog, exchange):
    wt_mock = mocker.patch.object(Path, "write_text", MagicMock())
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    sample_selections = {
        'max_open_trades': 3,
        'stake_currency': 'USDT',
        'stake_amount': 100,
        'fiat_display_currency': 'EUR',
        'ticker_interval': '15m',
        'dry_run': True,
        'exchange_name': exchange,
        'exchange_key': 'sampleKey',
        'exchange_secret': 'Samplesecret',
        'telegram': False,
        'telegram_token': 'asdf1244',
        'telegram_chat_id': '1144444',
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
    result = json.loads(wt_mock.call_args_list[0][0][0])
    assert result['exchange']['name'] == exchange
    assert result['ticker_interval'] == '15m'


def test_start_new_config_exists(mocker, caplog):
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    args = [
        "new-config",
        "--config",
        "coolconfig.json"
    ]
    with pytest.raises(OperationalException, match=r"Configuration .* already exists\."):
        start_new_config(get_args(args))


def test_ask_user_config():
    # TODO: Implement me
    pass
    # assert ask_user_config()
