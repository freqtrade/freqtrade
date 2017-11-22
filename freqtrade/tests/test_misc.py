# pragma pylint: disable=missing-docstring,C0103
import json
import os
import time
from argparse import Namespace
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from jsonschema import ValidationError

from freqtrade.misc import throttle, parse_args, start_backtesting, load_config


def test_throttle():

    def func():
        return 42

    start = time.time()
    result = throttle(func, 0.1)
    end = time.time()

    assert result == 42
    assert end - start > 0.1

    result = throttle(func, -1)
    assert result == 42


def test_parse_args_defaults():
    args = parse_args([])
    assert args is not None
    assert args.config == 'config.json'
    assert args.dynamic_whitelist is False
    assert args.loglevel == 20


def test_parse_args_invalid():
    with pytest.raises(SystemExit, match=r'2'):
        parse_args(['-c'])


def test_parse_args_config():
    args = parse_args(['-c', '/dev/null'])
    assert args is not None
    assert args.config == '/dev/null'

    args = parse_args(['--config', '/dev/null'])
    assert args is not None
    assert args.config == '/dev/null'


def test_parse_args_verbose():
    args = parse_args(['-v'])
    assert args is not None
    assert args.loglevel == 10


def test_parse_args_dynamic_whitelist():
    args = parse_args(['--dynamic-whitelist'])
    assert args is not None
    assert args.dynamic_whitelist is True


def test_parse_args_backtesting(mocker):
    backtesting_mock = mocker.patch('freqtrade.misc.start_backtesting', MagicMock())
    args = parse_args(['backtesting'])
    assert args is None
    assert backtesting_mock.call_count == 1

    call_args = backtesting_mock.call_args[0][0]
    assert call_args.config == 'config.json'
    assert call_args.live is False
    assert call_args.loglevel == 20
    assert call_args.subparser == 'backtesting'
    assert call_args.func is not None
    assert call_args.ticker_interval == 5


def test_parse_args_backtesting_invalid():
    with pytest.raises(SystemExit, match=r'2'):
        parse_args(['--ticker-interval'])

    with pytest.raises(SystemExit, match=r'2'):
        parse_args(['--ticker-interval', 'abc'])


def test_parse_args_backtesting_custom(mocker):
    backtesting_mock = mocker.patch('freqtrade.misc.start_backtesting', MagicMock())
    args = parse_args(['-c', 'test_conf.json', 'backtesting', '--live', '--ticker-interval', '1'])
    assert args is None
    assert backtesting_mock.call_count == 1

    call_args = backtesting_mock.call_args[0][0]
    assert call_args.config == 'test_conf.json'
    assert call_args.live is True
    assert call_args.loglevel == 20
    assert call_args.subparser == 'backtesting'
    assert call_args.func is not None
    assert call_args.ticker_interval == 1


def test_start_backtesting(mocker):
    pytest_mock = mocker.patch('pytest.main', MagicMock())
    env_mock = mocker.patch('os.environ', {})
    args = Namespace(
        config='config.json',
        live=True,
        loglevel=20,
        ticker_interval=1,
        limit_max_trades=True,
    )
    start_backtesting(args)
    assert env_mock == {
        'BACKTEST': 'true',
        'BACKTEST_LIVE': 'true',
        'BACKTEST_CONFIG': 'config.json',
        'BACKTEST_TICKER_INTERVAL': '1',
        'BACKTEST_LIMIT_MAX_TRADES': 'true',
    }
    assert pytest_mock.call_count == 1

    main_call_args = pytest_mock.call_args[0][0]
    assert main_call_args[0] == '-s'
    assert main_call_args[1].endswith(os.path.join('freqtrade', 'tests', 'test_backtesting.py'))


def test_load_config(default_conf, mocker):
    file_mock = mocker.patch('freqtrade.misc.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))
    validated_conf = load_config('somefile')
    assert file_mock.call_count == 1
    assert validated_conf.items() >= default_conf.items()


def test_load_config_invalid_pair(default_conf, mocker):
    conf = deepcopy(default_conf)
    conf['exchange']['pair_whitelist'].append('BTC-ETH')
    mocker.patch('freqtrade.misc.open', mocker.mock_open(read_data=json.dumps(conf)))
    with pytest.raises(ValidationError, match=r'.*does not match.*'):
        load_config('somefile')


def test_load_config_missing_attributes(default_conf, mocker):
    conf = deepcopy(default_conf)
    conf.pop('exchange')
    mocker.patch('freqtrade.misc.open', mocker.mock_open(read_data=json.dumps(conf)))
    with pytest.raises(ValidationError, match=r'.*\'exchange\' is a required property.*'):
        load_config('somefile')
