# pragma pylint: disable=missing-docstring, protected-access, invalid-name

import json
from argparse import Namespace
import logging
from unittest.mock import MagicMock

import pytest
from jsonschema import validate, ValidationError

from freqtrade import constants
from freqtrade import OperationalException
from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration, set_loggers
from freqtrade.constants import DEFAULT_DB_DRYRUN_URL, DEFAULT_DB_PROD_URL
from freqtrade.tests.conftest import log_has


def test_load_config_invalid_pair(default_conf) -> None:
    default_conf['exchange']['pair_whitelist'].append('ETH-BTC')

    with pytest.raises(ValidationError, match=r'.*does not match.*'):
        configuration = Configuration(Namespace())
        configuration._validate_config(default_conf)


def test_load_config_missing_attributes(default_conf) -> None:
    default_conf.pop('exchange')

    with pytest.raises(ValidationError, match=r'.*\'exchange\' is a required property.*'):
        configuration = Configuration(Namespace())
        configuration._validate_config(default_conf)


def test_load_config_incorrect_stake_amount(default_conf) -> None:
    default_conf['stake_amount'] = 'fake'

    with pytest.raises(ValidationError, match=r'.*\'fake\' does not match \'unlimited\'.*'):
        configuration = Configuration(Namespace())
        configuration._validate_config(default_conf)


def test_load_config_file(default_conf, mocker, caplog) -> None:
    file_mock = mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    configuration = Configuration(Namespace())
    validated_conf = configuration._load_config_file('somefile')
    assert file_mock.call_count == 1
    assert validated_conf.items() >= default_conf.items()
    assert 'internals' in validated_conf
    assert log_has('Validating configuration ...', caplog.record_tuples)


def test_load_config_max_open_trades_zero(default_conf, mocker, caplog) -> None:
    default_conf['max_open_trades'] = 0
    file_mock = mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    Configuration(Namespace())._load_config_file('somefile')
    assert file_mock.call_count == 1
    assert log_has('Validating configuration ...', caplog.record_tuples)


def test_load_config_file_exception(mocker) -> None:
    mocker.patch(
        'freqtrade.configuration.open',
        MagicMock(side_effect=FileNotFoundError('File not found'))
    )
    configuration = Configuration(Namespace())

    with pytest.raises(OperationalException, match=r'.*Config file "somefile" not found!*'):
        configuration._load_config_file('somefile')


def test_load_config(default_conf, mocker) -> None:
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    args = Arguments([], '').get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf.get('strategy') == 'DefaultStrategy'
    assert validated_conf.get('strategy_path') is None
    assert 'dynamic_whitelist' not in validated_conf


def test_load_config_with_params(default_conf, mocker) -> None:
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))
    arglist = [
        '--dynamic-whitelist', '10',
        '--strategy', 'TestStrategy',
        '--strategy-path', '/some/path',
        '--db-url', 'sqlite:///someurl',
    ]
    args = Arguments(arglist, '').get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf.get('dynamic_whitelist') == 10
    assert validated_conf.get('strategy') == 'TestStrategy'
    assert validated_conf.get('strategy_path') == '/some/path'
    assert validated_conf.get('db_url') == 'sqlite:///someurl'

    conf = default_conf.copy()
    conf["dry_run"] = False
    del conf["db_url"]
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(conf)
    ))

    arglist = [
        '--dynamic-whitelist', '10',
        '--strategy', 'TestStrategy',
        '--strategy-path', '/some/path'
    ]
    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get('db_url') == DEFAULT_DB_PROD_URL

    # Test dry=run with ProdURL
    conf = default_conf.copy()
    conf["dry_run"] = True
    conf["db_url"] = DEFAULT_DB_PROD_URL
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(conf)
    ))

    arglist = [
        '--dynamic-whitelist', '10',
        '--strategy', 'TestStrategy',
        '--strategy-path', '/some/path'
    ]
    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get('db_url') == DEFAULT_DB_DRYRUN_URL


def test_load_custom_strategy(default_conf, mocker) -> None:
    default_conf.update({
        'strategy': 'CustomStrategy',
        'strategy_path': '/tmp/strategies',
    })
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    args = Arguments([], '').get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf.get('strategy') == 'CustomStrategy'
    assert validated_conf.get('strategy_path') == '/tmp/strategies'


def test_show_info(default_conf, mocker, caplog) -> None:
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))
    arglist = [
        '--dynamic-whitelist', '10',
        '--strategy', 'TestStrategy',
        '--db-url', 'sqlite:///tmp/testdb',
    ]
    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    configuration.get_config()

    assert log_has(
        'Parameter --dynamic-whitelist detected. '
        'Using dynamically generated whitelist. '
        '(not applicable with Backtesting and Hyperopt)',
        caplog.record_tuples
    )
    assert log_has('Using DB: "sqlite:///tmp/testdb"', caplog.record_tuples)
    assert log_has('Dry run is enabled', caplog.record_tuples)


def test_setup_configuration_without_arguments(mocker, default_conf, caplog) -> None:
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))
    arglist = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        'backtesting'
    ]

    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has(
        'Using data folder: {} ...'.format(config['datadir']),
        caplog.record_tuples
    )
    assert 'ticker_interval' in config
    assert not log_has('Parameter -i/--ticker-interval detected ...', caplog.record_tuples)

    assert 'live' not in config
    assert not log_has('Parameter -l/--live detected ...', caplog.record_tuples)

    assert 'position_stacking' not in config
    assert not log_has('Parameter --enable-position-stacking detected ...', caplog.record_tuples)

    assert 'refresh_pairs' not in config
    assert not log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog.record_tuples)

    assert 'timerange' not in config
    assert 'export' not in config


def test_setup_configuration_with_arguments(mocker, default_conf, caplog) -> None:
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    arglist = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        '--datadir', '/foo/bar',
        'backtesting',
        '--ticker-interval', '1m',
        '--live',
        '--enable-position-stacking',
        '--disable-max-market-positions',
        '--refresh-pairs-cached',
        '--timerange', ':100',
        '--export', '/bar/foo'
    ]

    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has(
        'Using data folder: {} ...'.format(config['datadir']),
        caplog.record_tuples
    )
    assert 'ticker_interval' in config
    assert log_has('Parameter -i/--ticker-interval detected ...', caplog.record_tuples)
    assert log_has(
        'Using ticker_interval: 1m ...',
        caplog.record_tuples
    )

    assert 'live' in config
    assert log_has('Parameter -l/--live detected ...', caplog.record_tuples)

    assert 'position_stacking'in config
    assert log_has('Parameter --enable-position-stacking detected ...', caplog.record_tuples)

    assert 'use_max_market_positions' in config
    assert log_has('Parameter --disable-max-market-positions detected ...', caplog.record_tuples)
    assert log_has('max_open_trades set to unlimited ...', caplog.record_tuples)

    assert 'refresh_pairs'in config
    assert log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog.record_tuples)
    assert 'timerange' in config
    assert log_has(
        'Parameter --timerange detected: {} ...'.format(config['timerange']),
        caplog.record_tuples
    )

    assert 'export' in config
    assert log_has(
        'Parameter --export detected: {} ...'.format(config['export']),
        caplog.record_tuples
    )


def test_setup_configuration_with_stratlist(mocker, default_conf, caplog) -> None:
    """
    Test setup_configuration() function
    """
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    arglist = [
        '--config', 'config.json',
        'backtesting',
        '--ticker-interval', '1m',
        '--export', '/bar/foo',
        '--strategy-list',
        'DefaultStrategy',
        'TestStrategy'
    ]

    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has(
        'Using data folder: {} ...'.format(config['datadir']),
        caplog.record_tuples
    )
    assert 'ticker_interval' in config
    assert log_has('Parameter -i/--ticker-interval detected ...', caplog.record_tuples)
    assert log_has(
        'Using ticker_interval: 1m ...',
        caplog.record_tuples
    )

    assert 'strategy_list' in config
    assert log_has('Using strategy list of 2 Strategies', caplog.record_tuples)

    assert 'position_stacking' not in config

    assert 'use_max_market_positions' not in config

    assert 'timerange' not in config

    assert 'export' in config
    assert log_has(
        'Parameter --export detected: {} ...'.format(config['export']),
        caplog.record_tuples
    )


def test_hyperopt_with_arguments(mocker, default_conf, caplog) -> None:
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))
    arglist = [
        'hyperopt',
        '--epochs', '10',
        '--spaces', 'all',
    ]
    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()

    assert 'epochs' in config
    assert int(config['epochs']) == 10
    assert log_has('Parameter --epochs detected ...', caplog.record_tuples)
    assert log_has('Will run Hyperopt with for 10 epochs ...', caplog.record_tuples)

    assert 'spaces' in config
    assert config['spaces'] == ['all']
    assert log_has('Parameter -s/--spaces detected: [\'all\']', caplog.record_tuples)


def test_check_exchange(default_conf) -> None:
    configuration = Configuration(Namespace())

    # Test a valid exchange
    default_conf.get('exchange').update({'name': 'BITTREX'})
    assert configuration.check_exchange(default_conf)

    # Test a valid exchange
    default_conf.get('exchange').update({'name': 'binance'})
    assert configuration.check_exchange(default_conf)

    # Test a invalid exchange
    default_conf.get('exchange').update({'name': 'unknown_exchange'})
    configuration.config = default_conf

    with pytest.raises(
        OperationalException,
        match=r'.*Exchange "unknown_exchange" not supported.*'
    ):
        configuration.check_exchange(default_conf)


def test_cli_verbose_with_params(default_conf, mocker, caplog) -> None:
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)))
    # Prevent setting loggers
    mocker.patch('freqtrade.configuration.set_loggers', MagicMock)
    arglist = ['-vvv']
    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf.get('verbosity') == 3
    assert log_has('Verbosity set to 3', caplog.record_tuples)


def test_set_loggers() -> None:
    # Reset Logging to Debug, otherwise this fails randomly as it's set globally
    logging.getLogger('requests').setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.DEBUG)
    logging.getLogger('ccxt.base.exchange').setLevel(logging.DEBUG)
    logging.getLogger('telegram').setLevel(logging.DEBUG)

    previous_value1 = logging.getLogger('requests').level
    previous_value2 = logging.getLogger('ccxt.base.exchange').level
    previous_value3 = logging.getLogger('telegram').level

    set_loggers()

    value1 = logging.getLogger('requests').level
    assert previous_value1 is not value1
    assert value1 is logging.INFO

    value2 = logging.getLogger('ccxt.base.exchange').level
    assert previous_value2 is not value2
    assert value2 is logging.INFO

    value3 = logging.getLogger('telegram').level
    assert previous_value3 is not value3
    assert value3 is logging.INFO

    set_loggers(log_level=2)

    assert logging.getLogger('requests').level is logging.DEBUG
    assert logging.getLogger('ccxt.base.exchange').level is logging.INFO
    assert logging.getLogger('telegram').level is logging.INFO

    set_loggers(log_level=3)

    assert logging.getLogger('requests').level is logging.DEBUG
    assert logging.getLogger('ccxt.base.exchange').level is logging.DEBUG
    assert logging.getLogger('telegram').level is logging.INFO


def test_validate_default_conf(default_conf) -> None:
    validate(default_conf, constants.CONF_SCHEMA)
