# pragma pylint: disable=protected-access, invalid-name

"""
Unit test file for configuration.py
"""
import json

from copy import deepcopy
import pytest
from jsonschema import ValidationError

from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
import freqtrade.tests.conftest as tt  # test tools


def test_configuration_object() -> None:
    """
    Test the Constants object has the mandatory Constants
    """
    assert hasattr(Configuration, 'load_config')
    assert hasattr(Configuration, '_load_config_file')
    assert hasattr(Configuration, '_validate_config')
    assert hasattr(Configuration, '_load_common_config')
    assert hasattr(Configuration, '_load_backtesting_config')
    assert hasattr(Configuration, '_load_hyperopt_config')
    assert hasattr(Configuration, 'get_config')


def test_load_config_invalid_pair(default_conf, mocker) -> None:
    """
    Test the configuration validator with an invalid PAIR format
    """
    conf = deepcopy(default_conf)
    conf['exchange']['pair_whitelist'].append('BTC-ETH')

    with pytest.raises(ValidationError, match=r'.*does not match.*'):
        configuration = Configuration([])
        configuration._validate_config(conf)


def test_load_config_missing_attributes(default_conf, mocker) -> None:
    """
    Test the configuration validator with a missing attribute
    """
    conf = deepcopy(default_conf)
    conf.pop('exchange')

    with pytest.raises(ValidationError, match=r'.*\'exchange\' is a required property.*'):
        configuration = Configuration([])
        configuration._validate_config(conf)


def test_load_config_file(default_conf, mocker, caplog) -> None:
    """
    Test Configuration._load_config_file() method
    """
    file_mock = mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    configuration = Configuration([])
    validated_conf = configuration._load_config_file('somefile')
    assert file_mock.call_count == 1
    assert validated_conf.items() >= default_conf.items()
    assert 'internals' in validated_conf
    assert tt.log_has('Validating configuration ...', caplog.record_tuples)


def test_load_config(default_conf, mocker) -> None:
    """
    Test Configuration.load_config() without any cli params
    """
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    args = Arguments([], '').get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert 'strategy' in validated_conf
    assert validated_conf['strategy'] == 'default_strategy'
    assert 'dynamic_whitelist' not in validated_conf
    assert 'dry_run_db' not in validated_conf


def test_load_config_with_params(default_conf, mocker) -> None:
    """
    Test Configuration.load_config() with cli params used
    """
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    args = [
        '--dynamic-whitelist', '10',
        '--strategy', 'test_strategy',
        '--dry-run-db'
    ]
    args = Arguments(args, '').get_parsed_arg()

    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert 'dynamic_whitelist' in validated_conf
    assert validated_conf['dynamic_whitelist'] == 10
    assert 'strategy' in validated_conf
    assert validated_conf['strategy'] == 'test_strategy'
    assert 'dry_run_db' in validated_conf
    assert validated_conf['dry_run_db'] is True


def test_show_info(default_conf, mocker, caplog) -> None:
    """
    Test Configuration.show_info()
    """
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    args = [
        '--dynamic-whitelist', '10',
        '--strategy', 'test_strategy',
        '--dry-run-db'
    ]
    args = Arguments(args, '').get_parsed_arg()

    configuration = Configuration(args)
    configuration.get_config()

    assert tt.log_has(
        'Parameter --dynamic-whitelist detected. '
        'Using dynamically generated whitelist. '
        '(not applicable with Backtesting and Hyperopt)',
        caplog.record_tuples
    )

    assert tt.log_has(
        'Parameter --dry-run-db detected ...',
        caplog.record_tuples
    )

    assert tt.log_has(
        'Dry_run will use the DB file: "tradesv3.dry_run.sqlite"',
        caplog.record_tuples
    )

    # Test the Dry run condition
    configuration.config.update({'dry_run': False})
    configuration._load_common_config(configuration.config)
    assert tt.log_has(
        'Dry run is disabled. (--dry_run_db ignored)',
        caplog.record_tuples
    )


def test_setup_configuration_without_arguments(mocker, default_conf, caplog) -> None:
    """
    Test setup_configuration() function
    """
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    args = [
        '--config', 'config.json',
        '--strategy', 'default_strategy',
        'backtesting'
    ]

    args = Arguments(args, '').get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert tt.log_has(
        'Parameter --datadir detected: {} ...'.format(config['datadir']),
        caplog.record_tuples
    )
    assert 'ticker_interval' in config
    assert not tt.log_has('Parameter -i/--ticker-interval detected ...', caplog.record_tuples)

    assert 'live' not in config
    assert not tt.log_has('Parameter -l/--live detected ...', caplog.record_tuples)

    assert 'realistic_simulation' not in config
    assert not tt.log_has('Parameter --realistic-simulation detected ...', caplog.record_tuples)

    assert 'refresh_pairs' not in config
    assert not tt.log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog.record_tuples)

    assert 'timerange' not in config
    assert 'export' not in config


def test_setup_configuration_with_arguments(mocker, default_conf, caplog) -> None:
    """
    Test setup_configuration() function
    """
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    args = [
        '--config', 'config.json',
        '--strategy', 'default_strategy',
        '--datadir', '/foo/bar',
        'backtesting',
        '--ticker-interval', '1',
        '--live',
        '--realistic-simulation',
        '--refresh-pairs-cached',
        '--timerange', ':100',
        '--export', '/bar/foo'
    ]

    args = Arguments(args, '').get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert tt.log_has(
        'Parameter --datadir detected: {} ...'.format(config['datadir']),
        caplog.record_tuples
    )
    assert 'ticker_interval' in config
    assert tt.log_has('Parameter -i/--ticker-interval detected ...', caplog.record_tuples)
    assert tt.log_has(
        'Using ticker_interval: 1 ...',
        caplog.record_tuples
    )

    assert 'live' in config
    assert tt.log_has('Parameter -l/--live detected ...', caplog.record_tuples)

    assert 'realistic_simulation'in config
    assert tt.log_has('Parameter --realistic-simulation detected ...', caplog.record_tuples)
    assert tt.log_has('Using max_open_trades: 1 ...', caplog.record_tuples)

    assert 'refresh_pairs'in config
    assert tt.log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog.record_tuples)
    assert 'timerange' in config
    assert tt.log_has(
        'Parameter --timerange detected: {} ...'.format(config['timerange']),
        caplog.record_tuples
    )

    assert 'export' in config
    assert tt.log_has(
        'Parameter --export detected: {} ...'.format(config['export']),
        caplog.record_tuples
    )


def test_hyperopt_space_argument(mocker, default_conf, caplog) -> None:
    """
    Test setup_configuration() function
    """
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    args = [
        'hyperopt',
        '--spaces', 'all',
    ]

    args = Arguments(args, '').get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()
    assert 'spaces' in config
    assert config['spaces'] is 'all'
    assert tt.log_has('Parameter -s/--spaces detected: all', caplog.record_tuples)
