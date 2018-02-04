# pragma pylint: disable=protected-access, invalid-name, missing-docstring

"""
Unit test file for configuration.py
"""
import json

from copy import deepcopy
from unittest.mock import MagicMock
import pytest
from jsonschema import ValidationError

from freqtrade.arguments import Arguments
from freqtrade.configuration import Configuration
import freqtrade.tests.conftest as tt  # test tools


def test_configuration_object() -> None:
    """
    Test the Constants object has the mandatory Constants
    :return: None
    """
    assert hasattr(Configuration, '_load_config')
    assert hasattr(Configuration, '_load_config_file')
    assert hasattr(Configuration, '_validate_config')
    assert hasattr(Configuration, 'show_info')
    assert hasattr(Configuration, 'get_config')


def test_load_config_invalid_pair(default_conf, mocker) -> None:
    """
    Test the configuration validator with an invalid PAIR format
    :param default_conf: Configuration already read from a file (JSON format)
    :return: None
    """
    mocker.patch.multiple(
        'freqtrade.configuration.Configuration',
        _load_config=MagicMock(return_value=[])
    )
    conf = deepcopy(default_conf)
    conf['exchange']['pair_whitelist'].append('BTC-ETH')

    with pytest.raises(ValidationError, match=r'.*does not match.*'):
        configuration = Configuration([])
        configuration._validate_config(conf)


def test_load_config_missing_attributes(default_conf, mocker) -> None:
    """
    Test the configuration validator with a missing attribute
    :param default_conf: Configuration already read from a file (JSON format)
    :return: None
    """
    mocker.patch.multiple(
        'freqtrade.configuration.Configuration',
        _load_config=MagicMock(return_value=[])
    )
    conf = deepcopy(default_conf)
    conf.pop('exchange')

    with pytest.raises(ValidationError, match=r'.*\'exchange\' is a required property.*'):
        configuration = Configuration([])
        configuration._validate_config(conf)


def test_load_config_file(default_conf, mocker, caplog) -> None:
    """
    Test _load_config_file() method
    :return:
    """
    mocker.patch.multiple(
        'freqtrade.configuration.Configuration',
        _load_config=MagicMock(return_value=[])
    )
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
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    args = Arguments([], '').get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration._load_config()

    assert 'strategy' in validated_conf
    assert validated_conf['strategy'] == 'default_strategy'
    assert 'dynamic_whitelist' not in validated_conf
    assert 'dry_run_db' not in validated_conf


def test_load_config_with_params(default_conf, mocker) -> None:
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
    validated_conf = configuration._load_config()

    assert 'dynamic_whitelist' in validated_conf
    assert validated_conf['dynamic_whitelist'] == 10
    assert 'strategy' in validated_conf
    assert validated_conf['strategy'] == 'test_strategy'
    assert 'dry_run_db' in validated_conf
    assert validated_conf['dry_run_db'] is True


def test_show_info(default_conf, mocker, caplog) -> None:
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
    configuration.show_info()

    assert tt.log_has(
        'Using dynamically generated whitelist. (--dynamic-whitelist detected)',
        caplog.record_tuples
    )

    assert tt.log_has(
        'Dry_run will use the DB file: "tradesv3.dry_run.sqlite". '
        '(--dry_run_db detected)',
        caplog.record_tuples
    )

    # Test the Dry run condition
    configuration.config.update({'dry_run': False})
    configuration.show_info()
    assert tt.log_has(
        'Dry run is disabled. (--dry_run_db ignored)',
        caplog.record_tuples
    )
