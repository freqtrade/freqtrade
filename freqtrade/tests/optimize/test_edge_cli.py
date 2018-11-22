# pragma pylint: disable=missing-docstring, C0103, C0330
# pragma pylint: disable=protected-access, too-many-lines, invalid-name, too-many-arguments

from unittest.mock import MagicMock
import json
from typing import List
from freqtrade.edge import PairInfo
from freqtrade.arguments import Arguments
from freqtrade.optimize.edge_cli import (EdgeCli, setup_configuration, start)
from freqtrade.tests.conftest import log_has, patch_exchange


def get_args(args) -> List[str]:
    return Arguments(args, '').get_parsed_arg()


def test_setup_configuration_without_arguments(mocker, default_conf, caplog) -> None:
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    args = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        'edge'
    ]

    config = setup_configuration(get_args(args))
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

    assert 'refresh_pairs' not in config
    assert not log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog.record_tuples)

    assert 'timerange' not in config
    assert 'stoploss_range' not in config


def test_setup_configuration_with_arguments(mocker, edge_conf, caplog) -> None:
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(edge_conf)
    ))

    args = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        '--datadir', '/foo/bar',
        'edge',
        '--ticker-interval', '1m',
        '--refresh-pairs-cached',
        '--timerange', ':100',
        '--stoplosses=-0.01,-0.10,-0.001'
    ]

    config = setup_configuration(get_args(args))
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

    assert 'refresh_pairs' in config
    assert log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog.record_tuples)
    assert 'timerange' in config
    assert log_has(
        'Parameter --timerange detected: {} ...'.format(config['timerange']),
        caplog.record_tuples
    )


def test_start(mocker, fee, edge_conf, caplog) -> None:
    start_mock = MagicMock()
    mocker.patch('freqtrade.exchange.Exchange.get_fee', fee)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.edge_cli.EdgeCli.start', start_mock)
    mocker.patch('freqtrade.configuration.open', mocker.mock_open(
        read_data=json.dumps(edge_conf)
    ))
    args = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        'edge'
    ]
    args = get_args(args)
    start(args)
    assert log_has(
        'Starting freqtrade in Edge mode',
        caplog.record_tuples
    )
    assert start_mock.call_count == 1


def test_edge_init(mocker, edge_conf) -> None:
    patch_exchange(mocker)
    edge_cli = EdgeCli(edge_conf)
    assert edge_cli.config == edge_conf
    assert callable(edge_cli.edge.calculate)


def test_generate_edge_table(edge_conf, mocker):
    patch_exchange(mocker)
    edge_cli = EdgeCli(edge_conf)

    results = {}
    results['ETH/BTC'] = PairInfo(-0.01, 0.60, 2, 1, 3, 10, 60)

    assert edge_cli._generate_edge_table(results).count(':|') == 7
    assert edge_cli._generate_edge_table(results).count('| ETH/BTC |') == 1
    assert edge_cli._generate_edge_table(results).count(
        '|   risk reward ratio |   required risk reward |   expectancy |') == 1
