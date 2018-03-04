# pragma pylint: disable=missing-docstring, C0103

"""
Unit test file for arguments.py
"""

import argparse
import logging
import pytest

from freqtrade.arguments import Arguments


def test_arguments_object() -> None:
    """
    Test the Arguments object has the mandatory methods
    :return: None
    """
    assert hasattr(Arguments, 'get_parsed_arg')
    assert hasattr(Arguments, '_parse_args')
    assert hasattr(Arguments, 'parse_timerange')
    assert hasattr(Arguments, 'scripts_options')


# Parse common command-line-arguments. Used for all tools
def test_parse_args_none() -> None:
    arguments = Arguments([], '')
    assert isinstance(arguments, Arguments)
    assert isinstance(arguments.parser, argparse.ArgumentParser)
    assert isinstance(arguments.parser, argparse.ArgumentParser)


def test_parse_args_defaults() -> None:
    args = Arguments([], '').get_parsed_arg()
    assert args.config == 'config.json'
    assert args.dynamic_whitelist is None
    assert args.loglevel == logging.INFO


def test_parse_args_config() -> None:
    args = Arguments(['-c', '/dev/null'], '').get_parsed_arg()
    assert args.config == '/dev/null'

    args = Arguments(['--config', '/dev/null'], '').get_parsed_arg()
    assert args.config == '/dev/null'


def test_parse_args_verbose() -> None:
    args = Arguments(['-v'], '').get_parsed_arg()
    assert args.loglevel == logging.DEBUG

    args = Arguments(['--verbose'], '').get_parsed_arg()
    assert args.loglevel == logging.DEBUG


def test_scripts_options() -> None:
    arguments = Arguments(['-p', 'BTC_ETH'], '')
    arguments.scripts_options()
    args = arguments.get_parsed_arg()
    assert args.pair == 'BTC_ETH'


def test_parse_args_version() -> None:
    with pytest.raises(SystemExit, match=r'0'):
        Arguments(['--version'], '').get_parsed_arg()


def test_parse_args_invalid() -> None:
    with pytest.raises(SystemExit, match=r'2'):
        Arguments(['-c'], '').get_parsed_arg()


def test_parse_args_dynamic_whitelist() -> None:
    args = Arguments(['--dynamic-whitelist'], '').get_parsed_arg()
    assert args.dynamic_whitelist == 20


def test_parse_args_dynamic_whitelist_10() -> None:
    args = Arguments(['--dynamic-whitelist', '10'], '').get_parsed_arg()
    assert args.dynamic_whitelist == 10


def test_parse_args_dynamic_whitelist_invalid_values() -> None:
    with pytest.raises(SystemExit, match=r'2'):
        Arguments(['--dynamic-whitelist', 'abc'], '').get_parsed_arg()


def test_parse_timerange_incorrect() -> None:
    assert ((None, 'line'), None, -200) == Arguments.parse_timerange('-200')
    assert (('line', None), 200, None) == Arguments.parse_timerange('200-')
    with pytest.raises(Exception, match=r'Incorrect syntax.*'):
        Arguments.parse_timerange('-')


def test_parse_args_backtesting_invalid() -> None:
    with pytest.raises(SystemExit, match=r'2'):
        Arguments(['backtesting --ticker-interval'], '').get_parsed_arg()

    with pytest.raises(SystemExit, match=r'2'):
        Arguments(['backtesting --ticker-interval', 'abc'], '').get_parsed_arg()


def test_parse_args_backtesting_custom() -> None:
    args = [
        '-c', 'test_conf.json',
        'backtesting',
        '--live',
        '--ticker-interval', '1',
        '--refresh-pairs-cached']
    call_args = Arguments(args, '').get_parsed_arg()
    assert call_args.config == 'test_conf.json'
    assert call_args.live is True
    assert call_args.loglevel == logging.INFO
    assert call_args.subparser == 'backtesting'
    assert call_args.func is not None
    assert call_args.ticker_interval == 1
    assert call_args.refresh_pairs is True


def test_parse_args_hyperopt_custom() -> None:
    args = [
        '-c', 'test_conf.json',
        'hyperopt',
        '--epochs', '20',
        '--spaces', 'buy'
    ]
    call_args = Arguments(args, '').get_parsed_arg()
    assert call_args.config == 'test_conf.json'
    assert call_args.epochs == 20
    assert call_args.loglevel == logging.INFO
    assert call_args.subparser == 'hyperopt'
    assert call_args.spaces == ['buy']
    assert call_args.func is not None
