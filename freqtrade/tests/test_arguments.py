# pragma pylint: disable=missing-docstring, C0103

import argparse

import pytest

from freqtrade.arguments import Arguments, TimeRange


# Parse common command-line-arguments. Used for all tools
def test_parse_args_none() -> None:
    arguments = Arguments([], '')
    assert isinstance(arguments, Arguments)
    assert isinstance(arguments.parser, argparse.ArgumentParser)


def test_parse_args_defaults() -> None:
    args = Arguments([], '').get_parsed_arg()
    assert args.config == 'config.json'
    assert args.dynamic_whitelist is None
    assert args.loglevel == 0


def test_parse_args_config() -> None:
    args = Arguments(['-c', '/dev/null'], '').get_parsed_arg()
    assert args.config == '/dev/null'

    args = Arguments(['--config', '/dev/null'], '').get_parsed_arg()
    assert args.config == '/dev/null'


def test_parse_args_db_url() -> None:
    args = Arguments(['--db-url', 'sqlite:///test.sqlite'], '').get_parsed_arg()
    assert args.db_url == 'sqlite:///test.sqlite'


def test_parse_args_verbose() -> None:
    args = Arguments(['-v'], '').get_parsed_arg()
    assert args.loglevel == 1

    args = Arguments(['--verbose'], '').get_parsed_arg()
    assert args.loglevel == 1


def test_scripts_options() -> None:
    arguments = Arguments(['-p', 'ETH/BTC'], '')
    arguments.scripts_options()
    args = arguments.get_parsed_arg()
    assert args.pair == 'ETH/BTC'


def test_parse_args_version() -> None:
    with pytest.raises(SystemExit, match=r'0'):
        Arguments(['--version'], '').get_parsed_arg()


def test_parse_args_invalid() -> None:
    with pytest.raises(SystemExit, match=r'2'):
        Arguments(['-c'], '').get_parsed_arg()


def test_parse_args_strategy() -> None:
    args = Arguments(['--strategy', 'SomeStrategy'], '').get_parsed_arg()
    assert args.strategy == 'SomeStrategy'


def test_parse_args_strategy_invalid() -> None:
    with pytest.raises(SystemExit, match=r'2'):
        Arguments(['--strategy'], '').get_parsed_arg()


def test_parse_args_strategy_path() -> None:
    args = Arguments(['--strategy-path', '/some/path'], '').get_parsed_arg()
    assert args.strategy_path == '/some/path'


def test_parse_args_strategy_path_invalid() -> None:
    with pytest.raises(SystemExit, match=r'2'):
        Arguments(['--strategy-path'], '').get_parsed_arg()


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
    assert TimeRange(None, 'line', 0, -200) == Arguments.parse_timerange('-200')
    assert TimeRange('line', None, 200, 0) == Arguments.parse_timerange('200-')
    assert TimeRange('index', 'index', 200, 500) == Arguments.parse_timerange('200-500')

    assert TimeRange('date', None, 1274486400, 0) == Arguments.parse_timerange('20100522-')
    assert TimeRange(None, 'date', 0, 1274486400) == Arguments.parse_timerange('-20100522')
    timerange = Arguments.parse_timerange('20100522-20150730')
    assert timerange == TimeRange('date', 'date', 1274486400, 1438214400)

    # Added test for unix timestamp - BTC genesis date
    assert TimeRange('date', None, 1231006505, 0) == Arguments.parse_timerange('1231006505-')
    assert TimeRange(None, 'date', 0, 1233360000) == Arguments.parse_timerange('-1233360000')
    timerange = Arguments.parse_timerange('1231006505-1233360000')
    assert TimeRange('date', 'date', 1231006505, 1233360000) == timerange

    # TODO: Find solution for the following case (passing timestamp in ms)
    timerange = Arguments.parse_timerange('1231006505000-1233360000000')
    assert TimeRange('date', 'date', 1231006505, 1233360000) != timerange

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
        '--ticker-interval', '1m',
        '--refresh-pairs-cached']
    call_args = Arguments(args, '').get_parsed_arg()
    assert call_args.config == 'test_conf.json'
    assert call_args.live is True
    assert call_args.loglevel == 0
    assert call_args.subparser == 'backtesting'
    assert call_args.func is not None
    assert call_args.ticker_interval == '1m'
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
    assert call_args.loglevel == 0
    assert call_args.subparser == 'hyperopt'
    assert call_args.spaces == ['buy']
    assert call_args.func is not None


def test_testdata_dl_options() -> None:
    args = [
        '--pairs-file', 'file_with_pairs',
        '--export', 'export/folder',
        '--days', '30',
        '--exchange', 'binance'
    ]
    arguments = Arguments(args, '')
    arguments.testdata_dl_options()
    args = arguments.parse_args()
    assert args.pairs_file == 'file_with_pairs'
    assert args.export == 'export/folder'
    assert args.days == 30
    assert args.exchange == 'binance'
