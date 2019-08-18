# pragma pylint: disable=missing-docstring, protected-access, invalid-name
import json
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from jsonschema import Draft4Validator, ValidationError, validate

from freqtrade import OperationalException, constants
from freqtrade.configuration import Arguments, Configuration, validate_config_consistency
from freqtrade.configuration.check_exchange import check_exchange
from freqtrade.configuration.config_validation import validate_config_schema
from freqtrade.configuration.create_datadir import create_datadir
from freqtrade.configuration.load_config import load_config_file
from freqtrade.constants import DEFAULT_DB_DRYRUN_URL, DEFAULT_DB_PROD_URL
from freqtrade.loggers import _set_loggers
from freqtrade.state import RunMode
from freqtrade.tests.conftest import (log_has, log_has_re,
                                      patched_configuration_load_config_file)


@pytest.fixture(scope="function")
def all_conf():
    config_file = Path(__file__).parents[2] / "config_full.json.example"
    print(config_file)
    conf = load_config_file(str(config_file))
    return conf


def test_load_config_invalid_pair(default_conf) -> None:
    default_conf['exchange']['pair_whitelist'].append('ETH-BTC')

    with pytest.raises(ValidationError, match=r'.*does not match.*'):
        validate_config_schema(default_conf)


def test_load_config_missing_attributes(default_conf) -> None:
    default_conf.pop('exchange')

    with pytest.raises(ValidationError, match=r'.*\'exchange\' is a required property.*'):
        validate_config_schema(default_conf)


def test_load_config_incorrect_stake_amount(default_conf) -> None:
    default_conf['stake_amount'] = 'fake'

    with pytest.raises(ValidationError, match=r'.*\'fake\' does not match \'unlimited\'.*'):
        validate_config_schema(default_conf)


def test_load_config_file(default_conf, mocker, caplog) -> None:
    file_mock = mocker.patch('freqtrade.configuration.load_config.open', mocker.mock_open(
        read_data=json.dumps(default_conf)
    ))

    validated_conf = load_config_file('somefile')
    assert file_mock.call_count == 1
    assert validated_conf.items() >= default_conf.items()


def test__args_to_config(caplog):

    arg_list = ['--strategy-path', 'TestTest']
    args = Arguments(arg_list, '').get_parsed_arg()
    configuration = Configuration(args)
    config = {}
    with warnings.catch_warnings(record=True) as w:
        # No warnings ...
        configuration._args_to_config(config, argname="strategy_path", logstring="DeadBeef")
        assert len(w) == 0
        assert log_has("DeadBeef", caplog)
        assert config['strategy_path'] == "TestTest"

    configuration = Configuration(args)
    config = {}
    with warnings.catch_warnings(record=True) as w:
        # Deprecation warnings!
        configuration._args_to_config(config, argname="strategy_path", logstring="DeadBeef",
                                      deprecated_msg="Going away soon!")
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "DEPRECATED: Going away soon!" in str(w[-1].message)
        assert log_has("DeadBeef", caplog)
        assert config['strategy_path'] == "TestTest"


def test_load_config_max_open_trades_zero(default_conf, mocker, caplog) -> None:
    default_conf['max_open_trades'] = 0
    patched_configuration_load_config_file(mocker, default_conf)

    args = Arguments([], '').get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf['max_open_trades'] == 0
    assert 'internals' in validated_conf
    assert log_has('Validating configuration ...', caplog)


def test_load_config_combine_dicts(default_conf, mocker, caplog) -> None:
    conf1 = deepcopy(default_conf)
    conf2 = deepcopy(default_conf)
    del conf1['exchange']['key']
    del conf1['exchange']['secret']
    del conf2['exchange']['name']
    conf2['exchange']['pair_whitelist'] += ['NANO/BTC']

    config_files = [conf1, conf2]

    configsmock = MagicMock(side_effect=config_files)
    mocker.patch(
        'freqtrade.configuration.configuration.load_config_file',
        configsmock
    )

    arg_list = ['-c', 'test_conf.json', '--config', 'test2_conf.json', ]
    args = Arguments(arg_list, '').get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    exchange_conf = default_conf['exchange']
    assert validated_conf['exchange']['name'] == exchange_conf['name']
    assert validated_conf['exchange']['key'] == exchange_conf['key']
    assert validated_conf['exchange']['secret'] == exchange_conf['secret']
    assert validated_conf['exchange']['pair_whitelist'] != conf1['exchange']['pair_whitelist']
    assert validated_conf['exchange']['pair_whitelist'] == conf2['exchange']['pair_whitelist']

    assert 'internals' in validated_conf
    assert log_has('Validating configuration ...', caplog)


def test_from_config(default_conf, mocker, caplog) -> None:
    conf1 = deepcopy(default_conf)
    conf2 = deepcopy(default_conf)
    del conf1['exchange']['key']
    del conf1['exchange']['secret']
    del conf2['exchange']['name']
    conf2['exchange']['pair_whitelist'] += ['NANO/BTC']
    conf2['fiat_display_currency'] = "EUR"
    config_files = [conf1, conf2]

    configsmock = MagicMock(side_effect=config_files)
    mocker.patch(
        'freqtrade.configuration.configuration.load_config_file',
        configsmock
    )

    validated_conf = Configuration.from_files(['test_conf.json', 'test2_conf.json'])

    exchange_conf = default_conf['exchange']
    assert validated_conf['exchange']['name'] == exchange_conf['name']
    assert validated_conf['exchange']['key'] == exchange_conf['key']
    assert validated_conf['exchange']['secret'] == exchange_conf['secret']
    assert validated_conf['exchange']['pair_whitelist'] != conf1['exchange']['pair_whitelist']
    assert validated_conf['exchange']['pair_whitelist'] == conf2['exchange']['pair_whitelist']
    assert validated_conf['fiat_display_currency'] == "EUR"
    assert 'internals' in validated_conf
    assert log_has('Validating configuration ...', caplog)


def test_load_config_max_open_trades_minus_one(default_conf, mocker, caplog) -> None:
    default_conf['max_open_trades'] = -1
    patched_configuration_load_config_file(mocker, default_conf)

    args = Arguments([], '').get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf['max_open_trades'] > 999999999
    assert validated_conf['max_open_trades'] == float('inf')
    assert log_has('Validating configuration ...', caplog)
    assert "runmode" in validated_conf
    assert validated_conf['runmode'] == RunMode.DRY_RUN


def test_load_config_file_exception(mocker) -> None:
    mocker.patch(
        'freqtrade.configuration.configuration.open',
        MagicMock(side_effect=FileNotFoundError('File not found'))
    )

    with pytest.raises(OperationalException, match=r'.*Config file "somefile" not found!*'):
        load_config_file('somefile')


def test_load_config(default_conf, mocker) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    args = Arguments([], '').get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf.get('strategy') == 'DefaultStrategy'
    assert validated_conf.get('strategy_path') is None
    assert 'edge' not in validated_conf


def test_load_config_with_params(default_conf, mocker) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    arglist = [
        '--strategy', 'TestStrategy',
        '--strategy-path', '/some/path',
        '--db-url', 'sqlite:///someurl',
    ]
    args = Arguments(arglist, '').get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf.get('strategy') == 'TestStrategy'
    assert validated_conf.get('strategy_path') == '/some/path'
    assert validated_conf.get('db_url') == 'sqlite:///someurl'

    # Test conf provided db_url prod
    conf = default_conf.copy()
    conf["dry_run"] = False
    conf["db_url"] = "sqlite:///path/to/db.sqlite"
    patched_configuration_load_config_file(mocker, conf)

    arglist = [
        '--strategy', 'TestStrategy',
        '--strategy-path', '/some/path'
    ]
    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get('db_url') == "sqlite:///path/to/db.sqlite"

    # Test conf provided db_url dry_run
    conf = default_conf.copy()
    conf["dry_run"] = True
    conf["db_url"] = "sqlite:///path/to/db.sqlite"
    patched_configuration_load_config_file(mocker, conf)

    arglist = [
        '--strategy', 'TestStrategy',
        '--strategy-path', '/some/path'
    ]
    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get('db_url') == "sqlite:///path/to/db.sqlite"

    # Test args provided db_url prod
    conf = default_conf.copy()
    conf["dry_run"] = False
    del conf["db_url"]
    patched_configuration_load_config_file(mocker, conf)

    arglist = [
        '--strategy', 'TestStrategy',
        '--strategy-path', '/some/path'
    ]
    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    validated_conf = configuration.load_config()
    assert validated_conf.get('db_url') == DEFAULT_DB_PROD_URL
    assert "runmode" in validated_conf
    assert validated_conf['runmode'] == RunMode.LIVE

    # Test args provided db_url dry_run
    conf = default_conf.copy()
    conf["dry_run"] = True
    conf["db_url"] = DEFAULT_DB_PROD_URL
    patched_configuration_load_config_file(mocker, conf)

    arglist = [
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
    patched_configuration_load_config_file(mocker, default_conf)

    args = Arguments([], '').get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf.get('strategy') == 'CustomStrategy'
    assert validated_conf.get('strategy_path') == '/tmp/strategies'


def test_show_info(default_conf, mocker, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    arglist = [
        '--strategy', 'TestStrategy',
        '--db-url', 'sqlite:///tmp/testdb',
    ]
    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    configuration.get_config()

    assert log_has('Using DB: "sqlite:///tmp/testdb"', caplog)
    assert log_has('Dry run is enabled', caplog)


def test_setup_configuration_without_arguments(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

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
    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'ticker_interval' in config
    assert not log_has('Parameter -i/--ticker-interval detected ...', caplog)

    assert 'position_stacking' not in config
    assert not log_has('Parameter --enable-position-stacking detected ...', caplog)

    assert 'refresh_pairs' not in config
    assert not log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog)

    assert 'timerange' not in config
    assert 'export' not in config


def test_setup_configuration_with_arguments(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch(
        'freqtrade.configuration.configuration.create_datadir',
        lambda c, x: x
    )

    arglist = [
        '--config', 'config.json',
        '--strategy', 'DefaultStrategy',
        '--datadir', '/foo/bar',
        'backtesting',
        '--ticker-interval', '1m',
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
    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'ticker_interval' in config
    assert log_has('Parameter -i/--ticker-interval detected ... Using ticker_interval: 1m ...',
                   caplog)

    assert 'position_stacking'in config
    assert log_has('Parameter --enable-position-stacking detected ...', caplog)

    assert 'use_max_market_positions' in config
    assert log_has('Parameter --disable-max-market-positions detected ...', caplog)
    assert log_has('max_open_trades set to unlimited ...', caplog)

    assert 'refresh_pairs'in config
    assert log_has('Parameter -r/--refresh-pairs-cached detected ...', caplog)
    assert 'timerange' in config
    assert log_has('Parameter --timerange detected: {} ...'.format(config['timerange']), caplog)

    assert 'export' in config
    assert log_has('Parameter --export detected: {} ...'.format(config['export']), caplog)


def test_setup_configuration_with_stratlist(mocker, default_conf, caplog) -> None:
    """
    Test setup_configuration() function
    """
    patched_configuration_load_config_file(mocker, default_conf)

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

    configuration = Configuration(args, RunMode.BACKTEST)
    config = configuration.get_config()
    assert config['runmode'] == RunMode.BACKTEST
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'ticker_interval' in config
    assert log_has('Parameter -i/--ticker-interval detected ... Using ticker_interval: 1m ...',
                   caplog)

    assert 'strategy_list' in config
    assert log_has('Using strategy list of 2 Strategies', caplog)

    assert 'position_stacking' not in config

    assert 'use_max_market_positions' not in config

    assert 'timerange' not in config

    assert 'export' in config
    assert log_has('Parameter --export detected: {} ...'.format(config['export']), caplog)


def test_hyperopt_with_arguments(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    arglist = [
        'hyperopt',
        '--epochs', '10',
        '--spaces', 'all',
    ]
    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args, RunMode.HYPEROPT)
    config = configuration.get_config()

    assert 'epochs' in config
    assert int(config['epochs']) == 10
    assert log_has('Parameter --epochs detected ... Will run Hyperopt with for 10 epochs ...',
                   caplog)

    assert 'spaces' in config
    assert config['spaces'] == ['all']
    assert log_has('Parameter -s/--spaces detected: [\'all\']', caplog)
    assert "runmode" in config
    assert config['runmode'] == RunMode.HYPEROPT


def test_check_exchange(default_conf, caplog) -> None:
    # Test an officially supported by Freqtrade team exchange
    default_conf.get('exchange').update({'name': 'BITTREX'})
    assert check_exchange(default_conf)
    assert log_has_re(r"Exchange .* is officially supported by the Freqtrade development team\.",
                      caplog)
    caplog.clear()

    # Test an officially supported by Freqtrade team exchange
    default_conf.get('exchange').update({'name': 'binance'})
    assert check_exchange(default_conf)
    assert log_has_re(r"Exchange .* is officially supported by the Freqtrade development team\.",
                      caplog)
    caplog.clear()

    # Test an available exchange, supported by ccxt
    default_conf.get('exchange').update({'name': 'kraken'})
    assert check_exchange(default_conf)
    assert log_has_re(r"Exchange .* is supported by ccxt and .* not officially supported "
                      r"by the Freqtrade development team\. .*", caplog)
    caplog.clear()

    # Test a 'bad' exchange, which known to have serious problems
    default_conf.get('exchange').update({'name': 'bitmex'})
    with pytest.raises(OperationalException,
                       match=r"Exchange .* is known to not work with the bot yet.*"):
        check_exchange(default_conf)
    caplog.clear()

    # Test a 'bad' exchange with check_for_bad=False
    default_conf.get('exchange').update({'name': 'bitmex'})
    assert check_exchange(default_conf, False)
    assert log_has_re(r"Exchange .* is supported by ccxt and .* not officially supported "
                      r"by the Freqtrade development team\. .*", caplog)
    caplog.clear()

    # Test an invalid exchange
    default_conf.get('exchange').update({'name': 'unknown_exchange'})

    with pytest.raises(
        OperationalException,
        match=r'.*Exchange "unknown_exchange" is not supported by ccxt '
              r'and therefore not available for the bot.*'
    ):
        check_exchange(default_conf)


def test_cli_verbose_with_params(default_conf, mocker, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    # Prevent setting loggers
    mocker.patch('freqtrade.loggers._set_loggers', MagicMock)
    arglist = ['-vvv']
    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf.get('verbosity') == 3
    assert log_has('Verbosity set to 3', caplog)


def test_set_loggers() -> None:
    # Reset Logging to Debug, otherwise this fails randomly as it's set globally
    logging.getLogger('requests').setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.DEBUG)
    logging.getLogger('ccxt.base.exchange').setLevel(logging.DEBUG)
    logging.getLogger('telegram').setLevel(logging.DEBUG)

    previous_value1 = logging.getLogger('requests').level
    previous_value2 = logging.getLogger('ccxt.base.exchange').level
    previous_value3 = logging.getLogger('telegram').level

    _set_loggers()

    value1 = logging.getLogger('requests').level
    assert previous_value1 is not value1
    assert value1 is logging.INFO

    value2 = logging.getLogger('ccxt.base.exchange').level
    assert previous_value2 is not value2
    assert value2 is logging.INFO

    value3 = logging.getLogger('telegram').level
    assert previous_value3 is not value3
    assert value3 is logging.INFO

    _set_loggers(verbosity=2)

    assert logging.getLogger('requests').level is logging.DEBUG
    assert logging.getLogger('ccxt.base.exchange').level is logging.INFO
    assert logging.getLogger('telegram').level is logging.INFO

    _set_loggers(verbosity=3)

    assert logging.getLogger('requests').level is logging.DEBUG
    assert logging.getLogger('ccxt.base.exchange').level is logging.DEBUG
    assert logging.getLogger('telegram').level is logging.INFO


def test_set_logfile(default_conf, mocker):
    patched_configuration_load_config_file(mocker, default_conf)

    arglist = [
        '--logfile', 'test_file.log',
    ]
    args = Arguments(arglist, '').get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf['logfile'] == "test_file.log"
    f = Path("test_file.log")
    assert f.is_file()
    f.unlink()


def test_load_config_warn_forcebuy(default_conf, mocker, caplog) -> None:
    default_conf['forcebuy_enable'] = True
    patched_configuration_load_config_file(mocker, default_conf)

    args = Arguments([], '').get_parsed_arg()
    configuration = Configuration(args)
    validated_conf = configuration.load_config()

    assert validated_conf.get('forcebuy_enable')
    assert log_has('`forcebuy` RPC message enabled.', caplog)


def test_validate_default_conf(default_conf) -> None:
    validate(default_conf, constants.CONF_SCHEMA, Draft4Validator)


def test_create_datadir(mocker, default_conf, caplog) -> None:
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=False))
    md = mocker.patch.object(Path, 'mkdir', MagicMock())

    create_datadir(default_conf, '/foo/bar')
    assert md.call_args[1]['parents'] is True
    assert log_has('Created data directory: /foo/bar', caplog)


def test_validate_tsl(default_conf):
    default_conf['trailing_stop'] = True
    default_conf['trailing_stop_positive'] = 0
    default_conf['trailing_stop_positive_offset'] = 0

    default_conf['trailing_only_offset_is_reached'] = True
    with pytest.raises(OperationalException,
                       match=r'The config trailing_only_offset_is_reached needs '
                       'trailing_stop_positive_offset to be more than 0 in your config.'):
        validate_config_consistency(default_conf)

    default_conf['trailing_stop_positive_offset'] = 0.01
    default_conf['trailing_stop_positive'] = 0.015
    with pytest.raises(OperationalException,
                       match=r'The config trailing_stop_positive_offset needs '
                       'to be greater than trailing_stop_positive_offset in your config.'):
        validate_config_consistency(default_conf)

    default_conf['trailing_stop_positive'] = 0.01
    default_conf['trailing_stop_positive_offset'] = 0.015
    validate_config_consistency(default_conf)


def test_validate_edge(edge_conf):
    edge_conf.update({"pairlist": {
        "method": "VolumePairList",
    }})

    with pytest.raises(OperationalException,
                       match="Edge and VolumePairList are incompatible, "
                       "Edge will override whatever pairs VolumePairlist selects."):
        validate_config_consistency(edge_conf)

    edge_conf.update({"pairlist": {
        "method": "StaticPairList",
    }})
    validate_config_consistency(edge_conf)


def test_load_config_test_comments() -> None:
    """
    Load config with comments
    """
    config_file = Path(__file__).parents[0] / "config_test_comments.json"
    print(config_file)
    conf = load_config_file(str(config_file))

    assert conf


def test_load_config_default_exchange(all_conf) -> None:
    """
    config['exchange'] subtree has required options in it
    so it cannot be omitted in the config
    """
    del all_conf['exchange']

    assert 'exchange' not in all_conf

    with pytest.raises(ValidationError,
                       match=r'\'exchange\' is a required property'):
        validate_config_schema(all_conf)


def test_load_config_default_exchange_name(all_conf) -> None:
    """
    config['exchange']['name'] option is required
    so it cannot be omitted in the config
    """
    del all_conf['exchange']['name']

    assert 'name' not in all_conf['exchange']

    with pytest.raises(ValidationError,
                       match=r'\'name\' is a required property'):
        validate_config_schema(all_conf)


@pytest.mark.parametrize("keys", [("exchange", "sandbox", False),
                                  ("exchange", "key", ""),
                                  ("exchange", "secret", ""),
                                  ("exchange", "password", ""),
                                  ])
def test_load_config_default_subkeys(all_conf, keys) -> None:
    """
    Test for parameters with default values in sub-paths
    so they can be omitted in the config and the default value
    should is added to the config.
    """
    # Get first level key
    key = keys[0]
    # get second level key
    subkey = keys[1]

    del all_conf[key][subkey]

    assert subkey not in all_conf[key]

    validate_config_schema(all_conf)
    assert subkey in all_conf[key]
    assert all_conf[key][subkey] == keys[2]


def test_pairlist_resolving():
    arglist = [
        'download-data',
        '--pairs', 'ETH/BTC', 'XRP/BTC',
        '--exchange', 'binance'
    ]

    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()

    assert config['pairs'] == ['ETH/BTC', 'XRP/BTC']
    assert config['exchange']['name'] == 'binance'


def test_pairlist_resolving_with_config(mocker, default_conf):
    patched_configuration_load_config_file(mocker, default_conf)
    arglist = [
        '--config', 'config.json',
        'download-data',
    ]

    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()

    assert config['pairs'] == default_conf['exchange']['pair_whitelist']
    assert config['exchange']['name'] == default_conf['exchange']['name']

    # Override pairs
    arglist = [
        '--config', 'config.json',
        'download-data',
        '--pairs', 'ETH/BTC', 'XRP/BTC',
    ]

    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()

    assert config['pairs'] == ['ETH/BTC', 'XRP/BTC']
    assert config['exchange']['name'] == default_conf['exchange']['name']


def test_pairlist_resolving_with_config_pl(mocker, default_conf):
    patched_configuration_load_config_file(mocker, default_conf)
    load_mock = mocker.patch("freqtrade.configuration.configuration.json_load",
                             MagicMock(return_value=['XRP/BTC', 'ETH/BTC']))
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))

    arglist = [
        '--config', 'config.json',
        'download-data',
        '--pairs-file', 'pairs.json',
    ]

    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()

    assert load_mock.call_count == 1
    assert config['pairs'] == ['ETH/BTC', 'XRP/BTC']
    assert config['exchange']['name'] == default_conf['exchange']['name']


def test_pairlist_resolving_with_config_pl_not_exists(mocker, default_conf):
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("freqtrade.configuration.configuration.json_load",
                 MagicMock(return_value=['XRP/BTC', 'ETH/BTC']))
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))

    arglist = [
        '--config', 'config.json',
        'download-data',
        '--pairs-file', 'pairs.json',
    ]

    args = Arguments(arglist, '').get_parsed_arg()

    with pytest.raises(OperationalException, match=r"No pairs file found with path.*"):
        configuration = Configuration(args)
        configuration.get_config()


def test_pairlist_resolving_fallback(mocker):
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    mocker.patch("freqtrade.configuration.configuration.json_load",
                 MagicMock(return_value=['XRP/BTC', 'ETH/BTC']))
    arglist = [
        'download-data',
        '--exchange', 'binance'
    ]

    args = Arguments(arglist, '').get_parsed_arg()

    configuration = Configuration(args)
    config = configuration.get_config()

    assert config['pairs'] == ['ETH/BTC', 'XRP/BTC']
    assert config['exchange']['name'] == 'binance'
