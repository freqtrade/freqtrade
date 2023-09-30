# pragma pylint: disable=missing-docstring, protected-access, C0103
import logging
from base64 import urlsafe_b64encode
from pathlib import Path

import pytest
from pandas import DataFrame

from freqtrade.configuration import Configuration
from freqtrade.exceptions import OperationalException
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy.interface import IStrategy
from tests.conftest import CURRENT_TEST_STRATEGY, log_has, log_has_re


def test_search_strategy():
    default_location = Path(__file__).parent / 'strats'

    s, _ = StrategyResolver._search_object(
        directory=default_location,
        object_name=CURRENT_TEST_STRATEGY,
        add_source=True,
    )
    assert issubclass(s, IStrategy)

    s, _ = StrategyResolver._search_object(
        directory=default_location,
        object_name='NotFoundStrategy',
        add_source=True,
    )
    assert s is None


def test_search_all_strategies_no_failed():
    directory = Path(__file__).parent / "strats"
    strategies = StrategyResolver._search_all_objects(directory, enum_failed=False)
    assert isinstance(strategies, list)
    assert len(strategies) == 13
    assert isinstance(strategies[0], dict)


def test_search_all_strategies_with_failed():
    directory = Path(__file__).parent / "strats"
    strategies = StrategyResolver._search_all_objects(directory, enum_failed=True)
    assert isinstance(strategies, list)
    assert len(strategies) == 14
    # with enum_failed=True search_all_objects() shall find 2 good strategies
    # and 1 which fails to load
    assert len([x for x in strategies if x['class'] is not None]) == 13

    assert len([x for x in strategies if x['class'] is None]) == 1

    directory = Path(__file__).parent / "strats_nonexistingdir"
    strategies = StrategyResolver._search_all_objects(directory, enum_failed=True)
    assert len(strategies) == 0


def test_load_strategy(default_conf, dataframe_1m):
    default_conf.update({'strategy': 'SampleStrategy',
                         'strategy_path': str(Path(__file__).parents[2] / 'freqtrade/templates')
                         })
    strategy = StrategyResolver.load_strategy(default_conf)
    assert isinstance(strategy.__source__, str)
    assert 'class SampleStrategy' in strategy.__source__
    assert isinstance(strategy.__file__, str)
    assert 'rsi' in strategy.advise_indicators(dataframe_1m, {'pair': 'ETH/BTC'})


def test_load_strategy_base64(dataframe_1m, caplog, default_conf):
    filepath = Path(__file__).parents[2] / 'freqtrade/templates/sample_strategy.py'
    encoded_string = urlsafe_b64encode(filepath.read_bytes()).decode("utf-8")
    default_conf.update({'strategy': f'SampleStrategy:{encoded_string}'})

    strategy = StrategyResolver.load_strategy(default_conf)
    assert 'rsi' in strategy.advise_indicators(dataframe_1m, {'pair': 'ETH/BTC'})
    # Make sure strategy was loaded from base64 (using temp directory)!!
    assert log_has_re(r"Using resolved strategy SampleStrategy from '"
                      r".*(/|\\).*(/|\\)SampleStrategy\.py'\.\.\.", caplog)


def test_load_strategy_invalid_directory(caplog, default_conf):
    extra_dir = Path.cwd() / 'some/path'
    with pytest.raises(OperationalException, match=r"Impossible to load Strategy.*"):
        StrategyResolver._load_strategy('StrategyTestV333', config=default_conf,
                                        extra_dir=extra_dir)

    assert log_has_re(r'Path .*' + r'some.*path.*' + r'.* does not exist', caplog)


def test_load_not_found_strategy(default_conf):
    default_conf['strategy'] = 'NotFoundStrategy'
    with pytest.raises(OperationalException,
                       match=r"Impossible to load Strategy 'NotFoundStrategy'. "
                             r"This class does not exist or contains Python code errors."):
        StrategyResolver.load_strategy(default_conf)


def test_load_strategy_noname(default_conf):
    default_conf['strategy'] = ''
    with pytest.raises(OperationalException,
                       match="No strategy set. Please use `--strategy` to specify "
                             "the strategy class to use."):
        StrategyResolver.load_strategy(default_conf)


@ pytest.mark.filterwarnings("ignore:deprecated")
@ pytest.mark.parametrize('strategy_name', ['StrategyTestV2'])
def test_strategy_pre_v3(dataframe_1m, default_conf, strategy_name):
    default_conf.update({'strategy': strategy_name})

    strategy = StrategyResolver.load_strategy(default_conf)
    metadata = {'pair': 'ETH/BTC'}
    assert strategy.minimal_roi[0] == 0.04
    assert default_conf["minimal_roi"]['0'] == 0.04

    assert strategy.stoploss == -0.10
    assert default_conf['stoploss'] == -0.10

    assert strategy.timeframe == '5m'
    assert default_conf['timeframe'] == '5m'

    df_indicators = strategy.advise_indicators(dataframe_1m, metadata=metadata)
    assert 'adx' in df_indicators

    dataframe = strategy.advise_entry(df_indicators, metadata=metadata)
    assert 'buy' not in dataframe.columns
    assert 'enter_long' in dataframe.columns

    dataframe = strategy.advise_exit(df_indicators, metadata=metadata)
    assert 'sell' not in dataframe.columns
    assert 'exit_long' in dataframe.columns


def test_strategy_can_short(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
    })
    strat = StrategyResolver.load_strategy(default_conf)
    assert isinstance(strat, IStrategy)
    default_conf['strategy'] = 'StrategyTestV3Futures'
    with pytest.raises(ImportError, match=""):
        StrategyResolver.load_strategy(default_conf)

    default_conf['trading_mode'] = 'futures'
    strat = StrategyResolver.load_strategy(default_conf)
    assert isinstance(strat, IStrategy)


def test_strategy_override_minimal_roi(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
        'minimal_roi': {
            "20": 0.1,
            "0": 0.5
        }
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.minimal_roi[0] == 0.5
    assert log_has(
        "Override strategy 'minimal_roi' with value in config file: {'20': 0.1, '0': 0.5}.",
        caplog)


def test_strategy_override_stoploss(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
        'stoploss': -0.5
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.stoploss == -0.5
    assert log_has("Override strategy 'stoploss' with value in config file: -0.5.", caplog)


def test_strategy_override_max_open_trades(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
        'max_open_trades': 7
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.max_open_trades == 7
    assert log_has("Override strategy 'max_open_trades' with value in config file: 7.", caplog)


def test_strategy_override_trailing_stop(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
        'trailing_stop': True
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.trailing_stop
    assert isinstance(strategy.trailing_stop, bool)
    assert log_has("Override strategy 'trailing_stop' with value in config file: True.", caplog)


def test_strategy_override_trailing_stop_positive(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
        'trailing_stop_positive': -0.1,
        'trailing_stop_positive_offset': -0.2

    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.trailing_stop_positive == -0.1
    assert log_has("Override strategy 'trailing_stop_positive' with value in config file: -0.1.",
                   caplog)

    assert strategy.trailing_stop_positive_offset == -0.2
    assert log_has("Override strategy 'trailing_stop_positive' with value in config file: -0.1.",
                   caplog)


def test_strategy_override_timeframe(caplog, default_conf):
    caplog.set_level(logging.INFO)

    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
        'timeframe': 60,
        'stake_currency': 'ETH'
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.timeframe == 60
    assert strategy.stake_currency == 'ETH'
    assert log_has("Override strategy 'timeframe' with value in config file: 60.",
                   caplog)


def test_strategy_override_process_only_new_candles(caplog, default_conf):
    caplog.set_level(logging.INFO)

    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
        'process_only_new_candles': False
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert not strategy.process_only_new_candles
    assert log_has("Override strategy 'process_only_new_candles' with value in config file: False.",
                   caplog)


def test_strategy_override_order_types(caplog, default_conf):
    caplog.set_level(logging.INFO)

    order_types = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
    }
    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
        'order_types': order_types
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.order_types
    for method in ['entry', 'exit', 'stoploss', 'stoploss_on_exchange']:
        assert strategy.order_types[method] == order_types[method]

    assert log_has("Override strategy 'order_types' with value in config file:"
                   " {'entry': 'market', 'exit': 'limit', 'stoploss': 'limit',"
                   " 'stoploss_on_exchange': True}.", caplog)

    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
        'order_types': {'exit': 'market'}
    })
    # Raise error for invalid configuration
    with pytest.raises(ImportError,
                       match=r"Impossible to load Strategy '" + CURRENT_TEST_STRATEGY + "'. "
                             r"Order-types mapping is incomplete."):
        StrategyResolver.load_strategy(default_conf)


def test_strategy_override_order_tif(caplog, default_conf):
    caplog.set_level(logging.INFO)

    order_time_in_force = {
        'entry': 'FOK',
        'exit': 'GTC',
    }

    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
        'order_time_in_force': order_time_in_force
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.order_time_in_force
    for method in ['entry', 'exit']:
        assert strategy.order_time_in_force[method] == order_time_in_force[method]

    assert log_has("Override strategy 'order_time_in_force' with value in config file:"
                   " {'entry': 'FOK', 'exit': 'GTC'}.", caplog)

    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
        'order_time_in_force': {'entry': 'FOK'}
    })
    # Raise error for invalid configuration
    with pytest.raises(ImportError,
                       match=f"Impossible to load Strategy '{CURRENT_TEST_STRATEGY}'. "
                             "Order-time-in-force mapping is incomplete."):
        StrategyResolver.load_strategy(default_conf)


def test_strategy_override_use_exit_signal(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
    })
    strategy = StrategyResolver.load_strategy(default_conf)
    assert strategy.use_exit_signal
    assert isinstance(strategy.use_exit_signal, bool)
    # must be inserted to configuration
    assert 'use_exit_signal' in default_conf
    assert default_conf['use_exit_signal']

    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
        'use_exit_signal': False,
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert not strategy.use_exit_signal
    assert isinstance(strategy.use_exit_signal, bool)
    assert log_has("Override strategy 'use_exit_signal' with value in config file: False.", caplog)


def test_strategy_override_use_exit_profit_only(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
    })
    strategy = StrategyResolver.load_strategy(default_conf)
    assert not strategy.exit_profit_only
    assert isinstance(strategy.exit_profit_only, bool)
    # must be inserted to configuration
    assert 'exit_profit_only' in default_conf
    assert not default_conf['exit_profit_only']

    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
        'exit_profit_only': True,
    })
    strategy = StrategyResolver.load_strategy(default_conf)

    assert strategy.exit_profit_only
    assert isinstance(strategy.exit_profit_only, bool)
    assert log_has("Override strategy 'exit_profit_only' with value in config file: True.", caplog)


def test_strategy_max_open_trades_infinity_from_strategy(caplog, default_conf):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
    })
    del default_conf['max_open_trades']

    strategy = StrategyResolver.load_strategy(default_conf)

    # this test assumes -1 set to 'max_open_trades' in CURRENT_TEST_STRATEGY
    assert strategy.max_open_trades == float('inf')
    assert default_conf['max_open_trades'] == float('inf')


def test_strategy_max_open_trades_infinity_from_config(caplog, default_conf, mocker):
    caplog.set_level(logging.INFO)
    default_conf.update({
        'strategy': CURRENT_TEST_STRATEGY,
        'max_open_trades': -1,
        'exchange': 'binance'
    })

    configuration = Configuration(args=default_conf)
    parsed_config = configuration.get_config()

    assert parsed_config['max_open_trades'] == float('inf')

    strategy = StrategyResolver.load_strategy(parsed_config)

    assert strategy.max_open_trades == float('inf')


@ pytest.mark.filterwarnings("ignore:deprecated")
def test_missing_implements(default_conf, caplog):

    default_location = Path(__file__).parent / "strats"
    default_conf.update({'strategy': 'StrategyTestV2',
                         'strategy_path': default_location})
    StrategyResolver.load_strategy(default_conf)

    log_has_re(r"DEPRECATED: .*use_sell_signal.*use_exit_signal.", caplog)

    default_conf['trading_mode'] = 'futures'
    with pytest.raises(OperationalException,
                       match=r"DEPRECATED: .*use_sell_signal.*use_exit_signal."):
        StrategyResolver.load_strategy(default_conf)

    default_conf['trading_mode'] = 'spot'

    default_location = Path(__file__).parent / "strats/broken_strats"
    default_conf.update({'strategy': 'TestStrategyNoImplements',
                         'strategy_path': default_location})
    with pytest.raises(OperationalException,
                       match=r"`populate_entry_trend` or `populate_buy_trend`.*"):
        StrategyResolver.load_strategy(default_conf)

    default_conf['strategy'] = 'TestStrategyNoImplementSell'

    with pytest.raises(OperationalException,
                       match=r"`populate_exit_trend` or `populate_sell_trend`.*"):
        StrategyResolver.load_strategy(default_conf)

    # Futures mode is more strict ...
    default_conf['trading_mode'] = 'futures'

    with pytest.raises(OperationalException,
                       match=r"`populate_exit_trend` must be implemented.*"):
        StrategyResolver.load_strategy(default_conf)

    default_conf['strategy'] = 'TestStrategyNoImplements'
    with pytest.raises(OperationalException,
                       match=r"`populate_entry_trend` must be implemented.*"):
        StrategyResolver.load_strategy(default_conf)

    default_conf['strategy'] = 'TestStrategyImplementCustomSell'
    with pytest.raises(OperationalException,
                       match=r"Please migrate your implementation of `custom_sell`.*"):
        StrategyResolver.load_strategy(default_conf)

    default_conf['strategy'] = 'TestStrategyImplementBuyTimeout'
    with pytest.raises(OperationalException,
                       match=r"Please migrate your implementation of `check_buy_timeout`.*"):
        StrategyResolver.load_strategy(default_conf)

    default_conf['strategy'] = 'TestStrategyImplementSellTimeout'
    with pytest.raises(OperationalException,
                       match=r"Please migrate your implementation of `check_sell_timeout`.*"):
        StrategyResolver.load_strategy(default_conf)


def test_call_deprecated_function(default_conf):
    default_location = Path(__file__).parent / "strats/broken_strats/"
    del default_conf['timeframe']
    default_conf.update({'strategy': 'TestStrategyLegacyV1',
                         'strategy_path': default_location})
    with pytest.raises(OperationalException,
                       match=r"Strategy Interface v1 is no longer supported.*"):
        StrategyResolver.load_strategy(default_conf)


def test_strategy_interface_versioning(dataframe_1m, default_conf):
    default_conf.update({'strategy': 'StrategyTestV2'})
    strategy = StrategyResolver.load_strategy(default_conf)
    metadata = {'pair': 'ETH/BTC'}

    assert strategy.INTERFACE_VERSION == 2

    indicator_df = strategy.advise_indicators(dataframe_1m, metadata=metadata)
    assert isinstance(indicator_df, DataFrame)
    assert 'adx' in indicator_df.columns

    enterdf = strategy.advise_entry(dataframe_1m, metadata=metadata)
    assert isinstance(enterdf, DataFrame)

    assert 'buy' not in enterdf.columns
    assert 'enter_long' in enterdf.columns

    exitdf = strategy.advise_exit(dataframe_1m, metadata=metadata)
    assert isinstance(exitdf, DataFrame)
    assert 'sell' not in exitdf
    assert 'exit_long' in exitdf


def test_strategy_ft_load_params_from_file(mocker, default_conf):
    default_conf.update({'strategy': 'StrategyTestV2'})
    del default_conf['max_open_trades']
    mocker.patch('freqtrade.strategy.hyper.HyperStrategyMixin.load_params_from_file',
                 return_value={
                     'params': {
                         'max_open_trades':  {
                            'max_open_trades': -1
                         }
                         }
                     })
    strategy = StrategyResolver.load_strategy(default_conf)
    assert strategy.max_open_trades == float('inf')
    assert strategy.config['max_open_trades'] == float('inf')
