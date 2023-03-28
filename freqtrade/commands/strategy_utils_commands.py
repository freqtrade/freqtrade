import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

from freqtrade.configuration import setup_utils_configuration
from freqtrade.enums import RunMode
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy.backtest_lookahead_bias_checker import BacktestLookaheadBiasChecker
from freqtrade.strategy.strategyupdater import StrategyUpdater


logger = logging.getLogger(__name__)


def start_strategy_update(args: Dict[str, Any]) -> None:
    """
    Start the strategy updating script
    :param args: Cli args from Arguments()
    :return: None
    """

    if sys.version_info == (3, 8):  # pragma: no cover
        sys.exit("Freqtrade strategy updater requires Python version >= 3.9")

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    strategy_objs = StrategyResolver.search_all_objects(
        config, enum_failed=False, recursive=config.get('recursive_strategy_search', False))

    filtered_strategy_objs = []
    if args['strategy_list']:
        filtered_strategy_objs = [
            strategy_obj for strategy_obj in strategy_objs
            if strategy_obj['name'] in args['strategy_list']
        ]

    else:
        # Use all available entries.
        filtered_strategy_objs = strategy_objs

    processed_locations = set()
    for strategy_obj in filtered_strategy_objs:
        if strategy_obj['location'] not in processed_locations:
            processed_locations.add(strategy_obj['location'])
            start_conversion(strategy_obj, config)


def start_conversion(strategy_obj, config):
    print(f"Conversion of {Path(strategy_obj['location']).name} started.")
    instance_strategy_updater = StrategyUpdater()
    start = time.perf_counter()
    instance_strategy_updater.start(config, strategy_obj)
    elapsed = time.perf_counter() - start
    print(f"Conversion of {Path(strategy_obj['location']).name} took {elapsed:.1f} seconds.")

    # except:
    #    pass


def start_backtest_lookahead_bias_checker(args: Dict[str, Any]) -> None:
    """
    Start the backtest bias tester script
    :param args: Cli args from Arguments()
    :return: None
    """
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    if args['targeted_trade_amount'] < args['minimum_trade_amount']:
        # add logic that tells the user to check the configuration
        # since this combo doesn't make any sense.
        pass

    strategy_objs = StrategyResolver.search_all_objects(
        config, enum_failed=False, recursive=config.get('recursive_strategy_search', False))

    bias_checker_instances = []

    filtered_strategy_objs = []
    if 'strategy_list' in args and args['strategy_list'] is not None:
        for args_strategy in args['strategy_list']:
            for strategy_obj in strategy_objs:
                if (strategy_obj['name'] == args_strategy
                        and strategy_obj not in filtered_strategy_objs):
                    filtered_strategy_objs.append(strategy_obj)
                    break

        for filtered_strategy_obj in filtered_strategy_objs:
            bias_checker_instances = initialize_single_lookahead_bias_checker(
                filtered_strategy_obj, config, args)
    else:
        processed_locations = set()
        for strategy_obj in strategy_objs:
            if strategy_obj['location'] not in processed_locations:
                processed_locations.add(strategy_obj['location'])
                bias_checker_instances = initialize_single_lookahead_bias_checker(
                    strategy_obj, config, args)
    create_result_list(bias_checker_instances)


def create_result_list(bias_checker_instances):
    pass


def initialize_single_lookahead_bias_checker(strategy_obj, config, args):
    # try:
    print(f"Bias test of {Path(strategy_obj['location']).name} started.")
    instance_backtest_lookahead_bias_checker = BacktestLookaheadBiasChecker()
    start = time.perf_counter()
    current_instance = instance_backtest_lookahead_bias_checker.start(config, strategy_obj, args)
    elapsed = time.perf_counter() - start
    print(f"checking look ahead bias via backtests of {Path(strategy_obj['location']).name} "
          f"took {elapsed:.1f} seconds.")
    return current_instance
