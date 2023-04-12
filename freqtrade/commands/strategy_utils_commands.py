import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tabulate import tabulate

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
            bias_checker_instances.append(
                initialize_single_lookahead_bias_checker(filtered_strategy_obj, config, args))
    else:
        processed_locations = set()
        for strategy_obj in strategy_objs:
            if strategy_obj['location'] not in processed_locations:
                processed_locations.add(strategy_obj['location'])
                bias_checker_instances.append(
                    initialize_single_lookahead_bias_checker(strategy_obj, config, args))
    text_table_bias_checker_instances(bias_checker_instances)
    export_to_csv(args, bias_checker_instances)


def text_table_bias_checker_instances(bias_checker_instances):
    headers = ['strategy', 'has_bias',
               'total_signals', 'biased_entry_signals', 'biased_exit_signals', 'biased_indicators']
    data = []
    for current_instance in bias_checker_instances:
        data.append(
            [current_instance.strategy_obj['name'],
             current_instance.current_analysis.has_bias,
             current_instance.current_analysis.total_signals,
             current_instance.current_analysis.false_entry_signals,
             current_instance.current_analysis.false_exit_signals,
             ", ".join(current_instance.current_analysis.false_indicators)]
        )
    table = tabulate(data, headers=headers, tablefmt="orgtbl")
    print(table)


def export_to_csv(args, bias_checker_instances):
    def add_or_update_row(df, row_data):
        strategy_col_name = 'strategy'
        if row_data[strategy_col_name] in df[strategy_col_name].values:
            # create temporary dataframe with a single row
            # and use that to replace the previous data in there.
            index = (df.index[df[strategy_col_name] ==
                              row_data[strategy_col_name]][0])
            df.loc[index] = pd.Series(row_data, index='strategy')

        else:
            df = df.concat(row_data, ignore_index=True)
        return df

    csv_df = None

    if not Path.exists(args['exportfilename']):
        # If the file doesn't exist, create a new DataFrame from scratch
        csv_df = pd.DataFrame(columns=['filename', 'strategy', 'has_bias',
                                       'total_signals',
                                       'biased_entry_signals', 'biased_exit_signals',
                                       'biased_indicators'],
                              index='filename')
    else:
        # Read CSV file into a pandas dataframe
        csv_df = pd.read_csv(args['exportfilename'])

    for inst in bias_checker_instances:
        new_row_data = {'filename': inst.strategy_obj['location'].parts[-1],
                        'strategy': inst.strategy_obj['name'],
                        'has_bias': inst.current_analysis.has_bias,
                        'total_signals': inst.current_analysis.total_signals,
                        'biased_entry_signals': inst.current_analysis.false_entry_signals,
                        'biased_exit_signals': inst.current_analysis.false_exit_signals,
                        'biased_indicators': ", ".join(inst.current_analysis.false_indicators)}
        csv_df = add_or_update_row(csv_df, new_row_data)
    if len(bias_checker_instances) > 0:
        print(f"saving {args['exportfilename']}")
        csv_df.to_csv(args['exportfilename'])


def initialize_single_lookahead_bias_checker(strategy_obj, config, args):
    print(f"Bias test of {Path(strategy_obj['location']).name} started.")
    start = time.perf_counter()
    current_instance = BacktestLookaheadBiasChecker()
    current_instance.start(config, strategy_obj, args)
    elapsed = time.perf_counter() - start
    print(f"checking look ahead bias via backtests of {Path(strategy_obj['location']).name} "
          f"took {elapsed:.1f} seconds.")
    return current_instance
