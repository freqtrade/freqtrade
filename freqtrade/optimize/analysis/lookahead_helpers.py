import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.analysis.lookahead import LookaheadAnalysis
from freqtrade.resolvers import StrategyResolver


logger = logging.getLogger(__name__)


class LookaheadAnalysisSubFunctions:

    @staticmethod
    def text_table_lookahead_analysis_instances(
            config: Dict[str, Any],
            lookahead_instances: List[LookaheadAnalysis]):
        headers = ['filename', 'strategy', 'has_bias', 'total_signals',
                   'biased_entry_signals', 'biased_exit_signals', 'biased_indicators']
        data = []
        for inst in lookahead_instances:
            if config['minimum_trade_amount'] > inst.current_analysis.total_signals:
                data.append(
                    [
                        inst.strategy_obj['location'].parts[-1],
                        inst.strategy_obj['name'],
                        "too few trades caught "
                        f"({inst.current_analysis.total_signals}/{config['minimum_trade_amount']})."
                        f"Test failed."
                    ]
                )
            elif inst.failed_bias_check:
                data.append(
                    [
                        inst.strategy_obj['location'].parts[-1],
                        inst.strategy_obj['name'],
                        'error while checking'
                    ]
                )
            else:
                data.append(
                    [
                        inst.strategy_obj['location'].parts[-1],
                        inst.strategy_obj['name'],
                        inst.current_analysis.has_bias,
                        inst.current_analysis.total_signals,
                        inst.current_analysis.false_entry_signals,
                        inst.current_analysis.false_exit_signals,
                        ", ".join(inst.current_analysis.false_indicators)
                    ]
                )
        from tabulate import tabulate
        table = tabulate(data, headers=headers, tablefmt="orgtbl")
        print(table)
        return table, headers, data

    @staticmethod
    def export_to_csv(config: Dict[str, Any], lookahead_analysis: List[LookaheadAnalysis]):
        def add_or_update_row(df, row_data):
            if (
                    (df['filename'] == row_data['filename']) &
                    (df['strategy'] == row_data['strategy'])
            ).any():
                # Update existing row
                pd_series = pd.DataFrame([row_data])
                df.loc[
                    (df['filename'] == row_data['filename']) &
                    (df['strategy'] == row_data['strategy'])
                    ] = pd_series
            else:
                # Add new row
                df = pd.concat([df, pd.DataFrame([row_data], columns=df.columns)])

            return df

        if Path(config['lookahead_analysis_exportfilename']).exists():
            # Read CSV file into a pandas dataframe
            csv_df = pd.read_csv(config['lookahead_analysis_exportfilename'])
        else:
            # Create a new empty DataFrame with the desired column names and set the index
            csv_df = pd.DataFrame(columns=[
                'filename', 'strategy', 'has_bias', 'total_signals',
                'biased_entry_signals', 'biased_exit_signals', 'biased_indicators'
            ],
                index=None)

        for inst in lookahead_analysis:
            # only update if
            if (inst.current_analysis.total_signals > config['minimum_trade_amount']
                    and inst.failed_bias_check is not True):
                new_row_data = {'filename': inst.strategy_obj['location'].parts[-1],
                                'strategy': inst.strategy_obj['name'],
                                'has_bias': inst.current_analysis.has_bias,
                                'total_signals':
                                    int(inst.current_analysis.total_signals),
                                'biased_entry_signals':
                                    int(inst.current_analysis.false_entry_signals),
                                'biased_exit_signals':
                                    int(inst.current_analysis.false_exit_signals),
                                'biased_indicators':
                                    ",".join(inst.current_analysis.false_indicators)}
                csv_df = add_or_update_row(csv_df, new_row_data)

        # Fill NaN values with a default value (e.g., 0)
        csv_df['total_signals'] = csv_df['total_signals'].astype(int).fillna(0)
        csv_df['biased_entry_signals'] = csv_df['biased_entry_signals'].astype(int).fillna(0)
        csv_df['biased_exit_signals'] = csv_df['biased_exit_signals'].astype(int).fillna(0)

        # Convert columns to integers
        csv_df['total_signals'] = csv_df['total_signals'].astype(int)
        csv_df['biased_entry_signals'] = csv_df['biased_entry_signals'].astype(int)
        csv_df['biased_exit_signals'] = csv_df['biased_exit_signals'].astype(int)

        logger.info(f"saving {config['lookahead_analysis_exportfilename']}")
        csv_df.to_csv(config['lookahead_analysis_exportfilename'], index=False)

    @staticmethod
    def calculate_config_overrides(config: Config):
        if config['targeted_trade_amount'] < config['minimum_trade_amount']:
            # this combo doesn't make any sense.
            raise OperationalException(
                "Targeted trade amount can't be smaller than minimum trade amount."
            )
        if len(config['pairs']) > config['max_open_trades']:
            logger.info('Max_open_trades were less than amount of pairs. '
                        'Set max_open_trades to amount of pairs just to avoid false positives.')
            config['max_open_trades'] = len(config['pairs'])

        min_dry_run_wallet = 1000000000
        if config['dry_run_wallet'] < min_dry_run_wallet:
            logger.info('Dry run wallet was not set to 1 billion, pushing it up there '
                        'just to avoid false positives')
            config['dry_run_wallet'] = min_dry_run_wallet

        if 'timerange' not in config:
            # setting a timerange is enforced here
            raise OperationalException(
                "Please set a timerange. "
                "Usually a few months are enough depending on your needs and strategy."
            )
        # fix stake_amount to 10k.
        # in a combination with a wallet size of 1 billion it should always be able to trade
        # no matter if they use custom_stake_amount as a small percentage of wallet size
        # or fixate custom_stake_amount to a certain value.
        logger.info('fixing stake_amount to 10k')
        config['stake_amount'] = 10000

        # enforce cache to be 'none', shift it to 'none' if not already
        # (since the default value is 'day')
        if config.get('backtest_cache') is None:
            config['backtest_cache'] = 'none'
        elif config['backtest_cache'] != 'none':
            logger.info(f"backtest_cache = "
                        f"{config['backtest_cache']} detected. "
                        f"Inside lookahead-analysis it is enforced to be 'none'. "
                        f"Changed it to 'none'")
            config['backtest_cache'] = 'none'
        return config

    @staticmethod
    def initialize_single_lookahead_analysis(config: Config, strategy_obj: Dict[str, Any]):

        logger.info(f"Bias test of {Path(strategy_obj['location']).name} started.")
        start = time.perf_counter()
        current_instance = LookaheadAnalysis(config, strategy_obj)
        current_instance.start()
        elapsed = time.perf_counter() - start
        logger.info(f"Checking look ahead bias via backtests "
                    f"of {Path(strategy_obj['location']).name} "
                    f"took {elapsed:.0f} seconds.")
        return current_instance

    @staticmethod
    def start(config: Config):
        config = LookaheadAnalysisSubFunctions.calculate_config_overrides(config)

        strategy_objs = StrategyResolver.search_all_objects(
            config, enum_failed=False, recursive=config.get('recursive_strategy_search', False))

        lookaheadAnalysis_instances = []

        # unify --strategy and --strategy-list to one list
        if not (strategy_list := config.get('strategy_list', [])):
            if config.get('strategy') is None:
                raise OperationalException(
                    "No Strategy specified. Please specify a strategy via --strategy or "
                    "--strategy-list"
                )
            strategy_list = [config['strategy']]

        # check if strategies can be properly loaded, only check them if they can be.
        for strat in strategy_list:
            for strategy_obj in strategy_objs:
                if strategy_obj['name'] == strat and strategy_obj not in strategy_list:
                    lookaheadAnalysis_instances.append(
                        LookaheadAnalysisSubFunctions.initialize_single_lookahead_analysis(
                            config, strategy_obj))
                    break

        # report the results
        if lookaheadAnalysis_instances:
            LookaheadAnalysisSubFunctions.text_table_lookahead_analysis_instances(
                config, lookaheadAnalysis_instances)
            if config.get('lookahead_analysis_exportfilename') is not None:
                LookaheadAnalysisSubFunctions.export_to_csv(config, lookaheadAnalysis_instances)
        else:
            logger.error("There were no strategies specified neither through "
                         "--strategy nor through "
                         "--strategy-list "
                         "or timeframe was not specified.")
