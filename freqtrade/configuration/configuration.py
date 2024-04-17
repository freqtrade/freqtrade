"""
This module contains the configuration class
"""
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from freqtrade import constants
from freqtrade.configuration.deprecated_settings import process_temporary_deprecated_settings
from freqtrade.configuration.directory_operations import create_datadir, create_userdata_dir
from freqtrade.configuration.environment_vars import enironment_vars_to_dict
from freqtrade.configuration.load_config import load_file, load_from_files
from freqtrade.constants import Config
from freqtrade.enums import NON_UTIL_MODES, TRADE_MODES, CandleType, RunMode, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.loggers import setup_logging
from freqtrade.misc import deep_merge_dicts, parse_db_uri_for_logging


logger = logging.getLogger(__name__)


class Configuration:
    """
    Class to read and init the bot configuration
    Reuse this class for the bot, backtesting, hyperopt and every script that required configuration
    """

    def __init__(self, args: Dict[str, Any], runmode: Optional[RunMode] = None) -> None:
        self.args = args
        self.config: Optional[Config] = None
        self.runmode = runmode

    def get_config(self) -> Config:
        """
        Return the config. Use this method to get the bot config
        :return: Dict: Bot config
        """
        if self.config is None:
            self.config = self.load_config()

        return self.config

    @staticmethod
    def from_files(files: List[str]) -> Dict[str, Any]:
        """
        Iterate through the config files passed in, loading all of them
        and merging their contents.
        Files are loaded in sequence, parameters in later configuration files
        override the same parameter from an earlier file (last definition wins).
        Runs through the whole Configuration initialization, so all expected config entries
        are available to interactive environments.
        :param files: List of file paths
        :return: configuration dictionary
        """
        # Keep this method as staticmethod, so it can be used from interactive environments
        c = Configuration({'config': files}, RunMode.OTHER)
        return c.get_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load the bot configuration
        :return: Configuration dictionary
        """
        # Load all configs
        config: Config = load_from_files(self.args.get("config", []))

        # Load environment variables
        from freqtrade.commands.arguments import NO_CONF_ALLOWED
        if self.args.get('command') not in NO_CONF_ALLOWED:
            env_data = enironment_vars_to_dict()
            config = deep_merge_dicts(env_data, config)

        # Normalize config
        if 'internals' not in config:
            config['internals'] = {}

        if 'pairlists' not in config:
            config['pairlists'] = []

        # Keep a copy of the original configuration file
        config['original_config'] = deepcopy(config)

        self._process_logging_options(config)

        self._process_runmode(config)

        self._process_common_options(config)

        self._process_trading_options(config)

        self._process_optimize_options(config)

        self._process_plot_options(config)

        self._process_data_options(config)

        self._process_analyze_options(config)

        self._process_freqai_options(config)

        # Import check_exchange here to avoid import cycle problems
        from freqtrade.exchange.check_exchange import check_exchange

        # Check if the exchange set by the user is supported
        check_exchange(config, config.get('experimental', {}).get('block_bad_exchanges', True))

        self._resolve_pairs_list(config)

        process_temporary_deprecated_settings(config)

        return config

    def _process_logging_options(self, config: Config) -> None:
        """
        Extract information for sys.argv and load logging configuration:
        the -v/--verbose, --logfile options
        """
        # Log level
        config.update({'verbosity': self.args.get('verbosity', 0)})

        if 'logfile' in self.args and self.args['logfile']:
            config.update({'logfile': self.args['logfile']})

        setup_logging(config)

    def _process_trading_options(self, config: Config) -> None:
        if config['runmode'] not in TRADE_MODES:
            return

        if config.get('dry_run', False):
            logger.info('Dry run is enabled')
            if config.get('db_url') in [None, constants.DEFAULT_DB_PROD_URL]:
                # Default to in-memory db for dry_run if not specified
                config['db_url'] = constants.DEFAULT_DB_DRYRUN_URL
        else:
            if not config.get('db_url'):
                config['db_url'] = constants.DEFAULT_DB_PROD_URL
            logger.info('Dry run is disabled')

        logger.info(f'Using DB: "{parse_db_uri_for_logging(config["db_url"])}"')

    def _process_common_options(self, config: Config) -> None:

        # Set strategy if not specified in config and or if it's non default
        if self.args.get('strategy') or not config.get('strategy'):
            config.update({'strategy': self.args.get('strategy')})

        self._args_to_config(config, argname='strategy_path',
                             logstring='Using additional Strategy lookup path: {}')

        if ('db_url' in self.args and self.args['db_url'] and
                self.args['db_url'] != constants.DEFAULT_DB_PROD_URL):
            config.update({'db_url': self.args['db_url']})
            logger.info('Parameter --db-url detected ...')

        self._args_to_config(config, argname='db_url_from',
                             logstring='Parameter --db-url-from detected ...')

        if config.get('force_entry_enable', False):
            logger.warning('`force_entry_enable` RPC message enabled.')

        # Support for sd_notify
        if 'sd_notify' in self.args and self.args['sd_notify']:
            config['internals'].update({'sd_notify': True})

    def _process_datadir_options(self, config: Config) -> None:
        """
        Extract information for sys.argv and load directory configurations
        --user-data, --datadir
        """
        # Check exchange parameter here - otherwise `datadir` might be wrong.
        if 'exchange' in self.args and self.args['exchange']:
            config['exchange']['name'] = self.args['exchange']
            logger.info(f"Using exchange {config['exchange']['name']}")

        if 'pair_whitelist' not in config['exchange']:
            config['exchange']['pair_whitelist'] = []

        if 'user_data_dir' in self.args and self.args['user_data_dir']:
            config.update({'user_data_dir': self.args['user_data_dir']})
        elif 'user_data_dir' not in config:
            # Default to cwd/user_data (legacy option ...)
            config.update({'user_data_dir': str(Path.cwd() / 'user_data')})

        # reset to user_data_dir so this contains the absolute path.
        config['user_data_dir'] = create_userdata_dir(config['user_data_dir'], create_dir=False)
        logger.info('Using user-data directory: %s ...', config['user_data_dir'])

        config.update({'datadir': create_datadir(config, self.args.get('datadir'))})
        logger.info('Using data directory: %s ...', config.get('datadir'))

        if self.args.get('exportfilename'):
            self._args_to_config(config, argname='exportfilename',
                                 logstring='Storing backtest results to {} ...')
            config['exportfilename'] = Path(config['exportfilename'])
        else:
            config['exportfilename'] = (config['user_data_dir']
                                        / 'backtest_results')

        if self.args.get('show_sensitive'):
            logger.warning(
                "Sensitive information will be shown in the upcomming output. "
                "Please make sure to never share this output without redacting "
                "the information yourself.")

    def _process_optimize_options(self, config: Config) -> None:

        # This will override the strategy configuration
        self._args_to_config(config, argname='timeframe',
                             logstring='Parameter -i/--timeframe detected ... '
                                       'Using timeframe: {} ...')

        self._args_to_config(config, argname='position_stacking',
                             logstring='Parameter --enable-position-stacking detected ...')

        self._args_to_config(
            config, argname='enable_protections',
            logstring='Parameter --enable-protections detected, enabling Protections. ...')

        if 'use_max_market_positions' in self.args and not self.args["use_max_market_positions"]:
            config.update({'use_max_market_positions': False})
            logger.info('Parameter --disable-max-market-positions detected ...')
            logger.info('max_open_trades set to unlimited ...')
        elif 'max_open_trades' in self.args and self.args['max_open_trades']:
            config.update({'max_open_trades': self.args['max_open_trades']})
            logger.info('Parameter --max-open-trades detected, '
                        'overriding max_open_trades to: %s ...', config.get('max_open_trades'))
        elif config['runmode'] in NON_UTIL_MODES:
            logger.info('Using max_open_trades: %s ...', config.get('max_open_trades'))
        # Setting max_open_trades to infinite if -1
        if config.get('max_open_trades') == -1:
            config['max_open_trades'] = float('inf')

        if self.args.get('stake_amount'):
            # Convert explicitly to float to support CLI argument for both unlimited and value
            try:
                self.args['stake_amount'] = float(self.args['stake_amount'])
            except ValueError:
                pass

        configurations = [
            ('timeframe_detail',
             'Parameter --timeframe-detail detected, using {} for intra-candle backtesting ...'),
            ('backtest_show_pair_list', 'Parameter --show-pair-list detected.'),
            ('stake_amount',
             'Parameter --stake-amount detected, overriding stake_amount to: {} ...'),
            ('dry_run_wallet',
             'Parameter --dry-run-wallet detected, overriding dry_run_wallet to: {} ...'),
            ('fee', 'Parameter --fee detected, setting fee to: {} ...'),
            ('timerange', 'Parameter --timerange detected: {} ...'),
            ]

        self._args_to_config_loop(config, configurations)

        self._process_datadir_options(config)

        self._args_to_config(config, argname='strategy_list',
                             logstring='Using strategy list of {} strategies', logfun=len)

        configurations = [
            ('recursive_strategy_search',
             'Recursively searching for a strategy in the strategies folder.'),
            ('timeframe', 'Overriding timeframe with Command line argument'),
            ('export', 'Parameter --export detected: {} ...'),
            ('backtest_breakdown', 'Parameter --breakdown detected ...'),
            ('backtest_cache', 'Parameter --cache={} detected ...'),
            ('disableparamexport', 'Parameter --disableparamexport detected: {} ...'),
            ('freqai_backtest_live_models',
             'Parameter --freqai-backtest-live-models detected ...'),
        ]
        self._args_to_config_loop(config, configurations)

        # Edge section:
        if 'stoploss_range' in self.args and self.args["stoploss_range"]:
            txt_range = eval(self.args["stoploss_range"])
            config['edge'].update({'stoploss_range_min': txt_range[0]})
            config['edge'].update({'stoploss_range_max': txt_range[1]})
            config['edge'].update({'stoploss_range_step': txt_range[2]})
            logger.info('Parameter --stoplosses detected: %s ...', self.args["stoploss_range"])

        # Hyperopt section

        configurations = [
            ('hyperopt', 'Using Hyperopt class name: {}'),
            ('hyperopt_path', 'Using additional Hyperopt lookup path: {}'),
            ('hyperoptexportfilename', 'Using hyperopt file: {}'),
            ('lookahead_analysis_exportfilename', 'Saving lookahead analysis results into {} ...'),
            ('epochs', 'Parameter --epochs detected ... Will run Hyperopt with for {} epochs ...'),
            ('spaces', 'Parameter -s/--spaces detected: {}'),
            ('analyze_per_epoch', 'Parameter --analyze-per-epoch detected.'),
            ('print_all', 'Parameter --print-all detected ...'),
        ]
        self._args_to_config_loop(config, configurations)

        if 'print_colorized' in self.args and not self.args["print_colorized"]:
            logger.info('Parameter --no-color detected ...')
            config.update({'print_colorized': False})
        else:
            config.update({'print_colorized': True})

        configurations = [
            ('print_json', 'Parameter --print-json detected ...'),
            ('export_csv', 'Parameter --export-csv detected: {}'),
            ('hyperopt_jobs', 'Parameter -j/--job-workers detected: {}'),
            ('hyperopt_random_state', 'Parameter --random-state detected: {}'),
            ('hyperopt_min_trades', 'Parameter --min-trades detected: {}'),
            ('hyperopt_loss', 'Using Hyperopt loss class name: {}'),
            ('hyperopt_show_index', 'Parameter -n/--index detected: {}'),
            ('hyperopt_list_best', 'Parameter --best detected: {}'),
            ('hyperopt_list_profitable', 'Parameter --profitable detected: {}'),
            ('hyperopt_list_min_trades', 'Parameter --min-trades detected: {}'),
            ('hyperopt_list_max_trades', 'Parameter --max-trades detected: {}'),
            ('hyperopt_list_min_avg_time', 'Parameter --min-avg-time detected: {}'),
            ('hyperopt_list_max_avg_time', 'Parameter --max-avg-time detected: {}'),
            ('hyperopt_list_min_avg_profit', 'Parameter --min-avg-profit detected: {}'),
            ('hyperopt_list_max_avg_profit', 'Parameter --max-avg-profit detected: {}'),
            ('hyperopt_list_min_total_profit', 'Parameter --min-total-profit detected: {}'),
            ('hyperopt_list_max_total_profit', 'Parameter --max-total-profit detected: {}'),
            ('hyperopt_list_min_objective', 'Parameter --min-objective detected: {}'),
            ('hyperopt_list_max_objective', 'Parameter --max-objective detected: {}'),
            ('hyperopt_list_no_details', 'Parameter --no-details detected: {}'),
            ('hyperopt_show_no_header', 'Parameter --no-header detected: {}'),
            ('hyperopt_ignore_missing_space', 'Paramter --ignore-missing-space detected: {}'),
        ]

        self._args_to_config_loop(config, configurations)

    def _process_plot_options(self, config: Config) -> None:

        configurations = [
            ('pairs', 'Using pairs {}'),
            ('indicators1', 'Using indicators1: {}'),
            ('indicators2', 'Using indicators2: {}'),
            ('trade_ids', 'Filtering on trade_ids: {}'),
            ('plot_limit', 'Limiting plot to: {}'),
            ('plot_auto_open', 'Parameter --auto-open detected.'),
            ('trade_source', 'Using trades from: {}'),
            ('prepend_data', 'Prepend detected. Allowing data prepending.'),
            ('erase', 'Erase detected. Deleting existing data.'),
            ('no_trades', 'Parameter --no-trades detected.'),
            ('timeframes', 'timeframes --timeframes: {}'),
            ('days', 'Detected --days: {}'),
            ('include_inactive', 'Detected --include-inactive-pairs: {}'),
            ('download_trades', 'Detected --dl-trades: {}'),
            ('dataformat_ohlcv', 'Using "{}" to store OHLCV data.'),
            ('dataformat_trades', 'Using "{}" to store trades data.'),
            ('show_timerange', 'Detected --show-timerange'),
        ]
        self._args_to_config_loop(config, configurations)

    def _process_data_options(self, config: Config) -> None:
        self._args_to_config(config, argname='new_pairs_days',
                             logstring='Detected --new-pairs-days: {}')
        self._args_to_config(config, argname='trading_mode',
                             logstring='Detected --trading-mode: {}')
        config['candle_type_def'] = CandleType.get_default(
            config.get('trading_mode', 'spot') or 'spot')
        config['trading_mode'] = TradingMode(config.get('trading_mode', 'spot') or 'spot')
        self._args_to_config(config, argname='candle_types',
                             logstring='Detected --candle-types: {}')

    def _process_analyze_options(self, config: Config) -> None:
        configurations = [
            ('analysis_groups', 'Analysis reason groups: {}'),
            ('enter_reason_list', 'Analysis enter tag list: {}'),
            ('exit_reason_list', 'Analysis exit tag list: {}'),
            ('indicator_list', 'Analysis indicator list: {}'),
            ('timerange', 'Filter trades by timerange: {}'),
            ('analysis_rejected', 'Analyse rejected signals: {}'),
            ('analysis_to_csv', 'Store analysis tables to CSV: {}'),
            ('analysis_csv_path', 'Path to store analysis CSVs: {}'),
            # Lookahead analysis results
            ('targeted_trade_amount', 'Targeted Trade amount: {}'),
            ('minimum_trade_amount', 'Minimum Trade amount: {}'),
            ('lookahead_analysis_exportfilename', 'Path to store lookahead-analysis-results: {}'),
            ('startup_candle', 'Startup candle to be used on recursive analysis: {}'),
        ]
        self._args_to_config_loop(config, configurations)

    def _args_to_config_loop(self, config, configurations: List[Tuple[str, str]]) -> None:

        for argname, logstring in configurations:
            self._args_to_config(config, argname=argname, logstring=logstring)

    def _process_runmode(self, config: Config) -> None:

        self._args_to_config(config, argname='dry_run',
                             logstring='Parameter --dry-run detected, '
                             'overriding dry_run to: {} ...')

        if not self.runmode:
            # Handle real mode, infer dry/live from config
            self.runmode = RunMode.DRY_RUN if config.get('dry_run', True) else RunMode.LIVE
            logger.info(f"Runmode set to {self.runmode.value}.")

        config.update({'runmode': self.runmode})

    def _process_freqai_options(self, config: Config) -> None:

        self._args_to_config(config, argname='freqaimodel',
                             logstring='Using freqaimodel class name: {}')

        self._args_to_config(config, argname='freqaimodel_path',
                             logstring='Using freqaimodel path: {}')

        return

    def _args_to_config(self, config: Config, argname: str,
                        logstring: str, logfun: Optional[Callable] = None,
                        deprecated_msg: Optional[str] = None) -> None:
        """
        :param config: Configuration dictionary
        :param argname: Argumentname in self.args - will be copied to config dict.
        :param logstring: Logging String
        :param logfun: logfun is applied to the configuration entry before passing
                        that entry to the log string using .format().
                        sample: logfun=len (prints the length of the found
                        configuration instead of the content)
        """
        if (argname in self.args and self.args[argname] is not None
                and self.args[argname] is not False):

            config.update({argname: self.args[argname]})
            if logfun:
                logger.info(logstring.format(logfun(config[argname])))
            else:
                logger.info(logstring.format(config[argname]))
            if deprecated_msg:
                warnings.warn(f"DEPRECATED: {deprecated_msg}", DeprecationWarning)

    def _resolve_pairs_list(self, config: Config) -> None:
        """
        Helper for download script.
        Takes first found:
        * -p (pairs argument)
        * --pairs-file
        * whitelist from config
        """

        if "pairs" in config:
            config['exchange']['pair_whitelist'] = config['pairs']
            return

        if "pairs_file" in self.args and self.args["pairs_file"]:
            pairs_file = Path(self.args["pairs_file"])
            logger.info(f'Reading pairs file "{pairs_file}".')
            # Download pairs from the pairs file if no config is specified
            # or if pairs file is specified explicitly
            if not pairs_file.exists():
                raise OperationalException(f'No pairs file found with path "{pairs_file}".')
            config['pairs'] = load_file(pairs_file)
            if isinstance(config['pairs'], list):
                config['pairs'].sort()
            return

        if 'config' in self.args and self.args['config']:
            logger.info("Using pairlist from configuration.")
            config['pairs'] = config.get('exchange', {}).get('pair_whitelist')
        else:
            # Fall back to /dl_path/pairs.json
            pairs_file = config['datadir'] / 'pairs.json'
            if pairs_file.exists():
                logger.info(f'Reading pairs file "{pairs_file}".')
                config['pairs'] = load_file(pairs_file)
                if 'pairs' in config and isinstance(config['pairs'], list):
                    config['pairs'].sort()
