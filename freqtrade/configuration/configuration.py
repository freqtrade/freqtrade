"""
This module contains the configuration class
"""
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from freqtrade import OperationalException, constants
from freqtrade.configuration.check_exchange import check_exchange
from freqtrade.configuration.config_validation import (
    validate_config_consistency, validate_config_schema)
from freqtrade.configuration.directory_operations import (create_datadir,
                                                          create_userdata_dir)
from freqtrade.configuration.load_config import load_config_file
from freqtrade.loggers import setup_logging
from freqtrade.misc import deep_merge_dicts, json_load
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


class Configuration:
    """
    Class to read and init the bot configuration
    Reuse this class for the bot, backtesting, hyperopt and every script that required configuration
    """

    def __init__(self, args: Dict[str, Any], runmode: RunMode = None) -> None:
        self.args = args
        self.config: Optional[Dict[str, Any]] = None
        self.runmode = runmode

    def get_config(self) -> Dict[str, Any]:
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
        :param files: List of file paths
        :return: configuration dictionary
        """
        # Keep this method as staticmethod, so it can be used from interactive environments
        config: Dict[str, Any] = {}

        if not files:
            return deepcopy(constants.MINIMAL_CONFIG)

        # We expect here a list of config filenames
        for path in files:
            logger.info(f'Using config: {path} ...')

            # Merge config options, overwriting old values
            config = deep_merge_dicts(load_config_file(path), config)

        # Normalize config
        if 'internals' not in config:
            config['internals'] = {}

        # validate configuration before returning
        logger.info('Validating configuration ...')
        validate_config_schema(config)

        return config

    def load_config(self) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load the bot configuration
        :return: Configuration dictionary
        """
        # Load all configs
        config: Dict[str, Any] = Configuration.from_files(self.args["config"])

        self._process_common_options(config)

        self._process_optimize_options(config)

        self._process_plot_options(config)

        self._process_runmode(config)

        # Check if the exchange set by the user is supported
        check_exchange(config, config.get('experimental', {}).get('block_bad_exchanges', True))

        self._resolve_pairs_list(config)

        validate_config_consistency(config)

        return config

    def _process_logging_options(self, config: Dict[str, Any]) -> None:
        """
        Extract information for sys.argv and load logging configuration:
        the -v/--verbose, --logfile options
        """
        # Log level
        config.update({'verbosity': self.args.get("verbosity", 0)})

        if 'logfile' in self.args and self.args["logfile"]:
            config.update({'logfile': self.args["logfile"]})

        setup_logging(config)

    def _process_common_options(self, config: Dict[str, Any]) -> None:

        self._process_logging_options(config)

        # Set strategy if not specified in config and or if it's non default
        if self.args.get("strategy") != constants.DEFAULT_STRATEGY or not config.get('strategy'):
            config.update({'strategy': self.args.get("strategy")})

        self._args_to_config(config, argname='strategy_path',
                             logstring='Using additional Strategy lookup path: {}')

        if ('db_url' in self.args and self.args["db_url"] and
                self.args["db_url"] != constants.DEFAULT_DB_PROD_URL):
            config.update({'db_url': self.args["db_url"]})
            logger.info('Parameter --db-url detected ...')

        if config.get('dry_run', False):
            logger.info('Dry run is enabled')
            if config.get('db_url') in [None, constants.DEFAULT_DB_PROD_URL]:
                # Default to in-memory db for dry_run if not specified
                config['db_url'] = constants.DEFAULT_DB_DRYRUN_URL
        else:
            if not config.get('db_url', None):
                config['db_url'] = constants.DEFAULT_DB_PROD_URL
            logger.info('Dry run is disabled')

        logger.info(f'Using DB: "{config["db_url"]}"')

        if config.get('forcebuy_enable', False):
            logger.warning('`forcebuy` RPC message enabled.')

        # Setting max_open_trades to infinite if -1
        if config.get('max_open_trades') == -1:
            config['max_open_trades'] = float('inf')

        # Support for sd_notify
        if 'sd_notify' in self.args and self.args["sd_notify"]:
            config['internals'].update({'sd_notify': True})

    def _process_datadir_options(self, config: Dict[str, Any]) -> None:
        """
        Extract information for sys.argv and load directory configurations
        --user-data, --datadir
        """
        # Check exchange parameter here - otherwise `datadir` might be wrong.
        if "exchange" in self.args and self.args["exchange"]:
            config['exchange']['name'] = self.args["exchange"]
            logger.info(f"Using exchange {config['exchange']['name']}")

        if 'user_data_dir' in self.args and self.args["user_data_dir"]:
            config.update({'user_data_dir': self.args["user_data_dir"]})
        elif 'user_data_dir' not in config:
            # Default to cwd/user_data (legacy option ...)
            config.update({'user_data_dir': str(Path.cwd() / "user_data")})

        # reset to user_data_dir so this contains the absolute path.
        config['user_data_dir'] = create_userdata_dir(config['user_data_dir'], create_dir=False)
        logger.info('Using user-data directory: %s ...', config['user_data_dir'])

        config.update({'datadir': create_datadir(config, self.args.get("datadir", None))})
        logger.info('Using data directory: %s ...', config.get('datadir'))

    def _process_optimize_options(self, config: Dict[str, Any]) -> None:

        # This will override the strategy configuration
        self._args_to_config(config, argname='ticker_interval',
                             logstring='Parameter -i/--ticker-interval detected ... '
                             'Using ticker_interval: {} ...')

        self._args_to_config(config, argname='position_stacking',
                             logstring='Parameter --enable-position-stacking detected ...')

        if 'use_max_market_positions' in self.args and not self.args["use_max_market_positions"]:
            config.update({'use_max_market_positions': False})
            logger.info('Parameter --disable-max-market-positions detected ...')
            logger.info('max_open_trades set to unlimited ...')
        elif 'max_open_trades' in self.args and self.args["max_open_trades"]:
            config.update({'max_open_trades': self.args["max_open_trades"]})
            logger.info('Parameter --max_open_trades detected, '
                        'overriding max_open_trades to: %s ...', config.get('max_open_trades'))
        else:
            logger.info('Using max_open_trades: %s ...', config.get('max_open_trades'))

        self._args_to_config(config, argname='stake_amount',
                             logstring='Parameter --stake_amount detected, '
                             'overriding stake_amount to: {} ...')

        self._args_to_config(config, argname='timerange',
                             logstring='Parameter --timerange detected: {} ...')

        self._process_datadir_options(config)

        self._args_to_config(config, argname='refresh_pairs',
                             logstring='Parameter -r/--refresh-pairs-cached detected ...',
                             deprecated_msg='-r/--refresh-pairs-cached will be removed soon.')

        self._args_to_config(config, argname='strategy_list',
                             logstring='Using strategy list of {} Strategies', logfun=len)

        self._args_to_config(config, argname='ticker_interval',
                             logstring='Overriding ticker interval with Command line argument')

        self._args_to_config(config, argname='export',
                             logstring='Parameter --export detected: {} ...')

        self._args_to_config(config, argname='exportfilename',
                             logstring='Storing backtest results to {} ...')

        # Edge section:
        if 'stoploss_range' in self.args and self.args["stoploss_range"]:
            txt_range = eval(self.args["stoploss_range"])
            config['edge'].update({'stoploss_range_min': txt_range[0]})
            config['edge'].update({'stoploss_range_max': txt_range[1]})
            config['edge'].update({'stoploss_range_step': txt_range[2]})
            logger.info('Parameter --stoplosses detected: %s ...', self.args["stoploss_range"])

        # Hyperopt section
        self._args_to_config(config, argname='hyperopt',
                             logstring='Using Hyperopt file {}')

        self._args_to_config(config, argname='hyperopt_path',
                             logstring='Using additional Hyperopt lookup path: {}')

        self._args_to_config(config, argname='epochs',
                             logstring='Parameter --epochs detected ... '
                             'Will run Hyperopt with for {} epochs ...'
                             )

        self._args_to_config(config, argname='spaces',
                             logstring='Parameter -s/--spaces detected: {}')

        self._args_to_config(config, argname='print_all',
                             logstring='Parameter --print-all detected ...')

        if 'print_colorized' in self.args and not self.args["print_colorized"]:
            logger.info('Parameter --no-color detected ...')
            config.update({'print_colorized': False})
        else:
            config.update({'print_colorized': True})

        self._args_to_config(config, argname='print_json',
                             logstring='Parameter --print-json detected ...')

        self._args_to_config(config, argname='hyperopt_jobs',
                             logstring='Parameter -j/--job-workers detected: {}')

        self._args_to_config(config, argname='hyperopt_random_state',
                             logstring='Parameter --random-state detected: {}')

        self._args_to_config(config, argname='hyperopt_min_trades',
                             logstring='Parameter --min-trades detected: {}')

        self._args_to_config(config, argname='hyperopt_continue',
                             logstring='Hyperopt continue: {}')

        self._args_to_config(config, argname='hyperopt_loss',
                             logstring='Using loss function: {}')

    def _process_plot_options(self, config: Dict[str, Any]) -> None:

        self._args_to_config(config, argname='pairs',
                             logstring='Using pairs {}')

        self._args_to_config(config, argname='indicators1',
                             logstring='Using indicators1: {}')

        self._args_to_config(config, argname='indicators2',
                             logstring='Using indicators2: {}')

        self._args_to_config(config, argname='plot_limit',
                             logstring='Limiting plot to: {}')
        self._args_to_config(config, argname='trade_source',
                             logstring='Using trades from: {}')

        self._args_to_config(config, argname='erase',
                             logstring='Erase detected. Deleting existing data.')

        self._args_to_config(config, argname='timeframes',
                             logstring='timeframes --timeframes: {}')

        self._args_to_config(config, argname='days',
                             logstring='Detected --days: {}')

    def _process_runmode(self, config: Dict[str, Any]) -> None:

        if not self.runmode:
            # Handle real mode, infer dry/live from config
            self.runmode = RunMode.DRY_RUN if config.get('dry_run', True) else RunMode.LIVE
            logger.info(f"Runmode set to {self.runmode}.")

        config.update({'runmode': self.runmode})

    def _args_to_config(self, config: Dict[str, Any], argname: str,
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
        if argname in self.args and self.args[argname]:

            config.update({argname: self.args[argname]})
            if logfun:
                logger.info(logstring.format(logfun(config[argname])))
            else:
                logger.info(logstring.format(config[argname]))
            if deprecated_msg:
                warnings.warn(f"DEPRECATED: {deprecated_msg}", DeprecationWarning)

    def _resolve_pairs_list(self, config: Dict[str, Any]) -> None:
        """
        Helper for download script.
        Takes first found:
        * -p (pairs argument)
        * --pairs-file
        * whitelist from config
        """

        if "pairs" in config:
            return

        if "pairs_file" in self.args and self.args["pairs_file"]:
            pairs_file = Path(self.args["pairs_file"])
            logger.info(f'Reading pairs file "{pairs_file}".')
            # Download pairs from the pairs file if no config is specified
            # or if pairs file is specified explicitely
            if not pairs_file.exists():
                raise OperationalException(f'No pairs file found with path "{pairs_file}".')
            with pairs_file.open('r') as f:
                config['pairs'] = json_load(f)
                config['pairs'].sort()
            return

        if "config" in self.args and self.args["config"]:
            logger.info("Using pairlist from configuration.")
            config['pairs'] = config.get('exchange', {}).get('pair_whitelist')
        else:
            # Fall back to /dl_path/pairs.json
            pairs_file = Path(config['datadir']) / "pairs.json"
            if pairs_file.exists():
                with pairs_file.open('r') as f:
                    config['pairs'] = json_load(f)
                if 'pairs' in config:
                    config['pairs'].sort()
