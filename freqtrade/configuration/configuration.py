"""
This module contains the configuration class
"""
import logging
import warnings
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional

from freqtrade import OperationalException, constants
from freqtrade.configuration.check_exchange import check_exchange
from freqtrade.configuration.create_datadir import create_datadir
from freqtrade.configuration.json_schema import validate_config_schema
from freqtrade.configuration.load_config import load_config_file
from freqtrade.loggers import setup_logging
from freqtrade.misc import deep_merge_dicts
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


class Configuration(object):
    """
    Class to read and init the bot configuration
    Reuse this class for the bot, backtesting, hyperopt and every script that required configuration
    """

    def __init__(self, args: Namespace, runmode: RunMode = None) -> None:
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
        config: Dict[str, Any] = Configuration.from_files(self.args.config)

        self._validate_config_consistency(config)

        self._process_common_options(config)

        self._process_optimize_options(config)

        self._process_plot_options(config)

        self._process_runmode(config)

        return config

    def _process_logging_options(self, config: Dict[str, Any]) -> None:
        """
        Extract information for sys.argv and load logging configuration:
        the -v/--verbose, --logfile options
        """
        # Log level
        if 'verbosity' in self.args and self.args.verbosity:
            config.update({'verbosity': self.args.verbosity})
        else:
            config.update({'verbosity': 0})

        if 'logfile' in self.args and self.args.logfile:
            config.update({'logfile': self.args.logfile})

        setup_logging(config)

    def _process_strategy_options(self, config: Dict[str, Any]) -> None:

        # Set strategy if not specified in config and or if it's non default
        if self.args.strategy != constants.DEFAULT_STRATEGY or not config.get('strategy'):
            config.update({'strategy': self.args.strategy})

        self._args_to_config(config, argname='strategy_path',
                             logstring='Using additional Strategy lookup path: {}')

    def _process_common_options(self, config: Dict[str, Any]) -> None:

        self._process_logging_options(config)
        self._process_strategy_options(config)

        if ('db_url' in self.args and self.args.db_url and
                self.args.db_url != constants.DEFAULT_DB_PROD_URL):
            config.update({'db_url': self.args.db_url})
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
        if 'sd_notify' in self.args and self.args.sd_notify:
            config['internals'].update({'sd_notify': True})

        # Check if the exchange set by the user is supported
        check_exchange(config, config.get('experimental', {}).get('block_bad_exchanges', True))

    def _process_datadir_options(self, config: Dict[str, Any]) -> None:
        """
        Extract information for sys.argv and load datadir configuration:
        the --datadir option
        """
        if 'datadir' in self.args and self.args.datadir:
            config.update({'datadir': create_datadir(config, self.args.datadir)})
        else:
            config.update({'datadir': create_datadir(config, None)})
        logger.info('Using data directory: %s ...', config.get('datadir'))

    def _process_optimize_options(self, config: Dict[str, Any]) -> None:

        # This will override the strategy configuration
        self._args_to_config(config, argname='ticker_interval',
                             logstring='Parameter -i/--ticker-interval detected ... '
                             'Using ticker_interval: {} ...')

        self._args_to_config(config, argname='live',
                             logstring='Parameter -l/--live detected ...',
                             deprecated_msg='--live will be removed soon.')

        self._args_to_config(config, argname='position_stacking',
                             logstring='Parameter --enable-position-stacking detected ...')

        if 'use_max_market_positions' in self.args and not self.args.use_max_market_positions:
            config.update({'use_max_market_positions': False})
            logger.info('Parameter --disable-max-market-positions detected ...')
            logger.info('max_open_trades set to unlimited ...')
        elif 'max_open_trades' in self.args and self.args.max_open_trades:
            config.update({'max_open_trades': self.args.max_open_trades})
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
                             logstring='Parameter -r/--refresh-pairs-cached detected ...')

        self._args_to_config(config, argname='strategy_list',
                             logstring='Using strategy list of {} Strategies', logfun=len)

        self._args_to_config(config, argname='ticker_interval',
                             logstring='Overriding ticker interval with Command line argument')

        self._args_to_config(config, argname='export',
                             logstring='Parameter --export detected: {} ...')

        self._args_to_config(config, argname='exportfilename',
                             logstring='Storing backtest results to {} ...')

        # Edge section:
        if 'stoploss_range' in self.args and self.args.stoploss_range:
            txt_range = eval(self.args.stoploss_range)
            config['edge'].update({'stoploss_range_min': txt_range[0]})
            config['edge'].update({'stoploss_range_max': txt_range[1]})
            config['edge'].update({'stoploss_range_step': txt_range[2]})
            logger.info('Parameter --stoplosses detected: %s ...', self.args.stoploss_range)

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

    def _process_runmode(self, config: Dict[str, Any]) -> None:

        if not self.runmode:
            # Handle real mode, infer dry/live from config
            self.runmode = RunMode.DRY_RUN if config.get('dry_run', True) else RunMode.LIVE
            logger.info("Runmode set to {self.runmode}.")

        config.update({'runmode': self.runmode})

    def _validate_config_consistency(self, conf: Dict[str, Any]) -> None:
        """
        Validate the configuration consistency
        :param conf: Config in JSON format
        :return: Returns None if everything is ok, otherwise throw an OperationalException
        """
        # validating trailing stoploss
        self._validate_trailing_stoploss(conf)

    def _validate_trailing_stoploss(self, conf: Dict[str, Any]) -> None:

        # Skip if trailing stoploss is not activated
        if not conf.get('trailing_stop', False):
            return

        tsl_positive = float(conf.get('trailing_stop_positive', 0))
        tsl_offset = float(conf.get('trailing_stop_positive_offset', 0))
        tsl_only_offset = conf.get('trailing_only_offset_is_reached', False)

        if tsl_only_offset:
            if tsl_positive == 0.0:
                raise OperationalException(
                    f'The config trailing_only_offset_is_reached needs '
                    'trailing_stop_positive_offset to be more than 0 in your config.')
        if tsl_positive > 0 and 0 < tsl_offset <= tsl_positive:
            raise OperationalException(
                f'The config trailing_stop_positive_offset needs '
                'to be greater than trailing_stop_positive_offset in your config.')

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
        if argname in self.args and getattr(self.args, argname):

            config.update({argname: getattr(self.args, argname)})
            if logfun:
                logger.info(logstring.format(logfun(config[argname])))
            else:
                logger.info(logstring.format(config[argname]))
            if deprecated_msg:
                warnings.warn(f"DEPRECATED: {deprecated_msg}", DeprecationWarning)
