"""
This module contains the configuration class
"""
import json
import logging
import sys
from argparse import Namespace
from typing import Any, Callable, Dict, Optional

from freqtrade import OperationalException, constants
from freqtrade.configuration.check_exchange import check_exchange
from freqtrade.configuration.create_datadir import create_datadir
from freqtrade.configuration.json_schema import validate_config_schema
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

    def load_config(self) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load the bot configuration
        :return: Configuration dictionary
        """
        config: Dict[str, Any] = {}
        # Now expecting a list of config filenames here, not a string
        for path in self.args.config:
            logger.info('Using config: %s ...', path)

            # Merge config options, overwriting old values
            config = deep_merge_dicts(self._load_config_file(path), config)

        if 'internals' not in config:
            config['internals'] = {}

        logger.info('Validating configuration ...')
        validate_config_schema(config)
        self._validate_config_consistency(config)

        # Set strategy if not specified in config and or if it's non default
        if self.args.strategy != constants.DEFAULT_STRATEGY or not config.get('strategy'):
            config.update({'strategy': self.args.strategy})

        if self.args.strategy_path:
            config.update({'strategy_path': self.args.strategy_path})

        # Load Common configuration
        config = self._load_common_config(config)

        # Load Optimize configurations
        config = self._load_optimize_config(config)

        # Add plotting options if available
        config = self._load_plot_config(config)

        # Set runmode
        if not self.runmode:
            # Handle real mode, infer dry/live from config
            self.runmode = RunMode.DRY_RUN if config.get('dry_run', True) else RunMode.LIVE

        config.update({'runmode': self.runmode})

        return config

    def _load_config_file(self, path: str) -> Dict[str, Any]:
        """
        Loads a config file from the given path
        :param path: path as str
        :return: configuration as dictionary
        """
        try:
            # Read config from stdin if requested in the options
            with open(path) if path != '-' else sys.stdin as file:
                conf = json.load(file)
        except FileNotFoundError:
            raise OperationalException(
                f'Config file "{path}" not found!'
                ' Please create a config file or check whether it exists.')

        return conf

    def _load_logging_config(self, config: Dict[str, Any]) -> None:
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

    def _load_common_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load common configuration
        :return: configuration as dictionary
        """
        self._load_logging_config(config)

        # Support for sd_notify
        if 'sd_notify' in self.args and self.args.sd_notify:
            config['internals'].update({'sd_notify': True})

        # Add dynamic_whitelist if found
        if 'dynamic_whitelist' in self.args and self.args.dynamic_whitelist:
            # Update to volumePairList (the previous default)
            config['pairlist'] = {'method': 'VolumePairList',
                                  'config': {'number_assets': self.args.dynamic_whitelist}
                                  }
            logger.warning(
                'Parameter --dynamic-whitelist has been deprecated, '
                'and will be completely replaced by the whitelist dict in the future. '
                'For now: using dynamically generated whitelist based on VolumePairList. '
                '(not applicable with Backtesting and Hyperopt)'
            )

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

        if config.get('forcebuy_enable', False):
            logger.warning('`forcebuy` RPC message enabled.')

        # Setting max_open_trades to infinite if -1
        if config.get('max_open_trades') == -1:
            config['max_open_trades'] = float('inf')

        logger.info(f'Using DB: "{config["db_url"]}"')

        # Check if the exchange set by the user is supported
        check_exchange(config)

        return config

    def _args_to_config(self, config: Dict[str, Any], argname: str,
                        logstring: str, logfun: Optional[Callable] = None) -> None:
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

    def _load_datadir_config(self, config: Dict[str, Any]) -> None:
        """
        Extract information for sys.argv and load datadir configuration:
        the --datadir option
        """
        if 'datadir' in self.args and self.args.datadir:
            config.update({'datadir': create_datadir(config, self.args.datadir)})
        else:
            config.update({'datadir': create_datadir(config, None)})
        logger.info('Using data directory: %s ...', config.get('datadir'))

    def _load_optimize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load Optimize configuration
        :return: configuration as dictionary
        """

        # This will override the strategy configuration
        self._args_to_config(config, argname='ticker_interval',
                             logstring='Parameter -i/--ticker-interval detected ... '
                             'Using ticker_interval: {} ...')

        self._args_to_config(config, argname='live',
                             logstring='Parameter -l/--live detected ...')

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

        self._load_datadir_config(config)

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

        self._args_to_config(config, argname='loss_function',
                             logstring='Using loss function: {}')

        return config

    def _load_plot_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information for sys.argv Plotting configuration
        :return: configuration as dictionary
        """

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
        return config

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
