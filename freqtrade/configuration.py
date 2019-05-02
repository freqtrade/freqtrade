"""
This module contains the configuration class
"""
import json
import logging
import os
import sys
from argparse import Namespace
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, List, Optional

from jsonschema import Draft4Validator, validators
from jsonschema.exceptions import ValidationError, best_match

from freqtrade import OperationalException, constants
from freqtrade.exchange import is_exchange_supported, supported_exchanges
from freqtrade.misc import deep_merge_dicts
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


def set_loggers(log_level: int = 0) -> None:
    """
    Set the logger level for Third party libs
    :return: None
    """

    logging.getLogger('requests').setLevel(logging.INFO if log_level <= 1 else logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.INFO if log_level <= 1 else logging.DEBUG)
    logging.getLogger('ccxt.base.exchange').setLevel(
        logging.INFO if log_level <= 2 else logging.DEBUG)
    logging.getLogger('telegram').setLevel(logging.INFO)


def _extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for prop, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(prop, subschema["default"])

        for error in validate_properties(
            validator, properties, instance, schema,
        ):
            yield error

    return validators.extend(
        validator_class, {"properties": set_defaults},
    )


ValidatorWithDefaults = _extend_with_default(Draft4Validator)


class Configuration(object):
    """
    Class to read and init the bot configuration
    Reuse this class for the bot, backtesting, hyperopt and every script that required configuration
    """

    def __init__(self, args: Namespace, runmode: RunMode = None) -> None:
        self.args = args
        self.config: Optional[Dict[str, Any]] = None
        self.runmode = runmode

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
        self._validate_config_schema(config)
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
            with open(path) as file:
                conf = json.load(file)
        except FileNotFoundError:
            raise OperationalException(
                f'Config file "{path}" not found!'
                ' Please create a config file or check whether it exists.')

        return conf

    def _load_common_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load common configuration
        :return: configuration as dictionary
        """

        # Log level
        if 'loglevel' in self.args and self.args.loglevel:
            config.update({'verbosity': self.args.loglevel})
        else:
            config.update({'verbosity': 0})

        # Log to stdout, not stderr
        log_handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
        if 'logfile' in self.args and self.args.logfile:
            config.update({'logfile': self.args.logfile})

        # Allow setting this as either configuration or argument
        if 'logfile' in config:
            log_handlers.append(RotatingFileHandler(config['logfile'],
                                                    maxBytes=1024 * 1024,  # 1Mb
                                                    backupCount=10))

        logging.basicConfig(
            level=logging.INFO if config['verbosity'] < 1 else logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=log_handlers
        )
        set_loggers(config['verbosity'])
        logger.info('Verbosity set to %s', config['verbosity'])

        # Support for sd_notify
        if self.args.sd_notify:
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

        if self.args.db_url and self.args.db_url != constants.DEFAULT_DB_PROD_URL:
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
        self.check_exchange(config)

        return config

    def _create_datadir(self, config: Dict[str, Any], datadir: Optional[str] = None) -> str:
        if not datadir:
            # set datadir
            exchange_name = config.get('exchange', {}).get('name').lower()
            datadir = os.path.join('user_data', 'data', exchange_name)

        if not os.path.isdir(datadir):
            os.makedirs(datadir)
            logger.info(f'Created data directory: {datadir}')
        return datadir

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

        if 'datadir' in self.args and self.args.datadir:
            config.update({'datadir': self._create_datadir(config, self.args.datadir)})
        else:
            config.update({'datadir': self._create_datadir(config, None)})
        logger.info('Using data folder: %s ...', config.get('datadir'))

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

        return config

    def _validate_config_schema(self, conf: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the configuration follow the Config Schema
        :param conf: Config in JSON format
        :return: Returns the config if valid, otherwise throw an exception
        """
        try:
            ValidatorWithDefaults(constants.CONF_SCHEMA).validate(conf)
            return conf
        except ValidationError as exception:
            logger.critical(
                'Invalid configuration. See config.json.example. Reason: %s',
                exception
            )
            raise ValidationError(
                best_match(Draft4Validator(constants.CONF_SCHEMA).iter_errors(conf)).message
            )

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

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config. Use this method to get the bot config
        :return: Dict: Bot config
        """
        if self.config is None:
            self.config = self.load_config()

        return self.config

    def check_exchange(self, config: Dict[str, Any]) -> bool:
        """
        Check if the exchange name in the config file is supported by Freqtrade
        :return: True or raised an exception if the exchange if not supported
        """
        exchange = config.get('exchange', {}).get('name').lower()
        if not is_exchange_supported(exchange):

            exception_msg = f'Exchange "{exchange}" not supported.\n' \
                            f'The following exchanges are supported: ' \
                            f'{", ".join(supported_exchanges())}'

            logger.critical(exception_msg)
            raise OperationalException(
                exception_msg
            )

        logger.debug('Exchange "%s" supported', exchange)
        return True
