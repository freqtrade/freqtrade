"""
This module contains the configuration class
"""
import os
import json
import logging
from argparse import Namespace
from typing import Optional, Dict, Any
from jsonschema import Draft4Validator, validate
from jsonschema.exceptions import ValidationError, best_match
import ccxt

from freqtrade import OperationalException, constants


logger = logging.getLogger(__name__)


class Configuration(object):
    """
    Class to read and init the bot configuration
    Reuse this class for the bot, backtesting, hyperopt and every script that required configuration
    """
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.config: Optional[Dict[str, Any]] = None

    def load_config(self) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load the bot configuration
        :return: Configuration dictionary
        """
        logger.info('Using config: %s ...', self.args.config)
        config = self._load_config_file(self.args.config)

        # Set strategy if not specified in config and or if it's non default
        if self.args.strategy != constants.DEFAULT_STRATEGY or not config.get('strategy'):
            config.update({'strategy': self.args.strategy})

        if self.args.strategy_path:
            config.update({'strategy_path': self.args.strategy_path})

        # Load Common configuration
        config = self._load_common_config(config)

        # Load Backtesting
        config = self._load_backtesting_config(config)

        # Load Hyperopt
        config = self._load_hyperopt_config(config)

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
                'Config file "{}" not found!'
                ' Please create a config file or check whether it exists.'.format(path))

        if 'internals' not in conf:
            conf['internals'] = {}
        logger.info('Validating configuration ...')

        return self._validate_config(conf)

    def _load_common_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load common configuration
        :return: configuration as dictionary
        """

        # Log level
        if 'loglevel' in self.args and self.args.loglevel:
            config.update({'loglevel': self.args.loglevel})
            logging.basicConfig(
                level=config['loglevel'],
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            )
            logger.info('Log level set to %s', logging.getLevelName(config['loglevel']))

        # Add dynamic_whitelist if found
        if 'dynamic_whitelist' in self.args and self.args.dynamic_whitelist:
            config.update({'dynamic_whitelist': self.args.dynamic_whitelist})
            logger.info(
                'Parameter --dynamic-whitelist detected. '
                'Using dynamically generated whitelist. '
                '(not applicable with Backtesting and Hyperopt)'
            )

        if self.args.db_url != constants.DEFAULT_DB_PROD_URL:
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

        logger.info('Using DB: "{}"'.format(config['db_url']))

        # Check if the exchange set by the user is supported
        self.check_exchange(config)

        return config

    def _create_default_datadir(self, config: Dict[str, Any]) -> str:
        exchange_name = config.get('exchange', {}).get('name').lower()
        default_path = os.path.join('user_data', 'data', exchange_name)
        if not os.path.isdir(default_path):
            os.makedirs(default_path)
            logger.info(f'Created data directory: {default_path}')
        return default_path

    def _load_backtesting_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load Backtesting configuration
        :return: configuration as dictionary
        """

        # If -i/--ticker-interval is used we override the configuration parameter
        # (that will override the strategy configuration)
        if 'ticker_interval' in self.args and self.args.ticker_interval:
            config.update({'ticker_interval': self.args.ticker_interval})
            logger.info('Parameter -i/--ticker-interval detected ...')
            logger.info('Using ticker_interval: %s ...', config.get('ticker_interval'))

        # If -l/--live is used we add it to the configuration
        if 'live' in self.args and self.args.live:
            config.update({'live': True})
            logger.info('Parameter -l/--live detected ...')

        # If --realistic-simulation is used we add it to the configuration
        if 'realistic_simulation' in self.args and self.args.realistic_simulation:
            config.update({'realistic_simulation': True})
            logger.info('Parameter --realistic-simulation detected ...')
        logger.info('Using max_open_trades: %s ...', config.get('max_open_trades'))

        # If --timerange is used we add it to the configuration
        if 'timerange' in self.args and self.args.timerange:
            config.update({'timerange': self.args.timerange})
            logger.info('Parameter --timerange detected: %s ...', self.args.timerange)

        # If --datadir is used we add it to the configuration
        if 'datadir' in self.args and self.args.datadir:
            config.update({'datadir': self.args.datadir})
        else:
            config.update({'datadir': self._create_default_datadir(config)})
        logger.info('Using data folder: %s ...', config.get('datadir'))

        # If -r/--refresh-pairs-cached is used we add it to the configuration
        if 'refresh_pairs' in self.args and self.args.refresh_pairs:
            config.update({'refresh_pairs': True})
            logger.info('Parameter -r/--refresh-pairs-cached detected ...')

        # If --export is used we add it to the configuration
        if 'export' in self.args and self.args.export:
            config.update({'export': self.args.export})
            logger.info('Parameter --export detected: %s ...', self.args.export)

        # If --export-filename is used we add it to the configuration
        if 'export' in config and 'exportfilename' in self.args and self.args.exportfilename:
            config.update({'exportfilename': self.args.exportfilename})
            logger.info('Storing backtest results to %s ...', self.args.exportfilename)

        return config

    def _load_hyperopt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load Hyperopt configuration
        :return: configuration as dictionary
        """
        # If --realistic-simulation is used we add it to the configuration
        if 'epochs' in self.args and self.args.epochs:
            config.update({'epochs': self.args.epochs})
            logger.info('Parameter --epochs detected ...')
            logger.info('Will run Hyperopt with for %s epochs ...', config.get('epochs'))

        # If --spaces is used we add it to the configuration
        if 'spaces' in self.args and self.args.spaces:
            config.update({'spaces': self.args.spaces})
            logger.info('Parameter -s/--spaces detected: %s', config.get('spaces'))

        return config

    def _validate_config(self, conf: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the configuration follow the Config Schema
        :param conf: Config in JSON format
        :return: Returns the config if valid, otherwise throw an exception
        """
        try:
            validate(conf, constants.CONF_SCHEMA)
            return conf
        except ValidationError as exception:
            logger.critical(
                'Invalid configuration. See config.json.example. Reason: %s',
                exception
            )
            raise ValidationError(
                best_match(Draft4Validator(constants.CONF_SCHEMA).iter_errors(conf)).message
            )

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
        if exchange not in ccxt.exchanges:

            exception_msg = f'Exchange "{exchange}" not supported.\n' \
                            f'The following exchanges are supported: {", ".join(ccxt.exchanges)}'

            logger.critical(exception_msg)
            raise OperationalException(
                exception_msg
            )

        logger.debug('Exchange "%s" supported', exchange)
        return True
