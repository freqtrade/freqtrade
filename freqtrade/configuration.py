"""
This module contains the configuration class
"""

import json
import logging
from argparse import Namespace
from typing import Dict, Any

from jsonschema import Draft4Validator, validate
from jsonschema.exceptions import ValidationError, best_match

from freqtrade.constants import Constants


logger = logging.getLogger(__name__)


class Configuration(object):
    """
    Class to read and init the bot configuration
    Reuse this class for the bot, backtesting, hyperopt and every script that required configuration
    """
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.config = None

    def load_config(self) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load the bot configuration
        :return: Configuration dictionary
        """
        logger.info('Using config: %s ...', self.args.config)
        config = self._load_config_file(self.args.config)

        # Add the strategy file to use
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
            logger.critical(
                'Config file "%s" not found. Please create your config file',
                path
            )
            exit(0)

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

        # Add dry_run_db if found and the bot in dry run
        if self.args.dry_run_db and config.get('dry_run', False):
            config.update({'dry_run_db': True})
            logger.info('Parameter --dry-run-db detected ...')

        if config.get('dry_run_db', False):
            if config.get('dry_run', False):
                logger.info('Dry_run will use the DB file: "tradesv3.dry_run.sqlite"')
            else:
                logger.info('Dry run is disabled. (--dry_run_db ignored)')

        return config

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
            logger.info('Using ticker_interval: %d ...', config.get('ticker_interval'))

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
            logger.info('Parameter --datadir detected: %s ...', self.args.datadir)

        # If -r/--refresh-pairs-cached is used we add it to the configuration
        if 'refresh_pairs' in self.args and self.args.refresh_pairs:
            config.update({'refresh_pairs': True})
            logger.info('Parameter -r/--refresh-pairs-cached detected ...')

        # If --export is used we add it to the configuration
        if 'export' in self.args and self.args.export:
            config.update({'export': self.args.export})
            logger.info('Parameter --export detected: %s ...', self.args.export)

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

        # If --mongodb is used we add it to the configuration
        if 'mongodb' in self.args and self.args.mongodb:
            config.update({'mongodb': self.args.mongodb})
            logger.info('Parameter --use-mongodb detected ...')

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
            validate(conf, Constants.CONF_SCHEMA)
            return conf
        except ValidationError as exception:
            logger.fatal(
                'Invalid configuration. See config.json.example. Reason: %s',
                exception
            )
            raise ValidationError(
                best_match(Draft4Validator(Constants.CONF_SCHEMA).iter_errors(conf)).message
            )

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config. Use this method to get the bot config
        :return: Dict: Bot config
        """
        if self.config is None:
            self.config = self.load_config()

        return self.config
