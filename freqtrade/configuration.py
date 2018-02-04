"""
This module contains the configuration class
"""

import json

from typing import Dict, List, Any
from jsonschema import Draft4Validator, validate
from jsonschema.exceptions import ValidationError, best_match

from freqtrade.constants import Constants
from freqtrade.logger import Logger


class Configuration(object):
    """
    Class to read and init the bot configuration
    Reuse this class for the bot, backtesting, hyperopt and every script that required configuration
    """
    def __init__(self, args: List[str]) -> None:
        self.args = args
        self.logger = Logger(name=__name__).get_logger()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load the bot configuration
        :return: Configuration dictionary
        """
        config = self._load_config_file(self.args.config)

        # Add the strategy file to use
        config.update({'strategy': self.args.strategy})

        # Add dynamic_whitelist if found
        if self.args.dynamic_whitelist:
            config.update({'dynamic_whitelist': self.args.dynamic_whitelist})

        # Add dry_run_db if found and the bot in dry run
        if self.args.dry_run_db and config.get('dry_run', False):
            config.update({'dry_run_db': True})

        return config

    def _load_config_file(self, path: str) -> Dict[str, Any]:
        """
        Loads a config file from the given path
        :param path: path as str
        :return: configuration as dictionary
        """
        with open(path) as file:
            conf = json.load(file)

        if 'internals' not in conf:
            conf['internals'] = {}
        self.logger.info('Validating configuration ...')

        return self._validate_config(conf)

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
            self.logger.fatal(
                'Invalid configuration. See config.json.example. Reason: %s',
                exception
            )
            raise ValidationError(
                best_match(Draft4Validator(Constants.CONF_SCHEMA).iter_errors(conf)).message
            )

    def show_info(self) -> None:
        """
        Display info message to user depending of the configuration of the bot
        :return: None
        """
        if self.config.get('dynamic_whitelist', False):
            self.logger.info(
                'Using dynamically generated whitelist. (--dynamic-whitelist detected)'
            )

        if self.config.get('dry_run_db', False):
            if self.config.get('dry_run', False):
                self.logger.info(
                    'Dry_run will use the DB file: "tradesv3.dry_run.sqlite". '
                    '(--dry_run_db detected)'
                )
            else:
                self.logger.info('Dry run is disabled. (--dry_run_db ignored)')

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config. Use this method to get the bot config
        :return: Dict: Bot config
        """
        return self.config
