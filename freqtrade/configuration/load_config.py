"""
This module contain functions to load the configuration file
"""
import rapidjson
import logging
import sys
from typing import Any, Dict

from freqtrade import OperationalException


logger = logging.getLogger(__name__)


CONFIG_PARSE_MODE = rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS


def load_config_file(path: str) -> Dict[str, Any]:
    """
    Loads a config file from the given path
    :param path: path as str
    :return: configuration as dictionary
    """
    try:
        # Read config from stdin if requested in the options
        with open(path) if path != '-' else sys.stdin as file:
            config = rapidjson.load(file, parse_mode=CONFIG_PARSE_MODE)
    except FileNotFoundError:
        raise OperationalException(
            f'Config file "{path}" not found!'
            ' Please create a config file or check whether it exists.')

    return config
