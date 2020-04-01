"""
This module contain functions to load the configuration file
"""
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict

import rapidjson

from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)


CONFIG_PARSE_MODE = rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS


def log_config_error_range(path: str, errmsg: str) -> str:
    """
    Parses configuration file and prints range around error
    """
    if path != '-':
        offsetlist = re.findall(r'(?<=Parse\serror\sat\soffset\s)\d+', errmsg)
        if offsetlist:
            offset = int(offsetlist[0])
            text = Path(path).read_text()
            # Fetch an offset of 80 characters around the error line
            subtext = text[offset-min(80, offset):offset+80]
            segments = subtext.split('\n')
            if len(segments) > 3:
                # Remove first and last lines, to avoid odd truncations
                return '\n'.join(segments[1:-1])
            else:
                return subtext
    return ''


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
    except rapidjson.JSONDecodeError as e:
        err_range = log_config_error_range(path, str(e))
        raise OperationalException(
            f'{e}\n'
            f'Please verify the following segment of your configuration:\n{err_range}'
            if err_range else 'Please verify your configuration file for syntax errors.'
        )

    return config
